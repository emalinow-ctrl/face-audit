# relay-poller-v2.ps1 — hardened command relay for the face-audit PC.
#
# Improvements over v1:
#   - Command timeout: kills hung commands after 30 min (configurable per-command
#     via cmd.json "timeout_min"). No more blocked queue.
#   - Heartbeat: writes out/heartbeat.json every 5 min so Muse can verify the
#     poller is alive via `git show origin/relay:out/heartbeat.json`.
#   - Git recovery: if pull fails 3x in a row, does `git reset --hard` to recover
#     from corrupted state (e.g., unstaged changes blocking rebase).
#
#   Start: powershell -NoProfile -ExecutionPolicy Bypass -File tools\relay-poller-v2.ps1
#   Stop: kill the process, or issue {"id":"halt"}.

param(
  [string]$RelayDir = "$env:USERPROFILE\relay",
  [string]$Branch = "relay",
  [int]$IntervalSec = 20,
  [int]$DefaultTimeoutMin = 30
)

$ErrorActionPreference = "Continue"
$lastId = ""
$pullFailCount = 0
$loopCount = 0

function Log($m) {
  $line = "[$(Get-Date -Format 'HH:mm:ss')] $m"
  $line | Out-File -FilePath (Join-Path $RelayDir "poller.log") -Append -Encoding utf8
  Write-Host $line
}

function Git([string[]]$gArgs) {
  $argLine = ($gArgs | ForEach-Object { '"' + ($_ -replace '"', '\"') + '"' }) -join ' '
  $p = Start-Process git.exe -ArgumentList $argLine -WorkingDirectory $RelayDir `
       -NoNewWindow -Wait -PassThru `
       -RedirectStandardOutput "$RelayDir\git.out.log" `
       -RedirectStandardError "$RelayDir\git.err.log"
  return $p.ExitCode
}

function Write-Heartbeat($lastCmd) {
  $hb = @{
    ts = (Get-Date).ToUniversalTime().ToString("o")
    poller_pid = $PID
    last_cmd = $lastCmd
    version = "v2"
  } | ConvertTo-Json -Compress
  $hb | Out-File -FilePath (Join-Path $RelayDir "out\heartbeat.json") -Encoding utf8 -NoNewline
}

if (!(Test-Path (Join-Path $RelayDir ".git"))) {
  Log "ERROR: $RelayDir is not a git clone."
  exit 1
}

Git @("config", "credential.helper", "") | Out-Null
Git @("config", "user.name", "faceaudit-relay") | Out-Null
Git @("config", "user.email", "relay@local") | Out-Null
Git @("reset", "-q") | Out-Null

Log "relay poller v2 started (branch=$Branch, interval=${IntervalSec}s, timeout=${DefaultTimeoutMin}min)"
while ($true) {
  try {
    $loopCount++
    Git @("checkout", "-q", "--", "cmd.json") | Out-Null
    if ((Git @("fetch", "-q", "origin", $Branch)) -ne 0) {
      Log "fetch failed, retrying next loop"; Start-Sleep -Seconds $IntervalSec; continue
    }
    if ((Git @("pull", "-q", "--rebase", "origin", $Branch)) -ne 0) {
      $pullFailCount++
      Log "pull failed (attempt $pullFailCount), retrying next loop"
      if ($pullFailCount -ge 3) {
        Log "pull failed 3x, resetting hard to recover"
        Git @("reset", "--hard", "origin/$Branch") | Out-Null
        Git @("clean", "-fd") | Out-Null
        $pullFailCount = 0
      }
      Start-Sleep -Seconds $IntervalSec; continue
    }
    $pullFailCount = 0
    Git @("push", "-q", "origin", $Branch) | Out-Null

    & git.exe -C $RelayDir show "origin/${Branch}:cmd.json" 2>$null |
      Set-Content (Join-Path $RelayDir "cmd.json") -Encoding utf8

    $cmdPath = Join-Path $RelayDir "cmd.json"
    $cmd = Get-Content $cmdPath -Raw | ConvertFrom-Json
    if ($cmd.id -eq "halt" -and $cmd.status -eq "pending") {
      Log "halt received, stopping."; break
    }

    $outDir = Join-Path $RelayDir "out"
    New-Item -ItemType Directory -Force $outDir | Out-Null
    $metaPath = Join-Path $outDir "$($cmd.id).meta.json"

    # Heartbeat every 15 loops (~5 min)
    if ($loopCount % 15 -eq 0) {
      Write-Heartbeat $lastId
      Git @("add", "--", "out/heartbeat.json") | Out-Null
      Git @("commit", "-qm", "heartbeat", "--", "out/heartbeat.json") | Out-Null
      Git @("push", "-q", "origin", $Branch) | Out-Null
    }

    if ($cmd.id -and $cmd.id -ne $lastId -and $cmd.status -eq "pending") {
      $exitNote = ""
      if (Test-Path $metaPath) {
        Log "$($cmd.id) already has a result locally; ensuring it is pushed"
        $exitNote = "repush"
      } else {
        Log "executing $($cmd.id)"
        $timeoutMin = if ($cmd.timeout_min) { [int]$cmd.timeout_min } else { $DefaultTimeoutMin }
        $scriptFile = Join-Path $RelayDir "run.ps1"
        Set-Content -Path $scriptFile -Value $cmd.script -Encoding utf8
        $outFile = Join-Path $outDir "$($cmd.id).txt"
        $errFile = Join-Path $outDir "$($cmd.id).err.txt"
        $cwd = if ($cmd.cwd) { $cmd.cwd } else { "$env:USERPROFILE\Desktop\audit2" }

        $argLine = (@("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $scriptFile) |
          ForEach-Object { '"' + ($_ -replace '"', '\"') + '"' }) -join ' '
        $p = Start-Process powershell.exe -ArgumentList $argLine `
             -WorkingDirectory $cwd -NoNewWindow -PassThru `
             -RedirectStandardOutput $outFile -RedirectStandardError $errFile
        $timeoutMs = $timeoutMin * 60 * 1000
        if ($p.WaitForExit($timeoutMs)) {
          $exitNote = "exit $($p.ExitCode)"
          Log "finished $($cmd.id) ($exitNote)"
        } else {
          try { $p.Kill() } catch { }
          $exitNote = "TIMEOUT after ${timeoutMin}min (killed)"
          Log "$($cmd.id) $exitNote"
          "COMMAND TIMED OUT after ${timeoutMin} minutes and was killed." |
            Out-File -FilePath $outFile -Append -Encoding utf8
        }
      }
      $lastId = $cmd.id
      Write-Heartbeat $lastId
      Git @("add", "--", "out/") | Out-Null
      Git @("commit", "-qm", "done $($cmd.id) $exitNote", "--", "out/") | Out-Null
      if ((Git @("push", "-q", "origin", $Branch)) -ne 0) {
        Log "push failed for $($cmd.id); result stranded, will retry"
      } else {
        Log "pushed result for $($cmd.id)"
      }
    }
  } catch {
    Log "loop error: $_"
  }
  Start-Sleep -Seconds $IntervalSec
}
Log "relay poller stopped."
