# relay-poller.ps1 — store-and-forward command relay for the face-audit PC.
#
# Lets Muse drive this PC (run the pipeline, read logs, diagnose) without an
# interactive session. No admin needed; runs as the current user.
#
#   Sync protocol (conflict-free by construction):
#     - Muse is the ONLY writer of cmd.json. The poller restores a clean tree,
#       pulls, then takes cmd.json verbatim from the remote every loop.
#     - The poller is the ONLY writer of out/. It commits just out/ and pushes.
#
#   Start it once (survives SSH disconnect while this Windows session lives):
#     powershell -NoProfile -ExecutionPolicy Bypass -File tools\relay-poller.ps1
#   Stop it: Ctrl+C in its window, kill the process, or issue {"id":"halt"}.
#   Revoke it: revoke the GitHub token and/or delete the `relay` branch.

param(
  [string]$RelayDir = "$env:USERPROFILE\relay",
  [string]$Branch = "relay",
  [int]$IntervalSec = 20
)

$ErrorActionPreference = "Continue"
$lastId = ""

function Log($m) {
  $line = "[$(Get-Date -Format 'HH:mm:ss')] $m"
  $line | Out-File -FilePath (Join-Path $RelayDir "poller.log") -Append -Encoding utf8
  Write-Host $line
}

# PS 5.1's Start-Process mangles array args containing spaces, so build one
# properly-quoted command line instead.
function Git([string[]]$gArgs) {
  $argLine = ($gArgs | ForEach-Object { '"' + ($_ -replace '"', '\"') + '"' }) -join ' '
  $p = Start-Process git.exe -ArgumentList $argLine -WorkingDirectory $RelayDir `
       -NoNewWindow -Wait -PassThru `
       -RedirectStandardOutput "$RelayDir\git.out.log" `
       -RedirectStandardError "$RelayDir\git.err.log"
  return $p.ExitCode
}

if (!(Test-Path (Join-Path $RelayDir ".git"))) {
  Log "ERROR: $RelayDir is not a git clone. See tools/relay-setup.md first."
  exit 1
}

# The clone URL already carries the token; the Windows credential manager only
# adds noise ("Unable to persist credentials") — disable it for this clone.
Git @("config", "credential.helper", "") | Out-Null

Log "relay poller started (branch=$Branch, interval=${IntervalSec}s)"
while ($true) {
  try {
    # 1. Clean tree, then sync. (A dirty cmd.json from last loop must not block
    #    the pull — restore it first.)
    Git @("checkout", "-q", "--", "cmd.json") | Out-Null
    if ((Git @("fetch", "-q", "origin", $Branch)) -ne 0) {
      Log "fetch failed, retrying next loop"; Start-Sleep -Seconds $IntervalSec; continue
    }
    if ((Git @("pull", "-q", "--rebase", "origin", $Branch)) -ne 0) {
      Log "pull failed, retrying next loop"; Start-Sleep -Seconds $IntervalSec; continue
    }
    Git @("push", "-q", "origin", $Branch) | Out-Null  # flush any stranded results

    # 2. Take Muse's cmd.json verbatim from the remote (never merge, never commit it).
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

    if ($cmd.id -and $cmd.id -ne $lastId -and $cmd.status -eq "pending") {
      $exitNote = ""
      if (Test-Path $metaPath) {
        Log "$($cmd.id) already has a result locally; ensuring it is pushed"
        $exitNote = "repush"
      } else {
        Log "executing $($cmd.id)"
        $scriptFile = Join-Path $RelayDir "run.ps1"
        Set-Content -Path $scriptFile -Value $cmd.script -Encoding utf8
        $outFile = Join-Path $outDir "$($cmd.id).txt"
        $errFile = Join-Path $outDir "$($cmd.id).err.txt"
        $cwd = if ($cmd.cwd) { $cmd.cwd } else { "$env:USERPROFILE\Desktop\audit2" }

        $argLine = (@("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $scriptFile) |
          ForEach-Object { '"' + ($_ -replace '"', '\"') + '"' }) -join ' '
        $p = Start-Process powershell.exe -ArgumentList $argLine `
             -WorkingDirectory $cwd -NoNewWindow -Wait -PassThru `
             -RedirectStandardOutput $outFile -RedirectStandardError $errFile
        $exitNote = "exit $($p.ExitCode)"
        Log "finished $($cmd.id) ($exitNote)"
      }
      $lastId = $cmd.id
      # Commit ONLY out/ — never cmd.json (Muse owns that file).
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
