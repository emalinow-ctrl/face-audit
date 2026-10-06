# relay-poller.ps1 — store-and-forward command relay for the face-audit PC.
#
# Lets Muse drive this PC (run the pipeline, read logs, diagnose) without an
# interactive session. No admin needed; runs as the current user.
#
#   Sync protocol (conflict-free by construction):
#     - Muse is the ONLY writer of cmd.json. The poller takes it verbatim from
#       the remote every loop (no merges, no local commits of it).
#     - The poller is the ONLY writer of out/. It commits just out/ and
#       rebases before pushing, so simultaneous pushes never conflict.
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

function Git($gArgs) {
  $p = Start-Process git.exe -ArgumentList $gArgs -WorkingDirectory $RelayDir `
       -NoNewWindow -Wait -PassThru `
       -RedirectStandardOutput "$RelayDir\git.out.log" `
       -RedirectStandardError "$RelayDir\git.err.log"
  return $p.ExitCode
}

if (!(Test-Path (Join-Path $RelayDir ".git"))) {
  Log "ERROR: $RelayDir is not a git clone. See tools/relay-setup.md first."
  exit 1
}

Log "relay poller started (branch=$Branch, interval=${IntervalSec}s)"
while ($true) {
  try {
    if ((Git @("fetch", "-q", "origin", $Branch)) -ne 0) {
      Log "fetch failed, retrying next loop"; Start-Sleep -Seconds $IntervalSec; continue
    }
    # Take Muse's cmd.json verbatim — never merge it.
    Git @("checkout", "-q", "origin/$Branch", "--", "cmd.json") | Out-Null
    Git @("reset", "-q") | Out-Null  # keep it out of the index; we never commit it

    $cmdPath = Join-Path $RelayDir "cmd.json"
    if (Test-Path $cmdPath) {
      $cmd = Get-Content $cmdPath -Raw | ConvertFrom-Json
      if ($cmd.id -eq "halt" -and $cmd.status -eq "pending") {
        Log "halt received, stopping."; break
      }
      if ($cmd.id -and $cmd.id -ne $lastId -and $cmd.status -eq "pending") {
        Log "executing $($cmd.id)"
        $scriptFile = Join-Path $RelayDir "run.ps1"
        Set-Content -Path $scriptFile -Value $cmd.script -Encoding utf8
        $outDir = Join-Path $RelayDir "out"
        New-Item -ItemType Directory -Force $outDir | Out-Null
        $outFile = Join-Path $outDir "$($cmd.id).txt"
        $errFile = Join-Path $outDir "$($cmd.id).err.txt"
        $cwd = if ($cmd.cwd) { $cmd.cwd } else { "$env:USERPROFILE\Desktop\audit2" }

        $p = Start-Process powershell.exe -ArgumentList @(
               "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $scriptFile
             ) -WorkingDirectory $cwd -NoNewWindow -Wait -PassThru `
             -RedirectStandardOutput $outFile -RedirectStandardError $errFile

        @{id = $cmd.id; exit_code = $p.ExitCode;
          finished_utc = (Get-Date).ToUniversalTime().ToString("o")} |
          ConvertTo-Json | Set-Content (Join-Path $outDir "$($cmd.id).meta.json")

        # Commit ONLY out/ — never cmd.json (Muse owns that file).
        Git @("add", "--", "out/") | Out-Null
        Git @("commit", "-qm", "done $($cmd.id) exit $($p.ExitCode)", "--", "out/") | Out-Null
        Git @("pull", "-q", "--rebase", "origin", $Branch) | Out-Null
        if ((Git @("push", "-q", "origin", $Branch)) -ne 0) {
          Log "push failed for $($cmd.id); will retry next loop"
        } else {
          Log "done $($cmd.id) (exit $($p.ExitCode))"
          $lastId = $cmd.id
        }
      }
    }
  } catch {
    Log "loop error: $_"
  }
  Start-Sleep -Seconds $IntervalSec
}
Log "relay poller stopped."
