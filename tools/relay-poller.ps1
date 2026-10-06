# relay-poller.ps1 — store-and-forward command relay for the face-audit PC.
#
# Lets Muse drive this PC (run the pipeline, read logs, diagnose) without an
# interactive session. No admin needed; runs as the current user.
#
#   How it works: every $IntervalSec seconds it pulls the `relay` branch.
#   When cmd.json appears with status "pending", it runs the embedded PowerShell
#   script, captures stdout/stderr + exit code into out/, marks it done, pushes.
#
#   Start it once (survives SSH disconnect as long as this Windows session lives):
#     powershell -NoProfile -ExecutionPolicy Bypass -File tools\relay-poller.ps1
#   Stop it: Ctrl+C in its window, or kill the powershell process.
#   Revoke it: revoke the GitHub token and/or delete the `relay` branch.
#
#   A command {"id":"halt","status":"pending"} stops the poller cleanly.

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

if (!(Test-Path (Join-Path $RelayDir ".git"))) {
  Log "ERROR: $RelayDir is not a git clone. See tools/relay-setup.md first."
  exit 1
}

Log "relay poller started (branch=$Branch, interval=${IntervalSec}s)"
while ($true) {
  try {
    git -C $RelayDir fetch -q origin $Branch 2>$null
    git -C $RelayDir checkout -q $Branch 2>$null
    git -C $RelayDir pull -q --ff-only origin $Branch 2>$null

    $cmdPath = Join-Path $RelayDir "cmd.json"
    if (Test-Path $cmdPath) {
      $cmd = Get-Content $cmdPath -Raw | ConvertFrom-Json
      if ($cmd.id -eq "halt" -and $cmd.status -eq "pending") {
        Log "halt received, stopping."
        $cmd.status = "done" | Out-Null
        @{id = "halt"; status = "done"; script = ""; cwd = ""} |
          ConvertTo-Json | Set-Content $cmdPath
        git -C $RelayDir commit -qam "halt" 2>$null
        git -C $RelayDir push -q origin $Branch 2>$null
        break
      }
      if ($cmd.id -ne $lastId -and $cmd.status -eq "pending") {
        Log "executing $($cmd.id)"
        @{id = $cmd.id; status = "running"; script = $cmd.script; cwd = $cmd.cwd} |
          ConvertTo-Json -Depth 5 | Set-Content $cmdPath
        git -C $RelayDir commit -qam "running $($cmd.id)" 2>$null
        git -C $RelayDir push -q origin $Branch 2>$null

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

        @{id = $cmd.id; status = "done"; script = $cmd.script; cwd = $cmd.cwd} |
          ConvertTo-Json -Depth 5 | Set-Content $cmdPath
        git -C $RelayDir add -A 2>$null
        git -C $RelayDir commit -qm "done $($cmd.id) exit $($p.ExitCode)" 2>$null
        git -C $RelayDir push -q origin $Branch 2>$null
        Log "done $($cmd.id) (exit $($p.ExitCode))"
        $lastId = $cmd.id
      }
    }
  } catch {
    Log "loop error: $_"
  }
  Start-Sleep -Seconds $IntervalSec
}
Log "relay poller stopped."
