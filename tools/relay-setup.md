# Relay setup (PC side, one time, no admin needed)

Gives Muse autonomous command execution on this PC via the `relay` branch.
Round-trip is ~30–60s per command. Kill it anytime: Ctrl+C the poller window,
or `taskkill /F /IM powershell.exe` (careful: kills all PowerShell), or revoke
the GitHub token / delete the `relay` branch.

## 1. Clone the relay branch (needs the repo token once)

In PowerShell (regular, not admin):

```powershell
# The token below is YOUR fine-grained PAT (repo-scoped). It lives only in this
# clone's local git config on YOUR machine. Rotate it whenever you like:
# GitHub -> Settings -> Developer settings -> Personal access tokens.
$pat = 'PASTE_YOUR_TOKEN_HERE'
git clone --branch relay "https://x-access-token:${pat}@github.com/emalinow-ctrl/face-audit.git" "$env:USERPROFILE\relay"
```

## 2. Start the poller (detached, survives SSH disconnect)

The relay branch carries its own copy of the poller, so the relay clone is
self-contained:

```powershell
Start-Process powershell.exe -ArgumentList "-NoProfile","-ExecutionPolicy","Bypass","-File","$env:USERPROFILE\relay\tools\relay-poller.ps1" -WorkingDirectory "$env:USERPROFILE\relay"
```

(The poller defaults to running commands in `Desktop\audit2` — override
per-command with the `cwd` field.)

## 3. Optional: start automatically at logon (no admin needed)

```powershell
$act = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$env:USERPROFILE\Desktop\audit2\tools\relay-poller.ps1`""
$trg = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
Register-ScheduledTask -TaskName "FaceAuditRelay" -Action $act -Trigger $trg -Description "face-audit command relay (user-level, revocable)"
```

Remove with: `Unregister-ScheduledTask -TaskName "FaceAuditRelay" -Confirm:$false`

## Protocol (for Muse)

- Issue: write `cmd.json` = `{"id":"cmd-004","status":"pending","cwd":"...","script":"..."}`, commit **only** `cmd.json`, pull --rebase, push `relay`.
- Wait: fetch until `out/<id>.meta.json` exists, then read `out/<id>.txt`, `out/<id>.err.txt`, `out/<id>.meta.json`.
- Never write to `out/` (the poller owns it); the poller never writes `cmd.json` (you own it). This is what keeps pushes conflict-free.
- Stop the poller: issue `{"id":"halt","status":"pending","script":"","cwd":""}`.
- Keep scripts self-contained; prefer absolute paths; don't assume admin.
