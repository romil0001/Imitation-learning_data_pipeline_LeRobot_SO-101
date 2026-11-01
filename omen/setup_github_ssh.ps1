# Create an Omen→GitHub SSH key and point the repo remote at SSH.
# Run in PowerShell from the repo (or double-click setup_github_ssh.bat):
#   powershell -ExecutionPolicy Bypass -File omen\setup_github_ssh.ps1
#
# Then add the printed public key at:
#   https://github.com/settings/ssh/new
# Title e.g. "omen-windows"

$ErrorActionPreference = "Stop"
$sshDir = Join-Path $env:USERPROFILE ".ssh"
New-Item -ItemType Directory -Force -Path $sshDir | Out-Null

$key = Join-Path $sshDir "id_ed25519_github"
$pub = "$key.pub"

if (-not (Test-Path $key)) {
  Write-Host "Generating $key ..."
  ssh-keygen -t ed25519 -f $key -C "omen-github@$env:COMPUTERNAME" -N '""'
} else {
  Write-Host "Using existing $key"
}

# ssh config for github.com on this Windows user
$cfg = Join-Path $sshDir "config"
$block = @"
Host github.com
  HostName github.com
  User git
  IdentityFile ~/.ssh/id_ed25519_github
  IdentitiesOnly yes
"@
$cfgText = ""
if (Test-Path $cfg) { $cfgText = Get-Content -Raw $cfg }
if ($cfgText -notmatch '(?m)^Host github\.com\s*$') {
  Add-Content -Path $cfg -Value "`r`n$block"
  Write-Host "Appended github.com Host entry to $cfg"
} else {
  Write-Host "github.com Host entry already present in $cfg"
}

Write-Host ""
Write-Host "===== ADD THIS PUBLIC KEY TO GITHUB ====="
Write-Host (Get-Content -Raw $pub).Trim()
Write-Host "========================================="
Write-Host "Open: https://github.com/settings/ssh/new"
Write-Host ""

# Point local clone remotes to SSH if this looks like the SO-101 repo
$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot
try {
  $url = git remote get-url origin 2>$null
  if ($LASTEXITCODE -eq 0 -and $url) {
    Write-Host "Current origin: $url"
    if ($url -match 'github\.com[/:]([^/]+)/([^/.]+)') {
      $owner = $Matches[1]
      $name = $Matches[2]
      $sshUrl = "git@github.com:$owner/$name.git"
      if ($url -ne $sshUrl) {
        git remote set-url origin $sshUrl
        Write-Host "Updated origin -> $sshUrl"
      } else {
        Write-Host "origin already uses SSH."
      }
    }
  } else {
    Write-Host "No git origin in $repoRoot — skip remote rewrite."
  }
} finally {
  Pop-Location
}

Write-Host ""
Write-Host "After adding the key on GitHub, test:"
Write-Host "  ssh -T git@github.com"
Write-Host "  git -C `"$repoRoot`" pull"
Write-Host "From Mac after OpenSSH is up:"
Write-Host "  ssh omen `"cd $repoRoot; git pull; git status`""
