# Enable OpenSSH Server on this Omen PC and authorize the Mac key.
# Run once (Admin PowerShell recommended):
#   powershell -ExecutionPolicy Bypass -File omen\setup_openssh.ps1
#
# After this, from the Mac:
#   ssh omen
#   ssh omen "cd Developer\SO101_Visuomotor_Manipulation && git pull"

$ErrorActionPreference = "Stop"
$RepoSshDir = Join-Path $PSScriptRoot "ssh"
$PubKeyFile = Join-Path $RepoSshDir "mac_to_omen.pub"

Write-Host "=== OpenSSH Server setup ==="

$cap = Get-WindowsCapability -Online | Where-Object Name -like 'OpenSSH.Server*'
if (-not $cap -or $cap.State -ne 'Installed') {
  Write-Host "Installing OpenSSH.Server..."
  Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0 | Out-Null
} else {
  Write-Host "OpenSSH.Server already installed."
}

Start-Service sshd
Set-Service -Name sshd -StartupType Automatic
if (Get-Service ssh-agent -ErrorAction SilentlyContinue) {
  Set-Service -Name ssh-agent -StartupType Automatic
  Start-Service ssh-agent -ErrorAction SilentlyContinue
}

$fw = Get-NetFirewallRule -Name "OpenSSH-Server-In-TCP" -ErrorAction SilentlyContinue
if (-not $fw) {
  New-NetFirewallRule -Name "OpenSSH-Server-In-TCP" `
    -DisplayName "OpenSSH Server (sshd)" `
    -Enabled True -Direction Inbound -Protocol TCP -Action Allow -LocalPort 22 | Out-Null
  Write-Host "Firewall rule added for TCP 22."
} else {
  Enable-NetFirewallRule -Name "OpenSSH-Server-In-TCP"
  Write-Host "Firewall rule OpenSSH-Server-In-TCP enabled."
}

$sshDir = Join-Path $env:USERPROFILE ".ssh"
New-Item -ItemType Directory -Force -Path $sshDir | Out-Null
$auth = Join-Path $sshDir "authorized_keys"

if (-not (Test-Path $PubKeyFile)) {
  throw "Missing $PubKeyFile — pull/sync the repo so omen\ssh\mac_to_omen.pub exists."
}
$pub = (Get-Content -Raw $PubKeyFile).Trim()
if (-not $pub.StartsWith("ssh-")) {
  throw "Public key file does not look like an OpenSSH public key: $PubKeyFile"
}

$existing = ""
if (Test-Path $auth) {
  $existing = Get-Content -Raw $auth
}
if ($existing -notlike "*$pub*") {
  Add-Content -Path $auth -Value $pub
  Write-Host "Appended Mac pubkey to $auth"
} else {
  Write-Host "Mac pubkey already present in authorized_keys."
}

# Restrict ACL on authorized_keys (required by Windows OpenSSH)
icacls $auth /inheritance:r | Out-Null
icacls $auth /grant:r "$($env:USERNAME):(R)" | Out-Null
icacls $sshDir /inheritance:r | Out-Null
icacls $sshDir /grant:r "$($env:USERNAME):(OI)(CI)(F)" | Out-Null

# Optional: default shell PowerShell
$shellKey = "HKLM:\SOFTWARE\OpenSSH"
if (Test-Path $shellKey) {
  New-ItemProperty -Path $shellKey -Name DefaultShell `
    -Value "C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe" `
    -PropertyType String -Force | Out-Null
}

$ip = (Get-NetIPAddress -AddressFamily IPv4 |
  Where-Object { $_.IPAddress -notlike '127.*' -and $_.PrefixOrigin -ne 'WellKnown' } |
  Select-Object -First 1 -ExpandProperty IPAddress)
Write-Host ""
Write-Host "DONE. From the Mac run:"
Write-Host "  ssh omen"
Write-Host "Expected HostName in Mac ~/.ssh/config: 192.168.0.59 (update if IP is now $ip)"
Write-Host "Then set up GitHub SSH on this PC:"
Write-Host "  powershell -ExecutionPolicy Bypass -File omen\setup_github_ssh.ps1"
