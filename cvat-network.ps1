param(
    [Parameter(Position = 0)]
    [ValidateSet("start", "stop", "status")]
    [string]$Action
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

function Show-Usage {
    Write-Host "Usage: .\cvat-network.ps1 start | stop | status"
    Write-Host "  start   let other computers open CVAT on this PC"
    Write-Host "  stop    keep CVAT only on this PC (localhost)"
    Write-Host "  status  show which address port 8080 is using"
}

if (-not $Action) {
    Show-Usage
    exit 1
}

if ($Action -eq "status") {
    docker port traefik 8080
    exit $LASTEXITCODE
}

$bind = "127.0.0.1"
if ($Action -eq "start") {
    $bind = "0.0.0.0"
}

$envPath = Join-Path $PSScriptRoot ".env"
$kept = @()
if (Test-Path $envPath) {
    $kept = @(Get-Content $envPath | Where-Object { $_ -notmatch '^\s*CVAT_BIND=' })
}
Set-Content -Path $envPath -Value (@($kept) + "CVAT_BIND=$bind") -Encoding ascii
$env:CVAT_BIND = $bind

docker compose -f docker-compose.yml -f components/serverless/docker-compose.serverless.yml up -d traefik
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
docker compose -f analyzer-service/docker-compose.analyzer.yml up -d analyzer
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if ($Action -eq "start") {
    $lan = Get-NetIPAddress -AddressFamily IPv4 |
        Where-Object { $_.IPAddress -like "192.168.*" } |
        Select-Object -First 1 -ExpandProperty IPAddress
    if ($lan) {
        Write-Host "Network host is on. Other computers can open http://${lan}:8080"
    } else {
        Write-Host "Network host is on. Other computers can open http://<this-pc-ip>:8080"
    }
} else {
    Write-Host "Network host is off. CVAT stays at http://localhost:8080 on this PC."
}
