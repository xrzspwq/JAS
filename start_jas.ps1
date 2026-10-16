<#
.SYNOPSIS
  Starts llama-swap, Jas and the Jas monitor, each in its own terminal pane.

.EXAMPLE
  .\start_jas.ps1                      # Windows Terminal: Jas on the left, monitor top-right, llama-swap bottom-right
  .\start_jas.ps1 -Layout tabs         # three tabs in one Windows Terminal window
  .\start_jas.ps1 -Layout windows      # three separate PowerShell windows (also used if Windows Terminal is missing)
  .\start_jas.ps1 -NoMonitor           # only llama-swap and Jas
  .\start_jas.ps1 -Port 2004           # llama-swap on another port (Jas and the monitor follow automatically)
  .\start_jas.ps1 -Config D:\x\llama-swap.config.yaml
  .\start_jas.ps1 -Stop                # stop llama-swap and any leftover llama-server processes
  .\start_jas.ps1 -DryRun              # show exactly what would be started, start nothing

  Settings you have exported (JAS_PROXY, SEARXNG_URL, JAS_CTX, ... any JAS_*) are passed to all three terminals.
  Put this file in your Jas folder (next to jas.py). Double-click start_jas.bat if scripts are blocked on your PC.
#>
[CmdletBinding()]
param(
    [ValidateSet("panes", "tabs", "windows")][string]$Layout = "panes",
    [string]$Config,
    [int]$Port = 8080,
    [switch]$NoMonitor,
    [switch]$Stop,
    [switch]$DryRun
)
$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
function Q([string]$s) { "'" + ($s -replace "'", "''") + "'" }     # PowerShell single-quoted literal

# ---------------------------------------------------------------- -Stop
if ($Stop) {
    $n = 0
    foreach ($name in "llama-swap", "llama-server") {
        Get-Process -Name $name -ErrorAction SilentlyContinue | ForEach-Object { $_.Kill(); $n++ }
    }
    Write-Host "Stopped $n process(es)."
    return
}

# ---------------------------------------------------------------- find everything
$Py = @("env", "venv", ".venv") | ForEach-Object { Join-Path $Root (Join-Path $_ "Scripts\python.exe") } |
    Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $Py) { $Py = "python"; Write-Host "No virtual environment (env/venv/.venv) found; using 'python' from PATH." -ForegroundColor Yellow }

if (-not $Config) {
    $Config = @("models\llama-swap.config.yaml", "llama-swap.config.yaml", "models\llama-swap_config.yaml", "llama-swap_config.yaml") |
        ForEach-Object { Join-Path $Root $_ } | Where-Object { Test-Path $_ } | Select-Object -First 1
}
if (-not $Config -or -not (Test-Path $Config)) { throw "llama-swap config not found. Pass it with:  .\start_jas.ps1 -Config <path>" }
$Config = (Resolve-Path $Config).Path

$Swap = (Get-Command llama-swap -ErrorAction SilentlyContinue).Source
if (-not $Swap) {
    $Swap = @("llama-swap.exe", "llama.cpp\llama-swap.exe") | ForEach-Object { Join-Path $Root $_ } |
        Where-Object { Test-Path $_ } | Select-Object -First 1
}
if (-not $Swap) { throw "llama-swap not found. Put llama-swap.exe on your PATH or next to this script." }
foreach ($f in "jas.py") { if (-not (Test-Path (Join-Path $Root $f))) { throw "$f not found in $Root - put this script in your Jas folder." } }

# ---------------------------------------------------------------- what each terminal runs
$envLines = @()
Get-ChildItem Env: | Where-Object { $_.Name -like "JAS_*" -or $_.Name -eq "SEARXNG_URL" } |
    ForEach-Object { $envLines += "`$env:$($_.Name) = $(Q $_.Value)" }
if ($Port -ne 8080) { $envLines += "`$env:JAS_URL = 'http://127.0.0.1:$Port/v1'" }

function New-Script([string]$Title, [string[]]$Body) {
    $lines = @("`$Host.UI.RawUI.WindowTitle = $(Q $Title)", "Set-Location -LiteralPath $(Q $Root)") + $envLines + $Body
    $lines -join "`n"
}

$swapBody = @(
    "& $(Q $Swap) --config $(Q $Config) --listen 127.0.0.1:$Port",
    "Write-Host ('llama-swap exited (code ' + `$LASTEXITCODE + '). Scroll up for the reason.') -ForegroundColor Red"
)
$jasBody = @(
    "`$u = 'http://127.0.0.1:$Port/v1/models'",
    "Write-Host 'Waiting for llama-swap...' -ForegroundColor DarkGray",
    "`$up = `$false",
    "for (`$i = 0; `$i -lt 90 -and -not `$up; `$i++) {",
    "    try { Invoke-WebRequest -UseBasicParsing -Uri `$u -TimeoutSec 2 | Out-Null; `$up = `$true }",
    "    catch { if (`$_.Exception.Response) { `$up = `$true } else { Start-Sleep -Seconds 1 } }",
    "}",
    "if (-not `$up) { Write-Host 'llama-swap did not answer on port $Port - check its pane for errors.' -ForegroundColor Red }",
    "& $(Q $Py) jas.py"
)
$monBody = @("& $(Q $Py) jas_monitor.py")

$terms = @(
    [pscustomobject]@{ Title = "Jas"; Script = New-Script "Jas" $jasBody },
    [pscustomobject]@{ Title = "llama-swap"; Script = New-Script "llama-swap" $swapBody }
)
if (-not $NoMonitor) { $terms += [pscustomobject]@{ Title = "Monitor"; Script = New-Script "Jas monitor" $monBody } }
foreach ($t in $terms) { $t | Add-Member Enc ([Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($t.Script))) }
function T([string]$title) { $terms | Where-Object Title -eq $title }

# ---------------------------------------------------------------- launch
$wt = Get-Command wt.exe -ErrorAction SilentlyContinue
if ($Layout -ne "windows" -and -not $wt -and -not $DryRun) {
    Write-Host "Windows Terminal (wt.exe) not found; opening separate windows instead." -ForegroundColor Yellow
    $Layout = "windows"
}
function Pane([string]$verb, [string]$title, [string]$enc) {
    "$verb --title `"$title`" --suppressApplicationTitle powershell.exe -NoLogo -NoExit -EncodedCommand $enc"
}

if ($Layout -eq "windows") {
    $plan = "3 separate PowerShell windows"
} elseif ($Layout -eq "tabs") {
    $plan = "one Windows Terminal window, tabs: " + (($terms | ForEach-Object Title) -join ", ")
    $parts = $terms | ForEach-Object { Pane "new-tab" $_.Title $_.Enc }
    $wtArgs = ($parts -join " ; ") + " ; focus-tab -t 0"
} else {
    $plan = "one Windows Terminal window, panes: Jas (left) | " + $(if ($NoMonitor) { "llama-swap (right)" } else { "monitor (top right), llama-swap (bottom right)" })
    $parts = @((Pane "new-tab" "Jas" (T "Jas").Enc))
    if ($NoMonitor) {
        $parts += (Pane "split-pane -V" "llama-swap" (T "llama-swap").Enc)
    } else {
        $parts += (Pane "split-pane -V" "Monitor" (T "Monitor").Enc)
        $parts += (Pane "split-pane -H -s 0.4" "llama-swap" (T "llama-swap").Enc)
    }
    $wtArgs = ($parts -join " ; ") + " ; focus-pane -t 0"
}

Write-Host "Folder     : $Root"
Write-Host "Python     : $Py"
Write-Host "llama-swap : $Swap"
Write-Host "Config     : $Config"
Write-Host "Address    : http://127.0.0.1:$Port"
Write-Host "Layout     : $plan"
if ($envLines.Count) { Write-Host "Passing on : $(($envLines | ForEach-Object { ($_ -split ' = ')[0] -replace '^\$env:', '' }) -join ', ')" }

if ($DryRun) {
    foreach ($t in $terms) { Write-Host "`n--- $($t.Title) runs: ---" -ForegroundColor Cyan; Write-Host $t.Script }
    if ($wtArgs) { Write-Host "`n--- wt.exe arguments (script bodies shortened): ---" -ForegroundColor Cyan; Write-Host ($wtArgs -replace "-EncodedCommand \S+", "-EncodedCommand <...>") }
    return
}

if ($Layout -eq "windows") {
    foreach ($t in $terms) {
        Start-Process powershell.exe -ArgumentList @("-NoLogo", "-NoExit", "-EncodedCommand", $t.Enc)
        Start-Sleep -Milliseconds 300
    }
} else {
    Start-Process -FilePath $wt.Source -ArgumentList $wtArgs
}
Write-Host "`nStarted. In the Jas pane, wait for 'Warming up ... ready' and then type your first message." -ForegroundColor Green
Write-Host "To stop everything: close the panes (Ctrl+C in llama-swap first), or run  .\start_jas.ps1 -Stop"
