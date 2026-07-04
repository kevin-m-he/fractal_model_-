# Creates a "Fractal Model" shortcut on the Desktop pointing at the launcher.
# Run from anywhere:  powershell -ExecutionPolicy Bypass -File scripts\create_desktop_shortcut.ps1
$repo = Split-Path -Parent $PSScriptRoot
$desktop = [Environment]::GetFolderPath('Desktop')
$ws = New-Object -ComObject WScript.Shell
$lnk = $ws.CreateShortcut((Join-Path $desktop 'Fractal Model.lnk'))
$lnk.TargetPath = Join-Path $repo 'run_fractal_model.bat'
$lnk.WorkingDirectory = $repo
$lnk.Description = 'Fractal Model - self-similar valuation structure explorer'
$icon = Join-Path $repo 'assets\fractal_model.ico'
if (Test-Path $icon) { $lnk.IconLocation = $icon }
$lnk.Save()
Write-Host "Shortcut created: $(Join-Path $desktop 'Fractal Model.lnk')"
