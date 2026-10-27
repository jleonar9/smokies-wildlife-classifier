# Step 1 on your PC: install packages, run the tests, and build the photo list from iNaturalist (~3 min).
#   Set-ExecutionPolicy -Scope Process Bypass -Force; & "$HOME\Documents\portfolio\04-smokies-classifier\run.ps1"
# Training runs on a free Google Colab GPU afterwards (notebooks\train_colab.ipynb).
Set-Location $PSScriptRoot
$py = "$PSScriptRoot\..\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) { Write-Host "Can't find the portfolio Python environment at $py" -ForegroundColor Red; exit 1 }
& $py -m pip install -q timm requests matplotlib
if ($LASTEXITCODE -ne 0) { Write-Host "Package install failed. Copy the error above to Claude." -ForegroundColor Red; exit 1 }

foreach ($t in @("tests\test_data.py", "tests\test_model.py")) {
    $out = & $py $t 2>&1
    if ($LASTEXITCODE -ne 0) { $out; Write-Host "`n$t failed. Copy the error above to Claude." -ForegroundColor Red; exit 1 }
    Write-Host ($out | Select-Object -Last 1)
}

& $py src\build_manifest.py
if ($LASTEXITCODE -ne 0) { Write-Host "`nBuilding the photo list failed. Copy the error above to Claude." -ForegroundColor Red; exit 1 }
Write-Host "`nAll set. Tell Claude 'done' so it can check the photo list." -ForegroundColor Green
