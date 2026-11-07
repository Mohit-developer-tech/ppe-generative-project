# scripts/setup.ps1
# Setup script for PPE Generative AI Benchmark & Compliance Platform

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " PPE Generative AI Benchmark: Automated Environment Setup " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Check Python
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    Write-Error "Python 3 is required but was not found in PATH."
    exit 1
}
Write-Host "[+] Found Python: $($pythonCmd.Source)" -ForegroundColor Green

# 2. Virtual Environment
if (-not (Test-Path ".venv")) {
    Write-Host "[*] Creating virtual environment (.venv)..." -ForegroundColor Yellow
    python -m venv .venv
} else {
    Write-Host "[+] Existing virtual environment (.venv) detected." -ForegroundColor Green
}

# 3. Virtual Environment Python & Pip
$venvPython = ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    $venvPython = "python"
}

Write-Host "[*] Upgrading pip, setuptools, wheel..." -ForegroundColor Yellow
& $venvPython -m pip install --upgrade pip setuptools wheel --quiet

# 4. Install Dependencies
if (Test-Path "dashboard/requirements.txt") {
    Write-Host "[*] Installing dashboard and core dependencies..." -ForegroundColor Yellow
    & $venvPython -m pip install -r dashboard/requirements.txt --quiet
}

Write-Host "[*] Installing development & pre-commit tooling..." -ForegroundColor Yellow
& $venvPython -m pip install pytest nbformat nbstripout pre-commit pyyaml --quiet

# 5. Pre-commit Installation
if (Test-Path ".pre-commit-config.yaml") {
    Write-Host "[*] Installing pre-commit git hooks..." -ForegroundColor Yellow
    & ".venv\Scripts\pre-commit.exe" install
}

Write-Host ""
Write-Host "==========================================================" -ForegroundColor Green
Write-Host " Environment setup complete! " -ForegroundColor Green
Write-Host " Next steps: " -ForegroundColor Cyan
Write-Host "   1. Prepare data:    python scripts/prepare_data.py --from-zip crops_64.zip" -ForegroundColor White
Write-Host "   2. Fetch assets:    python scripts/fetch_assets.py --from-folder release_assets" -ForegroundColor White
Write-Host "   3. Run tests:       python -m pytest tests/" -ForegroundColor White
Write-Host "   4. Start dashboard: uvicorn dashboard.app:app --reload" -ForegroundColor White
Write-Host "==========================================================" -ForegroundColor Green

