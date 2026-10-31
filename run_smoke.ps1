$ErrorActionPreference = 'Stop'

$smokePython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $smokePython -PathType Leaf)) {
    throw 'Missing .venv\Scripts\python.exe. Create .venv with 64-bit Python 3.11 and install requirements-cpu.txt manually using the README commands.'
}

Push-Location -LiteralPath $PSScriptRoot
try {
    Write-Host '[1/4] Checking installed dependencies...'
    & $smokePython -m pip check
    if ($LASTEXITCODE -ne 0) {
        throw 'Dependency check failed. Fix the issues reported by pip in .venv, then rerun .\run_smoke.ps1.'
    }

    Write-Host '[2/4] Verifying the Python environment...'
    & $smokePython verify_smoke.py
    if ($LASTEXITCODE -ne 0) {
        throw 'Environment verification failed. Follow the error above to fix .venv, then rerun .\run_smoke.ps1.'
    }

    Write-Warning 'The first run downloads MNIST into data\ if needed. Training runs three communication rounds on the full MNIST training dataset.'
    Write-Host '[3/4] Running the two-client MNIST/LeNet smoke experiment...'
    & $smokePython resfed_main.py -n 2 -m 3 -c 10 -mn LeNet -ds MNIST -e 1 -bs 64 -uls 0.95 -dls 0.95 -ul 1 -dl 1 -ulp 1 -dlp 1 -qe 1 -rd results\smoke
    if ($LASTEXITCODE -ne 0) {
        throw 'Training failed. Review and resolve the error above before rerunning .\run_smoke.ps1.'
    }

    Write-Host '[4/4] Verifying saved results and saving the plot...'
    & $smokePython verify_smoke.py --results results\smoke --output results\smoke.png
    if ($LASTEXITCODE -ne 0) {
        throw 'Result verification failed. Review the error above and the files in results\smoke before retrying verification.'
    }

    Write-Host 'Smoke check completed. Plot saved to results\smoke.png.'
}
finally {
    Pop-Location
}
