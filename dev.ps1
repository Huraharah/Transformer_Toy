# Move to project root
Set-Location $PSScriptRoot

# Deactivate Conda if active
if ($env:CONDA_DEFAULT_ENV) {
    conda deactivate
}

# Activate local venv
& "$PSScriptRoot\.venv\Scripts\Activate.ps1"

# Make local package importable
$env:PYTHONPATH = "$PSScriptRoot\python;$env:PYTHONPATH"

Write-Host ""
Write-Host "Transformer_Toy environment ready." -ForegroundColor Green
Write-Host "Python:" (Get-Command python).Source
Write-Host "PYTHONPATH:" $env:PYTHONPATH
Write-Host ""