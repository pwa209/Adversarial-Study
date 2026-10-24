$ErrorActionPreference = 'Stop'
$figureRoot = $PSScriptRoot
$rscriptPath = 'C:\Program Files\R\R-4.6.1\bin\Rscript.exe'
$pythonPath = 'C:\Users\Gebruiker\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$env:LC_ALL = 'C'
$env:LANG = 'C'
& $pythonPath (Join-Path $figureRoot 'scripts\prepare_plot_data.py')
if ($LASTEXITCODE -ne 0) { throw 'Source table verification failed.' }
& $rscriptPath --vanilla (Join-Path $figureRoot 'scripts\draw_figures.R') $figureRoot
if ($LASTEXITCODE -ne 0) { throw 'R figure rendering failed.' }
& $pythonPath (Join-Path $figureRoot 'scripts\audit_exports.py')
if ($LASTEXITCODE -ne 0) { throw 'Export audit failed.' }
