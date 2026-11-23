param(
    [ValidateSet('test', 'train', 'report')] [string] $Action = 'test',
    [ValidateSet('vanilla', 's1', 'vanilla-ft', 's2')] [string] $Method = 's2',
    [string] $TorchPython = 'python',
    [string] $Distribution = 'Ubuntu'
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$runtimeDir = Join-Path (Split-Path -Parent $PSScriptRoot) 'runtime'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:TEMP = Join-Path $runtimeDir 'temp'
$env:TMP = $env:TEMP
$env:MPLCONFIGDIR = Join-Path $runtimeDir 'cache\matplotlib'
if ($Action -eq 'report') {
    & $TorchPython (Join-Path $PSScriptRoot 'report.py')
    exit $LASTEXITCODE
}
$root = [IO.Path]::GetFullPath($PSScriptRoot).Replace('\', '/')
if ($root -notmatch '^([a-zA-Z]):/(.+)$') { throw 'A local drive path is required.' }
$linuxRoot = '/mnt/' + $Matches[1].ToLowerInvariant() + '/' + $Matches[2]
if (-not (Test-Path -LiteralPath (Join-Path $runtimeDir 'python-memory.sh'))) {
    throw 'Local Jittor launcher missing. See README.md for a fresh Linux environment.'
}
$launcher = $linuxRoot + '/../runtime/python-memory.sh'
$extra = @()
if ($Action -eq 'test') { $extra += '--test-only' }
& wsl.exe -d $Distribution --exec env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 bash $launcher ($linuxRoot + '/train.py') --framework jittor --method $Method @extra
exit $LASTEXITCODE
