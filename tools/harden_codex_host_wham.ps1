[CmdletBinding()]
param(
    [switch]$VerifyOnly,
    [switch]$Rollback
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$package = Get-ChildItem 'C:\Program Files\WindowsApps' -Directory -Filter 'OpenAI.Codex_*' |
    Sort-Object Name -Descending | Select-Object -First 1
if (-not $package) { throw 'OpenAI Codex package was not found.' }
$asar = Join-Path $package.FullName 'app\resources\app.asar'
if (-not (Test-Path -LiteralPath $asar -PathType Leaf)) { throw "app.asar was not found: $asar" }

$backupRoot = Join-Path $PSScriptRoot '..\work\vendor-backups'
$backupRoot = [IO.Path]::GetFullPath($backupRoot)
$manifest = Join-Path $backupRoot 'codex-host-wham.json'
$needle = 'enabled:!0,placeholderData:i,queryFn:async()=>{try{return(await Ae.safeGet(`/wham/tasks/list`,{parameters:{query:{limit:20,task_filter:`current`}}})).items'
$replacement = $needle.Replace('enabled:!0,', 'enabled:!1,')

function Find-ByteSequence([string]$Path, [byte[]]$Needle) {
    $stream = [IO.File]::OpenRead($Path)
    try {
        $chunkSize = 1MB
        $buffer = New-Object byte[] ($chunkSize + $Needle.Length)
        $carry = 0L
        while (($read = $stream.Read($buffer, $carry, $chunkSize)) -gt 0) {
            $total = $carry + $read
            for ($i = 0; $i -le $total - $Needle.Length; $i++) {
                $match = $true
                for ($j = 0; $j -lt $Needle.Length; $j++) {
                    if ($buffer[$i + $j] -ne $Needle[$j]) { $match = $false; break }
                }
                if ($match) { return ($stream.Position - $read - $carry + $i) }
            }
            $carry = [Math]::Min($Needle.Length - 1, $total)
            [Array]::Copy($buffer, $total - $carry, $buffer, 0, $carry)
            $stream.Position = $stream.Position
        }
        return -1
    } finally { $stream.Dispose() }
}

$original = [Text.Encoding]::UTF8.GetBytes($needle)
$patched = [Text.Encoding]::UTF8.GetBytes($replacement)
$currentHash = (Get-FileHash -LiteralPath $asar -Algorithm SHA256).Hash.ToLowerInvariant()

if ($Rollback) {
    if (-not (Test-Path -LiteralPath $manifest)) { throw "Rollback manifest not found: $manifest" }
    $record = Get-Content -Raw -LiteralPath $manifest | ConvertFrom-Json
    if ($record.asar -ne $asar) { throw 'Rollback manifest points to a different installed package.' }
    if (-not (Test-Path -LiteralPath $record.backup)) { throw "Backup not found: $($record.backup)" }
    Copy-Item -LiteralPath $record.backup -Destination $asar -Force
    [pscustomobject]@{ status = 'ROLLED_BACK'; asar = $asar; backup = $record.backup } | ConvertTo-Json
    exit 0
}

$originalAt = Find-ByteSequence $asar $original
$replacementAt = Find-ByteSequence $asar $patched
if ($originalAt -lt 0 -and $replacementAt -lt 0) { throw 'Current Codex build does not contain the expected current-tasks query fingerprint.' }
if ($VerifyOnly) {
    [pscustomobject]@{
        status = if ($originalAt -ge 0) { 'UNPATCHED' } else { 'PATCHED' }
        package = $package.Name
        asar = $asar
        sha256 = $currentHash
        originalFingerprint = $originalAt -ge 0
        patchedFingerprint = $replacementAt -ge 0
    } | ConvertTo-Json
    exit 0
}
if ($replacementAt -ge 0 -and $originalAt -lt 0) {
    [pscustomobject]@{ status = 'ALREADY_PATCHED'; asar = $asar; sha256 = $currentHash } | ConvertTo-Json
    exit 0
}
if ($originalAt -lt 0 -or $replacementAt -ge 0) { throw 'Unexpected mixed patch state; refusing to modify the vendor archive.' }

New-Item -ItemType Directory -Path $backupRoot -Force | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$backup = Join-Path $backupRoot "app.asar.$stamp.$currentHash.bak"
Copy-Item -LiteralPath $asar -Destination $backup
$stream = [IO.File]::Open($asar, [IO.FileMode]::Open, [IO.FileAccess]::Write, [IO.FileShare]::Read)
try {
    $stream.Position = $originalAt
    $stream.Write($patched, 0, $patched.Length)
    $stream.Flush($true)
} finally { $stream.Dispose() }
$afterHash = (Get-FileHash -LiteralPath $asar -Algorithm SHA256).Hash.ToLowerInvariant()
if ((Find-ByteSequence $asar $original) -ge 0 -or (Find-ByteSequence $asar $patched) -lt 0) {
    Copy-Item -LiteralPath $backup -Destination $asar -Force
    throw 'Post-write fingerprint verification failed; original archive restored.'
}

[pscustomobject]@{
    status = 'PATCHED'
    package = $package.Name
    asar = $asar
    backup = $backup
    beforeSha256 = $currentHash
    afterSha256 = $afterHash
    changedBytes = $patched.Length
    behavior = 'Disable only host current-tasks background query; preserve all other features.'
} | ConvertTo-Json
