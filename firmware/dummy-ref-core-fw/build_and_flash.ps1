param(
    [string]$BuildType = "Debug",
    [string]$BuildDir = "build",
    [switch]$SkipFlash,
    [switch]$Reconfigure
)

$ErrorActionPreference = "Stop"

function Assert-Exists {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PathToCheck,
        [Parameter(Mandatory = $true)]
        [string]$What
    )
    if (-not (Test-Path -LiteralPath $PathToCheck)) {
        throw "$What not found: $PathToCheck"
    }
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptDir "..\..")
$openocdExe = Join-Path $repoRoot "tools\openocd\xpack-openocd-0.12.0-7\bin\openocd.exe"
$openocdScripts = Join-Path $repoRoot "tools\openocd\xpack-openocd-0.12.0-7\openocd\scripts"
$stlinkCfg = Join-Path $scriptDir "stlink.cfg"
$buildPath = Join-Path $scriptDir $BuildDir
$elfPath = Join-Path $buildPath "Core-STM32F4-fw.elf"
$elfPathForOpenOcd = "$BuildDir/Core-STM32F4-fw.elf"

Write-Host "==> Ref-core build/flash"
Write-Host "    Project: $scriptDir"
Write-Host "    Build dir: $buildPath"
Write-Host "    Build type: $BuildType"

Assert-Exists -PathToCheck $stlinkCfg -What "stlink config"
Assert-Exists -PathToCheck $openocdExe -What "OpenOCD executable"
Assert-Exists -PathToCheck $openocdScripts -What "OpenOCD scripts directory"

Push-Location $scriptDir
try {
    $needsConfigure = $Reconfigure -or -not (Test-Path -LiteralPath (Join-Path $buildPath "build.ninja"))
    if ($needsConfigure) {
        Write-Host "==> Configuring CMake"
        cmake -S . -B $BuildDir -G "Ninja" -DCMAKE_BUILD_TYPE=$BuildType
        if ($LASTEXITCODE -ne 0) { throw "CMake configure failed with exit code $LASTEXITCODE." }
    } else {
        Write-Host "==> Skipping configure (existing Ninja build detected)"
    }

    Write-Host "==> Building firmware"
    cmake --build $BuildDir
    if ($LASTEXITCODE -ne 0) { throw "Build failed with exit code $LASTEXITCODE." }

    Assert-Exists -PathToCheck $elfPath -What "Firmware ELF output"

    if ($SkipFlash) {
        Write-Host "==> Build complete. Flash skipped by -SkipFlash."
        exit 0
    }

    Write-Host "==> Flashing with OpenOCD"
    & $openocdExe `
        -s $openocdScripts `
        -f $stlinkCfg `
        -c "program $elfPathForOpenOcd verify reset exit"
    if ($LASTEXITCODE -ne 0) { throw "OpenOCD flash failed with exit code $LASTEXITCODE." }

    Write-Host "==> Flash complete."
}
finally {
    Pop-Location
}
