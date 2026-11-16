param(
    [string]$GodotExe = "C:\Godot\Godot 4.7\Godot_v4.7-stable_win64_console.exe",
    [string]$Suite = "",
    [string]$Case = "",
    [switch]$CpuOnly
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ClassCache = Join-Path $ProjectRoot ".godot\global_script_class_cache.cfg"

if (-not (Test-Path $ClassCache) -or -not (Select-String -Path $ClassCache -Pattern "JrOrreryTelemetry" -Quiet)) {
    & $GodotExe --path $ProjectRoot --headless --import | Out-Null
}

$UserArgs = @()
if ($Suite -ne "" -or $Case -ne "") {
    $UserArgs += "--"
    if ($Suite -ne "") {
        $UserArgs += "--suite=$Suite"
    }
    if ($Case -ne "") {
        $UserArgs += "--case=$Case"
    }
}

if ($CpuOnly) {
    & $GodotExe --path $ProjectRoot --headless -s res://benchmarks/jr_orrery_benchmark.gd @UserArgs
} else {
    & $GodotExe --path $ProjectRoot `
        --display-driver windows `
        --rendering-driver d3d12 `
        --audio-driver Dummy `
        --disable-vsync `
        -s res://benchmarks/jr_orrery_benchmark.gd @UserArgs
}
