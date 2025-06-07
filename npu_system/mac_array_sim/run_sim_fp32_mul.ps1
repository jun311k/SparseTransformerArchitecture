# FP32 multiplier testbench simulation script for PowerShell

# Get the directory where the script is located
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Get the current directory
$currentDir = Get-Location

# Only change directory if we're not already in the correct directory
if ($currentDir.Path -ne $scriptDir) {
    Write-Host "Changing to directory: $scriptDir"
    Set-Location $scriptDir
}

# Clean up old files
Remove-Item -Force -ErrorAction SilentlyContinue fp32_mul_sim, dump.vcd, sim.out, test_fp32_expected.txt, test_fp32_refc_details.txt

# Compile C reference model
Write-Host "Compiling C reference model..."
gcc -o fp32_ref_model_exec.exe fp32_ref_model.c -lm
if ($LASTEXITCODE -eq 0) {
    Write-Host "C reference model compiled successfully"
    ./fp32_ref_model_exec.exe
} else {
    Write-Host "Error compiling C reference model"
    exit 1
}

# Compile and run the simulation
Write-Host "Compiling and running simulation..."
iverilog -g2012 -o fp32_mul_sim fp32_mul_tb.sv ../mac_array/fp32_mul.v
if ($LASTEXITCODE -eq 0) {
    Write-Host "Simulation compiled successfully"
    vvp fp32_mul_sim
} else {
    Write-Host "Error compiling simulation"
    exit 1
} 