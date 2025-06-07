@echo off
REM Get the directory where the script is located
set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

REM Get the current directory
set "CURRENT_DIR=%CD%"

REM Only change directory if we're not already in the correct directory
if not "%CURRENT_DIR%"=="%SCRIPT_DIR%" (
    echo Changing to directory: %SCRIPT_DIR%
    cd /d "%SCRIPT_DIR%"
)

REM Clean up old files
del /f /q fp32_mul_sim dump.vcd sim.out fp32_ref_model_exec.exe test_fp32_expected.txt test_fp32_refc_details.txt 2>nul

REM Compile C reference model
echo Compiling C reference model...
gcc -o fp32_ref_model_exec.exe fp32_ref_model.c -lm
if %ERRORLEVEL% EQU 0 (
    echo C reference model compiled successfully
    fp32_ref_model_exec.exe
) else (
    echo Error compiling C reference model
    exit /b 1
)

REM Compile and run the simulation
echo Compiling and running simulation...
iverilog -g2012 -o fp32_mul_sim fp32_mul_tb.sv ..\mac_array\fp32_mul.v
if %ERRORLEVEL% EQU 0 (
    echo Simulation compiled successfully
    vvp fp32_mul_sim
) else (
    echo Error compiling simulation
    exit /b 1
) 