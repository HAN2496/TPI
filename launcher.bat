@echo off
setlocal enabledelayedexpansion

REM ==== USER CONFIG ====
set "DRIVER_NAME=b"
set "MODEL_NAME=online_polynomial"
set "TIME_RANGE=[5,7]"
set "DOWNSAMPLE=5"
set "DEVICE=cpu"
set "VERBOSE=1"

REM ==== TRAIN ====
@REM uv run python .\scripts\train_model.py ^
@REM     --driver-name "!DRIVER_NAME!" ^
@REM     --model-name "!MODEL_NAME!" ^
@REM     --time-range "!TIME_RANGE!" ^
@REM     --downsample "!DOWNSAMPLE!" ^
@REM     --device "!DEVICE!" ^
@REM     --verbose "!VERBOSE!"

uv run python .\scripts\optimize_model.py ^
    --driver-name "!DRIVER_NAME!" ^
    --model-name "!MODEL_NAME!" ^
    --time-range "!TIME_RANGE!" ^
    --downsample "!DOWNSAMPLE!" ^
    --n-trials 500 ^
    --device "!DEVICE!" ^
    --verbose "!VERBOSE!"