@echo off
:: scripts/build_cuda.bat — Build tb_infer.exe with CUDA (MSVC + nvcc)
:: Usage: scripts\build_cuda.bat [--debug]
::
:: Requires: VS 2022 Build Tools + CUDA Toolkit 13.x

setlocal EnableDelayedExpansion

set ROOT=%~dp0..
cd /d "%ROOT%"

:: ── Compiler paths ────────────────────────────────────────────────────────
set VCVARS="C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set CUDA_BIN=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin
set CUDA_INC=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\include
set CUDA_LIB=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\lib\x64

if not exist %VCVARS% (
    echo ERROR: VS2022 Build Tools not found at %VCVARS%
    exit /b 1
)

:: ── Activate MSVC environment ─────────────────────────────────────────────
call %VCVARS% 2>nul

mkdir bin 2>nul
mkdir obj 2>nul

:: ── Compile CUDA kernel object (nvcc + MSVC host) ────────────────────────
echo [1/3] Compiling CUDA kernels (sm_75)...
nvcc -arch=sm_75 -O2 ^
    -I"%CUDA_INC%" -Isrc -Ilayer4 ^
    -DTB_CUDA -DTB_INFER_TEST ^
    --compile layer4/tb_gguf_dequant.cu ^
    -Xcompiler "/MD /O2 /EHsc" ^
    -o obj\tb_gguf_dequant.obj
if errorlevel 1 ( echo CUDA compile FAILED & exit /b 1 )
echo    Done: obj\tb_gguf_dequant.obj

:: ── Compile all C sources with MSVC ──────────────────────────────────────
echo [2/3] Compiling C sources (MSVC)...

set INC=/Ilayer0 /Ilayer1 /Ilayer2 /Ilayer3 /Ilayer4 /Ilayer5 /Iinclude /Isrc /I"%CUDA_INC%"
set CFLAGS=/O2 /Ox /MD /EHsc /W3 /std:c11 /D_CRT_SECURE_NO_WARNINGS /DTB_CUDA /DTB_INFER_TEST

set SRCS=^
    layer4\tb_infer.c ^
    layer4\tb_gguf.c ^
    layer4\tb_tokenizer.c ^
    layer3\tb_orchestration.c ^
    layer5\tb_semantic_os.c ^
    layer0\tb_phi_lattice.c ^
    layer1\tb_tensor.c ^
    layer2\tb_graph.c ^
    src\sha256_minimal.c ^
    src\hdgl_bootloaderz.c ^
    src\hdgl_router.c ^
    src\vector_container.c ^
    src\analog_engine.c ^
    src\tb_analog_dispatch.c

cl %CFLAGS% %INC% /c %SRCS% /Fo"obj\\"
if errorlevel 1 ( echo C compile FAILED & exit /b 1 )
echo    Done: obj\*.obj

:: ── Link everything ───────────────────────────────────────────────────────
echo [3/3] Linking tb_infer.exe...

set OBJS=
for %%f in (obj\*.obj) do set OBJS=!OBJS! %%f

link /OUT:bin\tb_infer.exe /SUBSYSTEM:CONSOLE ^
    !OBJS! ^
    "%CUDA_LIB%\cudart.lib" ^
    ws2_32.lib kernel32.lib user32.lib advapi32.lib
if errorlevel 1 ( echo Link FAILED & exit /b 1 )

echo.
echo === Build complete: bin\tb_infer.exe (CUDA sm_75) ===
echo   Run:  bin\tb_infer.exe --model ^<path.gguf^> --prompt "Hello"
echo   GPU:  RTX 2060 (sm_75) -- CUDA kernels active
