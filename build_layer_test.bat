@echo off
cd /d C:\Users\Owner\Documents\trailblazer\v0.8
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" 2>nul
if errorlevel 1 ( echo ERROR: vcvars64 failed & exit /b 1 )

set CFLAGS=/O2 /MD /EHsc /W3 /std:c11 /D_CRT_SECURE_NO_WARNINGS /DTB_CUDA ^
    /Ilayer0 /Ilayer1 /Ilayer2 /Ilayer3 /Ilayer4 /Ilayer5 /Iinclude /Isrc ^
    /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\include"

echo Compiling tb_infer.c (no-main variant)...
cl %CFLAGS% /DTB_NO_MAIN /c layer4\tb_infer.c /Foobj\tb_infer_notest.obj
if errorlevel 1 ( echo COMPILE FAILED (tb_infer_notest) & exit /b 1 )

echo Compiling tb_layer_test.c...
cl %CFLAGS% /c tb_layer_test.c /Foobj\tb_layer_test.obj
if errorlevel 1 ( echo COMPILE FAILED (tb_layer_test) & exit /b 1 )
echo COMPILE OK

echo Linking tb_layer_test.exe...
link /OUT:bin\tb_layer_test.exe /SUBSYSTEM:CONSOLE ^
    obj\tb_layer_test.obj ^
    obj\tb_infer_notest.obj ^
    obj\tb_gguf.obj ^
    obj\tb_gguf_dequant.obj ^
    obj\tb_tensor.obj ^
    obj\tb_tokenizer.obj ^
    obj\tb_phi_lattice.obj ^
    obj\tb_graph.obj ^
    obj\tb_orchestration.obj ^
    obj\tb_semantic_os.obj ^
    obj\tb_analog_dispatch.obj ^
    obj\analog_engine.obj ^
    obj\hdgl_router.obj ^
    obj\hdgl_bootloaderz.obj ^
    obj\sha256_minimal.obj ^
    obj\vector_container.obj ^
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\lib\x64\cudart.lib" ^
    ws2_32.lib kernel32.lib user32.lib advapi32.lib
if errorlevel 1 ( echo LINK FAILED & exit /b 1 )
echo LINK OK
echo.
echo Usage: bin\tb_layer_test.exe ^<model.gguf^> [--layer N] [--batch 8] [--stop-first] [--verbose] [--inventory]
