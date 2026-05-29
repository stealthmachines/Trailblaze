@echo off
cd /d C:\Users\Owner\Documents\trailblazer\v0.8
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" 2>nul
if errorlevel 1 ( echo ERROR: vcvars64 failed & exit /b 1 )

set CFLAGS=/O2 /MD /EHsc /W3 /std:c11 /D_CRT_SECURE_NO_WARNINGS /DTB_CUDA /DTB_INFER_TEST ^
    /Ilayer0 /Ilayer1 /Ilayer2 /Ilayer3 /Ilayer4 /Ilayer5 /Iinclude /Isrc ^
    /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\include"

echo Compiling tb_gguf.c ...
cl %CFLAGS% /c layer4\tb_gguf.c /Foobj\tb_gguf.obj
if errorlevel 1 ( echo COMPILE FAILED tb_gguf.c & exit /b 1 )

echo Compiling tb_tokenizer.c ...
cl %CFLAGS% /c layer4\tb_tokenizer.c /Foobj\tb_tokenizer.obj
if errorlevel 1 ( echo COMPILE FAILED tb_tokenizer.c & exit /b 1 )

echo Compiling tb_infer.c ...
cl %CFLAGS% /c layer4\tb_infer.c /Foobj\tb_infer.obj
if errorlevel 1 ( echo COMPILE FAILED tb_infer.c & exit /b 1 )

echo All compiles OK -- linking ...
setlocal EnableDelayedExpansion
set OBJS=
for %%f in (obj\*.obj) do set OBJS=!OBJS! %%f
link /OUT:bin\tb_infer.exe /SUBSYSTEM:CONSOLE !OBJS! ^
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\lib\x64\cudart.lib" ^
    ws2_32.lib kernel32.lib user32.lib advapi32.lib
if errorlevel 1 ( echo LINK FAILED & exit /b 1 )
echo BUILD OK  --  bin\tb_infer.exe
