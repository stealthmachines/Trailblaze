@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" 2>nul
if errorlevel 1 ( echo ERROR: vcvars64 failed & exit /b 1 )
cd /d "C:\Users\Owner\Documents\trailblazer\v0.8"
echo Compiling CUDA kernel...
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\nvcc.exe" -arch=sm_75 -O2 -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\include" -Ilayer4 -DTB_CUDA -DTB_INFER_TEST --compile layer4/tb_gguf_dequant.cu -Xcompiler "/MD /O2 /EHsc" -o obj\tb_gguf_dequant.obj
if errorlevel 1 ( echo CUDA FAILED & exit /b 1 )
echo CUDA OK
