@echo off
setlocal
rem ---------------------------------------------------------------------------
rem Build helper for Windows/MSVC.
rem   build.bat                       -> compiles main.cpp to pt.exe
rem   build.bat some.cpp other.exe    -> compiles some.cpp to other.exe
rem On Linux/macOS just use:  g++ -O3 -std=c++17 -pthread main.cpp -o pt
rem ---------------------------------------------------------------------------

set SRC=%~1
if "%SRC%"=="" set SRC=main.cpp
set OUT=%~2
if "%OUT%"=="" set OUT=pt.exe

rem Locate the Visual Studio x64 developer environment (vswhere ships with VS).
set VSWHERE="%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist %VSWHERE% set VSWHERE="%ProgramFiles%\Microsoft Visual Studio\Installer\vswhere.exe"
set VCVARS=
if exist %VSWHERE% for /f "usebackq delims=" %%D in (`%VSWHERE% -latest -property installationPath`) do set VCVARS="%%D\VC\Auxiliary\Build\vcvars64.bat"
if not defined VCVARS (
  echo Could not locate vcvars64.bat - run this from a "Developer Command Prompt" instead. 1>&2
  exit /b 1
)
call %VCVARS% >nul

cl /nologo /W4 /O2 /Oi /Gy /EHsc /std:c++17 /MT %SRC% /Fe:%OUT%
