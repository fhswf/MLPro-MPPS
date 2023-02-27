@ECHO OFF

pushd %~dp0

if "%1" == "test" goto :test
if "%1" == "" (goto :end) else (goto :end)

:pytest
python -m pytest
goto end

:end
popd
