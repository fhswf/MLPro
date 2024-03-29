@ECHO OFF

pushd %~dp0

if "%1" == "test" goto :test
if "%1" == "doc" goto :doc
if "%1" == "doc-autobuild" goto :doc-autobuild
if "%1" == "" (goto :end) else (goto :end)

:pytest
python -m pytest
goto end

:docu
cd doc/rtd && make html
goto end

:docu-autobuild
cd doc/rtd && make autobuild

:end
popd
