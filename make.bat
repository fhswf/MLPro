@ECHO OFF

pushd %~dp0

if "%1" == "test" goto :test
if "%1" == "doc" goto :doc
if "%1" == "doc-autobuild" goto :doc-autobuild
if "%1" == "" (goto :end) else (goto :end)

:pytest
python -m pytest --ignore=src/mlpro/rl/pool/envs/ur5jointcontrol/src/ --ignore=src/mlpro/rl/pool/envs/multigeorobot/src/
goto end

:doc
cd doc/rtd && make html
goto end

:doc-autobuild
cd doc/rtd && make autobuild

:end
popd
