rmdir .\build
rmdir .\src\*.egg-info
del .\dist\*
pip install --upgrade build
python -m build

