rm -r ./build
rm -r ./src/*.egg-info
rm ./dist/*
pip3 install --upgrade build
python3 -m build
pip3 install ./dist/*.whl
