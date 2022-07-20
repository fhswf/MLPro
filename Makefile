pytest: Makefile
	python3 -m pytest --ignore=src/mlpro/rl/pool/envs/ur5jointcontrol/src/ --ignore=src/mlpro/rl/pool/envs/multigeorobot/src/

doc: Makefile
	cd doc/rtd && make html

doc-autobuild: Makefile
	cd doc/rtd && make autobuild
