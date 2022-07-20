pytest: Makefile
	python3 -m pytest test

doc: Makefile
	cd doc/rtd && make html

doc-autobuild: Makefile
	cd doc/rtd && make autobuild
