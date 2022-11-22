pytest: Makefile
	python3 -m pytest

docu: Makefile
	cd doc/rtd && make html

docu-autobuild: Makefile
	cd doc/rtd && make autobuild
