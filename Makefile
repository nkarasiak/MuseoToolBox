

PYTHON=python3
branch := $(shell git symbolic-ref --short -q HEAD)

help :
	@echo "The following make targets are available:"
	@echo "    help - print this message"
	@echo "    build - build python package"
	@echo "    install - install python package (local user)"
	@echo "    sinstall - install python package (system with sudo)"
	@echo "    remove - remove the package (local user)"
	@echo "    sremove - remove the package (system with sudo)"
	@echo "    clean - remove any temporary files"
	@echo "    notebook - launch ipython3 notebook"
build :
	$(PYTHON) setup.py build

buildext :
	$(PYTHON) setup.py build_ext --inplace

install :
	$(PYTHON) setup.py install --user

sinstall :
	sudo $(PYTHON) setup.py install

remove :
	$(PYTHON) setup.py install --user --record files.txt
	tr '\n' '\0' < files.txt | xargs -0 rm -f --
	rm files.txt

sremove :
	$(PYTHON) setup.py install  --record files.txt
	tr '\n' '\0' < files.txt | sudo xargs -0 rm -f --
	rm files.txt

clean : FORCE
	$(PYTHON) setup.py clean

uploadpypi :
	#python setup.py register
	$(PYTHON) setup.py sdist upload -r pypi

rdoc :
	pandoc --from=markdown --to=rst --output=docs/source/readme.rst README.md

notebook :
	ipython3 notebook --notebook-dir=notebooks/
	
bench : 
	@git stash  >/dev/null 2>&1
	@echo 'Branch master'
	@git checkout master >/dev/null 2>&1
	python3 $(script)
	@echo 'Branch $(branch)'
	@git checkout $(branch) >/dev/null 2>&1
	python3 $(script)
	@git stash apply >/dev/null 2>&1
	
autopep8 :
	autopep8 -ir MuseoToolBox --jobs -1

aautopep8 :
	autopep8 -air MuseoToolBox --jobs -1
