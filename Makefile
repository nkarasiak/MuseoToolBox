

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
	$(PYTHON) setup.py sdist bdist_wheel
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

doc :
	m2r README.md
	mv README.rst docs/source/
	rm -rf docs/source/auto_examples/ docs/sources/modules
	cd docs/ && make html

git-release:
	git add --all
	git commit -m "Version. `cat museotoolbox/__init__.py | awk -F '("|")' '{ print($$2)}'`"
	git tag `cat museotoolbox/__init__.py | awk -F '("|")' '{ print($$2)}'`
	git push
	git push --tags

autopep8 :
	autopep8 -ir museotoolbox --jobs -1

aautopep8 :
	autopep8 -air museotoolbox --jobs -1
