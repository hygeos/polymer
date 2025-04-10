# mark these make targets as non-file targets, so they'll always trigger when checked
.PHONY: main all rebuild clean test auxdata

main:
	python setup.py build_ext --inplace -j 4

all: auxdata main ancillary

rebuild: clean all

clean:
	rm -rfv build
	rm -fv polymer/*.so

tests: all
	pytest -n auto tests/test_olci.py -v

ancillary:
	mkdir -p ANCILLARY/METEO/

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

auxdata:
	python polymer/get_auxdata.py

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%:
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
