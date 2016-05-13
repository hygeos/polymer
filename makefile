all:
	python setup.py build_ext --inplace

rebuild: clean all

clean:
	rm -fv *.html
	rm -fv *.pyc
	rm -fv *.c
	rm -fv *.so

test: all
	nosetests -v
