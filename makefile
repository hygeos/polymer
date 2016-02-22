all:
	python setup.py build_ext --inplace

annotate:
	cython -a *.pyx

clean:
	rm -fv *.html
	rm -fv *.pyc
	rm -fv *.c
	rm -fv *.so
