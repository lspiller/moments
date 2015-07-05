default: moments

moments:
	python setup.py build_ext --inplace

clean:
	rm -rf build
	rm -f moments.so 
	rm -f moments.c
