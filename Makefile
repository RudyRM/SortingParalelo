all:
	nvcc -O3 -Xcompiler -fopenmp prog.cu -o prog -lcudart
