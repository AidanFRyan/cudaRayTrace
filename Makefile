main: main.o tracer.o
	nvcc -lIlmImf -lHalf -arch=compute_61 -o main main.o tracer.o
main.o: main.cu tracer.h
	nvcc -dc -arch=compute_61 -c main.cu
tracer.o: tracer.cu tracer.h
	nvcc -dc -arch=compute_61 -c tracer.cu

debug: dmain.o dtracer.o
	nvcc -lIlmImf -lHalf -g -G -arch=sm_61 -o debug dmain.o dtracer.o
dmain.o: main.cu tracer.h
	nvcc -dc -g -G -arch=sm_61 -o dmain.o -c main.cu
dtracer.o: tracer.cu tracer.h
	nvcc -dc -g -G -arch=sm_61 -o dtracer.o -c tracer.cu
