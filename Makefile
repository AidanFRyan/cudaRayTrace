main: main.o tracer.o
	nvcc -lIlmImf -arch=sm_61 main.o tracer.o -o main
main.o: main.cu tracer.h
	nvcc -dc -arch=sm_61 -c main.cu
tracer.o: tracer.cu tracer.h
	nvcc -dc -arch=sm_61 -c tracer.cu
# objRead.o: objRead.cpp objRead.h
# 	nvcc -dc -arch=sm_61 -c objRead.cpp

debug: dmain.o dtracer.o
	nvcc -g -G -lIlmImf -arch=sm_61 -o debug dmain.o dtracer.o
dmain.o: main.cu tracer.h
	nvcc -dc -g -G -arch=sm_61 -o dmain.o -c main.cu
dtracer.o: tracer.cu tracer.h
	nvcc -dc -g -G -arch=sm_61 -o dtracer.o -c tracer.cu
# dobjRead.o: objRead.cpp objRead.h
# 	nvcc -dc -g -G -arch=sm_61 -o dobjRead.o -c objRead.cpp