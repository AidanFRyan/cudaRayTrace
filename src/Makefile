main: main.o tracer.o
	nvcc -Xcompiler "-fopenmp" -rdc=true -lIlmImf -arch=sm_61 main.o tracer.o -o main
	mv main ..
main.o: main.cu tracer.h Makefile
	nvcc -Xcompiler "-fopenmp" -rdc=true -dc -arch=sm_61 -c main.cu
tracer.o: tracer.cu tracer.h Makefile
	nvcc -rdc=true -dc -arch=sm_61 -c tracer.cu
# objRead.o: objRead.cpp objRead.h
# 	nvcc -dc -arch=sm_61 -c objRead.cpp

debug: dmain.o dtracer.o
	nvcc -rdc=true -g -G -lIlmImf -arch=sm_61 -o debug dmain.o dtracer.o
	mv debug ..
dmain.o: main.cu tracer.h Makefile
	nvcc -rdc=true -dc -g -G -arch=sm_61 -o dmain.o -c main.cu
dtracer.o: tracer.cu tracer.h Makefile
	nvcc -rdc=true -dc -g -G -arch=sm_61 -o dtracer.o -c tracer.cu
# dobjRead.o: objRead.cpp objRead.h
# 	nvcc -dc -g -G -arch=sm_61 -o dobjRead.o -c objRead.cpp
