#ifndef OBJREAD_H
#define OBJREAD_H

#include <iostream>
#include <fstream>
#include <string>
// #include "tracer.h"

using namespace std;

class OBJ{
public:
    __host__ __device__ OBJ();
    __host__ __device__ OBJ(string fn);
    OBJ* copyToDevice();
    __host__ __device__ bool hit(const ray& r, const float& tmin, float& tmax, hit_record& rec) const;
// private:
    ifstream file, mtllib;
    vec3 *points, *text, *normals;
    int numP, numT, numN;
    void parse(char* line);
    void append(vec3*& list, int& size, const vec3& item);
    void append(const Face& item);
    Face* object;
    int numFaces;
};

#endif