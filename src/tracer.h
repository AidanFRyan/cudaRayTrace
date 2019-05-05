//Aidan Ryan, 2019

#ifndef TRACER_H
#define TRACER_H

#include <random>
#include <iostream>
#include <curand_kernel.h>
#include <cfloat>
#include <fstream>
#include <string>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#define CUDA_PI 3.141592654f

using namespace std;

class vec3{
public:
	__host__ __device__ vec3();
	__host__ __device__ vec3(float e0, float e1, float e2);
	__host__ __device__ vec3(const vec3& v);
	__host__ __device__ float x() const;
	__host__ __device__ float y() const;
	__host__ __device__ float z() const;
	__host__ __device__ float r() const;
	__host__ __device__ float g() const;
	__host__ __device__ float b() const;

	__host__ __device__ vec3& operator=(const vec3& v);
	__host__ __device__ const vec3& operator+() const;
	__host__ __device__ vec3 operator-() const;
	__host__ __device__ float operator[](int i) const;
	__host__ __device__ float& operator[](int i);

	__host__ __device__ vec3& operator+=(const vec3 &v2);
	__host__ __device__ vec3& operator-=(const vec3 &v2);
	__host__ __device__ vec3& operator*=(const vec3 &v2);
	__host__ __device__ vec3& operator/=(const vec3 &v2);
	__host__ __device__ vec3& operator*=(const float t);
	__host__ __device__ vec3& operator/=(const float t);

	__host__ __device__ float length() const;
	__host__ __device__ float squared_length() const;
	__host__ __device__ void make_unit_vector();

	__host__ __device__ float dot(const vec3 &v2);
	__host__ __device__ vec3 cross(const vec3 &v2);

	__host__ __device__ void set(float e0, float e1, float e2);

	float e[3];
};

istream& operator>>(istream &is, vec3 &t);
ostream& operator<<(ostream &os, vec3 &t);
__host__ __device__ vec3 operator+(const vec3 &v1, const vec3 &v2);
__host__ __device__ vec3 operator-(const vec3 &v1, const vec3 &v2);
__host__ __device__ vec3 operator*(const vec3 &v1, const vec3 &v2);
__host__ __device__ vec3 operator/(const vec3 &v1, const vec3 &v2);

__host__ __device__ vec3 operator*(const float t, const vec3 &v);
__host__ __device__ vec3 operator*(const vec3 &v, const float t);
__host__ __device__ vec3 operator/(const vec3 v, float t);

__host__ __device__ float dot(const vec3 &v1, const vec3 &v2);
__host__ __device__ vec3 cross(const vec3 &v1, const vec3 &v2);

__host__ __device__ vec3 unit_vector(vec3 v);



class ray{
public:
	__host__ __device__ ray();
	__host__ __device__ ray(const vec3& a, const vec3& b);
	__host__ __device__ ray& operator=(const ray& r);
	__host__ __device__ vec3 origin() const;
	__host__ __device__ vec3 direction() const;
	__host__ __device__ vec3 p(float t) const;

	vec3 A;
	vec3 B;
};


__host__ __device__ float hit_sphere(const vec3& center, float radius, const ray& r);

class material;
class hitable;
struct hit_record{
	float t;
	vec3 p;
	vec3 normal;
	material *mat;
	const hitable* obj = 0;
};


class hitable{
public:
	__device__ virtual bool hit(const ray& r, const float& t_min, float& t_max, hit_record& rec) const = 0;
	// __device__ virtual void insert(void* in) = 0;
};

class sphere: public hitable {
public:
	__host__ __device__ sphere();
	__host__ __device__ sphere(vec3 cen, float r, material* m);
	__device__ virtual bool hit(const ray& r, const float& tmin, float& tmax, hit_record& rec) const;
	// __device__ virtual void insert(void* in);
	vec3 center;
	float radius;
	material * mat;
};

class OBJ;
class TriTree;
class hitable_list{
public:
	__host__ __device__ hitable_list();
	__host__ __device__ hitable_list(int n);
	__host__ __device__ hitable_list(hitable **list, int n);
	__device__ bool hit(const ray& r, const float& tmin, float& tmax, hit_record& rec);//, bool* d_hits, hit_record* d_recs, float* d_dmax) const;
	__device__ bool hit(const ray& r, const float& tmin, float& tmax, hit_record& rec, int index);
	__device__ hitable_list(OBJ **list, int n, int additional);
	void copyDevice();
	// __device__ TriTree* toTree();

	hitable **list, **d_list;
	hitable_list* d_world;
	int list_size;
};

// __host__ __device__ vec3 color(const ray& r, hitable_list* world, curandState state);

class camera{
public:
	camera();
	camera(float, float);
	camera(vec3 o, vec3 lookAt, vec3 vup, float vfov, float aspect);
	camera(vec3 o, vec3 lookAt, vec3 vup, float vfov, float aspect, float aperture, float focus_dist);
	__device__ void get_ray(const float& u, const float& v, ray& r, curandState* state);
	void get_ray(const float& u, const float& v, ray& r, mt19937 state);

	vec3 origin;
	vec3 ulc;
	vec3 horizontal;
	vec3 vertical;
	vec3 u, v, w;
	float lens_radius;
};

class material{
public:
	__device__ virtual bool scatter(const ray& impacting, const hit_record& rec, vec3& att, ray& scattered, curandState* state) const = 0;
	bool emitter, transparent;
};

class lambertian : public material{
public:
	__host__ __device__ lambertian(const vec3& a);
	__device__ virtual bool scatter(const ray& impacting, const hit_record& rec, vec3& att, ray& scattered, curandState* state) const;
	vec3 albedo;
};

__device__ vec3 reflect(const vec3& v, const vec3& n);

class metal : public material{
public:
	__host__ __device__ metal(const vec3& a, const float& f);
	__device__ virtual bool scatter(const ray& impacting, const hit_record& rec, vec3& att, ray& scattered, curandState* state) const;
	vec3 albedo;
	float fuzzy;
};

__device__ vec3 random_in_unit_sphere(curandState* state);
__device__ bool refract(const vec3& v, const vec3& n, float ni_nt, vec3& refracted);

class dielectric : public material{
public:
	__host__ __device__ dielectric(const float& i);
	__device__ virtual bool scatter(const ray& impacting, const hit_record& rec, vec3& att, ray& scattered, curandState* state) const;
	__device__ float schlick(const float& cosine, const float& indor) const;
	float ior;
};

class volume : public hitable{
public:
	__device__ volume();
	__device__ virtual bool hit(const ray& r, const float& t_min, float& t_max, hit_record& rec) const = 0;
	// __device__ virtual void insert(void* in);
};

class light : public material{
public:
	__device__ light(vec3 att);
	__device__ virtual bool scatter(const ray& impacting, const hit_record& rec, vec3& att, ray& scattered, curandState* state) const;
	vec3 attenuation;
};

class hair : public material{
public:
	__device__ hair(vec3 color);
	__device__ virtual bool scatter(const ray& impacting, const hit_record& rec, vec3& att, ray& scattered, curandState* state) const;
	__device__ void sim();
	__device__ void move(const vec3& position, const float& time);
};

class sss : public material{
public:
	__device__ sss(material* surf, const float& d, const vec3& internalColor);
	__device__ virtual bool scatter(const ray& impacting, const hit_record& rec, vec3& att, ray& scattered, curandState* state) const;

private:
	material* surface;
	vec3 attenuation;
	float depth;
};

class Face:public hitable{
public:
    __host__ __device__ Face();
    __host__ __device__ Face(vec3 v1, vec3 v2, vec3 v3, vec3 t1, vec3 t2, vec3 t3, vec3 n1, vec3 n2, vec3 n3);
    __device__ virtual bool hit(const ray& r, const float& t_min, float& t_max, hit_record& rec) const;
    __host__ __device__ Face& operator=(const Face& in);
    __host__ __device__ Face(const Face& in);
	__host__ __device__ Face(const Face& in, material* m);
	// __device__ virtual void insert(void* in);
// private:
    vec3 verts[3], texts[3], normals[3], surfNorm, e[3], median;
    float max[3], min[3];
	material* mat;
};

class OBJ{
public:
    __host__ __device__ OBJ();
    OBJ(string fn);
    OBJ* copyToDevice();
    __device__ bool hit(const ray& r, const float& tmin, float& tmax, hit_record& rec) const;
// private:
    __device__ TriTree* toTree();
    ifstream file, mtllib;
    vec3 *points, *text, *normals;
    int numP, numT, numN, PBuf, TBuf, NBuf;
    void parse(char* line);
    void append(vec3*& list, int& size, int& bufSize, const vec3& item);
    void append(const Face& item);
    Face* object;
    int numFaces, faceBuffer;
};

class TreeNode : public hitable{
	friend class TriTree;
	friend class OBJ;
public:
	TreeNode *r, *l;
	__host__ __device__ TreeNode();
	__host__ __device__ TreeNode(Face* in, TreeNode* par);
	__device__ bool hit(const ray& r, const float& tmin, float& tmax, hit_record& rec) const;
	__device__ bool withinBB(const vec3& p);
	__device__ TreeNode* lt();
	__device__ TreeNode* gt();
// private:
	Face **contained;
	unsigned int within;
	float max[3], min[3], p;
	short dim;
	vec3 median;
	__device__ bool boxIntersect(const ray& r) const;
};

class TriTree : public hitable{
	friend class OBJ;
public:
	__host__ __device__ TriTree();
	__host__ __device__ void insert(Face* in);
	__device__ virtual bool hit(const ray& r, const float& tmin, float& tmax, hit_record& rec) const;
	hitable* copyToDevice();
	__device__ void print();
// private:
	
	int numNodes;
	TreeNode* head;
	TreeNode* descendCopy(TreeNode*);
};




#endif