#include <iostream>
#include <curand_kernel.h>
// #include <cfloat>
#include "cuda_fp16.h"
#include <OpenEXR/ImfNamespace.h>
#include <OpenEXR/ImfOutputFile.h>
#include <OpenEXR/ImfChannelList.h>
#include <OpenEXR/half.h>

#define CUDA_PI 3.141592654f

using namespace std;

class d_vec3{
public:
	__device__ d_vec3();
	__device__ d_vec3(cuhalf e0, cuhalf e1, cuhalf e2);
	__device__ d_vec3(const d_vec3& v);
	__device__ cuhalf x() const;
	__device__ cuhalf y() const;
	__device__ cuhalf z() const;
	__device__ cuhalf r() const;
	__device__ cuhalf g() const;
	__device__ cuhalf b() const;

	__device__ d_vec3& operator=(const d_vec3& v);
	__device__ const d_vec3& operator+() const;
	__device__ d_vec3 operator-() const;
	__device__ cuhalf operator[](int i) const;
	__device__ cuhalf& operator[](int i);

	__device__ d_vec3& operator+=(const d_vec3 &v2);
	__device__ d_vec3& operator-=(const d_vec3 &v2);
	__device__ d_vec3& operator*=(const d_vec3 &v2);
	__device__ d_vec3& operator/=(const d_vec3 &v2);
	__device__ d_vec3& operator*=(const cuhalf t);
	__device__ d_vec3& operator/=(const cuhalf t);

	__device__ cuhalf length() const;
	__device__ cuhalf squared_length() const;
	__device__ void make_unit_vector();

	__device__ cuhalf dot(const d_vec3 &v2);
	__device__ d_vec3 cross(const d_vec3 &v2);

	__device__ void set(cuhalf e0, cuhalf e1, cuhalf e2);

	cuhalf e[3];
};

class h_vec3{
	public:
	h_vec3();

	half e[3];
};

istream& operator>>(istream &is, d_vec3 &t);
ostream& operator<<(ostream &os, d_vec3 &t);
__device__ d_vec3 operator+(const d_vec3 &v1, const d_vec3 &v2);
__device__ d_vec3 operator-(const d_vec3 &v1, const d_vec3 &v2);
__device__ d_vec3 operator*(const d_vec3 &v1, const d_vec3 &v2);
__device__ d_vec3 operator/(const d_vec3 &v1, const d_vec3 &v2);

__device__ d_vec3 operator*(const cuhalf t, const d_vec3 &v);
__device__ d_vec3 operator*(const d_vec3 &v, const cuhalf t);
__device__ d_vec3 operator/(const d_vec3 v, cuhalf t);

__device__ cuhalf dot(const d_vec3 &v1, const d_vec3 &v2);
__device__ d_vec3 cross(const d_vec3 &v1, const d_vec3 &v2);

__device__ d_vec3 unit_vector(d_vec3 v);



class ray{
public:
	__device__ ray();
	__device__ ray(const d_vec3& a, const d_vec3& b);
	__device__ d_vec3 origin() const;
	__device__ d_vec3 direction() const;
	__device__ d_vec3 p(cuhalf t) const;

	d_vec3 A;
	d_vec3 B;
};


__device__ cuhalf hit_sphere(const d_vec3& center, cuhalf radius, const ray& r);

class material;
struct hit_record{
	cuhalf t;
	d_vec3 p;
	d_vec3 normal;
	material *mat;
};

class hitable{
public:
	__device__ virtual bool hit(const ray& r, const cuhalf& t_min, cuhalf& t_max, hit_record& rec) const = 0;
};

class sphere: public hitable {
public:
	__device__ sphere();
	__device__ sphere(d_vec3 cen, cuhalf r, material* m);
	__device__ virtual bool hit(const ray& r, const cuhalf& tmin, cuhalf& tmax, hit_record& rec) const;

	d_vec3 center;
	cuhalf radius;
	material * mat;
};

class hitable_list{
public:
	__device__ hitable_list();
	__device__ hitable_list(hitable **list, int n);
	__device__ bool hit(const ray& r, const cuhalf& tmin, cuhalf& tmax, hit_record& rec) const;
	void copyDevice();

	hitable **list, **d_list;
	hitable_list* d_world;
	int list_size;
};

// __device__ d_vec3 color(const ray& r, hitable_list* world, curandState state);

class camera{
public:
	__device__ camera();
	__device__ camera(cuhalf, cuhalf);
	__device__ camera(d_vec3 o, d_vec3 lookAt, d_vec3 vup, cuhalf vfov, cuhalf aspect);
	__device__ camera(d_vec3 o, d_vec3 lookAt, d_vec3 vup, cuhalf vfov, cuhalf aspect, cuhalf aperture, cuhalf focus_dist);
	__device__ void get_ray(const cuhalf& u, const cuhalf& v, ray& r, curandState* state);

	d_vec3 origin;
	d_vec3 ulc;
	d_vec3 horizontal;
	d_vec3 vertical;
	d_vec3 u, v, w;
	cuhalf lens_radius;
};

class material{
public:
	__device__ virtual bool scatter(const ray& impacting, const hit_record& rec, d_vec3& att, ray& scattered, curandState* state) const = 0;
	bool emitter;
};

class lambertian : public material{
public:
	__device__ lambertian(const d_vec3& a);
	__device__ virtual bool scatter(const ray& impacting, const hit_record& rec, d_vec3& att, ray& scattered, curandState* state) const;
	d_vec3 albedo;
};

__device__ d_vec3 reflect(const d_vec3& v, const d_vec3& n);

class metal : public material{
public:
	__device__ metal(const d_vec3& a, const cuhalf& f);
	__device__ virtual bool scatter(const ray& impacting, const hit_record& rec, d_vec3& att, ray& scattered, curandState* state) const;
	d_vec3 albedo;
	cuhalf fuzzy;
};

__device__ d_vec3 random_in_unit_sphere(curandState* state);
__device__ bool refract(const d_vec3& v, const d_vec3& n, cuhalf ni_nt, d_vec3& refracted);

class dielectric : public material{
public:
	__device__ dielectric(const cuhalf& i);
	__device__ virtual bool scatter(const ray& impacting, const hit_record& rec, d_vec3& att, ray& scattered, curandState* state) const;
	__device__ cuhalf schlick(const cuhalf& cosine, const cuhalf& indor) const;
	cuhalf ior;
};

class volume : public hitable{
public:
	__device__ volume();
	__device__ virtual bool hit(const ray& r, const cuhalf& t_min, cuhalf& t_max, hit_record& rec) const = 0;
};

class light : public material{
public:
	__device__ light(d_vec3 att);
	__device__ virtual bool scatter(const ray& impacting, const hit_record& rec, d_vec3& att, ray& scattered, curandState* state) const;
	d_vec3 attenuation;
};

class hair : public material{
public:
	__device__ hair(d_vec3 color);
	__device__ virtual bool scatter(const ray& impacting, const hit_record& rec, d_vec3& att, ray& scattered, curandState* state) const;
	__device__ void sim();
	__device__ void move(const d_vec3& position, const cuhalf& time);
};