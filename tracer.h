#include <iostream>
#include <curand_kernel.h>
#include <cfloat>
using namespace std;

class vec3{
public:
	__host__ __device__ vec3();
	__host__ __device__ vec3(float e0, float e1, float e2);
	__host__ __device__ float x() const;
	__host__ __device__ float y() const;
	__host__ __device__ float z() const;
	__host__ __device__ float r() const;
	__host__ __device__ float g() const;
	__host__ __device__ float b() const;

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
	__host__ __device__ vec3 origin() const;
	__host__ __device__ vec3 direction() const;
	__host__ __device__ vec3 p(float t) const;

	vec3 A;
	vec3 B;
};


__host__ __device__ float hit_sphere(const vec3& center, float radius, const ray& r);

class material;
struct hit_record{
	float t;
	vec3 p;
	vec3 normal;
	material *mat;
};

class hitable{
public:
	__host__ __device__ virtual bool hit(const ray& r, const float& t_min, float& t_max, hit_record& rec) const = 0;
};

class sphere: public hitable {
public:
	__host__ __device__ sphere();
	__host__ __device__ sphere(vec3 cen, float r, material* m);
	__host__ __device__ virtual bool hit(const ray& r, const float& tmin, float& tmax, hit_record& rec) const;

	vec3 center;
	float radius;
	material * mat;
};

class hitable_list{
public:
	__host__ __device__ hitable_list();
	__host__ __device__ hitable_list(hitable **list, int n);
	__host__ __device__ bool hit(const ray& r, const float& tmin, float& tmax, hit_record& rec) const;
	void copyDevice();

	hitable **list, **d_list;
	hitable_list* d_world;
	int list_size;
};

// __host__ __device__ vec3 color(const ray& r, hitable_list* world, curandState state);

class camera{
public:
	camera();
	__host__ __device__ void get_ray(const float& u, const float& v, ray& r);

	vec3 origin;
	vec3 ulc;
	vec3 horizontal;
	vec3 vertical;
};

class material{
public:
	__device__ virtual bool scatter(const ray& impacting, const hit_record& rec, vec3& att, ray& scattered, curandState* state) const = 0;
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
__device__ bool refract(const vec3& v, const vec3& n, const float& ni_nt, vec3& refracted);

class dielectric : public material{
public:
	__host__ __device__ dielectric(const float& i);
	__device__ virtual bool scatter(const ray& impacting, const hit_record& rec, vec3& att, ray& scattered, curandState* state) const;
	__device__ float schlick(const float& cosine, const float& indor) const;
	float ior;
};