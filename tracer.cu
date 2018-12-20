#include "tracer.h"
#include <typeinfo>

vec3::vec3(){
	e[0] = 0;
	e[1] = 0;
	e[2] = 0;
}
vec3::vec3(float e0, float e1, float e2){
	e[0] = e0;
	e[1] = e1;
	e[2] = e2;
}
float vec3::x() const{
	return e[0];
}
float vec3::y() const{
	return e[1];
}
float vec3::z() const{
	return e[2];
}
float vec3::r() const{
	return e[0];
}
float vec3::g() const{
	return e[1];
}
float vec3::b() const{
	return e[2];
}

const vec3& vec3::operator+() const{
	return *this;
}
vec3 vec3::operator-() const{
	return vec3(-e[0], -e[1], -e[2]);
}
float vec3::operator[](int i) const{
	if(i < 3 && i > 0)
		return e[i];
	else return 0;
}
float& vec3::operator[](int i){
	return e[i];
}

vec3& vec3::operator+=(const vec3 &v2){
	e[0] += v2.e[0];
	e[1] += v2.e[1];
	e[2] += v2.e[2];
	return *this;
}
vec3& vec3::operator-=(const vec3 &v2){
	e[0] -= v2.e[0];
	e[1] -= v2.e[1];
	e[2] -= v2.e[2];
	return *this;
}
vec3& vec3::operator*=(const vec3 &v2){
	e[0] *= v2.e[0];
	e[1] *= v2.e[1];
	e[2] *= v2.e[2];
	return *this;}
vec3& vec3::operator/=(const vec3 &v2){
	e[0] /= v2.e[0];
	e[1] /= v2.e[1];
	e[2] /= v2.e[2];
	return *this;
}
vec3& vec3::operator*=(const float t){
	e[0] *= t;
	e[1] *= t;
	e[2] *= t;
	return *this;
}
vec3& vec3::operator/=(const float t){
	e[0] /= t;
	e[1] /= t;
	e[2] /= t;
	return *this;
}

float vec3::length() const{
	return sqrtf(e[0]*e[0] + e[1]*e[1] + e[2]*e[2]);
}
float vec3::squared_length() const{
	return e[0]*e[0] + e[1]*e[1] + e[2]*e[2];
}
void vec3::make_unit_vector(){
	float k = 1.0f / sqrt(e[0]*e[0] + e[1]*e[1] + e[2] * e[2]);
	e[0] *= k;
	e[0] *= k;
	e[0] *= k;
}

float vec3::dot(const vec3 &v2){
	return e[0]*v2.e[0] + e[1]*v2.e[1] + e[2]*v2.e[2];
}
vec3 vec3::cross(const vec3 &v2){
	return vec3(e[1]*v2.e[2] - e[2]*v2.e[1], (-(e[0]*v2.e[2] - e[2]*v2.e[0])), e[0]*v2.e[1] - e[1]*v2.e[0]);
}

istream& operator>>(istream &is, vec3 &t){
	is>>t.e[0]>>t.e[1]>>t.e[2];
	return is;
}
ostream& operator<<(ostream &os, vec3 &t){
	os<<t.e[0]<<' '<<t.e[1]<<' '<<t.e[2];
	return os;
}
vec3 operator+(const vec3 &v1, const vec3 &v2){
	return vec3(v1.e[0]+v2.e[0], v1.e[1]+v2.e[1], v1.e[2]+v2.e[2]);
}
vec3 operator-(const vec3 &v1, const vec3 &v2){
	return vec3(v1.e[0]-v2.e[0], v1.e[1]-v2.e[1], v1.e[2]-v2.e[2]);
}
vec3 operator*(const vec3 &v1, const vec3 &v2){
	return vec3(v1.e[0]*v2.e[0], v1.e[1]*v2.e[1], v1.e[2]*v2.e[2]);
}
vec3 operator/(const vec3 &v1, const vec3 &v2){
	return vec3(v1.e[0]/v2.e[0], v1.e[1]/v2.e[1], v1.e[2]/v2.e[2]);
}

vec3 operator*(const float t, const vec3 &v){
	return vec3(v.e[0]*t, v.e[1]*t, v.e[2]*t);
}
vec3 operator*(const vec3 &v, const float t){
	return vec3(v.e[0]*t, v.e[1]*t, v.e[2]*t);
}
vec3 operator/(const vec3 v, float t){
	return vec3(v.e[0]/t, v.e[1]/t, v.e[2]/t);
}

float dot(const vec3 &v1, const vec3 &v2){
	return v1.e[0]*v2.e[0] + v1.e[1]*v2.e[1] + v1.e[2]*v2.e[2];
}
vec3 cross(const vec3 &v1, const vec3 &v2){
	return vec3(v1.e[1]*v2.e[2] - v1.e[2]*v2.e[1], (-(v1.e[0]*v2.e[2] - v1.e[2]*v2.e[0])), v1.e[0]*v2.e[1] - v1.e[1]*v2.e[0]);
}

vec3 unit_vector(vec3 v){
	return v/v.length();
}

void vec3::set(float e0, float e1, float e2){
	e[0] = e0;
	e[1] = e1;
	e[2] = e2;
}

ray::ray(){}
ray::ray(const vec3& a, const vec3& b){
	A = a;
	B = b;
}
vec3 ray::origin() const{
	return A;
}
vec3 ray::direction() const{
	return B;
}
vec3 ray::p(float t) const{
	return A + t*B;
}

sphere::sphere(){
	center = vec3(0,0,0);
	radius = 0;
}
sphere::sphere(vec3 cen, float r, material* m){
	center = cen;
	radius = r;
	mat = m;
}
bool sphere::hit(const ray& r, const float& tmin, float& tmax, hit_record& rec) const{
	vec3 oc = r.origin() - center;
	float a = dot(r.direction(), r.direction());
	float b = dot(oc, r.direction());
	float c = dot(oc, oc) - radius*radius;
	float discriminant = b*b-a*c;
	// printf("%f\n", radius);
	if (discriminant > 0){
		rec.mat = mat;
		float temp = (-b - sqrtf(b*b-a*c))/a;
		if (temp < tmax && temp > tmin){
			rec.t = temp;
			rec.p = r.p(rec.t);
			rec.normal = (rec.p - center) / radius;
			return true;
		}
		temp = (-b + sqrtf(b*b-a*c))/a;
		if(temp < tmax && temp > tmin){
			rec.t = temp;
			rec.p = r.p(rec.t);
			rec.normal = (rec.p - center) / radius;
			return true;
		}
	}
	return false;
}

hitable_list::hitable_list(){
	list = 0;
	list_size = 0;
}
hitable_list::hitable_list(hitable **list, int n){
	this->list = list;
	list_size = n;
}
// hitable** hitable_list::listPointer(){
// 	return d_list;
// }
bool hitable_list::hit(const ray& r, const float& tmin, float& tmax, hit_record& rec) const{
	hit_record temp_rec;
	bool anyHits = false;
	float closest = tmax;
	// printf("%p, %d\n", this, this->list_size);
	for(int i = 0; i < list_size; i++){
		// printf("%d %d\n", i, list_size);
		if(list[i]->hit(r, tmin, closest, temp_rec)){
			// printf("%f, %f, %f\n", r.direction().x(), r.direction().y(), r.direction().z());
			anyHits = true;
			closest = temp_rec.t;
			rec = temp_rec;
		}
	}
	return anyHits;
}

//apparently overrides of a parent's virtual functions don't work when the objects are instantiated on the host, instead must be instantiated through a backassward array of pointers and created entirely dynamically on the device
// void hitable_list::copyDevice(){
// 	hitable **h_list = new hitable*[list_size];
// 	// printf("%u\n", sizeof(hitable*));
// 	cudaMalloc((void**)&d_list, sizeof(hitable*)*list_size);
// 	cudaDeviceSynchronize();
// 	// printf("Done\n");
// 	for(int i = 0; i < list_size; i++){
// 		hitable* temp;
// 		// printf("%s %u\n", typeid(*list[i]).name(), sizeof(*list[i]));
// 		cudaMalloc((void**)&temp, sizeof(*list[i]));
// 		cudaDeviceSynchronize();
// 		cudaMemcpy(temp, list[i], sizeof(*list[i]), cudaMemcpyHostToDevice);
// 		cudaDeviceSynchronize();
// 		h_list[i] = temp;
// 	}
// 	cudaMemcpy(d_list, h_list, sizeof(hitable*)*list_size, cudaMemcpyHostToDevice);
// 	cudaDeviceSynchronize();
// 	// hitable_list* d_hlist;
// 	cudaMalloc((void**)&d_world, sizeof(hitable_list));
// 	cudaDeviceSynchronize();
// 	hitable **copy = list;
// 	list = d_list;
// 	// printf("%p %p %p\n", copy, list, h_list);
// 	// for(int i = 0; i < list_size; i++){
// 	// 	printf("copy: %p list: %p hlist: %p\n", copy[i], list[i], h_list[i]);
// 	// }
// 	cudaMemcpy(d_world, this, sizeof(hitable_list), cudaMemcpyHostToDevice);
// 	cudaDeviceSynchronize();
// 	list = copy;
// }

camera::camera(){
	ulc = vec3(-2, 1, -1);
	horizontal = vec3(4, 0, 0);
	vertical = vec3(0,2,0);
	origin = vec3(0,0,0);
}

void camera::get_ray(const float& u, const float& v, ray& r){
	r = ray(origin, ulc+u*horizontal-v*vertical-origin);
}

lambertian::lambertian(const vec3& a){
	albedo = a;
}

__device__ bool lambertian::scatter(const ray& impacting, const hit_record& rec, vec3& att, ray& scattered, curandState* state) const{
	vec3 target = rec.p+rec.normal+random_in_unit_sphere(state);
	scattered = ray(rec.p, target-rec.p);
	att = albedo;
	return true;
}

metal::metal(const vec3& a, const float& f){
	albedo = a;
	if(f<1)
		fuzzy = f;
	else
		fuzzy = 1;
}

__device__ vec3 reflect(const vec3& v, const vec3& n){
	return v - 2*dot(v,n)*n;
}

__device__ bool metal::scatter(const ray& impacting, const hit_record& rec, vec3& att, ray& scattered, curandState* state) const{
	vec3 reflected = reflect(unit_vector(impacting.direction()), rec.normal);
	if(fuzzy >= 0.01)
		scattered = ray(rec.p, reflected + fuzzy*random_in_unit_sphere(state));
	else
		scattered = ray(rec.p, reflected);
	att = albedo;
	return (dot(scattered.direction(), rec.normal) > 0);
}

dielectric::dielectric(const float& i){
	ior = i;
}

__device__  bool dielectric::scatter(const ray& impacting, const hit_record& rec, vec3& att, ray& scattered, curandState* state) const{
	vec3 outward_normal;
	vec3 reflected = reflect(impacting.direction(), rec.normal);
	float ni_nt;
	att = vec3(1.0f, 1.0f, 1.0f);
	vec3 refracted;
	float reflect_prob;
	float cosine;
	float dotted = dot(impacting.direction(), rec.normal);
	if(dotted>0){
		outward_normal = -rec.normal;
		ni_nt = ior;
		cosine = dotted/impacting.direction().length();
		cosine = sqrtf(1-ior*ior*(1-cosine*cosine));
	}
	else{
		outward_normal = rec.normal;
		ni_nt = 1.0f/ior;
		cosine = -dotted/impacting.direction().length();
		// cosine = sqrtf(1-ior*ior*(1-cosine*cosine));
	}
	if(refract(impacting.direction(), outward_normal, ni_nt, refracted)){
		reflect_prob = schlick(cosine, ior);
	}
	else{
		reflect_prob = 1;
	}
	if(curand_uniform(state) < reflect_prob){
		scattered = ray(rec.p, reflected);
	}
	else{
		scattered = ray(rec.p, refracted);
		// printf("refracted\n");
	}
	return true;
}

__device__ bool refract(const vec3& v, const vec3& n, const float& ni_nt, vec3& refracted){
	vec3 uv = unit_vector(v);
	// vec3 un = unit_vector(n);
	float dt = dot(uv, n);
	// pr  intf("%f\n", dt);
	float discriminant = 1.0f-ni_nt*ni_nt*(1.0f-dt*dt);
	if(discriminant > 0){
		refracted = ni_nt*(uv-n*dt) - n*sqrtf(discriminant);
		return true;
	}
	else return false;
}

__device__ float dielectric::schlick(const float& cosine, const float& indor) const{
	float r0 = (1-indor)/(1+indor);
	r0 *= r0;
	return r0 + (1-r0)*pow((1-cosine), 5);
}