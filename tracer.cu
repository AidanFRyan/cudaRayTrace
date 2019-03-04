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
vec3::vec3(const vec3& v){
	e[0] = v.e[0];
	e[1] = v.e[1];
	e[2] = v.e[2];
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

vec3& vec3::operator=(const vec3& v){
	e[0] = v.e[0];
	e[1] = v.e[1];
	e[2] = v.e[2];
	return *this;
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
	return vec3(v1.y()*v2.z() - v1.z()*v2.y(), (v1.z()*v2.x() - v1.x()*v2.z()), v1.x()*v2.y() - v1.y()*v2.x());
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
ray& ray::operator=(const ray& r){
	A = r.A;
	B = r.B;
	return *this;
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
__device__ bool sphere::hit(const ray& r, const float& tmin, float& tmax, hit_record& rec) const{
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
hitable_list::hitable_list(int n){
	list_size = n;
	this->list = new hitable*[n];
}

__global__ void listHits(int n, int cluster, bool* anyHits, const ray* r, hitable** list, hit_record* temp_rec, float* dist, float tmin, float tmax, bool* finished){
	int index = threadIdx.x + blockDim.x*blockIdx.x;
	int curIndex = index * cluster;
	while(curIndex < n){
		for(int i = 0; i < cluster && curIndex+i < n; i++){
			if(list[curIndex+i]->hit(*r, tmin, tmax, temp_rec[curIndex+i])){
				anyHits[curIndex+i] = true;
				dist[curIndex+i] = temp_rec[curIndex+i].t;
			}
			else{
				anyHits[curIndex+i] = false;
			}
		}
		curIndex += gridDim.x*blockDim.x;
	}
	__syncthreads();
	if(index == 0){
		float max = tmax;
		for(int i = 0; i < n; i++){
			if(anyHits[i]){
				if(dist[i] < max){
					max = dist[i];
					anyHits[0] = true;
					dist[0] = max;
					temp_rec[0] = temp_rec[i];
				}
			}
		}
		*finished = true;
	}
}
// hitable** hitable_list::listPointer(){
// 	return d_list;
// }
__device__ bool hitable_list::hit(const ray& r, const float& tmin, float& tmax, hit_record& rec){//}, bool* d_hits, hit_record* d_recs, float* d_dmax) const{
	hit_record temp_rec;
	bool anyHits = false;
	// bool* finished = new bool;
	// *finished = false;
	float closest = tmax;
	// bool* d_hits = new bool[list_size];
	// hit_record* d_recs = new hit_record[list_size];
	// float* d_dmax = new float[list_size];
	// printf("%p, %d\n", this, this->list_size);
	// listHits<<<1, 256>>>(list_size, 1, d_hits, &r, list, d_recs, d_dmax, tmin, tmax, finished);
	// cudaDeviceSynchronize();
	// while(!finished);
	// rec = d_recs[0];
	// anyHits = d_hits[0];

	for(int i = 0; i < list_size; i++){
		// printf("%d %d\n", i, list_size);
		// printf("%d %d\n", list_size, i);
		// printf("hh: %d %p %p %p %p\n", i, &r, &tmin, &closest, &temp_rec);
		if(list[i]->hit(r, tmin, closest, temp_rec)){
			// printf("%f, %f, %f\n", r.direction().x(), r.direction().y(), r.direction().z());
			anyHits = true;
			closest = temp_rec.t;
			rec = temp_rec;
			// if(temp_rec.mat)
			// break;
		}
	}
	// delete[] d_hits;
	// delete[] d_recs;
	// delete[] d_dmax;
	return anyHits;
}

__device__ bool hitable_list::hit(const ray& r, const float& tmin, float& tmax, hit_record& rec, int index){
	if(index<list_size){
		hit_record temp_rec;
		// bool anyHits = false;
		// bool* finished = new bool;
		// *finished = false;
		float closest = tmax;
		if(list[index]->hit(r, tmin, closest, temp_rec)){
			closest = temp_rec.t;
			rec = temp_rec;
			return true;
		}
	}
	return false;
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
__device__ vec3 random_in_unit_disk(curandState* state){
	// curandState state;
	// curand_init(1234, threadIdx.x+blockDim.x*blockIdx.x, 0, &state);
	vec3 p;
	do{
		p = 2.0f*vec3(curand_uniform(state), curand_uniform(state), 0) - vec3(1,1,0);
	}while(dot(p,p) >= 1.0f);
	return p;
}

vec3 random_in_unit_disk(mt19937 state){
	// curandState state;
	// curand_init(1234, threadIdx.x+blockDim.x*blockIdx.x, 0, &state);
	uniform_real_distribution<>dis(0,1);
	// printf("%f\n", state);
	vec3 p;
	do{
		p = 2.0f*vec3(dis(state), dis(state), 0) - vec3(1,1,0);
	}while(dot(p,p) >= 1.0f);
	// printf("exiting riud\n");
	return p;
}

camera::camera(){
	ulc = vec3(-2, 1, -1);
	horizontal = vec3(4, 0, 0);
	vertical = vec3(0,2,0);
	origin = vec3(0,0,0);
}

camera::camera(float vfov, float aspect){
	vfov *= CUDA_PI/180;
	float halfHeight = tanf(vfov/2);
	float halfWidth = aspect*halfHeight;
	ulc = vec3(-halfWidth, halfHeight, -1);
	horizontal = vec3(2*halfWidth, 0, 0);
	vertical = vec3(0,2*halfHeight,0);
	origin = vec3(0,0,0);
}
camera::camera(vec3 o, vec3 lookAt, vec3 vup, float vfov, float aspect){
	// vec3 u, v, w;
	lens_radius=0;
	vfov *= CUDA_PI/180;
	float halfHeight = tanf(vfov/2);
	float halfWidth = aspect*halfHeight;
	origin = o;
	w = unit_vector(o-lookAt);
	u = unit_vector(cross(vup, w));
	v = cross(w, u);
	ulc = vec3(-halfWidth, halfHeight, -1);
	ulc = origin - halfWidth*u + halfHeight*v - w;
	horizontal = 2*halfWidth*u;
	vertical = 2*halfHeight*v;
}
camera::camera(vec3 o, vec3 lookAt, vec3 vup, float vfov, float aspect, float aperture, float focus_dist){
	// vec3 u, v, w;
	lens_radius = aperture/2;
	vfov *= CUDA_PI/180;
	float halfHeight = tanf(vfov/2);
	float halfWidth = aspect*halfHeight;
	origin = o;
	w = unit_vector(o-lookAt);
	u = unit_vector(cross(vup, w));
	v = cross(w, u);
	// ulc = vec3(-halfWidth, halfHeight, -1);
	ulc = origin - halfWidth*focus_dist*u + halfHeight*focus_dist*v - focus_dist*w;
	horizontal = 2*halfWidth*u*focus_dist;
	vertical = 2*halfHeight*v*focus_dist;
}
__device__ void camera::get_ray(const float& s, const float& t, ray& r, curandState* state){
	vec3 rd;
	if(lens_radius > 0.001)
		rd = lens_radius * random_in_unit_disk(state);
	// printf("%f\n", v.y());
	vec3 offset = u*rd.x() + v*rd.y();
	r = ray(origin + offset, ulc+s*horizontal-t*vertical-origin-offset);
}

void camera::get_ray(const float& s, const float& t, ray& r, mt19937 state){
	vec3 rd;
	if(lens_radius > 0.001)
		rd = lens_radius * random_in_unit_disk(state);
	// printf("%f\n", v.y());
	vec3 offset = u*rd.x() + v*rd.y();
	r = ray(origin + offset, ulc+s*horizontal-t*vertical-origin-offset);
}

__host__ __device__ Face::Face(vec3 v1, vec3 v2, vec3 v3, vec3 t1, vec3 t2, vec3 t3, vec3 n1, vec3 n2, vec3 n3){
    verts[0] = v1;
    verts[1] = v2;
    verts[2] = v3;
    texts[0] = t1;
    texts[1] = t2;
    texts[2] = t3;
    normals[0] = n1;
    normals[1] = n2;
	normals[2] = n3;
	surfNorm = unit_vector(cross(verts[1]-verts[0], verts[2]-verts[1]));
	e[0] = verts[1] - verts[0];
    e[1] = verts[2] - verts[1];
    e[2] = verts[0] - verts[2];
    // median.set((v1.x()+v2.x()+v3.x())/3, (v1.y()+v2.y()+v3.y())/3, (v1.z()+v2.z()+v3.z())/3);
    float x[3], y[3], z[3];
   
	x[0] = v1.x();
	y[0] = v1.y();
	z[0] = v1.z();
	x[1] = v2.x();
	y[1] = v2.y();
	z[1] = v2.z();
	x[2] = v3.x();
	y[2] = v3.y();
	z[2] = v3.z();

	for(int i = 0; i < 2; i++){
		for(int j = i; j < 3; j++){
			if(x[i] > x[j]){
				float temp = x[i];
				x[i] = x[j];
				x[j] = temp;
			}
			if(y[i] > y[j]){
				float temp = y[i];
				y[i] = y[j];
				y[j] = temp;
			}
			if(z[i] > z[j]){
				float temp = z[i];
				z[i] = z[j];
				z[j] = temp;
			}
		}
	}
	median.set(x[1], y[1], z[1]);

    for(int i =0; i < 3; i++){
		max[i] = FLT_MIN;
		min[i] = FLT_MAX;
		for(int j = 0; j < 3; j++){
			if(verts[j].e[i] > max[i])
				max[i] = verts[j].e[i];
			if(verts[j].e[i] < min[i])
				min[i] = verts[j].e[i];
		}
	}
    mat = nullptr;
	// vec3 avgNorms = unit_vector((n1 + n2 + n3)/3);
	// printf("verts: %f %f %f, %f %f %f, %f %f %f\n", verts[0].x(), verts[0].y(), verts[0].z(), verts[1].x(), verts[1].y(), verts[1].z(), verts[2].x(), verts[2].y(), verts[2].z());
	// if(avgNorms.x() != surfNorm.x() || avgNorms.y() != surfNorm.y() || avgNorms.z() != surfNorm.z())
	// printf("normals: %f %f %f vs %f %f %f\n", surfNorm.x(), surfNorm.y(), surfNorm.z(), avgNorms.x(), avgNorms.y(), avgNorms.z());
}



lambertian::lambertian(const vec3& a){
	albedo = a;
	emitter = false;
}

__device__ bool lambertian::scatter(const ray& impacting, const hit_record& rec, vec3& att, ray& scattered, curandState* state) const{
	vec3 target = rec.p+rec.normal+random_in_unit_sphere(state);
	scattered = ray(rec.p, target-rec.p);
	att = albedo;
	return true;
}

metal::metal(const vec3& a, const float& f){
	emitter = false;
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
	emitter = false;
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
	if(dotted>0){//if normal and ray are facing same direction
		outward_normal = -rec.normal;
		ni_nt = ior;
		cosine = dotted/impacting.direction().length();
		cosine = sqrtf(1-ior*ior*(1-cosine*cosine));
	}
	else{
		outward_normal = rec.normal;
		ni_nt = 1.0f/ior;
		cosine = -dotted/impacting.direction().length();
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
	}
	return true;
}

__device__ bool refract(const vec3& v, const vec3& n, float ni_nt, vec3& refracted){
	vec3 uv = unit_vector(v);
	float dt = dot(uv, n);
	float discriminant = 1.0f-ni_nt*ni_nt*(1.0f-dt*dt);
	if(discriminant > 0){
		refracted = ni_nt*(uv-n*dt) - n*sqrtf(discriminant);
		return true;
	}
	else return false;
}

__device__ float dielectric::schlick(const float& cosine, const float& indor) const{
	float r0 = (1-indor)/(1+indor);
	r0 = r0*r0;
	return r0 + (1-r0)*pow((1-cosine), 5);
}

__device__ vec3 random_in_unit_sphere(curandState* state){
	vec3 p;
	// do {
		p = 2*vec3(curand_uniform(state),curand_uniform(state),curand_uniform(state)) - vec3(1,1,1);
	// } while(p.squared_length() >= 1);
	return unit_vector(p);
}

// __device__ bool refract(const vec3&  v, const vec3& n, const float& ni_over_nt, vec3& refracted){
// 	vec3 uv = unit_vector(v);
// 	float dt = dot(uv, n);
// 	float discriminant = 1.0-ni_over_nt*ni_over_nt*(1-dt*dt);
// 	if(discriminant > 0){
// 		refracted = ni_over_nt*(uv - n*dt) - n*sqrtf(discriminant);
// 		return true;
// 	}
// 	else return false;
// }

// __device__ bool dielectric::scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState* state) const{
// 	vec3 outward_normal;
// 	vec3 reflected = reflect(r_in.direction(), rec.normal);
// 	float ni_over_nt;
// 	attenuation = vec3(1.0f, 1.0f, 1.0f);
// 	vec3 refracted;
// 	if(dot(r_in.direction(), rec.normal)>0){
// 		outward_normal = -rec.normal;
// 		ni_over_nt = ref_idx;
// 	}
// 	else{
// 		outward_normal = rec.normal;
// 		ni_over_nt = 1.0f/ref_idx;
// 	}
// 	if(refract(r_in.direction(), outward_normal, ni_over_nt, refracted)){
// 		scattered = ray(rec.p, refracted);
// 	}
// 	else{
// 		scattered = ray(rec.p, reflected);
// 		return false;
// 	}
// 	return true;
// }

__device__ light::light(vec3 att){
	attenuation = att;
	emitter = true;
}

__device__ bool light::scatter(const ray& impacting, const hit_record& rec, vec3& att, ray& scattered, curandState* state) const{
	att = attenuation;
	scattered = impacting;
	// printf("light!\n");
	return true;
}

__device__ hitable_list::hitable_list(OBJ **in, int n, int additional){

	list_size = 0;
	// printf("%d\n", n);
	for(int i = 0; i < n; i++){
		// printf("hl i: %d\n", i);
		list_size += in[i]->numFaces;
	}
	// printf("%d\n", list_size);
	list = new hitable*[list_size+additional];
	// int z = 0;
	// for(int i = 0; i < n; i++){
	// 	for(int j = 0; j < in[i]->numFaces; j++){
	// 		// printf("j: %d\n", j);
	// 		list[z] = new Face(in[i]->object[j], new light(vec3(4, 2, 2)));
	// 		// *list[z] = in[z];
	// 		z++;
	// 		if(z%10000 == 0)
	// 			printf("%d\n", z);
	// 	}
	// }
}

OBJ::OBJ(){
    points = 0;
    text = 0;
    normals = 0;
    numP = 0;
    numT = 0;
    numN = 0;
    numFaces = 0;
}

OBJ::OBJ(string fn){
    file = ifstream(fn);
    numP = 0;
    numT = 0;
    numN = 0;
    points = 0;
    text = 0;
    normals = 0;
	numFaces = 0;
	int i = 0;
    while(!file.eof() && !file.fail()){
        char line[1000];
        file.getline(line, 1000);
		parse(line);
		if(i%10000 == 0)
			printf("%d\n", i);
		i++;
	}
	file.close();
}

void OBJ::parse(char* line){
	// printf(line);
	// printf("\n");
    string buf = "";
    bool pp = false, tt = false, nn = false, newFace = false;
    float vec[3] = {0,0,0};
    int index = 0;
    int set[9];
    for(int i = 0; ; i++){
        if(line[i] == '#')
            break;
        if(line[i] == ' ' || line[i] == '\t' || line[i] == '\0'){
            if(!pp && !tt && !nn && !newFace && buf.compare("v") == 0){
                pp = true;
            }
            else if(!tt && !nn && !newFace && buf.compare("vt") == 0){
                tt = true;
            }
            else if(!nn && !newFace && buf.compare("vn") == 0){
                nn = true;
            }
            else if(!newFace && buf.compare("f") == 0){
                newFace = true;
            }
            else if((pp || tt || nn) && index < 3){
                vec[index] = stof(buf);
                index++;
            }
            else if(newFace && index < 3){
                int count = 0;
                string petiteBuf = "";
                for(int j = 0; j < buf.length()+1; j++){
                    if(buf[j] == '/' || buf[j] == '\0'){
						set[index*3 + count] = stoi(petiteBuf)-1;
						// printf("%d\n", set[index*3+count]);
						petiteBuf = "";
                        count++;
                    }
                    else{
                        petiteBuf += buf[j];
                    }
                }
                index++;
            }
            buf = "";
            if(line[i] == '\0')
                break;
            continue;
        }
        buf += line[i];
    }
    if(pp){
        append(points, numP, PBuf, vec3(vec[0], vec[1], vec[2]));
        // numP++;
    }
    else if(tt){
        append(text, numT, TBuf, vec3(vec[0], vec[1], 0.0f));
        // numT++;
    }
    else if(nn){
        append(normals, numN, NBuf, vec3(vec[0], vec[1], vec[2]));
        // numN++;
    }
    else if(newFace){
		// printf("%d: %f %f %f\n", set[0], points[set[0]].x(), points[set[0]].y(), points[set[0]].z());
        // append(Face(points[set[0]], points[set[3]], points[set[6]], text[set[1]], text[set[4]], text[set[7]], normals[set[2]], normals[set[5]], normals[set[8]]));
        append(Face(points[set[0]], points[set[3]], points[set[6]], vec3(), vec3(), vec3(), vec3(), vec3(), vec3()));
		// exit(0);
	}
}

void OBJ::append(vec3*& list, int& size, int& bufSize, const vec3& item){
	if(size+1 > bufSize){
		vec3* temp = new vec3[bufSize+=1000];
		// printf("appending vectors\n");
		for(int i = 0; i < size; i++){
			temp[i] = list[i];
		}		
		if(size > 0)
			delete[] list;
		list = temp;
		// bufSize += 1000;
	}
	list[size] = item;
	size++;
}

void OBJ::append(const Face& item){
	if(numFaces + 1 > faceBuffer){
		Face* temp = new Face[faceBuffer+=1000];
		for(int i = 0; i < numFaces; i++){
			temp[i] = object[i];
		}
		// faceBuffer += 1000;
		if(numFaces > 0)
			delete[] object;
		object = temp;
	}
	object[numFaces] = item;
    numFaces++;
}

__host__ __device__ Face::Face(){
    verts[0] = vec3();
    verts[1] = vec3();
    verts[2] = vec3();
    texts[0] = vec3();
    texts[1] = vec3();
    texts[2] = vec3();
    normals[0] = vec3();
    normals[1] = vec3();
	normals[2] = vec3();
	e[0] = vec3();
	e[1] = vec3();
	e[2] = vec3();
	median = vec3();
	mat = nullptr;
	min[0] = 0;
	min[1] = 0;
	min[2] = 0;
	max[0] = 0;
	max[1] = 0;
	max[2] = 0;
}



__host__ __device__ Face& Face::operator=(const Face& in){
    verts[0] = in.verts[0];
    verts[1] = in.verts[1];
    verts[2] = in.verts[2];
    texts[0] = in.texts[0];
    texts[1] = in.texts[1];
    texts[2] = in.texts[2];
    normals[0] = in.normals[0];
    normals[1] = in.normals[1];
	normals[2] = in.normals[2];
	e[0] = verts[1] - verts[0];
    e[1] = verts[2] - verts[1];
    e[2] = verts[0] - verts[2];
	surfNorm = in.surfNorm;
	mat = in.mat;
	median = in.median;
	min[0] = in.min[0];
	min[1] = in.min[1];
	min[2] = in.min[2];
	max[0] = in.max[0];
	max[1] = in.max[1];
	max[2] = in.max[2];
	// surfNorm.make_unit_vector();
	// surfNorm = unit_vector(surfNorm);
	// vec3 temp = unit_vector(surfNorm);
	return *this;
}

OBJ* OBJ::copyToDevice(){
	// printf("entering ctd\n");
	gpuErrchk(cudaDeviceSynchronize());
	// printf("synching\n");
	// printf("%d\n", sizeof(Face));
	// printf("%p\n", &numFaces);
	// printf("trying to malloc %d bytes\n", (sizeof(Face))*(numFaces));
	Face *d_faces, *oldFaces;
	gpuErrchk(cudaMalloc((void**)&d_faces, sizeof(Face)*this->numFaces));
	// cout<<"mallocced faces\n";
    gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaMemcpy(d_faces, object, sizeof(Face)*this->numFaces, cudaMemcpyHostToDevice));
	// cout<<"copied faces\n";
    oldFaces = object;
    object = d_faces;
    gpuErrchk(cudaDeviceSynchronize());
    OBJ* d_obj;
    gpuErrchk(cudaMalloc((void**)&d_obj, sizeof(OBJ)));
    gpuErrchk(cudaMemcpy(d_obj, this, sizeof(OBJ), cudaMemcpyHostToDevice));
    gpuErrchk(cudaDeviceSynchronize());
    object = oldFaces;
    return d_obj;
}

__device__ bool OBJ::hit(const ray& r, const float& tmin, float& tmax, hit_record& rec) const{
    for(int i = 0; i < numFaces; i++){
        if(object[i].hit(r, tmin, tmax, rec))
            return true;
    }
    return false;
}

__host__ __device__ Face::Face(const Face& in){
    verts[0] = in.verts[0];
    verts[1] = in.verts[1];
    verts[2] = in.verts[2];
    texts[0] = in.texts[0];
    texts[1] = in.texts[1];
    texts[2] = in.texts[2];
    normals[0] = in.normals[0];
    normals[1] = in.normals[1];
	normals[2] = in.normals[2];
	e[0] = verts[1] - verts[0];
    e[1] = verts[2] - verts[1];
    e[2] = verts[0] - verts[2];
	// e[0] = in.e[0];
	// e[1] = in.e[1];
	// e[2] = in.e[2];
	median = in.median;
	min[0] = in.min[0];
	min[1] = in.min[1];
	min[2] = in.min[2];
	max[0] = in.max[0];
	max[1] = in.max[1];
	max[2] = in.max[2];
	// surfNorm.make_uni
	surfNorm = in.surfNorm;
	mat = in.mat;
}

__host__ __device__ Face::Face(const Face& in, material* m){
	// printf("creating mat %p\n", m);
	mat = m;
	verts[0] = in.verts[0];
    verts[1] = in.verts[1];
    verts[2] = in.verts[2];
    texts[0] = in.texts[0];
    texts[1] = in.texts[1];
    texts[2] = in.texts[2];
    normals[0] = in.normals[0];
    normals[1] = in.normals[1];
	normals[2] = in.normals[2];
	e[0] = verts[1] - verts[0];
    e[1] = verts[2] - verts[1];
    e[2] = verts[0] - verts[2];
    min[0] = in.min[0];
	min[1] = in.min[1];
	min[2] = in.min[2];
	max[0] = in.max[0];
	max[1] = in.max[1];
	max[2] = in.max[2];
	// surfNorm.make_uni
    median = in.median;
	// printf("%f %f, %f %f, %f %f\n", verts[0].x(), in.verts[0].x(), verts[1].x(), in.verts[1].x(), verts[2].x(), in.verts[2].x());
	surfNorm = in.surfNorm;
}

__device__ sss::sss(material* surf, const float& d, const vec3& internal){
	attenuation = internal;
	depth = d;
	surface = surf;
}

__device__ bool sss::scatter(const ray& impacting, const hit_record& rec, vec3& att, ray& scattered, curandState* state) const{
	if(dot(unit_vector(impacting.direction()), unit_vector(rec.normal)) > 0){
		// printf("inside out %f %f %f, %f %f %f\n", rec.p.x(), rec.p.y(), rec.p.z(), impacting.direction().x(), impacting.direction().y(), impacting.direction().z());
		vec3 temp = impacting.direction();
		vec3 tAtt;
		ray tScattered;
		surface->scatter(impacting, rec, tAtt, tScattered, state);
		do{
			temp = random_in_unit_sphere(state);
		}	while(dot(temp, rec.normal) <= 0);
		scattered = ray(rec.p, impacting.direction() + temp);
		float l = depth/(rec.p - impacting.origin()).length();
		if(l > 1)
			l = 1;
		att = vec3(l, l, l);
		att = tAtt;
		att /= 2;
		return true;
		// return surface->scatter(impacting, rec, att, scattered, state);
	}
	else if(curand_uniform(state) > 0.5f){//determines if reflecting off surface
		return surface->scatter(impacting, rec, att, scattered, state);
	}
	else{//or going inside
		// vec3 internalTarget = (dot(rec.p, -rec.normal)/rec.normal.length()) * unit_vector(-rec.normal);
		vec3 temp = impacting.direction();
		do{
			temp = random_in_unit_sphere(state);
		}	while(dot(temp, rec.normal) >= 0);
		// printf("%f %f %f, %f %f %f\n", rec.normal.x(), rec.normal.y(), rec.normal.z(), temp.x(), temp.y(), temp.z());
		scattered = ray(rec.p, impacting.direction() + temp);
		att = attenuation;
		return true;
	}
}

__device__ TreeNode::TreeNode(){
	parent = nullptr;
	obj = nullptr;
	dim = 0;
	r = nullptr;
	l = nullptr;
	within = 0;
	contained = nullptr;
	min[0] = FLT_MAX;
	min[1] = FLT_MAX;
	min[2] = FLT_MAX;
	max[0] = FLT_MIN;
	max[1] = FLT_MIN;
	max[2] = FLT_MIN;
}
__device__ TreeNode::TreeNode(Face* in, TreeNode* par){
	// printf("inserted\n");
	// obj = nullptr;
	l = r = nullptr;
	parent = par;
	min[0] = in->min[0];
	min[1] = in->min[1];
	min[2] = in->min[2];
	max[0] = in->max[0];
	max[1] = in->max[1];
	max[2] = in->max[2];
	within = 0;
	contained = nullptr;
	// surfNorm.make_uni
	if(par != nullptr)
		dim = parent->dim<2 ? par->dim+1 : 0;
	else{
		dim = 0;
	}
	median = in->median;
	p = median[dim];	//note that this is mean value, may have to use median (probably won't matter)
	obj = in;
}
__device__ bool TreeNode::hit(const ray& r, const float& tmin, float& tmax, hit_record& rec) const{
	if(obj != nullptr){
		return obj->hit(r, tmin, tmax, rec);
	}
	return false;
}

__device__ bool TreeNode::withinBB(const vec3& p){
	for(int i = 0; i < 2; i++){
		if(p.e[i] > max[i] || p.e[i] < min[i])
			return false;
	}
	return true;
}

__device__ TriTree::TriTree(){
	numNodes = 0;
	head = nullptr;
}
__device__ void TriTree::insert(Face* in){ //TODO: Need to preprocess total triangles, create bounding box at each node level and split, then create leaf nodes. This takes care of order of insertion problem	
	TreeNode* cur = head, *prev = nullptr;
	numNodes++;
	while(cur != nullptr){
		cur->within = 0;
		// cur->medSum += in->median;
		// cur->median = cur->medSum/cur->within;
		for(int i = 0; i < 3; i++){
			if(in->max[i] > cur->max[i])
				cur->max[i] = in->max[i];
			if(in->min[i] < cur->min[i])
				cur->min[i] = in->min[i];
		}
		prev = cur;
		if(in->median.e[cur->dim] < cur->p){
			cur = cur->l;
			if(cur == nullptr){
				prev->l = new TreeNode(in, prev);
				break;
			}
		}
		else{
			cur = cur->r;
			if(cur == nullptr){
				prev->r = new TreeNode(in, prev);
				break;
			}
		}
	}
	if(head == nullptr){
		head = new TreeNode(in, nullptr);
	}
}



__device__ bool Face::hit(const ray& r, const float& t_min, float& t_max, hit_record& rec) const{//need to store non-ray derived values to reduce comp time
	vec3 one = vec3(1,1,1);
	for(int i = 0; i < 3; i++){
		if(dot(verts[i], r.direction()) - dot(one, t_max*r.direction()) > 0)
			break;
		if(i == 2)
			return false;
	}
	float NdotDir = dot(surfNorm, r.direction());
	if(abs(NdotDir) < .001){
		return false;
	}
	float D = dot(surfNorm, verts[0]);
	
	float temp = -((dot(surfNorm, r.origin())-D)/NdotDir);
    vec3 p = (r.origin())+temp*(r.direction());
    vec3 diff[3];
    
    diff[0] = p - verts[0];
    diff[1] = p - verts[1];
	diff[2] = p - verts[2];

	for(int i = 0; i < 3; i++){
        if(dot(surfNorm, cross(e[i], diff[i])) < 0){
			return false;
		}
	}

    if(temp < t_max && temp > t_min){
		t_max = temp;
		rec.mat = mat;
		rec.t = temp;
		rec.p = p;
		rec.normal = surfNorm;
        return true;
    }
    return false;
}

__device__ bool TriTree::positionOnPlane(const ray& r, TreeNode* n, vec3& poi) const {	//TODO: fix error where half the triangles don't render at all (has to do with left/right in plane for each tri)
	vec3 planeNormal;//, planePos;
	int dimension = n->dim;
	dimension = dimension == 0 ? 2 : dimension-1;
	planeNormal.e[dimension] = 1;
	float denom = dot(r.direction(), planeNormal);

	if(abs(denom) < 0.0001f)
		return false;

	vec3 vAdjusted = r.direction()*(-dot((n->median - r.origin()), planeNormal)/denom);
	poi = r.origin() - vAdjusted;
	return true;
}

__device__ bool TriTree::hit(const ray& r, const float& tmin, float& tmax, hit_record& rec) const{	//Need to fix math/logic error in positionOnPlane
	TreeNode* cur = head;//, *t = nullptr;
	// printf("numNodes: %d\n", numNodes);
	TreeNode** stack = new TreeNode*[numNodes];
	int stackSize = 0;
	bool anyhit = false;
	vec3 pos;
	hit_record temprec;
	float closest = tmax;
	bool *position = new bool[numNodes];
	do{
		while(cur != nullptr){
			stack[stackSize++] = cur;
			if(position[stackSize-1] = positionOnPlane(r, cur, pos)){
				if(cur->l == nullptr && cur->r == nullptr){
					// printf("within: %d\n", cur->within);
					for(int i = 0; i < cur->within; i++){
						if(cur->contained[i]->hit(r, tmin, closest, temprec)){
							if(closest < tmax){
								rec = temprec;
								tmax = closest;
								anyhit = true;
							}
						}
					}
				}
				if(pos.e[cur->dim] < cur->median[cur->dim]){
					cur = cur->l;
				}
				else if(pos.e[cur->dim] >= cur->median[cur->dim]){
					cur = cur->r;
				}
			}
			else{
				cur = cur->l;
			}
		}
		while(stackSize > 0 && (position[stackSize-1] || stack[stackSize-1]->r == nullptr)){
			stackSize--;
		}
		if(stackSize > 0){
			cur = stack[stackSize-1]->r;
			stackSize--;
		}
	}	while(stackSize > 0);
	delete[] stack;
	delete[] position;
	return anyhit;
}

__device__ void sortInsertion(int max, float* mx, float* my, float* mz, const vec3& med){
	const float *median = med.e;

	for(int i = 0; i <= max; i++){
		if(i == max){
			mx[i] = median[0];
		}
		else{
			if(median[0] < mx[i]){
				for(int r = max; r > i; r--){
					mx[r] = mx[r-1];
				}
				mx[i] = median[0];
				break;
			}
		}
	}
	for(int i = 0; i <= max; i++){
		if(i == max){
			my[i] = median[1];
		}
		else{
			if(median[1] < my[i]){
				for(int r = max; r > i; r--){
					mx[r] = my[r-1];
				}
				my[i] = median[1];
				break;
			}
		}
	}
	for(int i = 0; i <= max; i++){
		if(i == max){
			mz[i] = median[2];
		}
		else{
			if(median[2] < mz[i]){
				for(int r = max; r > i; r--){
					mz[r] = mz[r-1];
				}
				mz[i] = median[2];
				break;
			}
		}
	}
}

__device__ TriTree* OBJ::toTree(){
	int list_size = numFaces;
	TreeNode** stack = new TreeNode*[list_size];
	unsigned int stackSize = 0, maxSize = 0;
	stack[stackSize++] = new TreeNode;
	stack[stackSize-1]->contained = new Face*[list_size];
	float *mx = new float[list_size], *my = new float[list_size], *mz = new float[list_size];
	
	for(int i = 0; i < list_size; i++){
		// stack[stackSize-1]->contained[i] = new Face(object[i], new lambertian(vec3(0.0f, 0.2f, 0.0f)));
		stack[stackSize-1]->contained[i] = new Face(object[i], new sss( new lambertian(vec3(0.0f, 0.2f, 0.0f)), 0.05f, vec3(1.0f, 0.25f, 0.2f)));

		sortInsertion(i, mx, my, mz, stack[stackSize-1]->contained[i]->median);
	}

	if(list_size % 2 != 0){
		stack[stackSize-1]->median.set(mx[list_size/2], my[list_size/2], mz[list_size/2]);
	}
	else{
		stack[stackSize-1]->median.set((mx[list_size/2]+mx[list_size/2-1])/2, (my[list_size/2]+my[list_size/2-1])/2, (mz[list_size/2]+mz[list_size/2-1])/2);
	}

	stack[stackSize-1]->within = list_size;
	stack[stackSize-1]->p = stack[stackSize-1]->median[stack[stackSize-1]->dim];
	
	while(stackSize > 0){
		while(stack[stackSize-1]->within > 1 && stack[stackSize-1]->l == nullptr){
			short d = stack[stackSize-1]->dim;
			Face** t = new Face*[stack[stackSize-1]->within];
			int s = 0;
			TreeNode* temp = new TreeNode();
			for(int i = 0; i < stack[stackSize-1]->within; i++){
				if(stack[stackSize-1]->contained[i]->median.e[d] < stack[stackSize-1]->median.e[d]){
					t[s] = stack[stackSize-1]->contained[i];
					sortInsertion(s, mx, my, mz, stack[stackSize-1]->contained[i]->median);
					for(int j = 0; j < 3; j++){
						if(t[s]->max[j] > temp->max[j])
							temp->max[j] = t[s]->max[j];
						if(t[s]->min[j] < temp->min[j])
							temp->min[j] = t[s]->min[j];
					}
					s++;
				}
			}
			Face** tt = new Face*[s];
			for(int i = 0; i < s; i++){
				tt[i] = t[i];
			}
			delete[] t;
			if(s > 0){
				temp->dim = d==2?0:d+1;
				if(s % 2 != 0){
					temp->median.set(mx[s/2], my[s/2], mz[s/2]);
				}
				else{
					temp->median.set((mx[s/2]+mx[s/2-1])/2, (my[s/2]+my[s/2-1])/2, (mz[s/2]+mz[s/2-1])/2);
				}
				temp->contained = tt;
				temp->parent = stack[stackSize-1];
				temp->within = s;
				temp->p = temp->median.e[temp->dim];
				stack[stackSize-1]->l = temp;
				if(stackSize < list_size-1){
					stack[stackSize++] = temp;
					maxSize++;
					if(stackSize > 3 && s == stack[stackSize-1]->parent->parent->parent->within){
						stackSize-=3;
						break;
					}
					if(s == 1)
						stack[stackSize-1]->obj = stack[stackSize-1]->contained[0];
				}
				else{
					stackSize--;
					break;
				}
			}
			else{
				break;
			}
			// printf("%d\n", maxSize);
		}
		if(stack[stackSize-1]->within > 1 && stack[stackSize-1]->r == nullptr){
			short d = stack[stackSize-1]->dim;
			Face** t = new Face*[stack[stackSize-1]->within];
			int s = 0;
			// vec3 m;
			TreeNode* temp = new TreeNode();
			// vec3 curMed = stack[stackSize-1]->median;
			for(int i = 0; i < stack[stackSize-1]->within; i++){
				if(stack[stackSize-1]->contained[i]->median.e[d] >= stack[stackSize-1]->median.e[d]){
					t[s] = stack[stackSize-1]->contained[i];
					sortInsertion(s, mx, my, mz, stack[stackSize-1]->contained[i]->median);
					for(int j = 0; j < 3; j++){
						if(t[s]->max[j] > temp->max[j])
							temp->max[j] = t[s]->max[j];
						if(t[s]->min[j] < temp->min[j])
							temp->min[j] = t[s]->min[j];
					}
					s++;
				}
			}
			Face** tt = new Face*[s];
			for(int i = 0; i < s; i++){
				tt[i] = t[i];
			}
			delete[] t;
			if(s > 0){
				temp->dim = d==2?0:d+1;
				if(s % 2 != 0){
					temp->median.set(mx[s/2], my[s/2], mz[s/2]);
				}
				else{
					temp->median.set((mx[s/2]+mx[s/2-1])/2, (my[s/2]+my[s/2-1])/2, (mz[s/2]+mz[s/2-1])/2);
				}
				temp->contained = tt;
				temp->parent = stack[stackSize-1];
				temp->within = s;
				temp->p = temp->median.e[temp->dim];
				stack[stackSize-1]->r = temp;
				if(stackSize < list_size-1){
					stack[stackSize++] = temp;
					maxSize++;
					if(stackSize > 3 && s == stack[stackSize-1]->parent->parent->parent->within){
						stackSize-=3;
						continue;
					}
					if(s == 1)
						stack[stackSize-1]->obj = stack[stackSize-1]->contained[0];
				}
				else stackSize--;
				
			}
			else stackSize--;
		}
		else stackSize--;
		// printf("%d\n", maxSize);
	}

	TriTree* tree = new TriTree;
	tree->head = stack[0];
	tree->numNodes = maxSize;
	return tree;
}

__device__ void TriTree::print(){
	TreeNode* cur = head;//, *t = nullptr;
	printf("numNodes: %d\n", numNodes);
	TreeNode** stack = new TreeNode*[numNodes];
	int stackSize = 0;
	do{
		while(cur != nullptr){
			stack[stackSize++] = cur;
			if(cur->r == nullptr && cur->l == nullptr){
				printf("%p %f %f %f %d\n", cur, cur->median.x(), cur->median.y(), cur->median.z(), cur->within);
				break;
			}
			printf("%p %p %p\n", cur, cur->l, cur->r);
			cur = cur->l;
		}
		while(stackSize > 0 && stack[stackSize-1]->r == nullptr){
			stackSize--;
		}
		if(stackSize > 0){
			cur = stack[stackSize-1]->r;
			stackSize--;
		}
	}	while(stackSize > 0);
	delete[] stack;
}

// hitable* TriTree::copyToDevice(){
// 	return descendCopy(head);
// }

// TreeNode* TriTree::descendCopy(TreeNode* in){
// 	TreeNode *r, *l, *par;
// 	TreeNode* d_in;
// 	Face* d_obj;
// 	TreeNode *oldR = in->r, *oldL = in->l;
// 	if(in->r != nullptr){
// 		r = descendCopy(in->r);
// 	}	else r = nullptr;
// 	if(in->l != nullptr){
// 		l = descendCopy(in->l);
// 	}	else l = nullptr;
// 	cudaMalloc((void**)&d_obj, sizeof(Face));
// 	cudaMemcpy(Face)
// 	cudaMalloc((void**)&d_in, sizeof(TriNode));
// 	// cudaMemcpy()
// }