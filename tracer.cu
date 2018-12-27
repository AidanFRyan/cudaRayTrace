#include "tracer.h"
#include <typeinfo>

__device__ d_vec3::d_vec3(){
	e[0] = 0;
	e[1] = 0;
	e[2] = 0;
}
__device__ d_vec3::d_vec3(cuhalf e0, cuhalf e1, cuhalf e2){
	e[0] = e0;
	e[1] = e1;
	e[2] = e2;
}
__device__ d_vec3::d_vec3(const d_vec3& v){
	e[0] = v.e[0];
	e[1] = v.e[1];
	e[2] = v.e[2];
}
__device__ cuhalf d_vec3::x() const{
	return e[0];
}
__device__ cuhalf d_vec3::y() const{
	return e[1];
}
__device__ cuhalf d_vec3::z() const{
	return e[2];
}
__device__ cuhalf d_vec3::r() const{
	return e[0];
}
__device__ cuhalf d_vec3::g() const{
	return e[1];
}
__device__ cuhalf d_vec3::b() const{
	return e[2];
}

__device__ d_vec3& d_vec3::operator=(const d_vec3& v){
	e[0] = v.e[0];
	e[1] = v.e[1];
	e[2] = v.e[2];
	return *this;
}
__device__ const d_vec3& d_vec3::operator+() const{
	return *this;
}
__device__ d_vec3 d_vec3::operator-() const{
	return d_vec3(-e[0], -e[1], -e[2]);
}
__device__ cuhalf d_vec3::operator[](int i) const{
	if(i < 3 && i > 0)
		return e[i];
	else return 0;
}
__device__ cuhalf& d_vec3::operator[](int i){
	return e[i];
}

__device__ d_vec3& d_vec3::operator+=(const d_vec3 &v2){
	e[0] += v2.e[0];
	e[1] += v2.e[1];
	e[2] += v2.e[2];
	return *this;
}
__device__ d_vec3& d_vec3::operator-=(const d_vec3 &v2){
	e[0] -= v2.e[0];
	e[1] -= v2.e[1];
	e[2] -= v2.e[2];
	return *this;
}
__device__ d_vec3& d_vec3::operator*=(const d_vec3 &v2){
	e[0] *= v2.e[0];
	e[1] *= v2.e[1];
	e[2] *= v2.e[2];
	return *this;}
	__device__ d_vec3& d_vec3::operator/=(const d_vec3 &v2){
	e[0] /= v2.e[0];
	e[1] /= v2.e[1];
	e[2] /= v2.e[2];
	return *this;
}
__device__ d_vec3& d_vec3::operator*=(const cuhalf t){
	e[0] *= t;
	e[1] *= t;
	e[2] *= t;
	return *this;
}
__device__ d_vec3& d_vec3::operator/=(const cuhalf t){
	e[0] /= t;
	e[1] /= t;
	e[2] /= t;
	return *this;
}

__device__ cuhalf d_vec3::length() const{
	return sqrtf(e[0]*e[0] + e[1]*e[1] + e[2]*e[2]);
}
__device__ cuhalf d_vec3::squared_length() const{
	return e[0]*e[0] + e[1]*e[1] + e[2]*e[2];
}
__device__ void d_vec3::make_unit_vector(){
	cuhalf k = 1.0f / hsqrt(e[0]*e[0] + e[1]*e[1] + e[2] * e[2]);
	e[0] *= k;
	e[0] *= k;
	e[0] *= k;
}

__device__ cuhalf d_vec3::dot(const d_vec3 &v2){
	return e[0]*v2.e[0] + e[1]*v2.e[1] + e[2]*v2.e[2];
}
__device__ d_vec3 d_vec3::cross(const d_vec3 &v2){
	return d_vec3(e[1]*v2.e[2] - e[2]*v2.e[1], (-(e[0]*v2.e[2] - e[2]*v2.e[0])), e[0]*v2.e[1] - e[1]*v2.e[0]);
}

// istream& operator>>(istream &is, d_vec3 &t){
// 	is>>t.e[0]>>t.e[1]>>t.e[2];
// 	return is;
// }
// ostream& operator<<(ostream &os, d_vec3 &t){
// 	os<<t.e[0]<<' '<<t.e[1]<<' '<<t.e[2];
// 	return os;
// }
__device__ d_vec3 operator+(const d_vec3 &v1, const d_vec3 &v2){
	return d_vec3(v1.e[0]+v2.e[0], v1.e[1]+v2.e[1], v1.e[2]+v2.e[2]);
}
__device__ d_vec3 operator-(const d_vec3 &v1, const d_vec3 &v2){
	return d_vec3(v1.e[0]-v2.e[0], v1.e[1]-v2.e[1], v1.e[2]-v2.e[2]);
}
__device__ d_vec3 operator*(const d_vec3 &v1, const d_vec3 &v2){
	return d_vec3(v1.e[0]*v2.e[0], v1.e[1]*v2.e[1], v1.e[2]*v2.e[2]);
}
__device__ d_vec3 operator/(const d_vec3 &v1, const d_vec3 &v2){
	return d_vec3(v1.e[0]/v2.e[0], v1.e[1]/v2.e[1], v1.e[2]/v2.e[2]);
}

__device__ d_vec3 operator*(const cuhalf t, const d_vec3 &v){
	return d_vec3(v.e[0]*t, v.e[1]*t, v.e[2]*t);
}
__device__ d_vec3 operator*(const d_vec3 &v, const cuhalf t){
	return d_vec3(v.e[0]*t, v.e[1]*t, v.e[2]*t);
}
__device__ d_vec3 operator/(const d_vec3 v, cuhalf t){
	return d_vec3(v.e[0]/t, v.e[1]/t, v.e[2]/t);
}

__device__ cuhalf dot(const d_vec3 &v1, const d_vec3 &v2){
	return v1.e[0]*v2.e[0] + v1.e[1]*v2.e[1] + v1.e[2]*v2.e[2];
}
__device__ d_vec3 cross(const d_vec3 &v1, const d_vec3 &v2){
	return d_vec3(v1.e[1]*v2.e[2] - v1.e[2]*v2.e[1], (-(v1.e[0]*v2.e[2] - v1.e[2]*v2.e[0])), v1.e[0]*v2.e[1] - v1.e[1]*v2.e[0]);
}

__device__ d_vec3 unit_vector(d_vec3 v){
	return v/v.length();
}

__device__ void d_vec3::set(cuhalf e0, cuhalf e1, cuhalf e2){
	e[0] = e0;
	e[1] = e1;
	e[2] = e2;
}

__device__ ray::ray(){}
__device__ ray::ray(const d_vec3& a, const d_vec3& b){
	A = a;
	B = b;
}
__device__ d_vec3 ray::origin() const{
	return A;
}
__device__ d_vec3 ray::direction() const{
	return B;
}
__device__ d_vec3 ray::p(cuhalf t) const{
	return A + t*B;
}

__device__ sphere::sphere(){
	center = d_vec3(0,0,0);
	radius = 0;
}
__device__ sphere::sphere(d_vec3 cen, cuhalf r, material* m){
	center = cen;
	radius = r;
	mat = m;
}
__device__ bool sphere::hit(const ray& r, const cuhalf& tmin, cuhalf& tmax, hit_record& rec) const{
	d_vec3 oc = r.origin() - center;
	cuhalf a = dot(r.direction(), r.direction());
	cuhalf b = dot(oc, r.direction());
	cuhalf c = dot(oc, oc) - radius*radius;
	cuhalf discriminant = b*b-a*c;
	// printf("%f\n", radius);
	if (discriminant > 0){
		rec.mat = mat;
		cuhalf temp = (-b - sqrtf(b*b-a*c))/a;
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

__device__ hitable_list::hitable_list(){
	list = 0;
	list_size = 0;
}
__device__ hitable_list::hitable_list(hitable **list, int n){
	this->list = list;
	list_size = n;
}
// hitable** hitable_list::listPointer(){
// 	return d_list;
// }
__device__ bool hitable_list::hit(const ray& r, const cuhalf& tmin, cuhalf& tmax, hit_record& rec) const{
	hit_record temp_rec;
	bool anyHits = false;
	cuhalf closest = tmax;
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
__device__ d_vec3 random_in_unit_disk(curandState* state){
	// curandState state;
	// curand_init(1234, threadIdx.x+blockDim.x*blockIdx.x, 0, &state);
	d_vec3 p;
	do{
		p = 2.0f*d_vec3(curand_uniform(state), curand_uniform(state), 0) - d_vec3(1,1,0);
	}while(dot(p,p) >= 1.0f);
	return p;
}

__device__ camera::camera(){
	ulc = d_vec3(-2, 1, -1);
	horizontal = d_vec3(4, 0, 0);
	vertical = d_vec3(0,2,0);
	origin = d_vec3(0,0,0);
}

__device__ camera::camera(cuhalf vfov, cuhalf aspect){
	vfov *= CUDA_PI/180;
	cuhalf cuhalfHeight = tanf(vfov/2);
	cuhalf cuhalfWidth = aspect*cuhalfHeight;
	ulc = d_vec3(-cuhalfWidth, cuhalfHeight, -1);
	horizontal = d_vec3(2*cuhalfWidth, 0, 0);
	vertical = d_vec3(0,2*cuhalfHeight,0);
	origin = d_vec3(0,0,0);
}
__device__ camera::camera(d_vec3 o, d_vec3 lookAt, d_vec3 vup, cuhalf vfov, cuhalf aspect){
	// d_vec3 u, v, w;
	lens_radius=0;
	vfov *= CUDA_PI/180;
	cuhalf cuhalfHeight = tanf(vfov/2);
	cuhalf cuhalfWidth = aspect*cuhalfHeight;
	origin = o;
	w = unit_vector(o-lookAt);
	u = unit_vector(cross(vup, w));
	v = cross(w, u);
	ulc = d_vec3(-cuhalfWidth, cuhalfHeight, -1);
	ulc = origin - cuhalfWidth*u + cuhalfHeight*v - w;
	horizontal = 2*cuhalfWidth*u;
	vertical = 2*cuhalfHeight*v;
}
__device__ camera::camera(d_vec3 o, d_vec3 lookAt, d_vec3 vup, cuhalf vfov, cuhalf aspect, cuhalf aperture, cuhalf focus_dist){
	// d_vec3 u, v, w;
	lens_radius = aperture/2;
	vfov *= CUDA_PI/180;
	cuhalf cuhalfHeight = tanf(vfov/2);
	cuhalf cuhalfWidth = aspect*cuhalfHeight;
	origin = o;
	w = unit_vector(o-lookAt);
	u = unit_vector(cross(vup, w));
	v = cross(w, u);
	ulc = d_vec3(-cuhalfWidth, cuhalfHeight, -1);
	ulc = origin - cuhalfWidth*focus_dist*u + cuhalfHeight*focus_dist*v - focus_dist*w;
	horizontal = 2*cuhalfWidth*u*focus_dist;
	vertical = 2*cuhalfHeight*v*focus_dist;
}
__device__ void camera::get_ray(const cuhalf& s, const cuhalf& t, ray& r, curandState* state){
	d_vec3 rd = lens_radius * random_in_unit_disk(state);
	d_vec3 offset = u*rd.x() + v*rd.y();
	r = ray(origin + offset, ulc+s*horizontal-t*vertical-origin-offset);
}

__device__ lambertian::lambertian(const d_vec3& a){
	albedo = a;
	emitter = false;
}

__device__ bool lambertian::scatter(const ray& impacting, const hit_record& rec, d_vec3& att, ray& scattered, curandState* state) const{
	d_vec3 target = rec.p+rec.normal+random_in_unit_sphere(state);
	scattered = ray(rec.p, target-rec.p);
	att = albedo;
	return true;
}

__device__ metal::metal(const d_vec3& a, const cuhalf& f){
	emitter = false;
	albedo = a;
	if(f<1)
		fuzzy = f;
	else
		fuzzy = 1;
}

__device__ d_vec3 reflect(const d_vec3& v, const d_vec3& n){
	return v - 2*dot(v,n)*n;
}

__device__ bool metal::scatter(const ray& impacting, const hit_record& rec, d_vec3& att, ray& scattered, curandState* state) const{
	d_vec3 reflected = reflect(unit_vector(impacting.direction()), rec.normal);
	if(fuzzy >= 0.01f)
		scattered = ray(rec.p, reflected + fuzzy*random_in_unit_sphere(state));
	else
		scattered = ray(rec.p, reflected);
	att = albedo;
	return (dot(scattered.direction(), rec.normal) > 0);
}

__device__ dielectric::dielectric(const cuhalf& i){
	ior = i;
	emitter = false;
}

__device__  bool dielectric::scatter(const ray& impacting, const hit_record& rec, d_vec3& att, ray& scattered, curandState* state) const{
	d_vec3 outward_normal;
	d_vec3 reflected = reflect(impacting.direction(), rec.normal);
	cuhalf ni_nt;
	att = d_vec3(1.0f, 1.0f, 1.0f);
	d_vec3 refracted;
	cuhalf reflect_prob;
	cuhalf cosine;
	cuhalf dotted = dot(impacting.direction(), rec.normal);
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

__device__ bool refract(const d_vec3& v, const d_vec3& n, cuhalf ni_nt, d_vec3& refracted){
	d_vec3 uv = unit_vector(v);
	cuhalf dt = dot(uv, n);
	cuhalf discriminant = 1.0f-ni_nt*ni_nt*(1.0f-dt*dt);
	if(discriminant > 0){
		refracted = ni_nt*(uv-n*dt) - n*sqrtf(discriminant);
		return true;
	}
	else return false;
}

__device__ cuhalf dielectric::schlick(const cuhalf& cosine, const cuhalf& indor) const{
	cuhalf r0 = (1-indor)/(1+indor);
	r0 = r0*r0;
	return r0 + (1-r0)*powf((1-cosine), 5);
}

__device__ d_vec3 random_in_unit_sphere(curandState* state){
	d_vec3 p;
	do {
		p = 2*d_vec3(curand_uniform(state),curand_uniform(state),curand_uniform(state)) - d_vec3(1,1,1);
	} while(p.squared_length() >= 1);
	return p;
}

// __device__ bool refract(const d_vec3&  v, const d_vec3& n, const cuhalf& ni_over_nt, d_vec3& refracted){
// 	d_vec3 uv = unit_vector(v);
// 	cuhalf dt = dot(uv, n);
// 	cuhalf discriminant = 1.0-ni_over_nt*ni_over_nt*(1-dt*dt);
// 	if(discriminant > 0){
// 		refracted = ni_over_nt*(uv - n*dt) - n*sqrtf(discriminant);
// 		return true;
// 	}
// 	else return false;
// }

// __device__ bool dielectric::scatter(const ray& r_in, const hit_record& rec, d_vec3& attenuation, ray& scattered, curandState* state) const{
// 	d_vec3 outward_normal;
// 	d_vec3 reflected = reflect(r_in.direction(), rec.normal);
// 	cuhalf ni_over_nt;
// 	attenuation = d_vec3(1.0f, 1.0f, 1.0f);
// 	d_vec3 refracted;
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

__device__ light::light(d_vec3 att){
	attenuation = att;
	emitter = true;
}

__device__ bool light::scatter(const ray& impacting, const hit_record& rec, d_vec3& att, ray& scattered, curandState* state) const{
	att = attenuation;
	scattered = impacting;
	return true;
}

h_vec3::h_vec3(){
	e[0] = 0;
	e[1] = 1;
	e[2] = 2;
}