#include "tracer.h"
// float hit_sphere(const vec3& center, float radius, const ray& r){
// 	vec3 oc = r.origin() - center;
// 	float a = dot(r.direction(), r.direction());
// 	float b = 2 * dot(oc, r.direction());
// 	float c = dot(oc, oc) - radius*radius;
// 	float discriminant = b*b - 4*a*c;
// 	if (discriminant < 0){
// 		return -1;
// 	}
// 	else{
// 		return (-b - sqrtf(discriminant))/(2*a);
// 	}
// }
__device__ vec3 random_in_unit_sphere(curandState* state){
	// curandState state;
	// printf("Finding rand\n");
	vec3 p;
	do {
		p = 2*vec3(curand_uniform(state),curand_uniform(state),curand_uniform(state)) - vec3(1,1,1);
	} while(p.squared_length() >= 1);
	return p;
}



__global__ void worldGenerator(hitable** list, hitable_list** world, int wSize, curandState* state){
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	if(i==0){
		// hitable* list[2];
		list[0] = new sphere(vec3(0,0,-1), 0.5, new lambertian(vec3(0.8f, 0.3f, 0.3f), state));
		list[1] = new sphere(vec3(0,-100.5, -1), 100, new lambertian(vec3(0.8f, 0.8f, 0.0f), state));
		list[2] = new sphere(vec3(1, 0, -1), 0.5, new metal(vec3(0.8f, 0.6f, 0.2f), 0, state));
		// list[3] = new sphere(vec3(-1, 0, -1), 0.5, new metal(vec3(0.8f, 0.8f, 0.8f), 1.0f, state));
		list[3] = new sphere(vec3(-1, 0, -1), 0.5f, new dielectric(1.5f, state));
		// list[4] = new sphere(vec3(0,1,-1), 0.5f, new metal(vec3(0.8f, 0.8f, 0.9f), 0, state));
		*world = new hitable_list(list, wSize);
		// printf("wrldGen: %p %p\n", (*world)->list, (*world)->list[0]);
	}
}

__global__ void initRand(int n, int cluster, int aa, float* aaRands, curandState* state){
	int index = threadIdx.x+blockDim.x*blockIdx.x;
	// int pixelNum = index*cluster;	
	// while(pixelNum < n){
		
		// for(int i = 0; i < cluster && (pixelNum+i) < n; i++){
			// printf("%d\n", pixelNum+i);
	curand_init(clock64(), index, 0, state);
		// }
		// pixelNum += blockDim.x*gridDim.x;
	// }
	for(int i = 0; i < aa; i++){
		aaRands[i] = curand_uniform(state);
	}
}

__device__ vec3 color(const ray& r, hitable_list* world, curandState* state){
	
	// printf("Color %d\n", iters);
	// float t = hit_sphere(vec3(0,0,-1), 0.5, r);
	// printf("%p, %d\n", world, world->list_size);
	float max = FLT_MAX;
	ray curRay = r;
	vec3 curLight = vec3(1,1,1);
	for(int i = 0; i < 10; i++){
		hit_record rec;
		// vec3 N = unit_vector(r.p(t) - vec3(0,0,-1));
		if(world->hit(curRay, 0.001, max, rec)){
			ray scattered;
			vec3 attenuation;
			if(rec.mat->scatter(r, rec, attenuation, scattered)){
				curLight *= attenuation;
				curRay = scattered;
			}
			else{
				return vec3(0,0,0);
			}

		}
		else{
			vec3 unit_direction = unit_vector(curRay.direction());
			float t = 0.5f*(unit_direction.y()+1.0f);
			vec3 c = (1.0f-t)*vec3(1, 1, 1) + t*vec3(0.5f, 0.7f, 1);
			return curLight * c;
		}
	}
	return vec3(0.0f, 0.0f, 0.0f);
}

__global__ void imageGenerator(int x, int y, int cluster, camera cam, int aa, float* aaRands, hitable_list** world, vec3* img, curandState* state){
	
	int index = threadIdx.x + blockDim.x*blockIdx.x;
	int pixelNum = index*cluster;
	// printf("%p, %p, %f\n", world, world->list, world->list[0]->radius);
	// curandState state;
	
	while(pixelNum < x*y){

		for(int i = 0; i < cluster && (pixelNum+i) < x*y; i++){
			float pixX = (pixelNum+i)%x, pixY = (pixelNum+i)/x;

			
			
			vec3 col;
			for(int j = 0; j < aa; j++){
				// if(index == 0)
					// printf("%d\n", j);
				// printf("%f\n",curand_uniform(&state));
				float u, v;
				u = (pixX+aaRands[j]) / x;
				v = (pixY+aaRands[j]) / y;
				// if(j == 0){
				// 	u = (pixX) / x;
				// 	v = (pixY) / y;
				// }
				// else if(j==1){
				// 	u = (pixX+0.5f) / x;
				// 	v = (pixY+0.5f) / y;
				// }
				// else if(j==2){
				// 	u = (pixX+1) / x;
				// 	v = (pixY) / y;
				// }
				// else if(j==3){
				// 	u = (pixX) / x;
				// 	v = (pixY+1) / y;
				// }
				// else if(j==4){
				// 	u = (pixX+1) / x;
				// 	v = (pixY+1) / y;
				// }
				ray r;
				cam.get_ray(u, v, r);
				col += color(r, *world, state);
			}
			col /= aa;
			img[pixelNum+i].set(int(255.99f*col[0]), int(255.99f*col[1]), int(255.99f*col[2]));
		}
		pixelNum += blockDim.x*gridDim.x;
	}
}

int main(){
	int count;
	cudaGetDeviceCount(&count);
	cudaSetDevice(--count);
	int x = 2000;
	int y = 1000;
	int aaSamples = 128;
	float* aaRands;
	vec3 *imgBuf, *d_img;//, origin(0,0,0), ulc(-2,1,-1), hor(4,0,0), vert(0,2,0);
	curandState* state;
	
	camera cam;
	// hitable *list[2];
	hitable ** list;
	hitable_list **world;// = new hitable_list(list, 2);
	int worldSize = 4;
	cudaMalloc((void**)&state, sizeof(curandState));
	cudaMalloc((void**)&world, sizeof(hitable_list*));
	cudaMalloc((void**)&list, worldSize*sizeof(hitable*));
	cudaMalloc((void**)&aaRands, aaSamples*sizeof(float));
	cudaDeviceSynchronize();
	initRand<<<1,1>>>(1, 1, aaSamples, aaRands, state);
	cudaDeviceSynchronize();
	worldGenerator<<<1,1>>>(list, world, worldSize, state);
	// world->copyDevice();
	imgBuf = new vec3[x*y];
	cudaDeviceSynchronize();
	// printf("Done initting\n");
	cudaMalloc((void**)&d_img, sizeof(vec3)*x*y);
	cudaDeviceSynchronize();
	imageGenerator<<<4, 512>>>(x, y, 16, cam, aaSamples, aaRands, world, d_img, state);
	cudaDeviceSynchronize();
	cudaMemcpy(imgBuf, d_img, sizeof(vec3)*x*y, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	cout<<"P3\n"<<x<<' '<<y<<"\n255\n";
	for(int i = 0; i < x*y; i++){
		cout<<imgBuf[i].r()<<' '<<imgBuf[i].g()<<' '<<imgBuf[i].b()<<'\n';
	}
	return 0;
}