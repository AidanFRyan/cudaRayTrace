#include "tracer.h"
// #include "objRead.h"

#include <OpenEXR/ImfNamespace.h>
#include <OpenEXR/ImfOutputFile.h>
#include <OpenEXR/ImfChannelList.h>
#include <cooperative_groups.h>


using namespace OPENEXR_IMF_NAMESPACE;
using namespace cooperative_groups;
__global__ void worldGenerator(hitable** list, hitable_list** world, int wSize, OBJ** objs, int numOBJs, int cluster){
	int index = threadIdx.x + blockDim.x*blockIdx.x;
	if(index==0){
		// hitable* list[2];
		// list[0] = new sphere(vec3(0,0,-1), 0.5, new lambertian(vec3(0.8f, 0.3f, 0.3f)));
		// list[1] = new sphere(vec3(0,-100.5, -1), 100, new lambertian(vec3(0.8f, 0.8f, 0.0f)));
		// list[2] = new sphere(vec3(1, 0, -1), 0.5, new metal(vec3(0.8f, 0.6f, 0.2f), 0.0f));
		// // list[3] = new sphere(vec3(-1, 0, -1), 0.5, new metal(vec3(0.8f, 0.8f, 0.8f), 1.0f))
		// list[5] = new sphere(vec3(-1, 0, -1), 0.5f, new dielectric(1.5f));
		// list[3] = new sphere(vec3(2, 1, 0), 0.5f, new light(vec3(2, 2, 2)));
		// list[4] = new sphere(vec3(-1, 1, -2), 0.5f, new light(vec3(4, 2, 2)));
		// list[4] = new sphere(vec3(0,1,-1), 0.5f, new metal(vec3(0.8f, 0.8f, 0.9f), 0));
		// int totalFaces = 0;
		// for(int j = 0; j < numOBJs; j++){
		// 	totalFaces += objs[j]->numFaces;
		// }
		// hitable** worldFaces = new hitable*[totalFaces];
		// int zz = 0;
		// for(int j = 0; j < numOBJs; j++){
		// 	for(int z = 0; z < objs[j]->numFaces; z++){
		// 		worldFaces[zz] = new Face(objs[j]->object[z]);
		// 	}
		// }
		// printf("Trying to create hitable_list\n");
		// printf("numObjs: %d\n", numOBJs);
		
		*world = new hitable_list(objs, numOBJs, wSize);
		
	}
	__syncthreads();
	int curIndex = index*cluster;
	
	while(curIndex < (*world)->list_size-wSize){
		for(int i = 0; i < cluster && (curIndex+i) < (*world)->list_size-wSize; i++){
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
			int totalFaces = 0, offset = 0;
			for(int j = 0; j < numOBJs; j++){
				totalFaces += objs[j]->numFaces;
				if(curIndex+i < totalFaces){
					for(int z = 0; z < j; z++){
						offset += objs[j]->numFaces;
					}
					(*world)->list[curIndex+i] = new Face(objs[j]->object[curIndex+i-offset], new sss( new lambertian(vec3(0.0f, 0.2f, 0.0f)), 0.05f, vec3(1.0f, 0.25f, 0.2f)));
					// printf("%d %p\n", curIndex+i, (*world)->list[curIndex+i]);
				}
			}
		}
		curIndex += gridDim.x*blockDim.x;
	}
	__syncthreads();
	if(index==0){
		(*world)->list[(*world)->list_size-10] = new sphere(vec3(4, 4, 0), 2, new lambertian(vec3(0.2f, 0.3f, 0.4f)));
		(*world)->list[(*world)->list_size-9] = new sphere(vec3(3, 1, 0), 0.5f, new metal(vec3(0.2f, 0.6f, 0.8f), 1.4f));
		(*world)->list[(*world)->list_size-8] = new sphere(vec3(3, 0, 1), 0.5f, new dielectric(1.5f));
		(*world)->list[(*world)->list_size-7] = new sphere(vec3(5, -2, -5), 0.5f, new lambertian(vec3(0.5f, 0.2f, 0.8f)));
		(*world)->list[(*world)->list_size-6] = new sphere(vec3(5, -2, 5), 0.5f, new dielectric(1.78f));

		(*world)->list[(*world)->list_size-5] = new sphere(vec3(-10, 4, 0), 1, new lambertian(vec3(0.2f, 1, 0.4f)));
		(*world)->list[(*world)->list_size-4] = new sphere(vec3(0, 1, 0), 0.5f, new metal(vec3(1.0f, 0.78f, 0.8f), 0));
		(*world)->list[(*world)->list_size-3] = new sphere(vec3(0, 0, 4), 0.5f, new dielectric(1.5f));
		(*world)->list[(*world)->list_size-2] = new sphere(vec3(2, -1, 0), 0.5f, new lambertian(vec3(1.0f, 0.78f, 0.8f)));
		(*world)->list[(*world)->list_size-1] = new sphere(vec3(-5, -5, -5), 3, new light(vec3(6, 6, 6)));
	}
}

__global__ void initRand(int n, int cluster, int aa, curandState* state){
	int index = threadIdx.x+blockDim.x*blockIdx.x;
	int pixelNum = index*cluster;
	while(pixelNum < n){
		
		for(int i = 0; i < cluster && (pixelNum+i) < n; i++){
			curand_init(pixelNum+i, pixelNum+i, 0, &state[pixelNum+i]);
		}
		pixelNum += blockDim.x*gridDim.x;
	}
}

__device__ vec3 color(const ray& r, hitable_list* world, curandState* state){//}, bool* d_hits, hit_record* d_recs, float* d_dmax){
	
	float max = FLT_MAX;
	ray curRay = r;
	vec3 curLight = vec3(1,1,1);
	for(int i = 0; i < 10; i++){
		hit_record rec;
		// const ray& r, const float& tmin, float& tmax, hit_record& rec, bool* d_hits, hit_record* d_recs, float* d_dmax
		if(world->hit(curRay, 0.00001f, max, rec)){//}, d_hits, d_recs, d_dmax)){
			ray scattered;
			vec3 attenuation;
			if(rec.mat->emitter && rec.mat->scatter(r, rec, attenuation, scattered, state)){
				// printf("hit a big ol' light\n");
				curLight *= attenuation;
				return curLight;
			}
			else if(rec.mat->scatter(r, rec, attenuation, scattered, state)){
				// printf("scattered\n");
				curLight *= attenuation;
				curRay = scattered;
			}
			else{
				// printf("hit but not scattered\n");
				return vec3(0,0,0);
			}

		}
		else{
			// return vec3(0,0,0);
			// printf("to infinity!\n");
			vec3 unit_direction = unit_vector(curRay.direction());
			float t = 0.5f*(unit_direction.y()+1.0f);
			vec3 c = (1.0f-t)*vec3(1, 0.7f, 0.6f) + t*vec3(0.5f, 0.2f, 1);
			return curLight * c;
		}
	}
	// printf("exceeded bounce count\n");
	return vec3(0.0f, 0.0f, 0.0f);
}

__global__ void imageGenerator(int x, int y, int cluster, camera cam, int aa, hitable_list** world, vec3* img, curandState* state){//}, bool** d_hits, hit_record** d_recs, float ** d_dmax){
	
	int index = threadIdx.x + blockDim.x*blockIdx.x;
	int pixelNum = index*cluster;
	
	while(pixelNum < x*y){
		for(int i = 0; i < cluster && (pixelNum+i) < x*y; i++){
			float pixX = (pixelNum+i)%x, pixY = (pixelNum+i)/x;

			
			
			vec3 col;
			for(int j = 0; j < aa; j++){
				float u, v;
				u = (pixX+curand_uniform(&state[pixelNum+i])) / x;
				v = (pixY+curand_uniform(&state[pixelNum+i])) / y;
				ray r;
				cam.get_ray(u, v, r, &state[pixelNum+i]);
				col += color(r, *world, &state[pixelNum+i]);//, d_hits[index], d_recs[index], d_dmax[index]);
			}
			col /= aa;
			img[pixelNum+i].set(col[0], col[1], col[2]);
		}
		if(index == 0)
			printf("%f%% finished\n", (float(pixelNum)/(x*y))*100);
		pixelNum += blockDim.x*gridDim.x;
	}
}

__global__ void averageImgs(vec3* fin, vec3** img1, int count, int x, int y, float* r, float* g, float* b, float* a){
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	int pixelNum = index;
	while(pixelNum < x*y){
		for(int i = 0; i < count; i++){
			fin[pixelNum] += img1[i][pixelNum];
		}
		fin[pixelNum]/=count;
		r[pixelNum] = fin[pixelNum].r();
		g[pixelNum] = fin[pixelNum].g();
		b[pixelNum] = fin[pixelNum].b();
		a[pixelNum] = 1.0f;
		pixelNum += gridDim.x*blockDim.x;
		
	}
}

__global__ void averageImgs(vec3* img, int x, int y, float* r, float* g, float* b, float* a){
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	int pixelNum = index;
	while(pixelNum < x*y){
		// fin[pixelNum]/=count;
		r[pixelNum] = img[pixelNum].r();
		g[pixelNum] = img[pixelNum].g();
		b[pixelNum] = img[pixelNum].b();
		a[pixelNum] = 1.0f;
		pixelNum += gridDim.x*blockDim.x;
		
	}
}

__global__ void clearWorld(hitable_list ** world, int cluster){
	int index = threadIdx.x + blockDim.x*blockIdx.x;
	int curIndex = index*cluster;
	// printf("%d\n", (*world)->list_size);
	while(curIndex < (*world)->list_size){
		for(int i = 0; i < cluster && (curIndex+i) < (*world)->list_size; i++){
			// printf("%d %d\n", index, (curIndex+i));
			delete (*world)->list[curIndex+i];
		}
		curIndex+=gridDim.x*blockDim.x;
	}
	__syncthreads();
	if(index == 0)
		delete[] (*world)->list;
}
//  bool hit(const ray& r, const float& t_min, float& t_max, hit_record& rec) const = 0;
__global__ void getColor(int x, int y, int aaSamples, camera cam, vec3* img, ray* curRay, hitable_list** world, curandState* state, vec3* color, hit_record* hitRec, bool* hits, bool* returned){//}, bool* d_hits, hit_record* d_recs, float* d_dmax){
	// grid_group g = this_grid();
	int l_aaSamples = aaSamples;
	int index = threadIdx.x;//+blockDim.x*blockIdx.x;
	int worldSize = (*world)->list_size;
	ray tRay;
	hit_record rec;
	int powa;
	camera l_cam = cam;
	curandState l_state;
	// int block = blockIdx.x;
	// if(index==0){
	// 	curRay = new ray();
	// 	color = new vec3();
	// }
	// __shared__ bool aHit[(*world)->list_size];
	for(int k = blockIdx.x; k < x*y; k+=gridDim.x){
		l_state = state[k];
		// g.sync();
		__syncthreads();
		for(int aa = 0; aa < l_aaSamples; aa++){
			if(index == 0){
				// printf("%d\n", i);
				float pixX = k%x, pixY = k/x;
				// *returned=false;
				color[blockIdx.x] = vec3(1,1,1);
				pixX += curand_uniform(&l_state);
				pixY += curand_uniform(&l_state);
				
				float u = pixX/x, v = pixY/y;
				
				l_cam.get_ray(u, v, curRay[blockIdx.x], &l_state);
				returned[blockIdx.x] = false;
				// printf("%d\n", worldSize);

				// printf("%f %f %f\n", curRay->B.x(), curRay->B.y(), curRay->B.z());
			}
			__syncthreads();
			for(int i = 0; i < 10 && !returned[blockIdx.x]; i++){
				
				float max = FLT_MAX;
				// __syncthreads();
				tRay = curRay[blockIdx.x];
				bool anyHits = false;
				for(int j = index; j < worldSize; j+= blockDim.x){
					if((*world)->list[j]->hit(tRay, 0.0001f, max, rec)){
						hitRec[blockIdx.x*blockDim.x+index] = rec;
						anyHits = true;
					}
				}
				
				hits[blockDim.x*blockIdx.x + index] = anyHits;
				__syncthreads();
				// int powa;
				for(unsigned int powa = 2; powa<=blockDim.x; powa=powa<<1){
					// g.sync();
					__syncthreads();
					// powa = blockDim.x/int(powf(2,z));
					int tp = blockDim.x/powa;
					if(hits[index+blockIdx.x*blockDim.x])
						max = hitRec[index+blockIdx.x*blockDim.x].t;
					else
						max = FLT_MAX;
					for(int j = index+tp; j < tp*2 && j < blockDim.x; j+=tp){
						if(hits[j+blockIdx.x*blockDim.x]){
							if(hitRec[j+blockIdx.x*blockDim.x].t < max){
								max = hitRec[j+blockIdx.x*blockDim.x].t;
								hits[index+blockIdx.x*blockDim.x] = true;
								hitRec[index+blockIdx.x*blockDim.x] = hitRec[j+blockIdx.x*blockDim.x];
								// printf("hits\n");
							}
						}
					}
					__syncthreads();
					// g.sync();
				}

				if(index == 0){
					// hit_record record;
					// bool hit = false;
					// max = FLT_MAX;
					// for(int j = 0; j < blockDim.x; j++){
					// 	if(hits[j+blockIdx.x*blockDim.x] && hitRec[j+blockIdx.x*blockDim.x].t < max){
					// 		// max = hitRec[j+blockIdx.x*blockDim.x].t;
					// 		// printf("%d %f\n", hits[j+blockIdx.x*blockDim.x], hitRec[j+blockIdx.x*blockDim.x].t);
					// 		rec = hitRec[j+blockIdx.x*blockDim.x];
					// 		max = rec.t;
					// 		hit = true;
					// 	}
					// }

					if(hits[blockDim.x*blockIdx.x]){
						ray scattered;
						vec3 attenuation;
						rec = hitRec[blockDim.x*blockIdx.x];
						if(rec.mat->scatter(tRay, rec, attenuation, scattered, &l_state)){
							color[blockIdx.x] *= attenuation;
							curRay[blockIdx.x] = scattered;
							if(rec.mat->emitter){
								returned[blockIdx.x] = true;
							}
						}
						else{
							color[blockIdx.x] = vec3();
							returned[blockIdx.x] = true;
						}
					}
					else{
						vec3 unit_direction = unit_vector(tRay.direction());
						float t = 0.5f*(unit_direction.y()+1.0f);
						vec3 c = (1.0f-t)*vec3(1, 0.1f, 0.1f) + t*vec3(0.2f, 0.1f, 1);
						color[blockIdx.x] *= c;
						returned[blockIdx.x] = true;
					}
				}

				// for(int j = index; j < worldSize; j+=blockDim.x){
					
				// 	// printf("%d\n", j);
				// 	if((*world)->list[j]->hit( tRay, 0.0001f, max, rec)){
				// 		// printf("hit %d\n", i);
						
				// 		// printf("%d\n", j+blockIdx.x*worldSize);
				// 		// curRay[blockIdx.x] = tRay;
				// 		hitRec[j+blockIdx.x*worldSize] = rec;
				// 		hits[j+blockIdx.x*worldSize] = true;
				// 		// printf("hit detected %d %d %d\n", index, j, hits[j+blockIdx.x*worldSize]);
				// 	}
				// 	else{
				// 		hits[j+blockIdx.x*worldSize] = false;
				// 	}
				// }
				// __syncthreads();
				// max = FLT_MAX;
				// // g.sync();
				// for(int j = index; j < worldSize; j+=blockDim.x){
				// 	// vec3 curLight = vec3(1,1,1);
				// 	// for(int j = 0; j < (*world)->list_size; j++){
				// 		if(hits[j+blockIdx.x*worldSize]){
				// 			if(hitRec[j+blockIdx.x*worldSize].t < max){
				// 				max = hitRec[j+blockIdx.x*worldSize].t;
				// 				hits[index+blockIdx.x*worldSize] = true;
				// 				hitRec[index+blockIdx.x*worldSize] = hitRec[j+blockIdx.x*worldSize];
								
				// 			}
				// 		}
				// 	// }
				// }
				// __syncthreads();
				// // g.sync();
				
				// for(int z = 1; int(powf(2, z))<=blockDim.x; z++){
				// 	// g.sync();
				// 	__syncthreads();
				// 	powa = blockDim.x/int(powf(2,z));
				// 	if(index < worldSize)
				// 		if(hits[index+blockIdx.x*worldSize])
				// 			max = hitRec[index+blockIdx.x*worldSize].t;
				// 	else
				// 		max = FLT_MAX;
				// 	for(int j = index+powa; j < powa*2 && j < worldSize; j+=powa){
				// 		if(hits[j+blockIdx.x*worldSize]){
				// 			if(hitRec[j+blockIdx.x*worldSize].t < max){
				// 				max = hitRec[j+blockIdx.x*worldSize].t;
				// 				hits[index+blockIdx.x*worldSize] = true;
				// 				hitRec[index+blockIdx.x*worldSize] = hitRec[j+blockIdx.x*worldSize];
				// 				// printf("hits\n");
				// 			}
				// 		}
				// 	}
				// 	__syncthreads();
				// 	// g.sync();
				// }
				// // g.sync();
				// // __syncthreads();
				// if(index == 0){
				// 	rec = hitRec[blockIdx.x*worldSize];
					
				// 	if(hits[blockIdx.x*worldSize]){//}, d_hits, d_recs, d_dmax)){
				// 		// printf("%d\n", i);
				// 		ray scattered;
				// 		vec3 attenuation;
				// 		if(rec.mat->scatter(tRay, rec, attenuation, scattered, &state[k])){
				// 			// printf("scattered\n");
				// 			color[blockIdx.x] *= attenuation;
				// 			curRay[blockIdx.x] = scattered;
				// 			if(rec.mat->emitter){
				// 				// printf("light\n");
				// 				returned[blockIdx.x] = true;
				// 			}
				// 		}
				// 		else{
				// 			color[blockIdx.x] = vec3(0,0,0);
				// 			returned[blockIdx.x] = true;
				// 		}

				// 	}
				// 	else{
				// 		// *color = vec3(0,0,0);
				// 		// returned[blockIdx.x] = true;
				// 		// printf("infinity\n");
				// 		vec3 unit_direction = unit_vector(tRay.direction());
				// 		float t = 0.5f*(unit_direction.y()+1.0f);
				// 		vec3 c = (1.0f-t)*vec3(deactivate amazon account1, 0.1f, 0.1f) + t*vec3(0.2f, 0.1f, 1);
				// 		color[blockIdx.x] *= c;
				// 		returned[blockIdx.x] = true;
				// 	}
				// 	// printf("%d %f %f %f %f %f %f\n", i, curRay->A.x(), curRay->A.y(), curRay->A.z(), curRay->B.x(), curRay->B.y(), curRay->B.z());
				// }
				// g.sync();
				__syncthreads();
				// if(index == 0)
					// printf("%d\n", returned[blockIdx.x]);

			}
			__syncthreads();
			if(index == 0)
				img[k] += color[blockIdx.x];
			// g.sync();
		}
		// __syncthreads();
		if(index==0){
			// printf("%d %f%% completed\n", blockIdx.x, float(k)/(x*y)*100);
			img[k] /= aaSamples;
		}
		// __syncthreads;
	}
}

int main(int argc, char* argv[]){
	int pixelCluster = 64;

	// printf("%d\n", sizeof(Face));

	// printf("%d\n", argc);
	size_t totalSize = 0, *curSize = new size_t;
	int numOBJs = argc-1;
	OBJ ***d_objs, **objs = new OBJ*[numOBJs], ***h_d_objs;// = new OBJ*[numOBJs];
	
	printf("Read .objs\n");
	curandState** state;
	hitable *** list;
	hitable_list ***world;// = new hitable_list(list, 2);
	int worldSize = 10;
	int count, firstDevice = 0;
	gpuErrchk(cudaGetDeviceCount(&count));
	// printf("numDevices: %d\n", count);
	state = new curandState*[count];
	list = new hitable**[count];
	world = new hitable_list**[count];

	int numBlocks = 100, numThreads = 512;
	int x = 2000;
	int y = 1000;
	int aaSamples = 32;

	vec3 **imgBuf, **d_img;//, origin(0,0,0), ulc(-2,1,-1), hor(4,0,0), vert(0,2,0);
	d_img = new vec3*[count];
	imgBuf = new vec3*[count];
	d_objs = new OBJ**[count];
	h_d_objs = new OBJ**[count];
	vec3 lookFrom(5, 2, 5);
	vec3 lookAt(0,0,0);
	float dist = (lookFrom-lookAt).length();
	float ap = 0.0f;
	camera cam(lookFrom, lookAt, vec3(0, 1, 0), 60, float(x)/float(y), ap, dist);
	// hitable *list[2];
	int numObjs = worldSize;
	for(int i = 0; i < numOBJs; i++){
		objs[i] = new OBJ(argv[i+1]);
		totalSize += objs[i]->numFaces*sizeof(Face) + objs[i]->numP*sizeof(vec3) + objs[i]->numT*sizeof(vec3) + objs[i]->numN*sizeof(vec3) + objs[i]->numFaces*sizeof(hit_record);//+objs[i]->numFaces*sizeof(bool)+objs[i]->numFaces*sizeof(hit_record)+objs[i]->numFaces*sizeof(float);// + x*y*(objs[i]->numFaces*(sizeof(bool)+sizeof(hit_record)+sizeof(float)));
		numObjs += objs[i]->numFaces;
	}
	// numObjs+=worldSize;
	totalSize*=4;
	printf("Beginning World Allocation, allocating %u bytes\n", totalSize);
	for(int i = 0; i < count; i++){
		// printf("%d\n", i);
		
		gpuErrchk(cudaSetDevice(i));
		gpuErrchk(cudaDeviceSetLimit(cudaLimitMallocHeapSize, totalSize));
		cudaDeviceSynchronize();
		cudaDeviceGetLimit(curSize, cudaLimitMallocHeapSize);
		gpuErrchk(cudaMalloc((void**)&state[i], x*y*sizeof(curandState)));
		gpuErrchk(cudaMalloc((void**)&world[i], sizeof(hitable_list*)));
		gpuErrchk(cudaMalloc((void**)&list[i], worldSize*sizeof(hitable*)));
		h_d_objs[i] = new OBJ*[numOBJs];
	}
	cudaDeviceSynchronize();
	printf("Beginning Rand Generation, %u bytes allocated\n", totalSize);
	for(int i = 0; i < count; i++){
		// printf("%d\n", i);
		gpuErrchk(cudaSetDevice(i));
		initRand<<<4,512>>>(x*y, 1, aaSamples/count, state[i]);
	}
	gpuErrchk(cudaDeviceSynchronize());
	printf("Beginning Copy of Faces to Device\n");
	for(int i = 0; i < count; i++){
		cudaSetDevice(i);
		for(int j = 0; j < numOBJs; j++){
			// printf("%d %d\n", i, j);
			h_d_objs[i][j] = objs[j]->copyToDevice();
		}
		cudaMalloc((void**)&d_objs[i], sizeof(OBJ*)*numOBJs);
		cudaMemcpy(d_objs[i], h_d_objs[i], sizeof(OBJ*)*numOBJs, cudaMemcpyHostToDevice);
		cudaDeviceSynchronize();
	}	
	printf("worldGenerator Beginning\n");
	for(int i = 0; i < count; i++){
		cudaSetDevice(i);
		
		worldGenerator<<<1,512>>>(list[i], world[i], worldSize, d_objs[i], numOBJs, 1);
		cudaMalloc((void**)&d_img[i], sizeof(vec3)*x*y);
	}
	// printf("Allocating Space for Hit Search\n");
	cudaDeviceSynchronize();
	bool** hits = new bool*[count];
	hit_record** hitRec = new hit_record*[count];//, ***host_record = new hit_record**[count];
	vec3** color = new vec3*[count];
	ray** d_ray = new ray*[count];
	bool** cuRet = new bool*[count];
	for(int i = 0; i < count; i++){
		cudaSetDevice(i);
		cudaMalloc((void**)&hitRec[i], sizeof(hit_record)*numThreads*numBlocks);
		// host_record[i] = new hit_record*[numObjs];
		cudaMalloc((void**)&hits[i], sizeof(bool)*numThreads*numBlocks);
		cudaMalloc((void**)&d_ray[i], sizeof(ray)*numBlocks);
		// cudaMalloc((void**)d_ray[i], sizeof(ray));
		cudaMalloc((void**)&color[i], numBlocks*sizeof(vec3));
		// cudaMalloc((void**)color[i], sizeof(vec3));
		cudaMalloc((void**)&cuRet[i], sizeof(bool)*numBlocks);
		// ray* tempRay;
		// cudaMalloc((void**)&tempRay, sizeof(ray));
		// vec3* d_color;
		// cudaMalloc((void**)&d_color, sizeof(vec3));
		// cudaMemcpy(d_ray[i], tempRay, sizeof(ray*), cudaMemcpyHostToDevice);
		// cudaMemcpy(color[i], d_color, sizeof(vec3*), cudaMemcpyHostToDevice);
	}
	cudaDeviceSynchronize();
	// for(int d = 0; d < count; d++){
	// 	hit_record* temp = new hit_record;
	// 	for(int i = 0; i < numObjs; i++){
	// 		cudaSetDevice(d);
	// 		cudaMalloc((void**)&host_record[d][i], sizeof(hit_record)*numBlocks);
	// 		// cudaMemcpy(hitRec[d], host_record[d], sizeof())
	// 	}
	// 	cudaMemcpy(hitRec[d], host_record[d], sizeof(hit_record*)*numObjs, cudaMemcpyDeviceToHost);
	// }
	printf("Beginning Render\n");
	for(int i = 0; i < count; i++){
		cudaSetDevice(i);
		// imageGenerator<<<1, 512>>>(x, y, 1, cam, aaSamples/count, world[i], d_img[i], state[i]);//, d_hits[i], d_recs[i], d_dmax[i]);
		getColor<<<numBlocks, numThreads>>>(x, y, aaSamples/count, cam, d_img[i], d_ray[i], world[i], state[i], color[i], hitRec[i], hits[i], cuRet[i]);//, d_hits[index], d_recs[index], d_dmax[index]);
		imgBuf[i] = new vec3[x*y];
	}
	// random_device rd;
	// mt19937 gen(rd());
	// uniform_real_distribution<>dis(0,1);
	// imgBuf = new vec3[x*y];
	// // color(const ray& r, hitable_list* world, curandState* state, int pixelNum, vec3* color, hit_record* hitRec, bool* hits)
	
	
	// #pragma omp parallel for//openmp parallelization for cpu instances
	// for(int j = 0; j < x*y; j+=count){
	// 		// vec3 col, *color, *d_color;
	// 		// color = new vec3();
	// 	vec3* col = new vec3[count];
	// 	vec3*** color = new vec3**[count];
	// 	vec3*** d_color = new vec3**[count];
	// 	ray *** d_ray = new ray**[count], ***ra = new ray**[count];
	// 	for(int i = 0; i < count; i++){
	// 		d_color[i] = new vec3*[aaSamples];
	// 		color[i] = new vec3*[aaSamples];
	// 		ra[i] = new ray*[aaSamples];
	// 		d_ray[i] = new ray*[aaSamples];
	// 		cudaSetDevice(i);
	// 		for(int z = 0; z < aaSamples; z++){
	// 			// vec3* temp = new vec3();
	// 			// printf("%d %d\n", i, z);
	// 			cudaMalloc((void**)&d_color[i][z], sizeof(vec3));
	// 			// d_color[i][z] = temp;
	// 			// printf("%p\n", d_color[i][z]);
	// 			color[i][z] = new vec3();
	// 			cudaMalloc((void**)&d_ray[i][z], sizeof(ray));
	// 			ra[i][z] = new ray();
	// 		}
	// 	}
	// 	bool** cuRet = new bool*[count];
	// 	for(int i = 0; i < count; i++){
	// 		cudaSetDevice(i);
	// 		cudaMalloc((void**)&cuRet[i], sizeof(bool));
	// 	}
	// 	cudaDeviceSynchronize();
	// 	for(int i = 0; i < count; i++){
	// 		for(int z = 0; z < aaSamples; z++){
	// 			cudaSetDevice(i);
				
	// 			int pixX = (j+i)%x, pixY = (j+i)/x;
	// 			// printf("%d %d\n", pixX, pixY);
	// 			float offsetX = 0, offsetY = 0;
	// 			if(z<aaSamples/4){
	// 				// printf("less than 1/4\n");
	// 				do{offsetX = dis(gen);}	while(offsetX < 0.5);
	// 				do{offsetY = dis(gen);} while(offsetY < 0.5);
	// 			}
	// 			else if(z<aaSamples/2){
	// 				do{offsetX = dis(gen);}	while(offsetX > 0.5);
	// 				do{offsetY = dis(gen);} while(offsetY < 0.5);
	// 			}
	// 			else if(z<3*aaSamples/4){
	// 				do{offsetX = dis(gen);}	while(offsetX < 0.5);
	// 				do{offsetY = dis(gen);} while(offsetY > 0.5);
	// 			}
	// 			else{
	// 				do{offsetX = dis(gen);}	while(offsetX > 0.5);
	// 				do{offsetY = dis(gen);} while(offsetY > 0.5);
	// 			}
	// 			// printf("%d\n", i);
	// 			float u = (pixX+offsetX) / x, v = (pixY+offsetY) / y;
	// 			// float u = (pixX+1/aaSamples) / x, v = (pixY+dis(gen)) / y;
	// 			// ray r;
				
	// 			cam.get_ray(u, v, *ra[i][z], gen);

	// 			// printf("%d %d\n", i, z);
	// 			// ray* d_r;
	// 			// gpuErrchk(cudaMalloc((void**)&d_r, sizeof(ray)));
	// 			cudaMemcpy(d_ray[i][z], ra[i][z], sizeof(ray), cudaMemcpyHostToDevice);
	// 		}
	// 	}
	// 	// printf("%d\n", j);
	// 	cudaDeviceSynchronize();
	// 	for(int z = 0; z < aaSamples; z++){
	// 		for(int i = 0; i < count; i++){
	// 			cudaSetDevice(i);
	// 			// void** args = new void*[8];
	// 			// args[0]=(void*)&d_ray[i][z];
	// 			// args[1]=(void*)&world[i];
	// 			// args[2]=(void*)&state[i];
	// 			// args[3]=(void*)&j;
	// 			// args[4]=(void*)&d_color[i][z];
	// 			// args[5]=(void*)&hitRec[i];
	// 			// args[6]=(void*)&hits[i];
	// 			// args[7]=(void*)&cuRet[i];
	// 			// cudaLaunchCooperativeKernel((void*)getColor, dim3(1,1,1), dim3(1024,1,1), args);
	// 			getColor<<<1, 1024>>>(d_ray[i][z], world[i], state, j, d_color[i][z], hitRec[i], hits[i], cuRet[i]);//, d_hits[index], d_recs[index], d_dmax[index]);
	// 			// cudaMemcpy(color, d_color, sizeof(vec3), cudaMemcpyDeviceToHost);
	// 			// col += *color;
	// 			// printf("%d %d\n", i, z);
	// 		}
	// 		cudaDeviceSynchronize();
	// 		// printf("pixel %d\n", j);	
	// 		// j++;
	// 	}
		

	// 	// cudaDeviceSynchronize();
	// 	for(int i = 0; i < count; i++){
	// 		cudaSetDevice(i);
	// 		for(int z = 0; z < aaSamples; z++){
	// 			// printf("%d %d\n", i, z);
	// 			// vec3* temp = new vec3();
	// 			// printf("%p\n", d_color[i][z]);
	// 			cudaMemcpy(color[i][z], d_color[i][z], sizeof(vec3), cudaMemcpyDeviceToHost);
	// 			cudaDeviceSynchronize();
	// 			// color[i][z] = temp;
	// 		}
			
	// 	}
	// 	for(int i = 0; i < count; i++){
	// 		for(int z = 0; z < aaSamples; z++){
	// 			col[i] += *color[i][z];
	// 		}
			
	// 		col[i] /= aaSamples;
	// 		imgBuf[j+i].set(col[i].x(), col[i].y(), col[i].z());
	// 		// printf("%f %f %f\n", imgBuf[j+i].r(), imgBuf[j+i].g(), imgBuf[j+i].b());
	// 	}
		
		
	// 	// cudaDeviceSynchronize();
	// 	// printf("%f%% finished\n", (float(j+count-1)/(x*y))*100);
	// 	for(int i = 0; i < count; i++){
	// 		for(int z = 0; z < aaSamples; z++){
	// 			cudaFree(d_color[i][z]);
	// 			delete color[i][z];
	// 			delete ra[i][z];
	// 			cudaFree(d_ray[i][z]);
	// 		}
	// 		delete[] color[i];
	// 		delete[] d_color[i];
	// 		delete[] d_ray[i];
	// 		delete[] ra[i];	
	// 	}
	// 	delete[] col;
	// 	delete[] color;
	// 	delete[] d_color;
	// 	delete[] ra;
	// 	delete[] d_ray;
	// }
	
	cudaDeviceSynchronize();
	printf("Done With Rendering, Copying to Disk/Cleaning\n");
	for(int i = 0; i < count; i++){
		cudaSetDevice(i);
		clearWorld<<<1, 256>>>(world[i], 1);		
		cudaMemcpy(imgBuf[i], d_img[i], sizeof(vec3)*x*y, cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		cudaFree(state[i]);
		cudaFree(world[i]);
		cudaFree(list[i]);
		cudaFree(d_img[i]);
	}
	printf("Done Cleaning, Merging from devices\n");
	delete[] state;
	delete[] world;
	delete[] list;
	delete[] d_img;

	cudaSetDevice(count-1);
	cudaDeviceSynchronize();
	
	vec3** d_imgs, **imgs;
	vec3* finImg, *img;
	// cudaMalloc((void**)&d_imgs, sizeof(vec3)*x*y);
	// cudaMemcpy(d_imgs, imgBuf, sizeof(vec3)*x*y, cudaMemcpyHostToDevice);
	
	imgs = new vec3*[count];
	cudaMalloc((void**)&d_imgs, count*sizeof(vec3*));
	cudaMalloc((void**)&finImg, sizeof(vec3)*x*y);
	img = new vec3[x*y];

	for(int i = 0; i < count; i++){
		cudaMalloc((void**)&imgs[i], x*y*sizeof(vec3));
	}
	cudaDeviceSynchronize();
	for(int i = 0; i < count; i++){
		cudaMemcpy(imgs[i], imgBuf[i], sizeof(vec3)*x*y, cudaMemcpyHostToDevice);
	}
	cudaMemcpy(d_imgs, imgs, count*sizeof(vec3*), cudaMemcpyHostToDevice);
	
	float *d_r, *d_g, *d_b, *d_a;
	float *r, *g, *b, *a;
	cudaMalloc((void**)&d_r, sizeof(float)*x*y);
	cudaMalloc((void**)&d_g, sizeof(float)*x*y);
	cudaMalloc((void**)&d_b, sizeof(float)*x*y);
	cudaMalloc((void**)&d_a, sizeof(float)*x*y);
	cudaDeviceSynchronize();

	// averageImgs<<<4, 512>>>(d_imgs, x, y, d_r, d_g, d_b, d_a);
	// __global__ void averageImgs(vec3* fin, vec3** img1, int count, int x, int y, float* r, float* g, float* b, float* a){
	averageImgs<<<1,1024>>>(finImg, d_imgs, count, x, y, d_r, d_g, d_b, d_a);
	r = new float[x*y];
	g = new float[x*y];
	b = new float[x*y];
	a = new float[x*y];
	cudaDeviceSynchronize();

	// cudaMemcpy(imgBuf, imgBuf, sizeof(vec3)*x*y, cudaMemcpyDeviceToHost);
	cudaMemcpy(r, d_r, sizeof(float)*x*y, cudaMemcpyDeviceToHost);
	cudaMemcpy(g, d_g, sizeof(float)*x*y, cudaMemcpyDeviceToHost);
	cudaMemcpy(b, d_b, sizeof(float)*x*y, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	cudaFree(d_r);
	cudaFree(d_g);
	cudaFree(d_b);
	cudaFree(d_a);
	// cudaFree(d_imgs);
	// cudaFree(finImg);
	// delete[] imgs;

	Header header(x, y);
	header.channels().insert("R", Channel(FLOAT));
	header.channels().insert("G", Channel(FLOAT));
	header.channels().insert("B", Channel(FLOAT));
	header.channels().insert("A", Channel(FLOAT));

	OutputFile file("out.exr", header);

	FrameBuffer frameBuffer;
	frameBuffer.insert("R", Slice(FLOAT, (char*)r, sizeof(*r)*1, sizeof(*r)*x));
	frameBuffer.insert("G", Slice(FLOAT, (char*)g, sizeof(*g)*1, sizeof(*g)*x));
	frameBuffer.insert("B", Slice(FLOAT, (char*)b, sizeof(*b)*1, sizeof(*b)*x));
	frameBuffer.insert("A", Slice(FLOAT, (char*)a, sizeof(*a)*1, sizeof(*a)*x));
	file.setFrameBuffer(frameBuffer);
	file.writePixels(y);
	
	// cout<<"P3\n"<<x<<' '<<y<<"\n255\n";
	// for(int i = 0; i < x*y; i++){
	// 	cout<<img[i].r()<<' '<<img[i].g()<<' '<<img[i].b()<<'\n';
	// }
	// delete[] imgBuf;

	delete[] r;
	delete[] g;
	delete[] b;
	delete[] a;
	// delete[] img;
	return 0;
}