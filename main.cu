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
	
	while(curIndex < (*world)->list_size){
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
					(*world)->list[curIndex+i] = new Face(objs[j]->object[curIndex+i-offset], new metal(vec3(0.5f, 0.5f, 0.5f), 0));
					// printf("%d %p\n", curIndex+i, (*world)->list[curIndex+i]);
				}
			}
		}
		curIndex += gridDim.x*blockDim.x;
	}
	__syncthreads();
	if(index==0){
		(*world)->list[(*world)->list_size-1] = new sphere(vec3(20, 0, 0), 5.0f, new light(vec3(200, 200, 200)));
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
	while(curIndex < (*world)->list_size){
		for(int i = 0; i < cluster && (curIndex+i) < (*world)->list_size; i++){
			delete (*world)->list[curIndex+i];
		}
		curIndex+=gridDim.x*blockDim.x;
	}
	__syncthreads();
	if(index == 0)
	delete[] (*world)->list;
}
//  bool hit(const ray& r, const float& t_min, float& t_max, hit_record& rec) const = 0;
__global__ void getColor(ray* curRay, hitable_list** world, curandState* state, int pixelNum, vec3* color, hit_record* hitRec, bool* hits, bool* returned){//}, bool* d_hits, hit_record* d_recs, float* d_dmax){
	grid_group g = this_grid();
	int index = threadIdx.x+blockDim.x*blockIdx.x;
	if(index == 0)
		*returned=false;
	g.sync();
	// hit_record* hitRec = new hit_record[world->list_size];
	// bool* hits = new bool[world->list_size];
	// printf("%d\n", index);
	// ray curRay(A, B);
	// g.sync();
	vec3 curLight = vec3(1,1,1);
	for(int i = 0; i < 10 && !*returned; i++){
		// if(index==1 || index==0)
		// printf("%d\n", index);
		// printf("%f %f %f\n", curRay->B.x(), curRay->B.y(), curRay->B.z());
		
		
		float max = FLT_MAX;
		// const ray& r, const float& tmin, float& tmax, hit_record& rec, bool* d_hits, hit_record* d_recs, float* d_dmax
		g.sync();
		for(int j = index; j < (*world)->list_size; j+=gridDim.x*blockDim.x){
			hits[j] = false;
			hit_record rec;
			if((*world)->list[j]->hit(*curRay, 0.0001f, max, rec)){
				hitRec[j] = rec;
				hits[j] = true;
				// if(j >= (*world)->list_size-1)
				// 	printf("%d\n", j);
			}
			// printf("%d\n", j);
		}
		// __syncthreads();
		g.sync();
		for(int j = index; j < (*world)->list_size; j+=gridDim.x*blockDim.x){
			// bool anyHit = false;
			// hit_record rec;
			vec3 curLight = vec3(1,1,1);
			// for(int j = 0; j < (*world)->list_size; j++){
				if(hits[j]){
					if(hitRec[j].t < max){
						max = hitRec[j].t;
						hits[index] = true;
						hitRec[index] = hitRec[i];
					}
				}
			// }
		}
		// __syncthreads();
		g.sync();
		int powa;
		for(int z = 1; int(powf(2, z))<=gridDim.x*blockDim.x; z++){
			g.sync();
			powa = gridDim.x*blockDim.x/int(powf(2,z));
			// if(index==123)
			// printf("%d\n", powa);
			if(hits[index])
				max = hitRec[index].t;
			else
				max = FLT_MAX;
			// printf("pow: %d\n", int(powf(2, z)));
			// if(index==1){
			// 	printf("%d %d\n", index, int(powf(2, z)));
			// }
			for(int j = index+powa; j < powa*2 && j < (*world)->list_size; j+=powa){
				// if(index == 1)
				// 	printf("%d\n", index);
				if(hits[j]){
					if(hitRec[j].t < max){
						max = hitRec[j].t;
						hits[index] = true;
						hitRec[index] = hitRec[j];
						// if(index == 0)
						// printf("%d Hit Detected at %d, moving to %d\n", z, j, index);
						
					}
				}
				// if(index == 2){
				// 	printf("%d %d %d\n", z, index, j);
				// }
				// if(index == 1){
				// 	printf("%d %d %d %d\n", z, index, j, gridDim.x*blockDim.x/int(powf(2, z-1)));
				// }
				// if(index == 0)
				// 	printf("%d bool* cuRet = bool[count];%d %d %d %d\n", z, index, j, hits[index], hits[j]);

					
			}
			// __syncthreads();
			// g.sync();
		}
		// __syncthreads();
		// if(index == 0){
		// 	printf("0: %d\n", i);
		// }
		g.sync();
		if(index == 0){
			hit_record rec = hitRec[0];
			
			if(hits[0]){//}, d_hits, d_recs, d_dmax)){
				// printf("hit %p\n", rec.mat);
				ray scattered;
				vec3 attenuation;
				// if(rec.mat->emitter && rec.mat->scatter(*curRay, rec, attenuation, scattered, &state[pixelNum])){
				// 	// printf("hit a big ol' light\n");
				// 	curLight *= attenuation;
				// 	*color = curLight;
				// 	return;
				// }
				if(rec.mat->scatter(*curRay, rec, attenuation, scattered, &state[pixelNum])){
					// printf("scattered\n");
					curLight *= attenuation;
					// printf("scattered %d: %f %f %f\n", i, attenuation.x(), attenuation.y(), attenuation.z());
					*curRay = scattered;
					if(rec.mat->emitter){
						// printf("%f %f %f\n", rec.normal.x(), rec.normal.y(), rec.normal.z());
						// assert(0);
						*returned = true;
					}
				}
				else{
					// printf("hit but not scattered\n");
					*color = vec3(0,0,0);
					*returned = true;
					// assert(0);
				}

			}
			else{
				// return vec3(0,0,0);
				// printf("to infinity!\n");
				vec3 unit_direction = unit_vector(curRay->direction());
				float t = 0.5f*(unit_direction.y()+1.0f);
				vec3 c = (1.0f-t)*vec3(1, 0.7f, 0.6f) + t*vec3(0.5f, 0.2f, 1);
				*color = curLight * c;
				*returned = true;
				// assert(0);
			}
		}
		g.sync();
		// if(index == 0){
		// 	printf("0: %d\n", i);
		// }
		// g.sync();
		// __syncthreads();
		// if(index == 123)
		// printf("%d: %f %f %f\n", i, curRay->B.x(), curRay->B.y(), curRay->B.z());
	}

	// if (index==0){
	// 	printf("color: %f %f %f\n", color->x(), color->y(), color->z());
	// }
	// 	*color = vec3(0.0f, 0.0f, 0.0f);
	// }
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
	int worldSize = 1;
	int count, firstDevice = 0;
	gpuErrchk(cudaGetDeviceCount(&count));
	// printf("numDevices: %d\n", count);
	state = new curandState*[count];
	list = new hitable**[count];
	world = new hitable_list**[count];

	int x = 100;
	int y = 50;
	int aaSamples = 32;

	vec3 *imgBuf, **d_img;//, origin(0,0,0), ulc(-2,1,-1), hor(4,0,0), vert(0,2,0);
	d_img = new vec3*[count];
	// imgBuf = new vec3*[count];
	d_objs = new OBJ**[count];
	h_d_objs = new OBJ**[count];
	vec3 lookFrom(5, 0, 0);
	vec3 lookAt(0,0,0);
	float dist = (lookFrom-lookAt).length();
	float ap = 0.0f;
	camera cam(lookFrom, lookAt, vec3(0, 1, 0), 60, float(x)/float(y), ap, dist);
	// hitable *list[2];
	int numObjs = worldSize;
	for(int i = 0; i < numOBJs; i++){
		objs[i] = new OBJ(argv[i+1]);
		totalSize += objs[i]->numFaces*sizeof(Face) + objs[i]->numP*sizeof(vec3) + objs[i]->numT*sizeof(vec3) + objs[i]->numN*sizeof(vec3);//+objs[i]->numFaces*sizeof(bool)+objs[i]->numFaces*sizeof(hit_record)+objs[i]->numFaces*sizeof(float);// + x*y*(objs[i]->numFaces*(sizeof(bool)+sizeof(hit_record)+sizeof(float)));
		numObjs += objs[i]->numFaces;
	}
	// numObjs+=worldSize;
	totalSize*=2;
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
		gpuErrchk(cudaSetDevice(i));
		for(int j = 0; j < numOBJs; j++){
			// printf("%d %d\n", i, j);
			h_d_objs[i][j] = objs[j]->copyToDevice();
		}
		gpuErrchk(cudaMalloc((void**)&d_objs[i], sizeof(OBJ*)*numOBJs));
		gpuErrchk(cudaMemcpy(d_objs[i], h_d_objs[i], sizeof(OBJ*)*numOBJs, cudaMemcpyHostToDevice));
		gpuErrchk(cudaDeviceSynchronize());
	}	
	printf("worldGenerator Beginning\n");
	for(int i = 0; i < count; i++){
		cudaSetDevice(i);
		
		worldGenerator<<<1,1024>>>(list[i], world[i], worldSize, d_objs[i], numOBJs, 1);
		cudaMalloc((void**)&d_img[i], sizeof(vec3)*x*y);
	}
	// printf("Allocating Space for Hit Search\n");
	cudaDeviceSynchronize();
	bool** hits = new bool*[count];
	hit_record** hitRec = new hit_record*[count];
	// vec3* color;
	for(int i = 0; i < count; i++){
		cudaSetDevice(i);
		cudaMalloc((void**)&hitRec[i], sizeof(hit_record)*numObjs);
		cudaMalloc((void**)&hits[i], sizeof(bool)*numObjs);
	}
	cudaDeviceSynchronize();
	printf("Beginning Render\n");
	// for(int i = 0; i < count; i++){
	// 	cudaSetDevice(i);
	// 	imageGenerator<<<1, 512>>>(x, y, 1, cam, aaSamples/count, world[i], d_img[i], state[i]);//, d_hits[i], d_recs[i], d_dmax[i]);
	// 	imgBuf[i] = new vec3[x*y];
	// }
	random_device rd;
	mt19937 gen(rd());
	uniform_real_distribution<>dis(0,1);
	imgBuf = new vec3[x*y];
	// color(const ray& r, hitable_list* world, curandState* state, int pixelNum, vec3* color, hit_record* hitRec, bool* hits)
	vec3* col = new vec3[count];
	vec3*** color = new vec3**[count];
	vec3*** d_color = new vec3**[count];
	ray *** d_ray = new ray**[count], ***ra = new ray**[count];
	for(int i = 0; i < count; i++){
		d_color[i] = new vec3*[aaSamples];
		color[i] = new vec3*[aaSamples];
		ra[i] = new ray*[aaSamples];
		d_ray[i] = new ray*[aaSamples];
		cudaSetDevice(i);
		for(int z = 0; z < aaSamples; z++){
			vec3* temp = new vec3();
			// printf("%d %d\n", i, z);
			gpuErrchk(cudaMalloc((void**)&temp, sizeof(vec3)));
			d_color[i][z] = temp;
			// printf("%p\n", d_color[i][z]);
			color[i][z] = new vec3();
			gpuErrchk(cudaMalloc((void**)&d_ray[i][z], sizeof(ray)));
			ra[i][z] = new ray();
		}
	}
	bool** cuRet = new bool*[count];
	for(int i = 0; i < count; i++){
		cudaSetDevice(i);
		cudaMalloc((void**)&cuRet[i], sizeof(bool));
	}
	for(int j = 0; j < x*y; j+=count){
			// vec3 col, *color, *d_color;
			// color = new vec3();
		cudaDeviceSynchronize();
		for(int i = 0; i < count; i++){
			for(int z = 0; z < aaSamples; z++){
				cudaSetDevice(i);
				
				int pixX = (j+i)%x, pixY = (j+i)/x;

				
				float u, v;
				u = (pixX+dis(gen)) / x;
				v = (pixY+dis(gen)) / y;
				// ray r;
				
				cam.get_ray(u, v, *ra[i][z], gen);

				// printf("%d %d\n", i, z);
				// ray* d_r;
				// gpuErrchk(cudaMalloc((void**)&d_r, sizeof(ray)));
				cudaMemcpy(d_ray[i][z], ra[i][z], sizeof(ray), cudaMemcpyHostToDevice);
			}
		}
		cudaDeviceSynchronize();
		for(int z = 0; z < aaSamples; z++){
			for(int i = 0; i < count; i++){
				cudaSetDevice(i);
				void** args = new void*[8];
				args[0]=(void*)&d_ray[i][z];
				args[1]=(void*)&world[i];
				args[2]=(void*)&state[i];
				args[3]=(void*)&j;
				args[4]=(void*)&d_color[i][z];
				args[5]=(void*)&hitRec[i];
				args[6]=(void*)&hits[i];
				args[7]=(void*)&cuRet[i];
				gpuErrchk(cudaLaunchCooperativeKernel((void*)getColor, dim3(8,1,1), dim3(512,1,1), args));
				// getColor<<<16, 1024>>>(d_ray[i][z], world[i], state[i], j, d_color[i][z], hitRec[i], hits[i]);//, d_hits[index], d_recs[index], d_dmax[index]);
				// cudaMemcpy(color, d_color, sizeof(vec3), cudaMemcpyDeviceToHost);
				// col += *color;
				// printf("%d %d\n", i, z);
			}
			cudaDeviceSynchronize();	
			// j++;
		}
		

		// cudaDeviceSynchronize();
		for(int i = 0; i < count; i++){
			cudaSetDevice(i);
			for(int z = 0; z < aaSamples; z++){
				// printf("%d %d\n", i, z);
				// vec3* temp = new vec3();
				// printf("%p\n", d_color[i][z]);
				gpuErrchk(cudaMemcpy(color[i][z], d_color[i][z], sizeof(vec3), cudaMemcpyDeviceToHost));
				cudaDeviceSynchronize();
				// color[i][z] = temp;
			}
			
		}
		for(int i = 0; i < count; i++){
			for(int z = 0; z < aaSamples; z++){
				col[i] += *color[i][z];
			}
			
			col[i] /= aaSamples;
			imgBuf[j+i].set(col[i].x(), col[i].y(), col[i].z());
			// printf("%f %f %f\n", imgBuf[j+i].r(), imgBuf[j+i].g(), imgBuf[j+i].b());
		}
		
		
		// cudaDeviceSynchronize();
		printf("%f%% finished\n", (float(j+count-1)/(x*y))*100);
	}
	for(int i = 0; i < count; i++){
		for(int z = 0; z < aaSamples; z++){
			gpuErrchk(cudaFree(d_color[i][z]));
			delete color[i][z];
			delete ra[i][z];
			cudaFree(d_ray[i][z]);
		}
		delete[] color[i];
		delete[] d_color[i];
		delete[] d_ray[i];
		delete[] ra[i];	
	}
	delete[] col;
	delete[] color;
	delete[] d_color;
	delete[] ra;
	delete[] d_ray;
	cudaDeviceSynchronize();
	printf("Done With Rendering, Copying to Disk/Cleaning\n");
	for(int i = 0; i < count; i++){
		cudaSetDevice(i);
		
		
		// cudaMemcpy(imgBuf[i], d_img[i], sizeof(vec3)*x*y, cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		cudaFree(state[i]);
		cudaFree(world[i]);
		cudaFree(list[i]);
		cudaFree(d_img[i]);
		clearWorld<<<1, 1024>>>(world[i], 1);
	}
	printf("Done Cleaning, Merging from devices\n");
	delete[] state;
	delete[] world;
	delete[] list;
	delete[] d_img;

	cudaSetDevice(count-1);
	cudaDeviceSynchronize();
	
	vec3* d_imgs;
	// vec3* finImg, *img;
	cudaMalloc((void**)&d_imgs, sizeof(vec3)*x*y);
	cudaMemcpy(d_imgs, imgBuf, sizeof(vec3)*x*y, cudaMemcpyHostToDevice);
	
	// imgs = new vec3[count];
	// cudaMalloc((void**)&d_imgs, count*sizeof(vec3*));
	// cudaMalloc((void**)&finImg, sizeof(vec3)*x*y);
	// img = new vec3[x*y];

	// for(int i = 0; i < count; i++){
	// 	cudaMalloc((void**)&imgs, x*y*sizeof(vec3));
	// }
	// cudaDeviceSynchronize();
	// for(int i = 0; i < count; i++){
	// 	cudaMemcpy(imgs, imgBuf, sizeof(vec3)*x*y, cudaMemcpyHostToDevice);
	// }
	// cudaMemcpy(d_imgs, imgs, count*sizeof(vec3*), cudaMemcpyHostToDevice);
	
	float *d_r, *d_g, *d_b, *d_a;
	float *r, *g, *b, *a;
	cudaMalloc((void**)&d_r, sizeof(float)*x*y);
	cudaMalloc((void**)&d_g, sizeof(float)*x*y);
	cudaMalloc((void**)&d_b, sizeof(float)*x*y);
	cudaMalloc((void**)&d_a, sizeof(float)*x*y);
	cudaDeviceSynchronize();

	averageImgs<<<4, 512>>>(d_imgs, x, y, d_r, d_g, d_b, d_a);
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