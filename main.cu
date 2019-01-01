#include "tracer.h"
// #include "objRead.h"
#include <OpenEXR/ImfNamespace.h>
#include <OpenEXR/ImfOutputFile.h>
#include <OpenEXR/ImfChannelList.h>



using namespace OPENEXR_IMF_NAMESPACE;

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
		
		*world = new hitable_list(objs, numOBJs);
	}
	__syncthreads();
	int curIndex = index*cluster;
	
	while(curIndex < (*world)->list_size){
		for(int i = 0; i < cluster && (curIndex+i) < (*world)->list_size; i++){
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
					(*world)->list[curIndex+i] = new Face(objs[j]->object[curIndex+i-offset], new light(vec3(4, 2, 2)));
					// printf("%d %p\n", curIndex+i, (*world)->list[curIndex+i]);
				}				
			}
		}
		curIndex += gridDim.x*blockDim.x;
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

__device__ vec3 color(const ray& r, hitable_list* world, curandState* state){
	
	float max = FLT_MAX;
	ray curRay = r;
	vec3 curLight = vec3(1,1,1);
	for(int i = 0; i < 10; i++){
		hit_record rec;
		if(world->hit(curRay, 0.00001f, max, rec)){
			ray scattered;
			vec3 attenuation;
			if(rec.mat->emitter && rec.mat->scatter(r, rec, attenuation, scattered, state)){
				curLight *= attenuation;
				return curLight;
			}
			else if(rec.mat->scatter(r, rec, attenuation, scattered, state)){
				curLight *= attenuation;
				curRay = scattered;
			}
			else{
				return vec3(0,0,0);
			}

		}
		else{
			return vec3(0,0,0);
			vec3 unit_direction = unit_vector(curRay.direction());
			float t = 0.5f*(unit_direction.y()+1.0f);
			vec3 c = (1.0f-t)*vec3(1, 1, 1) + t*vec3(0.5f, 0.7f, 1);
			return curLight * c;
		}
	}
	return vec3(0.0f, 0.0f, 0.0f);
}

__global__ void imageGenerator(int x, int y, int cluster, camera cam, int aa, hitable_list** world, vec3* img, curandState* state){
	
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
				col += color(r, *world, &state[pixelNum+i]);
			}
			col /= aa;
			img[pixelNum+i].set(col[0], col[1], col[2]);
		}
		// if(index == 0)
		// 	printf("%f%% finished\n", (float(pixelNum)/(x*y))*100);
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

int main(int argc, char* argv[]){
	// printf("%d\n", sizeof(Face));

	// printf("%d\n", argc);
	size_t totalSize = 0, *curSize = new size_t;
	int numOBJs = argc-1;
	OBJ ***d_objs, **objs = new OBJ*[numOBJs], ***h_d_objs;// = new OBJ*[numOBJs];
	for(int i = 0; i < numOBJs; i++){
		objs[i] = new OBJ(argv[i+1]);
		totalSize += objs[i]->numFaces*sizeof(Face) + objs[i]->numP*sizeof(vec3) + objs[i]->numT*sizeof(vec3) + objs[i]->numN*sizeof(vec3);
	}
	printf("Read .objs\n");
	curandState** state;
	hitable *** list;
	hitable_list ***world;// = new hitable_list(list, 2);
	int worldSize = 6;
	int count, firstDevice = 0;
	gpuErrchk(cudaGetDeviceCount(&count));
	// printf("numDevices: %d\n", count);
	state = new curandState*[count];
	list = new hitable**[count];
	world = new hitable_list**[count];

	int x = 720;
	int y = 480;
	int aaSamples = 2;

	vec3 **imgBuf, **d_img;//, origin(0,0,0), ulc(-2,1,-1), hor(4,0,0), vert(0,2,0);
	d_img = new vec3*[count];
	imgBuf = new vec3*[count];
	d_objs = new OBJ**[count];
	h_d_objs = new OBJ**[count];
	vec3 lookFrom(-10, -10, 10);
	vec3 lookAt(0,0,0);
	float dist = (lookFrom-lookAt).length();
	float ap = 0.0f;
	camera cam(lookFrom, lookAt, vec3(0, 1, 0), 40, float(x)/float(y), ap, dist);
	// hitable *list[2];
	printf("Beginning World Allocation, allocating %d bytes\n", totalSize);
	totalSize*=4;
	for(int i = firstDevice; i < count; i++){
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
	printf("Beginning Rand Generation, %d bytes allocated\n", totalSize);
	for(int i = firstDevice; i < count; i++){
		// printf("%d\n", i);
		gpuErrchk(cudaSetDevice(i));
		initRand<<<4,512>>>(x*y, 1, aaSamples/count, state[i]);
	}
	gpuErrchk(cudaDeviceSynchronize());
	printf("Beginning Copy of Faces to Device\n");
	for(int i = firstDevice; i < count; i++){
		gpuErrchk(cudaSetDevice(i));
		for(int j = 0; j < numOBJs; j++){
			// printf("%d %d\n", i, j);
			h_d_objs[i][j] = objs[j]->copyToDevice();
		}
		gpuErrchk(cudaMalloc((void**)&d_objs[i], sizeof(OBJ*)*numOBJs));
		gpuErrchk(cudaMemcpy(d_objs[i], h_d_objs[i], sizeof(OBJ*)*numOBJs, cudaMemcpyHostToDevice));
		gpuErrchk(cudaDeviceSynchronize());
	}
	cudaDeviceSynchronize();
	printf("worldGenerator Beginning\n");
	for(int i = firstDevice; i < count; i++){
		cudaSetDevice(i);
		
		worldGenerator<<<1,1024>>>(list[i], world[i], worldSize, d_objs[i], numOBJs, 1);
		cudaMalloc((void**)&d_img[i], sizeof(vec3)*x*y);
	}
	cudaDeviceSynchronize();
	printf("Beginning Render\n");
	for(int i = firstDevice; i < count; i++){
		cudaSetDevice(i);
		imageGenerator<<<4, 512>>>(x, y, 1, cam, aaSamples/count, world[i], d_img[i], state[i]);
		imgBuf[i] = new vec3[x*y];	
	}
	cudaDeviceSynchronize();
	printf("Done With Rendering, Copying to Disk/Cleaning\n");
	for(int i = firstDevice; i < count; i++){
		cudaSetDevice(i);
		
		
		cudaMemcpy(imgBuf[i], d_img[i], sizeof(vec3)*x*y, cudaMemcpyDeviceToHost);
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
	
	vec3** d_imgs, **imgs;
	vec3* finImg, *img;

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

	averageImgs<<<4, 512>>>(finImg, d_imgs, count, x, y, d_r, d_g, d_b, d_a);
	r = new float[x*y];
	g = new float[x*y];
	b = new float[x*y];
	a = new float[x*y];
	cudaDeviceSynchronize();

	cudaMemcpy(img, finImg, sizeof(vec3)*x*y, cudaMemcpyDeviceToHost);
	cudaMemcpy(r, d_r, sizeof(float)*x*y, cudaMemcpyDeviceToHost);
	cudaMemcpy(g, d_g, sizeof(float)*x*y, cudaMemcpyDeviceToHost);
	cudaMemcpy(b, d_b, sizeof(float)*x*y, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	cudaFree(d_r);
	cudaFree(d_g);
	cudaFree(d_b);
	cudaFree(d_a);
	cudaFree(d_imgs);
	cudaFree(finImg);
	delete[] imgs;

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
	delete[] img;
	return 0;
}