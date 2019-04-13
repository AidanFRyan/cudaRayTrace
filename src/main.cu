//Aidan Ryan, 2019

#include "tracer.h"
// #include "objRead.h"
#include "kerns.hu"

#include <OpenEXR/ImfNamespace.h>
#include <OpenEXR/ImfOutputFile.h>
#include <OpenEXR/ImfChannelList.h>
#include <cooperative_groups.h>


using namespace OPENEXR_IMF_NAMESPACE;
using namespace cooperative_groups;

int main(int argc, char* argv[]){
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

	int numBlocks = 1, numThreads = 512;
	
	int x = 200;
	int y = 100;

	// x = 1000;
	// y = 500;
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

	TreeNode**** h_d_nodes = new TreeNode***[count], ****d_nodes = new TreeNode***[count];

	for(int i = 0; i < numOBJs; i++){
		objs[i] = new OBJ(argv[i+1]);
		totalSize += objs[i]->numFaces*sizeof(TreeNode) + objs[i]->numFaces*objs[i]->numFaces*sizeof(Face*) + objs[i]->numFaces*sizeof(TreeNode*) + numBlocks*numThreads*objs[i]->numFaces*sizeof(TreeNode*) + numBlocks*numThreads*objs[i]->numFaces*sizeof(bool) + objs[i]->numP*sizeof(vec3) + objs[i]->numT*sizeof(vec3) + objs[i]->numN*sizeof(vec3) + objs[i]->numFaces*sizeof(hit_record);//+objs[i]->numFaces*sizeof(bool)+objs[i]->numFaces*sizeof(hit_record)+objs[i]->numFaces*sizeof(float);// + x*y*(objs[i]->numFaces*(sizeof(bool)+sizeof(hit_record)+sizeof(float)));
		numObjs += objs[i]->numFaces;
	}
	printf("Beginning World Allocation, allocating %u bytes\n", totalSize);
	for(int i = 0; i < count; i++){
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
		// cudaMalloc((void**)&h_d_nodes[i], sizeof(TreeNode**)*numOBJs);
		h_d_nodes[i] = new TreeNode**[numOBJs];
		for(int j = 0; j < numOBJs; j++){
			h_d_objs[i][j] = objs[j]->copyToDevice();
			
			cudaMalloc((void**)&h_d_nodes[i][j], sizeof(TreeNode*)*(3*(objs[j]->numFaces*2+1)));
		}

		cudaMalloc((void**)&d_nodes[i], sizeof(TreeNode**)*numOBJs);
		cudaMemcpy(d_nodes[i], h_d_nodes[i], sizeof(TreeNode**)*numOBJs, cudaMemcpyHostToDevice);

		cudaMalloc((void**)&d_objs[i], sizeof(OBJ*)*numOBJs);
		cudaMemcpy(d_objs[i], h_d_objs[i], sizeof(OBJ*)*numOBJs, cudaMemcpyHostToDevice);

		cudaDeviceSynchronize();
	}	
	printf("worldGenerator Beginning\n");
	for(int i = 0; i < count; i++){
		cudaSetDevice(i);
		parTreeConstruction<<<1, 128>>>(list[i], world[i], worldSize, d_objs[i], numOBJs, d_nodes[i]);
		// worldGenerator<<<1,1>>>(list[i], world[i], worldSize, d_objs[i], numOBJs, 1);
		cudaMalloc((void**)&d_img[i], sizeof(vec3)*x*y);
	}
	cudaDeviceSynchronize();

	for(int i = 0; i < count; i++){
		cudaSetDevice(i);
		for(int j = 0; j < numOBJs; j++){
			cudaFree(h_d_nodes[i][j]);
		}
		delete[] h_d_nodes[i];
		cudaFree(d_nodes[i]);
	}
	delete[] d_nodes;
	delete[] h_d_nodes;

	printf("Beginning Render\n");
	for(int i = 0; i < count; i++){
		cudaSetDevice(i);
		imageGenerator<<<numBlocks, numThreads>>>(x, y, 1, cam, aaSamples/count, world[i], d_img[i], state[i]);//, d_hits[i], d_recs[i], d_dmax[i]);
		// getColor<<<numBlocks, numThreads>>>(x, y, aaSamples/count, cam, d_img[i], d_ray[i], world[i], state[i], color[i], hitRec[i], hits[i], cuRet[i]);//, d_hits[index], d_recs[index], d_dmax[index]);
		imgBuf[i] = new vec3[x*y];
	}

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

	delete[] r;
	delete[] g;
	delete[] b;
	delete[] a;
	// delete[] img;
	return 0;
}