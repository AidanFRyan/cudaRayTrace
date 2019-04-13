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

	x = 1000;
	y = 500;
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

//leftover from parallel hit detection, one call per pixel

	// printf("Allocating Space for Hit Search\n");
	cudaDeviceSynchronize();
	// bool** hits = new bool*[count];
	// hit_record** hitRec = new hit_record*[count];//, ***host_record = new hit_record**[count];
	// vec3** color = new vec3*[count];
	// ray** d_ray = new ray*[count];
	// bool** cuRet = new bool*[count];
	// for(int i = 0; i < count; i++){
	// 	cudaSetDevice(i);
	// 	cudaMalloc((void**)&hitRec[i], sizeof(hit_record)*numThreads*numBlocks);
	// 	// host_record[i] = new hit_record*[numObjs];
	// 	cudaMalloc((void**)&hits[i], sizeof(bool)*numThreads*numBlocks);
	// 	cudaMalloc((void**)&d_ray[i], sizeof(ray)*numBlocks);
	// 	// cudaMalloc((void**)d_ray[i], sizeof(ray));
	// 	cudaMalloc((void**)&color[i], numBlocks*sizeof(vec3));
	// 	// cudaMalloc((void**)color[i], sizeof(vec3));
	// 	cudaMalloc((void**)&cuRet[i], sizeof(bool)*numBlocks);
	// 	// ray* tempRay;
	// 	// cudaMalloc((void**)&tempRay, sizeof(ray));
	// 	// vec3* d_color;
	// 	// cudaMalloc((void**)&d_color, sizeof(vec3));
	// 	// cudaMemcpy(d_ray[i], tempRay, sizeof(ray*), cudaMemcpyHostToDevice);
	// 	// cudaMemcpy(color[i], d_color, sizeof(vec3*), cudaMemcpyHostToDevice);
	// }
	// cudaDeviceSynchronize();
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
		imageGenerator<<<numBlocks, numThreads>>>(x, y, 1, cam, aaSamples/count, world[i], d_img[i], state[i]);//, d_hits[i], d_recs[i], d_dmax[i]);
		// getColor<<<numBlocks, numThreads>>>(x, y, aaSamples/count, cam, d_img[i], d_ray[i], world[i], state[i], color[i], hitRec[i], hits[i], cuRet[i]);//, d_hits[index], d_recs[index], d_dmax[index]);
		imgBuf[i] = new vec3[x*y];
	}

//leftover from parallel hit detection, also experimenting with multithreading GPU memory copies

	// random_device rd;
	// mt19937 gen(rd());
	// uniform_real_distribution<>dis(0,1);
	// imgBuf = new vec3[x*y];
	// // color(const ray& r, hitable_list* world, curandState* state, int pixelNum, vec3* color, hit_record* hitRec, bool* hits)
	
	
	// #pragma omp parallel for 			//openmp parallelization for cpu instances
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

	delete[] r;
	delete[] g;
	delete[] b;
	delete[] a;
	// delete[] img;
	return 0;
}