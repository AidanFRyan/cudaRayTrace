#include "tracer.h"
// #include <OpenEXR/ImfStandardAttributes.h>
#include <OpenEXR/ImfNamespace.h>
#include <OpenEXR/ImfOutputFile.h>
#include <OpenEXR/ImfChannelList.h>
// #include <OpenEXR/ImfExport.h>

using namespace OPENEXR_IMF_NAMESPACE;
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




__global__ void worldGenerator(hitable** list, hitable_list** world, int wSize){
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	if(i==0){
		// hitable* list[2];
		list[0] = new sphere(vec3(0,0,-1), 0.5, new lambertian(vec3(0.8f, 0.3f, 0.3f)));
		list[1] = new sphere(vec3(0,-100.5, -1), 100, new lambertian(vec3(0.8f, 0.8f, 0.0f)));
		list[2] = new sphere(vec3(1, 0, -1), 0.5, new metal(vec3(0.8f, 0.6f, 0.2f), 0.2f));
		// list[3] = new sphere(vec3(-1, 0, -1), 0.5, new metal(vec3(0.8f, 0.8f, 0.8f), 1.0f));
		// list[3] = new sphere(vec3(-1, 0, -1), 0.5f, new dielectric(1.5f));
		list[3] = new sphere(vec3(2, 1, 0), 0.5f, new light(vec3(2, 2, 2)));
		list[4] = new sphere(vec3(-1, 1, -2), 0.5f, new light(vec3(4, 2, 2)));
		// list[4] = new sphere(vec3(0,1,-1), 0.5f, new metal(vec3(0.8f, 0.8f, 0.9f), 0));
		*world = new hitable_list(list, wSize);
		// printf("wrldGen: %p %p\n", (*world)->list, (*world)->list[0]);
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
	
	// printf("Color %d\n", iters);
	// float t = hit_sphere(vec3(0,0,-1), 0.5, r);
	// printf("%p, %d\n", world, world->list_size);
	float max = FLT_MAX;
	ray curRay = r;
	vec3 curLight = vec3(1,1,1);
	for(int i = 0; i < 20; i++){
		hit_record rec;
		// vec3 N = unit_vector(r.p(t) - vec3(0,0,-1));
		if(world->hit(curRay, 0.001, max, rec)){
			ray scattered;
			vec3 attenuation;
			// int index = threadIdx.x + blockIdx.x*blockDim.x;
			// if (index == 0)
			// 	printf("scattering %f, %f, %f\n", r.direction().e[0], r.direction().e[1], r.direction().e[2]);//, outward_normal.e[0], outward_normal.e[1], outward_normal.e[2]);
			if(rec.mat->emitter && rec.mat->scatter(r, rec, attenuation, scattered, state)){
				curLight *= attenuation;
				// if(curLight.x() > 255){
				// 	curLight.e[0] = 255;
				// }
				// if(curLight.y() > 255){
				// 	curLight.e[1] = 255;
				// }
				// if(curLight.z() > 255){
				// 	curLight.e[2] = 255;
				// }
				return curLight;
				// printf("%f %f %f\n", attenuation.r(), attenuation.g(), attenuation.b());
				// break;
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
			
			// return curLight;
		}
	}
	return vec3(0.0f, 0.0f, 0.0f);
}

__global__ void imageGenerator(int x, int y, int cluster, camera cam, int aa, hitable_list** world, vec3* img, curandState* state){
	
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
				u = (pixX+curand_uniform(&state[pixelNum+i])) / x;
				v = (pixY+curand_uniform(&state[pixelNum+i])) / y;
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
				cam.get_ray(u, v, r, &state[pixelNum+i]);
				col += color(r, *world, &state[pixelNum+i]);
			}
			col /= aa;
			// img[pixelNum+i].set(int(255.99f*col[0]), int(255.99f*col[1]), int(255.99f*col[2]));
			img[pixelNum+i].set(col[0], col[1], col[2]);
		}
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

int main(){
	curandState** state;
	hitable *** list;
	hitable_list ***world;// = new hitable_list(list, 2);
	int worldSize = 5;
	int count;
	cudaGetDeviceCount(&count);

	state = new curandState*[count];
	list = new hitable**[count];
	world = new hitable_list**[count];
	// cudaSetDevice(--count);
	int x = 1920;
	int y = 1080;
	int aaSamples = 1024;
	// float** aaRands;
	vec3 **imgBuf, **d_img;//, origin(0,0,0), ulc(-2,1,-1), hor(4,0,0), vert(0,2,0);
	d_img = new vec3*[count];
	imgBuf = new vec3*[count];
	vec3 lookFrom(-3,3,2);
	vec3 lookAt(0,0,-1);
	float dist = (lookFrom-lookAt).length();
	float ap = 2.0f;
	camera cam(lookFrom, lookAt, vec3(0, 1, 0), 20, float(x)/float(y), ap, dist);
	// hitable *list[2];
	for(int i = 0; i < count; i++){
		
		cudaSetDevice(i);
		
		cudaMalloc((void**)&state[i], x*y*sizeof(curandState));
		cudaMalloc((void**)&world[i], sizeof(hitable_list*));
		cudaMalloc((void**)&list[i], worldSize*sizeof(hitable*));
		// cudaMalloc((void**)&aaRands, x*y*sizeof(float*));
		// cudaDeviceSynchronize();
		// cudaDeviceSetLimit(cudaLimitMallocHeapSize, 2*x*y*aaSamples*sizeof(float));
	}
	cudaDeviceSynchronize();
	for(int i = 0; i < count; i++){
		cudaSetDevice(i);
		initRand<<<4,512>>>(x*y, 1, aaSamples/count, state[i]);
	}
	cudaDeviceSynchronize();
	for(int i = 0; i < count; i++){
		cudaSetDevice(i);
		
		// cout<<"Done initting\n";
		worldGenerator<<<1,1>>>(list[i], world[i], worldSize);
		// world->copyDevice();
		
		// cudaDeviceSynchronize();
		// printf("Done initting\n");
		cudaMalloc((void**)&d_img[i], sizeof(vec3)*x*y);
		// cudaDeviceSynchronize();
	}
	cudaDeviceSynchronize();
	for(int i = 0; i < count; i++){
		cudaSetDevice(i);
		imageGenerator<<<4, 512>>>(x, y, 1, cam, aaSamples/count, world[i], d_img[i], state[i]);
		imgBuf[i] = new vec3[x*y];	
	}

	for(int i = 0; i < count; i++){
		cudaSetDevice(i);
		cudaDeviceSynchronize();
		
		cudaMemcpy(imgBuf[i], d_img[i], sizeof(vec3)*x*y, cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		cudaFree(state[i]);
		cudaFree(world[i]);
		cudaFree(list[i]);
		cudaFree(d_img[i]);
		// cudaDeviceSynchronize();
	}
	// for(int i = 0; i < ; i++){
	// 	cudaSetDevice()
	// }
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
	// cudaFree(d_img);
	return 0;
}