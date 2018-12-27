#include "tracer.h"


using namespace OPENEXR_IMF_NAMESPACE;

__global__ void worldGenerator(hitable** list, hitable_list** world, int wSize, camera* cam, float x, float y){
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	if(i==0){
		d_vec3 lookFrom(-3,3,2);
		d_vec3 lookAt(0,0,-1);
		cuhalf dist = (lookFrom-lookAt).length();
		cuhalf ap = 0.0f;
		cam = new camera(lookFrom, lookAt, d_vec3(0, 1, 0), 20, float(x)/float(y), ap, dist);
		// hitable* list[2];
		list[0] = new sphere(d_vec3(0,0,-1), 0.5, new lambertian(d_vec3(0.8f, 0.3f, 0.3f)));
		list[1] = new sphere(d_vec3(0,-100.5, -1), 100, new lambertian(d_vec3(0.8f, 0.8f, 0.0f)));
		list[2] = new sphere(d_vec3(1, 0, -1), 0.5, new metal(d_vec3(0.8f, 0.6f, 0.2f), 0.0f));
		// list[3] = new sphere(d_vec3(-1, 0, -1), 0.5, new metal(d_vec3(0.8f, 0.8f, 0.8f), 1.0f))
		list[5] = new sphere(d_vec3(-1, 0, -1), 0.5f, new dielectric(1.5f));
		list[3] = new sphere(d_vec3(2, 1, 0), 0.5f, new light(d_vec3(2, 2, 2)));
		list[4] = new sphere(d_vec3(-1, 1, -2), 0.5f, new light(d_vec3(4, 2, 2)));
		// list[4] = new sphere(d_vec3(0,1,-1), 0.5f, new metal(d_vec3(0.8f, 0.8f, 0.9f), 0));
		*world = new hitable_list(list, wSize);
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

__device__ d_vec3 color(const ray& r, hitable_list* world, curandState* state){
	
	cuhalf max = FLT_MAX;
	ray curRay = r;
	d_vec3 curLight = d_vec3(1,1,1);
	for(int i = 0; i < 20; i++){
		hit_record rec;
		if(world->hit(curRay, 0.00001, max, rec)){
			ray scattered;
			d_vec3 attenuation;
			if(rec.mat->emitter && rec.mat->scatter(r, rec, attenuation, scattered, state)){
				curLight *= attenuation;
				return curLight;
			}
			else if(rec.mat->scatter(r, rec, attenuation, scattered, state)){
				curLight *= attenuation;
				curRay = scattered;
			}
			else{
				return d_vec3(0,0,0);
			}

		}
		else{
			return d_vec3(0,0,0);
			d_vec3 unit_direction = unit_vector(curRay.direction());
			cuhalf t = 0.5f*(unit_direction.y()+1);
			d_vec3 c = (1.0f-t)*d_vec3(1, 1, 1) + t*d_vec3(0.5f, 0.7f, 1);
			return curLight * c;
		}
	}
	return d_vec3(0.0f, 0.0f, 0.0f);
}

__global__ void imageGenerator(int x, int y, int cluster, camera* cam, int aa, hitable_list** world, d_vec3* img, curandState* state){
	
	int index = threadIdx.x + blockDim.x*blockIdx.x;
	int pixelNum = index*cluster;
	
	while(pixelNum < x*y){

		for(int i = 0; i < cluster && (pixelNum+i) < x*y; i++){
			cuhalf pixX = (pixelNum+i)%x, pixY = (pixelNum+i)/x;
			// printf("%p %p\n", &pixX, &pixY);
			
			
			d_vec3 col;
			for(int j = 0; j < aa; j++){
				cuhalf u, v;
				u = (pixX+curand_uniform(&state[pixelNum+i])) / x;
				v = (pixY+curand_uniform(&state[pixelNum+i])) / y;
				// printf("%f %f %f\n", pixX, pixY, curand_uniform(&state[pixelNum+i]));
				ray r;
				cam->get_ray(u, v, r, &state[pixelNum+i]);
				col += color(r, *world, &state[pixelNum+i]);
			}
			col /= aa;
			img[pixelNum+i].set(col[0], col[1], col[2]);
			// printf("%f %f %f\n", col[0], col[1], col[2]);
		}
		pixelNum += blockDim.x*gridDim.x;
	}
}

__global__ void averageImgs(d_vec3* fin, d_vec3** img1, int count, int x, int y, cuhalf* r, cuhalf* g, cuhalf* b, cuhalf* a){
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
	int worldSize = 6;
	int count;
	cudaGetDeviceCount(&count);

	state = new curandState*[count];
	list = new hitable**[count];
	world = new hitable_list**[count];
	int x = 1920;
	int y = 1080;
	int aaSamples = 1024;
	h_vec3 **imgBuf, **d_img;//, origin(0,0,0), ulc(-2,1,-1), hor(4,0,0), vert(0,2,0);
	d_img = new h_vec3*[count];
	imgBuf = new h_vec3*[count];
	
	// camera cam(lookFrom, lookAt, d_vec3(0, 1, 0), 20, float(x)/float(y), ap, dist);
	camera* cam;
	// hitable *list[2];
	for(int i = 0; i < count; i++){
		
		cudaSetDevice(i);
		cudaMalloc((void**)&cam, sizeof(camera));
		cudaMalloc((void**)&state[i], x*y*sizeof(curandState));
		cudaMalloc((void**)&world[i], sizeof(hitable_list*));
		cudaMalloc((void**)&list[i], worldSize*sizeof(hitable*));
	}
	cudaDeviceSynchronize();
	for(int i = 0; i < count; i++){
		cudaSetDevice(i);
		initRand<<<4,512>>>(x*y, 1, aaSamples/count, state[i]);
	}
	cudaDeviceSynchronize();
	for(int i = 0; i < count; i++){
		cudaSetDevice(i);
		
		worldGenerator<<<1,1>>>(list[i], world[i], worldSize, cam, x, y);
		cudaMalloc((void**)&d_img[i], sizeof(d_vec3)*x*y);
	}
	cudaDeviceSynchronize();
	for(int i = 0; i < count; i++){
		cudaSetDevice(i);
		imageGenerator<<<4, 512>>>(x, y, 1, cam, aaSamples/count, world[i], (d_vec3*)d_img[i], state[i]);
		imgBuf[i] = new h_vec3[x*y];	
	}

	for(int i = 0; i < count; i++){
		cudaSetDevice(i);
		cudaDeviceSynchronize();
		
		cudaMemcpy(imgBuf[i], d_img[i], sizeof(d_vec3)*x*y, cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		cudaFree(state[i]);
		cudaFree(world[i]);
		cudaFree(list[i]);
		cudaFree(d_img[i]);
	}
	delete[] state;
	delete[] world;
	delete[] list;
	delete[] d_img;

	cudaSetDevice(count-1);
	cudaDeviceSynchronize();
	
	d_vec3** d_imgs, *finImg;
	h_vec3 *img, **imgs;

	imgs = new h_vec3*[count];
	cudaMalloc((void**)&d_imgs, count*sizeof(d_vec3*));
	cudaMalloc((void**)&finImg, sizeof(d_vec3)*x*y);
	img = new h_vec3[x*y];

	for(int i = 0; i < count; i++){
		cudaMalloc((void**)&imgs[i], x*y*sizeof(d_vec3));
	}
	cudaDeviceSynchronize();
	for(int i = 0; i < count; i++){
		cudaMemcpy(imgs[i], imgBuf[i], sizeof(d_vec3)*x*y, cudaMemcpyHostToDevice);
	}
	cudaMemcpy(d_imgs, imgs, count*sizeof(d_vec3*), cudaMemcpyHostToDevice);
	
	cuhalf *d_r, *d_g, *d_b, *d_a;
	cuhalf *r, *g, *b, *a;
	cudaMalloc((void**)&d_r, sizeof(cuhalf)*x*y);
	cudaMalloc((void**)&d_g, sizeof(cuhalf)*x*y);
	cudaMalloc((void**)&d_b, sizeof(cuhalf)*x*y);
	cudaMalloc((void**)&d_a, sizeof(cuhalf)*x*y);
	cudaDeviceSynchronize();

	averageImgs<<<4, 512>>>(finImg, d_imgs, count, x, y, d_r, d_g, d_b, d_a);
	r = new cuhalf[x*y];
	g = new cuhalf[x*y];
	b = new cuhalf[x*y];
	a = new cuhalf[x*y];
	cudaDeviceSynchronize();

	cudaMemcpy(img, finImg, sizeof(d_vec3)*x*y, cudaMemcpyDeviceToHost);
	cudaMemcpy(r, d_r, sizeof(cuhalf)*x*y, cudaMemcpyDeviceToHost);
	cudaMemcpy(g, d_g, sizeof(cuhalf)*x*y, cudaMemcpyDeviceToHost);
	cudaMemcpy(b, d_b, sizeof(cuhalf)*x*y, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	cudaFree(d_r);
	cudaFree(d_g);
	cudaFree(d_b);
	cudaFree(d_a);
	cudaFree(d_imgs);
	cudaFree(finImg);
	delete[] imgs;

	Header header(x, y);
	header.channels().insert("R", Channel(HALF));
	header.channels().insert("G", Channel(HALF));
	header.channels().insert("B", Channel(HALF));
	header.channels().insert("A", Channel(HALF));

	OutputFile file("out.exr", header);

	FrameBuffer frameBuffer;
	frameBuffer.insert("R", Slice(HALF, (char*)r, sizeof(*r)*1, sizeof(*r)*x));
	frameBuffer.insert("G", Slice(HALF, (char*)g, sizeof(*g)*1, sizeof(*g)*x));
	frameBuffer.insert("B", Slice(HALF, (char*)b, sizeof(*b)*1, sizeof(*b)*x));
	frameBuffer.insert("A", Slice(HALF, (char*)a, sizeof(*a)*1, sizeof(*a)*x));
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