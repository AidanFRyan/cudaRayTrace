//Aidan Ryan, 2019
#include "tracer.h"

__global__ void parTreeConstruction(hitable** list, hitable_list** world, int wSize, OBJ** objs, int numOBJs, TreeNode*** tArr){
    if(threadIdx.x == 0){
        *world = new hitable_list(wSize+1);
        // printf("in parTree\n");
        for(unsigned int i = 0; i < numOBJs; i++){
            // printf("%p\n", tArr[i][0]);
            tArr[i][0] = new TreeNode;
            tArr[i][0]->contained = new Face*[objs[i]->numFaces];
            tArr[i][0]->within = objs[i]->numFaces;
            float min[3];
            float max[3];
            for(int j = 0; j < tArr[i][0]->within; j++){
                tArr[i][0]->contained[j] = new Face(objs[i]->object[j], new lambertian(vec3(0.0f, 0.2f, 0.0f)));
                tArr[i][0]->median += tArr[i][0]->contained[j]->median;
                if(j == 0){
                    for(int l = 0; l < 3; ++l){
                        min[l] = tArr[i][0]->contained[j]->min[l];
                        max[l] = tArr[i][0]->contained[j]->max[l];
                    }
                }
                else{
                    for(int l = 0; l < 3; ++l){
                        if(tArr[i][0]->contained[j]->min[l] < min[l])
                            min[l] = tArr[i][0]->contained[j]->min[l];

                        if(tArr[i][0]->contained[j]->max[l] > min[l])
                            max[l] = tArr[i][0]->contained[j]->max[l];
                    }
                }
            }
            tArr[i][0]->median /= tArr[i][0]->within;
            tArr[i][0]->dim = 0;
            // printf("%p\n", tArr[i][0]);
        }

    }
    for(int i = 0; i < numOBJs; i++){
        unsigned int numNodes = 0;
		for(unsigned int j = 1; j < objs[i]->numFaces; j=j<<1){
			for(int t = threadIdx.x; t < j; t += blockDim.x){
                unsigned int index = (j)-1 + t;
                // printf("%d %d %d\n", (j)-1+t, index, objs[i]->numFaces);
                TreeNode* curNode = tArr[i][index];
                if(curNode == nullptr){
                    // printf("curNode is NULL\n");
                    continue;
                }
                TreeNode *l = curNode->lt(), *r = curNode->gt();
                // printf("%d\n", (j<<1) + 2*t-1);
                tArr[i][(j<<1) + 2*t-1] = l;
                tArr[i][(j<<1) + 2*t] = r;
                if(l != nullptr)
                    ++numNodes;
                if(r != nullptr)
                    ++numNodes;
                
            }
        }
        if(threadIdx.x == 0){
            TriTree *tt = new TriTree();
            tt->numNodes = numNodes;
            tt->head = tArr[i][0];
            (*world)->list[i] = tt;
        }
    }
    
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
			if(rec.mat->emitter){
				if(rec.mat->scatter(r, rec, attenuation, scattered, state)){
				// printf("hit a big ol' light\n");
				curLight *= attenuation;
				return curLight;
				}
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
		// if(threadIdx.x == 0)
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

__global__ void averageImgs(vec3* img, int x, int y, float* r, float* g, float* b, float* a){
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	int pixelNum = index;
	while(pixelNum < x*y){
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

__global__ void renderRawFaceSearch(int x, int y, int aaSamples, camera cam, vec3* img, ray* curRay, hitable_list** world, curandState* state, vec3* color, hit_record* hitRec, bool* hits, bool* returned){//}, bool* d_hits, hit_record* d_recs, float* d_dmax){
	// grid_group g = this_grid();
	int l_aaSamples = aaSamples;
	int index = threadIdx.x;//+blockDim.x*blockIdx.x;
	int worldSize = (*world)->list_size;
	ray tRay;
	hit_record rec;
	camera l_cam = cam;
	curandState l_state;
	
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
			}
			__syncthreads();
			for(int i = 0; i < 10 && !returned[blockIdx.x]; i++){
				
				float max = FLT_MAX;
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
				for(unsigned int powa = 2; powa<=blockDim.x; powa=powa<<1){
					__syncthreads();
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
							}
						}
					}
					__syncthreads();
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

//begin old hit detection method, used gpu to traverse through array of triangles/objects in parallel, one pixel per kernel call, rather than current
//implementation which is one thread/pixel, executing on 512 pixels in parallel

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