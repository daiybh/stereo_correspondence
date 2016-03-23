#include "asw.h"
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cstdlib>
#include <math.h>
#include <iostream>
#include <cstdio>

#define GAMMA_C1 30.91
#define GAMMA_G1 28.21

#define GAMMA_C2 10.94
#define GAMMA_G2 118.78

#define ALPHA 0.81

using namespace std;

__forceinline__  __device__ int getPixel(unsigned char * image, int x, int y, int width,
		int height) {
	if (x < 0) {
		return 0;
	}

	if (y < 0) {
		return 0;
	}
	if (x >= width) {
		return 0;
	}

	if (y >= height) {
		return 0;
	}
	return image[y * width + x];
}

__device__ int getPixelD(int * image, int x, int y, int width, int height) {
	if (x < 0) {
		return 0;
	}

	if (y < 0) {
		return 0;
	}
	if (x >= width) {
		return 0;
	}

	if (y >= height) {
		return 0;
	}
	return image[y * width + x];
}

__global__ void verticalCostAggregation(unsigned char* source,
		unsigned char* target, float* V, int disp, int max_disp, int width,
		int height, int blocks_x) {
	int x = (blockIdx.x * blockDim.x + threadIdx.x);
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x >= width || y >= height) {
		__syncthreads();
		__syncthreads();
		return;
	}

	__shared__ unsigned char block_region_source[48][16];

	__shared__ unsigned char block_region_target[48][16];

	//	if (threadIdx.y < 3) {
	//		int xb = blockIdx.x * blockDim.x;
	//		int yb = (blockIdx.y) * blockDim.y + threadIdx.y * blockDim.x
	//				+ threadIdx.x - 16;
	//		for (int i = 0; i < 16; i++) {
	//			block_region_source[threadIdx.y * blockDim.x + threadIdx.x][i] =
	//					getPixel(source, xb + i, yb, width, height);
	//
	//		}
	//	}
	//	if ((threadIdx.y >= 3) && (threadIdx.y < 6)) {
	//		int xb = blockIdx.x * blockDim.x + disp;
	//		int yb = (blockIdx.y) * blockDim.y + (threadIdx.y - 3) * blockDim.x
	//				+ threadIdx.x - 16;
	//		for (int i = 0; i < 16; i++) {
	//			block_region_target[(threadIdx.y - 3) * blockDim.x + threadIdx.x][i] =
	//					getPixel(target, xb + i, yb, width, height);
	//		}
	//	}

	int yb = y - 16;
	block_region_source[threadIdx.y][threadIdx.x] = getPixel(source, x, yb,
			width, height);
	block_region_source[threadIdx.y + 16][threadIdx.x] = getPixel(source, x,
			yb + 16, width, height);
	block_region_source[threadIdx.y + 32][threadIdx.x] = getPixel(source, x,
			yb + 32, width, height);
	block_region_target[threadIdx.y][threadIdx.x] = getPixel(target, x + disp,
			yb, width, height);
	block_region_target[threadIdx.y + 16][threadIdx.x] = getPixel(target,
			x + disp, yb + 16, width, height);
	block_region_target[threadIdx.y + 32][threadIdx.x] = getPixel(target,
			x + disp, yb + 32, width, height);
	__syncthreads();
	float sum1 = 0;
	float sum2 = 0;

	int p_s = block_region_source[threadIdx.y + 16][threadIdx.x];
	//	int p_s = getPixel(source,x,y,width,height);

	int p_t = block_region_target[threadIdx.y + 16][threadIdx.x];
	//	int p_t = getPixel(target,x+disp,y,width,height);

	for (int i = -16; i < 16; i++) {
		if (i != 0) {
			int y_q = y + i;

			int q_s = block_region_source[threadIdx.y + 16 + i][threadIdx.x];
			//			int q_s = getPixel(source,x,y_q,width,height);
			int q_t = block_region_target[threadIdx.y + 16 + i][threadIdx.x];
			//			int q_t = getPixel(target,x+disp,y_q,width,height);

			float delta_c = float(abs(p_s - q_s));
			float delta_g = float(abs(y_q - y));
			float w1 = __expf(
					__fdividef(-delta_c, GAMMA_C1)
							- __fdividef(delta_g, GAMMA_G1));

			delta_c = float(abs(p_t - q_t));
			float w2 = __expf(
					__fdividef(-delta_c, GAMMA_C1)
							- __fdividef(delta_g, GAMMA_G1));

			float d = (abs(q_s - q_t));
			sum1 += w1 * w2 * d;
			sum2 += w1 * w2;
		}

	}
	//printf("%f\n",sum1);
	//			V[y * width * max_disp + x * (max_disp) + abs(disp)] = sum1 / sum2;
	V[abs(disp) * width * height + y * width + x] = __fdividef(sum1, sum2);
	__syncthreads();
	//
}

__global__ void horizontalCostAggregation(unsigned char* source,
		unsigned char* target, float* V, int disp, int max_disp, int width,
		int height, int blocks_x) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x >= width || y >= height) {
		__syncthreads();
		__syncthreads();
		return;
	}

	__shared__ unsigned char block_region_source[16][48];
	__shared__ unsigned char block_region_target[16][48];

	//	if (threadIdx.y < 3) {
	//		int xb = blockIdx.x * blockDim.x + threadIdx.y * blockDim.x
	//				+ threadIdx.x - 16;
	//		int yb = blockIdx.y * blockDim.y;
	//		int xbl = threadIdx.y * blockDim.x + threadIdx.x;
	//		for (int i = 0; i < 16; i++) {
	//			block_region_source[i][xbl] = getPixel(source, xb, yb + i, width,
	//					height);
	//		}
	//	}
	//	if (threadIdx.y >= 3 && threadIdx.y < 6) {
	//		int xb = blockIdx.x * blockDim.x + disp + (threadIdx.y - 3) * blockDim.x
	//				+ threadIdx.x - 16;
	//		int yb = blockIdx.y * blockDim.y;
	//		int xbl = (threadIdx.y - 3) * blockDim.x + threadIdx.x;
	//		for (int i = 0; i < 16; i++) {
	//			block_region_target[i][xbl] = getPixel(target, xb, yb + i, width,
	//					height);
	//		}
	//	}
	int xb = x - 16;
	int xbt = xb + disp;
	block_region_source[threadIdx.y][threadIdx.x] = getPixel(source, xb, y,
			width, height);
	block_region_source[threadIdx.y][threadIdx.x + 16] = getPixel(source,
			xb + 16, y, width, height);
	block_region_source[threadIdx.y][threadIdx.x + 32] = getPixel(source,
			xb + 32, y, width, height);
	block_region_target[threadIdx.y][threadIdx.x] = getPixel(target, xbt, y,
			width, height);
	block_region_target[threadIdx.y][threadIdx.x + 16] = getPixel(target,
			xbt + 16, y, width, height);
	block_region_target[threadIdx.y][threadIdx.x + 32] = getPixel(target,
			xbt + 32, y, width, height);
	__syncthreads();
	float sum1 = 0;
	float sum2 = 0;

	int p_s = block_region_source[threadIdx.y][threadIdx.x + 16];
	//	int p_s = getPixel(source,x,y,width,height);

	int p_t = block_region_target[threadIdx.y][threadIdx.x + 16];
	//	int p_t=getPixel(target,x_pt,y,width,height);
	for (int i = -16; i < 16; i++) {
		if (i != 0) {
			int x_q = x + i;

			int q_s = block_region_source[threadIdx.y][threadIdx.x + 16 + i];
			//			int q_s = getPixel(source,x_q,y,width,height);

			int q_t = block_region_target[threadIdx.y][threadIdx.x + 16 + i];
			//			int q_t = getPixel(target,x_qt,y,width,height);

			float delta_c = float(abs(p_s - q_s));
			float delta_g = float(abs(x_q - x));
			//			float w1 = exp(-delta_c / GAMMA_C1 - delta_g / GAMMA_G1);
			float w1 = __expf(
					__fdividef(-delta_c, GAMMA_C1)
							- __fdividef(delta_g, GAMMA_G1));
			delta_c = float(abs(p_t - q_t));
			float w2 = __expf(
					__fdividef(-delta_c, GAMMA_C1)
							- __fdividef(delta_g, GAMMA_G1));

			float d = (abs(q_s - q_t));
			sum1 += w1 * w2 * d;
			sum2 += w1 * w2;
		}

	}
	//		V[y * width * max_disp + x * (max_disp) + abs(disp)] += sum1 / sum2;
	V[abs(disp) * width * height + y * width + x] = __fdividef(sum1, sum2);
	__syncthreads();
	//    V[y * width + x * max_disp + disp] = 1.0;
}

__global__ void WTA1(int* d, float* F, float* V, int height, int width,
		int max_disp) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height) {
		//printf("%d : %d\n",x,y);
		__syncthreads();
		return;
	}
	int d_idx = y * width + x;
	float min1 = CUDART_INF;
	float min2 = CUDART_INF;
	for (int i = 0; i < max_disp; i++) {
		//		int idx = y * width * max_disp + x * max_disp + i;
		int idx = i * width * height + y * width + x;
		if (V[idx] < min1) {
			min2 = min1;
			min1 = V[idx];
			d[d_idx] = i;
		} else if (min2 > V[idx]) {
			min2 = V[idx];
		}
	}
	F[d_idx] = __fdividef((min2 - min1), min2);
	//d[y*width +x]=img[y*width +x];
	__syncthreads();
	//printf("%f %f\n",min1,min2);
}

__global__ void consistencyCheck(int* d_left, int* d_right, float* F_left,
		float* F_right, int width, int height, bool zero_disp) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height) {
		//printf("%d : %d\n",x,y);
		__syncthreads();
		return;
	}

	int d_p = d_left[y * width + x];
	int d_pt = d_right[y * width + x - d_p];

	if (abs(d_p - d_pt) > 1) {
		F_left[y * width + x] = 0;
		F_right[y * width + x - d_p] = 0;
		if (zero_disp) {
			d_left[y * width + x] = 0;
			d_right[y * width + x - d_p] = 0;
		}
		//        d_left[y * width + x] = 0;
		//        d_right[y * width + x - d_p] = 0;
	}
	__syncthreads();
}

__global__ void verticalRefinement(unsigned char* image, int* disp, float* F,
		float* D, float* N, float* eps, int width, int height) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x >= width || y >= height) {
		__syncthreads();
		__syncthreads();
		return;
	}

	__shared__ char block_region_source[48][16];

	//	if (threadIdx.y < 3) {
	//		int xb = blockIdx.x * blockDim.x;
	//		int yb = (blockIdx.y) * blockDim.y + threadIdx.y * blockDim.x
	//				+ threadIdx.x - 16;
	//		for (int i = 0; i < 16; i++) {
	//			block_region_source[threadIdx.y * blockDim.x + threadIdx.x][i] =
	//					getPixel(image, xb + i, yb, width, height);
	//
	//		}
	//	}
	int yb = y - 16;
	block_region_source[threadIdx.y][threadIdx.x] = getPixel(image, x, yb,
			width, height);
	block_region_source[threadIdx.y + 16][threadIdx.x] = getPixel(image, x,
			yb + 16, width, height);
	block_region_source[threadIdx.y + 32][threadIdx.x] = getPixel(image, x,
			yb + 32, width, height);
	__syncthreads();
	int idx = y * width + x;
	N[idx] = 0;
	D[idx] = 0;
	int p_s = block_region_source[threadIdx.y + 16][threadIdx.x];

	for (int i = -16; i < 16; i++) {
		if (i != 0) {
			int y_q = y + i;
			if (y_q < 0 || y_q >= height) {
				y_q = y;
			}
			int q_s = block_region_source[threadIdx.y + 16 + i][threadIdx.x];

			float delta_c = float(abs(p_s - q_s));
			float delta_g = float(abs(y_q - y));
			float w1 = __expf(
					__fdividef(-delta_c, GAMMA_C2)
							- __fdividef(delta_g, GAMMA_G2));

			N[idx] += w1 * F[y_q * width + x] * disp[y_q * width + x];
			D[idx] += w1 * F[y_q * width + x];
		}
	}
	//eps[y * width + x] = N[y * width + x] / D[y * width + x];
	eps[idx] = __fdividef(N[idx], D[idx]);
	__syncthreads();
}

__global__ void horizontalRefinement(unsigned char* image, int* disp, float* F,
		float* D, float* N, float* eps, float* D_v, float* N_v, float* eps_v,
		int width, int height) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x >= width || y >= height) {
		__syncthreads();
		__syncthreads();
		return;
	}

	__shared__ char block_region_source[16][48];

	//	if (threadIdx.y < 3) {
	//		int xb = blockIdx.x * blockDim.x + threadIdx.y * blockDim.x
	//				+ threadIdx.x - 16;
	//		int yb = blockIdx.y * blockDim.y;
	//		for (int i = 0; i < 16; i++) {
	//			block_region_source[i][threadIdx.y * blockDim.x + threadIdx.x] =
	//					getPixel(image, xb, yb + i, width, height);
	//		}
	//	}

	int xb = x - 16;
	block_region_source[threadIdx.y][threadIdx.x] = getPixel(image, xb, y,
			width, height);
	block_region_source[threadIdx.y][threadIdx.x + 16] = getPixel(image,
			xb + 16, y, width, height);
	block_region_source[threadIdx.y][threadIdx.x + 32] = getPixel(image,
			xb + 32, y, width, height);
	__syncthreads();
	int idx = y * width + x;
	N[idx] = 0;
	D[idx] = 0;
	int p_s = block_region_source[threadIdx.y][threadIdx.x + 16];

	for (int i = -16; i < 16; i++) {
		if (i != 0) {
			int x_q = x + i;
			if (x_q < 0 || x_q >= width) {
				x_q = x;
			}
			int q_s = block_region_source[threadIdx.y][threadIdx.x + 16 + i];

			float delta_c = float(abs(p_s - q_s));
			float delta_g = float(abs(x_q - x));
			float w1 = __expf(
					__fdividef(-delta_c, GAMMA_C2)
							- __fdividef(delta_g, GAMMA_G2));

			N[idx] += w1 * F[y * width + x_q] * D_v[y * width + x_q]
					* eps_v[y * width + x_q];
			D[idx] += w1 * F[y * width + x_q] * D_v[y * width + x_q];
		}
	}
	eps[idx] = __fdividef(N[idx], D[idx]);
	__syncthreads();
}

__global__ void WTA2(float* V, int* disp, float* D, float* F, float* eps,
		int width, int height, int max_disp) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < 0 || x >= width || y < 0 || y >= height) {
		__syncthreads();
		return;
	}
	int idx = y * width + x;
	float min1 = CUDART_INF;
	float min2 = CUDART_INF;
	for (int d = 0; d < max_disp; d++) {
		float lambda = ALPHA * D[idx] * abs(eps[idx] - d);
		//		int vidx = y * width * max_disp + x * max_disp + d;
		int vidx = d * width * height + y * width + x;
		if (min1 > (V[vidx] + lambda)) {
			disp[idx] = d;
			min2 = min1;
			min1 = V[vidx] + lambda;
		} else if (min2 > (V[vidx] + lambda)) {
			min2 = V[vidx] + lambda;
		}
		F[idx] = __fdividef((min2 - min1), min2);
	}
	__syncthreads();
}

__global__ void disparityMedianFilter(int * disp, int* out_disp, int width,
		int height) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	__shared__ int block_window[18][18];
	block_window[threadIdx.y + 1][threadIdx.x + 1] = getPixelD(disp, x, y,
			width, height);

	if (threadIdx.x == 0 && threadIdx.y == 0)
		block_window[0][0] = getPixelD(disp, x - 1, y - 1, width, height);
	if (threadIdx.x == 15 && threadIdx.y == 0)
		block_window[0][17] = getPixelD(disp, x + 1, y - 1, width, height);
	if (threadIdx.x == 0 && threadIdx.y == 15)
		block_window[17][0] = getPixelD(disp, x - 1, y + 1, width, height);
	if (threadIdx.x == 15 && threadIdx.y == 15)
		block_window[17][17] = getPixelD(disp, x + 1, y + 1, width, height);
	if (threadIdx.x == 0)
		block_window[threadIdx.y + 1][0] = getPixelD(disp, x - 1, y + 1, width,
				height);
	if (threadIdx.x == 15)
		block_window[threadIdx.y + 1][17] = getPixelD(disp, x + 1, y + 1, width,
				height);
	if (threadIdx.y == 0)
		block_window[0][threadIdx.x + 1] = getPixelD(disp, x + 1, y - 1, width,
				height);
	if (threadIdx.y == 15)
		block_window[17][threadIdx.x + 1] = getPixelD(disp, x + 1, y + 1, width,
				height);

	__syncthreads();
	int v[9] = { block_window[threadIdx.y][threadIdx.x],
			block_window[threadIdx.y][threadIdx.x + 1],
			block_window[threadIdx.y][threadIdx.x + 2], block_window[threadIdx.y
					+ 1][threadIdx.x], block_window[threadIdx.y + 1][threadIdx.x
					+ 1], block_window[threadIdx.y + 1][threadIdx.x + 2],
			block_window[threadIdx.y + 2][threadIdx.x], block_window[threadIdx.y
					+ 2][threadIdx.x + 1],
			block_window[threadIdx.y + 2][threadIdx.x + 2] };

	for (int i = 0; i < 5; i++) {
		for (int j = i + 1; j < 9; j++) {
			if (v[i] > v[j]) {
				int tmp = v[i];
				v[i] = v[j];
				v[j] = tmp;
			}
		}
	}
	out_disp[y * width + x] = v[4];
}

__global__ void disparityFill(int* disp, int * out_disp, int width,
		int height) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	__shared__ int block_region[16][48];
	int xb = x - 16;
	block_region[threadIdx.y][threadIdx.x] = getPixelD(disp, xb, y, width,
			height);
	block_region[threadIdx.y][threadIdx.x + 16] = getPixelD(disp, xb + 16, y,
			width, height);
	block_region[threadIdx.y][threadIdx.x + 32] = getPixelD(disp, xb + 32, y,
			width, height);
	__syncthreads();
	int sum=0;
	for (int i = -16; i < 16; i++) {
		sum+=block_region[threadIdx.y][threadIdx.x + 16 + i];
	}
	int res=sum/33;
	if(block_region[threadIdx.y][threadIdx.x + 16]==0){
		out_disp[y*width+x]=res;
	}else{
		out_disp[y*width+x]=block_region[threadIdx.y][threadIdx.x + 16];
	}
}

int* disparity(const unsigned char* left, const unsigned char* right,
		int max_disp, int width, int height, int iterations) {
	size_t img_size = width * height;
	unsigned char *left_d;
	unsigned char *right_d;
	cudaError_t cudaStatus;
	cudaMalloc((void**) &left_d, img_size * sizeof(char));
	cudaMalloc((void**) &right_d, img_size * sizeof(char));
	cudaMemcpy(left_d, left, img_size * sizeof(char), cudaMemcpyHostToDevice);
	cudaMemcpy(right_d, right, img_size * sizeof(char), cudaMemcpyHostToDevice);

	float* V_left;
	float* V_right;

	cudaMalloc((void**) &V_left, width * height * max_disp * sizeof(float));
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		cout << "CUDA error occured :" << cudaGetErrorString(cudaStatus)
				<< endl;
	}
	cudaMalloc((void**) &V_right, width * height * max_disp * sizeof(float));
	cudaMemset(V_left, 0, width * height * max_disp * sizeof(float));
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		cout << "CUDA error occured :" << cudaGetErrorString(cudaStatus)
				<< endl;
	}
	cudaMemset(V_right, 0, width * height * max_disp * sizeof(float));
	int blocks_x = width / 16;
	if (width % 16 != 0) {
		blocks_x++;
	}
	int blocks_y = height / 16;
	if (height % 16 != 0) {
		blocks_y++;
	}
	cudaDeviceSynchronize();
	dim3 blocks(blocks_x, blocks_y);
	dim3 threads(16, 16);
	for (int i = 0; i < max_disp; i++) {
		verticalCostAggregation<<<blocks, threads>>>(left_d, right_d, V_left,
				-i, max_disp, width, height, blocks_x);
		verticalCostAggregation<<<blocks, threads>>>(right_d, left_d, V_right,
				i, max_disp, width, height, blocks_x);
		cudaDeviceSynchronize();
		horizontalCostAggregation<<<blocks, threads>>>(left_d, right_d, V_left,
				-i, max_disp, width, height, blocks_x);

		horizontalCostAggregation<<<blocks, threads>>>(right_d, left_d, V_right,
				i, max_disp, width, height, blocks_x);
		cudaDeviceSynchronize();
	}
	float* F_left;
	float* F_right;
	int* d_left;
	int* d_right;
	cudaMalloc((void**) &F_left, width * height * sizeof(float));
	cudaMalloc((void**) &F_right, width * height * sizeof(float));
	cudaMalloc((void**) &d_left, width * height * sizeof(int));
	cudaMalloc((void**) &d_right, width * height * sizeof(int));
	cudaMemset(d_left, 0, width * height * sizeof(int));
	cudaMemset(d_right, 0, width * height * sizeof(int));
	cudaDeviceSynchronize();
	WTA1<<<blocks, threads>>>(d_left, F_left, V_left, height, width, max_disp);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		cout << "CUDA error occured :" << cudaGetErrorString(cudaStatus)
				<< endl;
	}
	cudaDeviceSynchronize();
	WTA1<<<blocks, threads>>>(d_right, F_right, V_right, height, width,
			max_disp);
	cudaDeviceSynchronize();
	consistencyCheck<<<blocks, threads>>>(d_left, d_right, F_left, F_right,
			width, height, false);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		cout << "CUDA error occured :" << cudaGetErrorString(cudaStatus)
				<< endl;
	}
	cudaDeviceSynchronize();

	float *N_v_left, *N_h_left, *D_v_left, *D_h_left, *eps_v_left, *eps_h_left;
	float *N_v_right, *N_h_right, *D_v_right, *D_h_right, *eps_v_right,
			*eps_h_right;
	cudaMalloc((void**) &N_v_left, width * height * sizeof(float));
	cudaMalloc((void**) &N_h_left, width * height * sizeof(float));
	cudaMalloc((void**) &D_v_left, width * height * sizeof(float));
	cudaMalloc((void**) &D_h_left, width * height * sizeof(float));
	cudaMalloc((void**) &eps_v_left, width * height * sizeof(float));
	cudaMalloc((void**) &eps_h_left, width * height * sizeof(float));

	cudaMalloc((void**) &N_v_right, width * height * sizeof(float));
	cudaMalloc((void**) &N_h_right, width * height * sizeof(float));
	cudaMalloc((void**) &D_v_right, width * height * sizeof(float));
	cudaMalloc((void**) &D_h_right, width * height * sizeof(float));
	cudaMalloc((void**) &eps_v_right, width * height * sizeof(float));
	cudaMalloc((void**) &eps_h_right, width * height * sizeof(float));

	for (int i = 0; i < iterations; i++) {
		verticalRefinement<<<blocks, threads>>>(left_d, d_left, F_left,
				D_v_left, N_v_left, eps_v_left, width, height);
		verticalRefinement<<<blocks, threads>>>(right_d, d_right, F_right,
				D_v_right, N_v_right, eps_v_right, width, height);
		cudaDeviceSynchronize();

		horizontalRefinement<<<blocks, threads>>>(left_d, d_left, F_left,
				D_h_left, N_h_left, eps_h_left, D_v_left, N_v_left, eps_v_left,
				width, height);
		horizontalRefinement<<<blocks, threads>>>(right_d, d_right, F_right,
				D_h_right, N_h_right, eps_h_right, D_v_right, N_v_right,
				eps_v_right, width, height);
		cudaDeviceSynchronize();
		WTA2<<<blocks, threads>>>(V_left, d_left, D_h_left, F_left, eps_h_left,
				width, height, max_disp);
		WTA2<<<blocks, threads>>>(V_right, d_right, D_h_right, F_right,
				eps_h_right, width, height, max_disp);
		cudaDeviceSynchronize();
		if (i == (iterations - 1)) {
			consistencyCheck<<<blocks, threads>>>(d_left, d_right, F_left,
					F_right, width, height, true);
		} else {
			consistencyCheck<<<blocks, threads>>>(d_left, d_right, F_left,
					F_right, width, height, false);
		}
		cudaDeviceSynchronize();
	}
	int* outDisp;
	cudaMalloc((void**) &outDisp, width * height * sizeof(int));
	disparityFill<<<blocks, threads>>>(d_left,outDisp,width,height);
	cudaMemcpy(d_left,outDisp,width*height*sizeof(int),cudaMemcpyDeviceToDevice);
	cudaDeviceSynchronize();
	disparityMedianFilter<<<blocks, threads>>>(d_left, outDisp, width, height);
	int* resv;
	resv = new int[width * height];
	cudaMemcpy(resv, outDisp, width * height * sizeof(int),
			cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	//	for (int i = 0; i < height; i++) {
	//		for (int j = 0; j < width; j++) {
	//			std::cout << i << "," << j << ": "
	//					<< resv[i * width + j] << std::endl;
	//		}
	//
	//	}
	cudaFree(left_d);
	cudaFree(right_d);
	cudaFree(V_left);
	cudaFree(V_right);
	cudaFree(F_left);
	cudaFree(F_right);
	cudaFree(d_left);
	cudaFree(d_right);
	cudaFree(N_v_left);
	cudaFree(N_h_left);
	cudaFree(D_v_left);
	cudaFree(D_h_left);
	cudaFree(eps_v_left);
	cudaFree(eps_h_left);
	cudaFree(N_v_right);
	cudaFree(N_h_right);
	cudaFree(D_v_right);
	cudaFree(D_h_right);
	cudaFree(eps_v_right);
	cudaFree(eps_h_right);
	return resv;
}
