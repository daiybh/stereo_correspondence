#include "sncc.h"
#include <cuda_runtime.h>
#include <math_constants.h>
#include <iostream>

__global__ void boxFilter3x3(float* in, float* out, int width) {
    int x = (blockIdx.x * blockDim.x + threadIdx.x);
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int img_x = x + 16;
    int img_y = y + 16;
    int bx = threadIdx.x + 1;
    int by = threadIdx.y + 1;
    __shared__ float block_window[18][18];

    block_window[threadIdx.y + 1][threadIdx.x + 1] = in[(img_y) * (width + 32) + img_x];
    if (threadIdx.x == 0 && threadIdx.y == 0)
        block_window[0][0] = in[(img_y - 1) * (width + 32) + img_x - 1];
    if (threadIdx.x == 15 && threadIdx.y == 0)
        block_window[0][17] = in[(img_y - 1) * (width + 32) + img_x + 1];
    if (threadIdx.x == 0 && threadIdx.y == 15)
        block_window[17][0] = in[(img_y + 1) * (width + 32) + img_x - 1];
    if (threadIdx.x == 15 && threadIdx.y == 15)
        block_window[17][17] = in[(img_y + 1) * (width + 32) + img_x + 1];
    if (threadIdx.x == 0)
        block_window[threadIdx.y + 1][0] = in[(img_y + 1) * (width + 32) + img_x - 1];
    if (threadIdx.x == 15)
        block_window[threadIdx.y + 1][17] = in[(img_y + 1) * (width + 32) + img_x + 1];
    if (threadIdx.y == 0)
        block_window[0][threadIdx.x + 1] = in[(img_y - 1) * (width + 32) + img_x + 1];
    if (threadIdx.y == 15)
        block_window[17][threadIdx.x + 1] = in[(img_y + 1) * (width + 32) + img_x + 1];
    __syncthreads();

    float sum = 0.0;
    sum += block_window[by - 1][bx - 1];
    sum += block_window[by - 1][bx];
    sum += block_window[by - 1][bx + 1];
    sum += block_window[by][bx - 1];
    sum += block_window[by][bx];
    sum += block_window[by][bx + 1];
    sum += block_window[by + 1][bx - 1];
    sum += block_window[by + 1][bx];
    sum += block_window[by + 1][bx + 1];
    //    out[y * width + x] = __fdividef(sum, 9.0);
    out[y * width + x] = sum / 9.0;
    __syncthreads();

}

__global__ void sdFilter3x3(float* in, float* out, float* means, int width) {
    int x = (blockIdx.x * blockDim.x + threadIdx.x);
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int img_x = x + 16;
    int img_y = y + 16;
    int bx = threadIdx.x + 1;
    int by = threadIdx.y + 1;
    __shared__ float block_window[18][18];


    block_window[threadIdx.y + 1][threadIdx.x + 1] = in[(img_y) * (width + 32) + img_x];
    if (threadIdx.x == 0 && threadIdx.y == 0)
        block_window[0][0] = in[(img_y - 1) * (width + 32) + img_x - 1];
    if (threadIdx.x == 15 && threadIdx.y == 0)
        block_window[0][17] = in[(img_y - 1) * (width + 32) + img_x + 1];
    if (threadIdx.x == 0 && threadIdx.y == 15)
        block_window[17][0] = in[(img_y + 1) * (width + 32) + img_x - 1];
    if (threadIdx.x == 15 && threadIdx.y == 15)
        block_window[17][17] = in[(img_y + 1) * (width + 32) + img_x + 1];
    if (threadIdx.x == 0)
        block_window[threadIdx.y + 1][0] = in[(img_y + 1) * (width + 32) + img_x - 1];
    if (threadIdx.x == 15)
        block_window[threadIdx.y + 1][17] = in[(img_y + 1) * (width + 32) + img_x + 1];
    if (threadIdx.y == 0)
        block_window[0][threadIdx.x + 1] = in[(img_y - 1) * (width + 32) + img_x + 1];
    if (threadIdx.y == 15)
        block_window[17][threadIdx.x + 1] = in[(img_y + 1) * (width + 32) + img_x + 1];
    __syncthreads();
    float mean = means[y * width + x];
    float sum = 0.0;
    sum += __powf(block_window[by - 1][bx - 1], 2);
    sum += __powf(block_window[by - 1][bx], 2);
    sum += __powf(block_window[by - 1][bx + 1], 2);
    sum += __powf(block_window[by][bx - 1], 2);
    sum += __powf(block_window[by][bx], 2);
    sum += __powf(block_window[by][bx + 1], 2);
    sum += __powf(block_window[by + 1][bx - 1], 2);
    sum += __powf(block_window[by + 1][bx], 2);
    sum += __powf(block_window[by + 1][bx + 1], 2);
    sum /= 9.0;
    sum -= __powf(mean, 2.0);
    out[y * width + x] = sqrtf(sum);
    __syncthreads();

}

__global__ void setCorrelation(float* correlation, int width) {
    int x = (blockIdx.x * blockDim.x + threadIdx.x);
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    correlation[y * width + x] = -CUDART_INF_F;
}

__global__ void correlationFilter3x3(float *source, float *target, float *cor, float *means_source, float *means_target, float *sds_source, float *sds_target, int disp, int height, int width) {
    int x = (blockIdx.x * blockDim.x + threadIdx.x);
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int img_x = x + 16;
    int img_y = y + 16;
    int img_dx = img_x - disp;
    int bx = threadIdx.x + 1;
    int by = threadIdx.y + 1;

    __shared__ float block_source_window[18][18];
    __shared__ float block_target_window[18][18];
    if ((x - disp) < 0) {
        cor[img_y * (width + 32) + img_x] = 0;
        __syncthreads();
        return;
    }
    block_source_window[by][bx] = source[(img_y) * (width + 32) + img_x];
    block_target_window[by][bx] = target[(img_y) * (width + 32) + img_dx];
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        block_source_window[0][0] = source[(img_y - 1) * (width + 32) + img_x - 1];
        block_target_window[0][0] = target[(img_y - 1) * (width + 32) + img_dx - 1];
    }
    if (threadIdx.x == 15 && threadIdx.y == 0) {
        block_source_window[0][17] = source[(img_y - 1) * (width + 32) + img_x + 1];
        block_target_window[0][17] = target[(img_y - 1) * (width + 32) + img_dx + 1];
    }
    if (threadIdx.x == 0 && threadIdx.y == 15) {
        block_source_window[17][0] = source[(img_y + 1) * (width + 32) + img_x - 1];
        block_target_window[17][0] = target[(img_y + 1) * (width + 32) + img_dx - 1];
    }
    if (threadIdx.x == 15 && threadIdx.y == 15) {
        block_source_window[17][17] = source[(img_y + 1) * (width + 32) + img_x + 1];
        block_target_window[17][17] = target[(img_y + 1) * (width + 32) + img_dx + 1];
    }
    if (threadIdx.x == 0) {
        block_source_window[threadIdx.y + 1][0] = source[(img_y + 1) * (width + 32) + img_x - 1];
        block_target_window[threadIdx.y + 1][0] = target[(img_y + 1) * (width + 32) + img_dx - 1];
    }
    if (threadIdx.x == 15) {
        block_source_window[threadIdx.y + 1][17] = source[(img_y + 1) * (width + 32) + img_x + 1];
        block_target_window[threadIdx.y + 1][17] = target[(img_y + 1) * (width + 32) + img_dx + 1];
    }
    if (threadIdx.y == 0) {
        block_source_window[0][threadIdx.x + 1] = source[(img_y - 1) * (width + 32) + img_x + 1];
        block_target_window[0][threadIdx.x + 1] = target[(img_y - 1) * (width + 32) + img_dx + 1];
    }
    if (threadIdx.y == 15) {
        block_source_window[17][threadIdx.x + 1] = source[(img_y + 1) * (width + 32) + img_x + 1];
        block_target_window[17][threadIdx.x + 1] = target[(img_y + 1) * (width + 32) + img_dx + 1];
    }
    __syncthreads();

    float mean_source = means_source[y * width + x];
    float mean_target = means_target[y * width + x - disp];
    float sd_source = sds_source[y * width + x];
    float sd_target = sds_target[y * width + x - disp];

    //    float mean_source = 1;
    //    float mean_target = 1;
    //    float sd_source = 1;
    //    float sd_target = 1;
    float sum = 0.0;
    sum += block_source_window[by - 1][bx - 1] * block_target_window[by - 1][bx - 1];
    sum += block_source_window[by - 1][bx] * block_target_window[by - 1][bx];
    sum += block_source_window[by - 1][bx + 1] * block_target_window[by - 1][bx + 1];
    sum += block_source_window[by][bx - 1] * block_target_window[by][bx - 1];
    sum += block_source_window[by][bx] * block_target_window[by][bx];
    sum += block_source_window[by][bx + 1] * block_target_window[by][bx + 1];
    sum += block_source_window[by + 1][bx - 1] * block_target_window[by + 1][bx - 1];
    sum += block_source_window[by + 1][bx] * block_target_window[by + 1][bx];
    sum += block_source_window[by + 1][bx + 1] * block_target_window[by + 1][bx + 1];
    sum = sum/ 9.0;
    sum -= mean_source*mean_target;
    sum = sum / (sd_source * sd_target);
    cor[img_y * (width + 32) + img_x] = sum;
}

__global__ void WTA(float *cor, float *bestCor, unsigned char *disparity, int width, int disp, float *image) {
    int x = (blockIdx.x * blockDim.x + threadIdx.x);
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int idx = y * width + x;
    float correlation = cor[idx];
    if (correlation > bestCor[idx] && correlation <= 1.0) {
        bestCor[idx] = correlation;
        disparity[idx] = disp;
    }
    //disparity[idx] = image[(y+16)*(width+32)+x+16];
}

__global__ void boxFilter5x9(float *in, float *out, int width) {
    int x = (blockIdx.x * blockDim.x + threadIdx.x);
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int in_x = x + 16;
    int in_y = y + 16;
    int b_x = threadIdx.x + 4;
    int b_y = threadIdx.y + 2;
    
    __shared__ float block_region[20][24];
    // loading center part
    block_region[b_y][b_x]=in[in_y*(width+32)+in_x];
    // loading corners
    if(threadIdx.x < 4 && threadIdx.y < 2){
        block_region[b_y-2][b_x-4]=in[(in_y-2)*(width+32)+(in_x-4)];
    }
    if(threadIdx.x >= 12 && threadIdx.y < 2){
        block_region[b_y-2][b_x+4]=in[(in_y-2)*(width+32)+(in_x+4)];
    }
    if(threadIdx.x < 4 && threadIdx.y >= 14){
        block_region[b_y+2][b_x-4]=in[(in_y+2)*(width+32)+(in_x-4)];
    }
    if(threadIdx.x >= 12 && threadIdx.y >= 14){
        block_region[b_y+2][b_x+4]=in[(in_y+2)*(width+32)+(in_x+4)];
    }
    // loading sides
    if(threadIdx.x < 4){
        block_region[b_y][b_x-4]=in[(in_y)*(width+32)+(in_x-4)];
    }
    if(threadIdx.x >= 12){
        block_region[b_y][b_x+4]=in[(in_y)*(width+32)+(in_x+4)];
    }
    // loading bottom and top
    if(threadIdx.y < 2){
        block_region[b_y - 2][b_x]=in[(in_y-2)*(width+32)+(in_x)];
    }
    if(threadIdx.y >= 14){
        block_region[b_y + 2][b_x]=in[(in_y+2)*(width+32)+(in_x)];
    }
    __syncthreads();

    float sum = 0.0;
    for (int i = b_y - 2; i <= (b_y + 2); i++) {
        for (int j = b_x - 4; j <= (b_x + 4); j++) {
            sum += block_region[i][j];
        }
    }
    float avg = __fdividef(sum, 45.0);
    out[y * width + x] = avg;
} 

__global__ void boxFilterMxN(float *in, float *out, int width, int M, int N, int mHalf, int nHalf) {
    int x = (blockIdx.x * blockDim.x + threadIdx.x);
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int in_x = x + 16;
    int in_y = y + 16;
    int b_x = threadIdx.x + nHalf;
    int b_y = threadIdx.y + mHalf;
    int blockWidth = 16 + N -1;
    extern __shared__ float block_region[];
    block_region[b_y*blockWidth + b_x]=in[in_y*(width+32)+in_x];
    
    // loading corners
    if(threadIdx.x < nHalf && threadIdx.y < mHalf){
        block_region[(b_y-mHalf)*blockWidth + b_x-nHalf]=in[(in_y-mHalf)*(width+32)+(in_x-nHalf)];
    }
    if(threadIdx.x >= (16-nHalf) && threadIdx.y < mHalf){
        block_region[(b_y-mHalf)*blockWidth + b_x+nHalf]=in[(in_y-mHalf)*(width+32)+(in_x+nHalf)];
    }
    if(threadIdx.x < nHalf && threadIdx.y >= (16-mHalf)){
        block_region[(b_y+mHalf)*blockWidth + b_x-nHalf]=in[(in_y+mHalf)*(width+32)+(in_x-nHalf)];
    }
    if(threadIdx.x >= (16-nHalf) && threadIdx.y >= (16-mHalf)){
        block_region[(b_y+mHalf)*blockWidth + b_x+nHalf]=in[(in_y+mHalf)*(width+32)+(in_x+nHalf)];
    }
    // loading sides
    if(threadIdx.x < nHalf){
        block_region[b_y*blockWidth+b_x-nHalf]=in[(in_y)*(width+32)+(in_x-nHalf)];
    }
    if(threadIdx.x >= (16-nHalf)){
        block_region[b_y*blockWidth+b_x+nHalf]=in[(in_y)*(width+32)+(in_x+nHalf)];
    }
    // loading bottom and top
    if(threadIdx.y < mHalf){
        block_region[(b_y-mHalf)*blockWidth + b_x]=in[(in_y-mHalf)*(width+32)+(in_x)];
    }
    if(threadIdx.y >= (16-mHalf)){
        block_region[(b_y+mHalf)*blockWidth + b_x]=in[(in_y+mHalf)*(width+32)+(in_x)];
    }
    __syncthreads();
    
    float sum = 0.0;
    for (int i = b_y - mHalf; i <= (b_y + mHalf); i++) {
        for (int j = b_x - nHalf; j <= (b_x + nHalf); j++) {
            sum += block_region[i*blockWidth + j];
        }
    }
    int size=M*N;
//    float avg = __fdividef(sum, size);
    float avg = sum/ float(size);
    out[y * width + x] = avg; 
}

__global__ void consistencyCheck(unsigned char* d_left, unsigned char* d_right, int width) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;


    int d_p = d_left[y * width + x];
    if ((x - d_p) < 0) {
        __syncthreads();
        return;
    }
    int d_pt = d_right[y * width + x - d_p];

    if (abs(d_p - d_pt) > 1) {
        d_left[y * width + x] = 0;
        d_right[y * width + x - d_p] = 0;
    }
    __syncthreads();
}

__device__ unsigned char getPixelD(unsigned char* image, int x, int y, int width, int height) {
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

__global__ void disparityFill(unsigned char* disp, unsigned char* out_disp, int width, int height, int maxDisp) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    __shared__ unsigned char block_region[16][48];
    int xb = x - 16;
    block_region[threadIdx.y][threadIdx.x] = getPixelD(disp, xb, y, width,
            height);
    block_region[threadIdx.y][threadIdx.x + 16] = getPixelD(disp, xb + 16, y,
            width, height);
    block_region[threadIdx.y][threadIdx.x + 32] = getPixelD(disp, xb + 32, y,
            width, height);
    __syncthreads();

    int left_i = -1, right_i = 1;
    unsigned char possible_disp = 0;
    while ((left_i > -16) && (right_i < 16)) {
        int left_v = block_region[threadIdx.y][threadIdx.x + 16 + left_i];
        int right_v = block_region[threadIdx.y][threadIdx.x + 16 + right_i];
        if (left_v != 0 && right_v != 0) {
            //possible_disp = (left_v + right_v) / 2;
            if (abs(left_i) < right_i) {
                possible_disp = left_v;
            } else {
                possible_disp = right_v;
            }
            break;
        }
        if (left_v == 0) {
            left_i--;
        }
        if (right_v == 0) {
            right_i++;
        }
    }
    if (block_region[threadIdx.y][threadIdx.x + 16] == 0) {
        out_disp[y * width + x] = possible_disp;
    } else {
        out_disp[y * width + x] = block_region[threadIdx.y][threadIdx.x + 16];
    }
    if(x < maxDisp){
        out_disp[y * width + x] = 0;
    }

}

__global__ void disparityMedianFilter(unsigned char * disp, unsigned char* out_disp, int width,
        int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    __shared__ unsigned char block_window[18][18];
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
    unsigned char v[9] = {block_window[threadIdx.y][threadIdx.x],
        block_window[threadIdx.y][threadIdx.x + 1],
        block_window[threadIdx.y][threadIdx.x + 2], block_window[threadIdx.y
        + 1][threadIdx.x], block_window[threadIdx.y + 1][threadIdx.x
        + 1], block_window[threadIdx.y + 1][threadIdx.x + 2],
        block_window[threadIdx.y + 2][threadIdx.x], block_window[threadIdx.y
        + 2][threadIdx.x + 1],
        block_window[threadIdx.y + 2][threadIdx.x + 2]};

    for (int i = 0; i < 5; i++) {
        for (int j = i + 1; j < 9; j++) {
            if (v[i] > v[j]) {
                unsigned char tmp = v[i];
                v[i] = v[j];
                v[j] = tmp;
            }
        }
    }
    out_disp[y * width + x] = v[4];
}

unsigned char* disparity(float* source, float* target, int width, int height, int numDisparities, int avgWindowHeight, int avgWindowWidth) {
    size_t padded_size = (width + 32) * (height + 32);
    size_t size = width*height;
    float *left_image_dev;
    float *right_image_dev;
    cudaError_t cudaStatus;
    cudaMalloc((void**) &left_image_dev, padded_size * sizeof (float));
    cudaMalloc((void**) &right_image_dev, padded_size * sizeof (float));
    cudaMemcpy(left_image_dev, source, padded_size * sizeof (float),
            cudaMemcpyHostToDevice);
    cudaMemcpy(right_image_dev, target, padded_size * sizeof (float),
            cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cout << "CUDA error occured after copy :" << cudaGetErrorString(cudaStatus)
                << std::endl;
    }
    int blocks_x = width / 16;
    int blocks_y = height / 16;
    dim3 blocks(blocks_x, blocks_y);
    dim3 threads(16, 16);

    float *means_source, *means_target;
    cudaMalloc((void**) &means_source, size * sizeof (float));
    cudaMalloc((void**) &means_target, size * sizeof (float));
    cudaDeviceSynchronize();
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cout << "CUDA error occured after means alloc :" << cudaGetErrorString(cudaStatus)
                << std::endl;
    }
    boxFilter3x3 << <blocks, threads>>>(left_image_dev, means_source, width);
//    boxFilterMxN <<<blocks, threads,sizeof(float)*18*18>>>(left_image_dev, means_source, width,3,3,1,1);
    cudaDeviceSynchronize();
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cout << "CUDA error occured after first box filter :" << cudaGetErrorString(cudaStatus)
                << std::endl;
    }
    boxFilter3x3 << <blocks, threads>>>(right_image_dev, means_target, width);
//    boxFilterMxN <<<blocks, threads,sizeof(float)*18*18>>>(right_image_dev, means_target, width,3,3,1,1);
    cudaDeviceSynchronize();
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cout << "CUDA error occured after box filter :" << cudaGetErrorString(cudaStatus)
                << std::endl;
    }
    float *sds_source, *sds_target;
    cudaMalloc((void**) &sds_source, size * sizeof (float));
    cudaMalloc((void**) &sds_target, size * sizeof (float));
    sdFilter3x3 << <blocks, threads>>>(left_image_dev, sds_source, means_source, width);
    sdFilter3x3 << <blocks, threads>>>(right_image_dev, sds_target, means_target, width);
    cudaDeviceSynchronize();
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cout << "CUDA error occured after sd filters :" << cudaGetErrorString(cudaStatus)
                << std::endl;
    }
    float *bestCorrelation_left, *bestCorrelation_right;
    cudaMalloc((void**) &bestCorrelation_left, size * sizeof (float));
    cudaMalloc((void**) &bestCorrelation_right, size * sizeof (float));
    setCorrelation << <blocks, threads>>>(bestCorrelation_left, width);
    setCorrelation << <blocks, threads>>>(bestCorrelation_right, width);
    cudaDeviceSynchronize();
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cout << "CUDA error occured after setting best cor :" << cudaGetErrorString(cudaStatus)
                << std::endl;
    }
    float *cor_left, *cor_right;
    float *cor2_left, *cor2_right;
    unsigned char *disparityMap_left, *disparityMap_right;
    cudaMalloc((void**) &cor_left, padded_size * sizeof (float));
    cudaMalloc((void**) &cor2_left, size * sizeof (float));
    cudaMalloc((void**) &disparityMap_left, size * sizeof (unsigned char));
    cudaMalloc((void**) &cor_right, padded_size * sizeof (float));
    cudaMalloc((void**) &cor2_right, size * sizeof (float));
    cudaMalloc((void**) &disparityMap_right, size * sizeof (unsigned char));
    cudaMemset(disparityMap_left, 0, size * sizeof (unsigned char));
    for (int disp = 0; disp < numDisparities; disp++) {
        correlationFilter3x3 << <blocks, threads>>>(left_image_dev, right_image_dev, cor_left, means_source, means_target, sds_source, sds_target, disp, height, width);
        correlationFilter3x3 << <blocks, threads>>>(right_image_dev, left_image_dev, cor_right, means_target, means_source, sds_target, sds_source, -disp, height, width);
        cudaDeviceSynchronize();
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            std::cout << "CUDA error occured after correlation filter :" << cudaGetErrorString(cudaStatus)
                    << std::endl;
        }
//        boxFilter5x9 << <blocks, threads>>>(cor_left, cor2_left, width);
//        boxFilter5x9 << <blocks, threads>>>(cor_right, cor2_right, width);
        int mHalf=avgWindowHeight/2;
        int nHalf=avgWindowWidth/2;
        boxFilterMxN<<<blocks,threads,sizeof(float)*(16+avgWindowHeight-1)*(16+avgWindowWidth-1)>>>(cor_left,cor2_left,width,avgWindowHeight,avgWindowWidth,mHalf,nHalf);
        boxFilterMxN<<<blocks,threads,sizeof(float)*(16+avgWindowHeight-1)*(16+avgWindowWidth-1)>>>(cor_right,cor2_right,width,avgWindowHeight,avgWindowWidth,mHalf,nHalf);
        cudaStatus = cudaGetLastError();
        cudaDeviceSynchronize();
        
        if (cudaStatus != cudaSuccess) {
            std::cout << "CUDA error occured after box :" << cudaGetErrorString(cudaStatus)
                    << std::endl;
        }
        WTA << <blocks, threads>>>(cor2_left, bestCorrelation_left, disparityMap_left, width, disp, left_image_dev);
        WTA << <blocks, threads>>>(cor2_right, bestCorrelation_right, disparityMap_right, width, disp, right_image_dev);
        cudaDeviceSynchronize();
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            std::cout << "CUDA error occured after WTA :" << cudaGetErrorString(cudaStatus)
                    << std::endl;
        }
    }
    consistencyCheck << <blocks, threads>>>(disparityMap_left, disparityMap_right, width);
    cudaDeviceSynchronize();
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cout << "CUDA error occured after WTA :" << cudaGetErrorString(cudaStatus)
                << std::endl;
    }
    unsigned char *disparityMap_out;
    cudaMalloc((void**) &disparityMap_out, size * sizeof (unsigned char));
    disparityMedianFilter<<<blocks,threads>>>(disparityMap_left,disparityMap_out,width,height);
    cudaMemcpy(disparityMap_left,disparityMap_out,sizeof(unsigned char)*width*height, cudaMemcpyDeviceToDevice);
    disparityFill<<<blocks,threads>>>(disparityMap_left,disparityMap_out,width,height, numDisparities);
    cudaDeviceSynchronize();
    unsigned char* disparityMapHost;
    disparityMapHost = new unsigned char[width * height];
    cudaMemcpy(disparityMapHost, disparityMap_out, size * sizeof (unsigned char), cudaMemcpyDeviceToHost);
    //    unsigned char *out;
    //    out=new unsigned char[size];
    //    for(int i=0;i<size;i++){
    //        out[i]=char(disparityMapHost[i]);
    //    }
    cudaDeviceSynchronize();
    cudaFree(left_image_dev);
    cudaFree(right_image_dev);
    cudaFree(means_source);
    cudaFree(means_target);
    cudaFree(sds_source);
    cudaFree(sds_target);
    cudaFree(bestCorrelation_left);
    cudaFree(bestCorrelation_right);
    cudaFree(disparityMap_left);
    cudaFree(disparityMap_right);
    cudaFree(disparityMap_out);
    cudaFree(cor_left);
    cudaFree(cor_right);
    cudaFree(cor2_left);
    cudaFree(cor2_right);
    return disparityMapHost;

}