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
    out[y * width + x] = sum/9.0;
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
    sum -= __powf(mean,2.0);
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
    sum = sum/9.0;
    sum -= mean_source*mean_target;
    sum = sum / (sd_source * sd_target);
    cor[img_y * (width + 32) + img_x] = sum;
}

__global__ void WTA(float *cor, float *bestCor, unsigned char *disparity, int width, int disp, float *image) {
    int x = (blockIdx.x * blockDim.x + threadIdx.x);
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int idx = y * width + x;
    float correlation = cor[idx];
    if (correlation > bestCor[idx]) {
        bestCor[idx] = correlation;
        disparity[idx] = disp;
    }
    //disparity[idx] = image[(y+16)*(width+32)+x+16];
}

__global__ void boxFilter(float *in, float *out, int width, int avgWindow){
    int x = (blockIdx.x * blockDim.x + threadIdx.x);
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int in_x=x+16;
    int in_y=y+16;
    
    int win2=avgWindow/2;
    float sum=0.0;
    for(int i=in_y-win2; i<=(in_y+win2);i++){
        for(int j=in_x-win2; j<=(in_x+win2);j++){
            sum+=in[i*(width+32)+j];
        }
    }
    float avg=sum/(avgWindow*avgWindow);
    out[y*width+x]=avg;
}

unsigned char* disparity(float* source, float* target, int width, int height, int numDisparities, int avgWindow) {
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
    cudaDeviceSynchronize();
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cout << "CUDA error occured after first box filter :" << cudaGetErrorString(cudaStatus)
                << std::endl;
    }
    boxFilter3x3 << <blocks, threads>>>(right_image_dev, means_target, width);
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
    float *bestCorrelation;
    cudaMalloc((void**) &bestCorrelation, size * sizeof (float));
    setCorrelation << <blocks, threads>>>(bestCorrelation, width);
    cudaDeviceSynchronize();
    cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            std::cout << "CUDA error occured after setting best cor :" << cudaGetErrorString(cudaStatus)
                    << std::endl;
        }
    float *cor;
    float *cor2;
    unsigned char *disparityMap;
    cudaMalloc((void**) &cor, padded_size * sizeof (float));
    cudaMalloc((void**) &cor2, size * sizeof (float));
    cudaMalloc((void**) &disparityMap, size * sizeof (unsigned char));
    cudaMemset(disparityMap,0,size*sizeof(unsigned char));
    for (int disp = 0; disp < numDisparities; disp++) {
        correlationFilter3x3 << <blocks, threads>>>(left_image_dev, right_image_dev, cor, means_source, means_target, sds_source, sds_target, disp, height, width);
        cudaDeviceSynchronize();
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            std::cout << "CUDA error occured after correlation filter :" << cudaGetErrorString(cudaStatus)
                    << std::endl;
        }
        boxFilter<< <blocks, threads>>>(cor, cor2, width, 15);
        cudaDeviceSynchronize();
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            std::cout << "CUDA error occured after box :" << cudaGetErrorString(cudaStatus)
                    << std::endl;
        }
        WTA << <blocks, threads>>>(cor2, bestCorrelation, disparityMap, width, disp, left_image_dev);
        cudaDeviceSynchronize();
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            std::cout << "CUDA error occured after WTA :" << cudaGetErrorString(cudaStatus)
                    << std::endl;
        }
    }
    unsigned char* disparityMapHost;
    disparityMapHost = new unsigned char[width * height];
    cudaMemcpy(disparityMapHost, disparityMap, size * sizeof (unsigned char), cudaMemcpyDeviceToHost);
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
    cudaFree(bestCorrelation);
    cudaFree(disparityMap);
    cudaFree(cor);
    cudaFree(cor2);
    return disparityMapHost;

}