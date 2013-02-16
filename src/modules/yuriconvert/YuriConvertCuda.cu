/*!
 * @file 		YuriConvertCuda.cu
 * @author 		Zdenek Travnicek
 * @date 		13.8.2010
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2010 - 2013
 * 				Distributed under GNU Public License 3.0
 *
 */


#include <cuda.h>
//include "yuri/video/YuriConvertor.h"

__device__ void _RGB2YUV(char*s, float *Y, float *Cb, float*Cr, float Wb, float Wr, float Wg, float Kb, float Kr)
{
	float r,g,b;
	r = (float)((unsigned char)(s[0]))/255.0f;
	g = (float)((unsigned char)(s[1]))/255.0f;
	b = (float)((unsigned char)(s[2]))/255.0f;
	*Y=r*Wr + g*Wg + b*Wb;
	*Cb=(b-*Y)*Kb;
	*Cr=(r-*Y)*Kr;
}
__device__ unsigned int luma(float Y) {
	return (unsigned int)(64 + Y  * 876);
}
__device__ unsigned int chroma(float Y) {
	return (unsigned int)(512 + Y  * 896);
}
__global__ void RGB2YUV(char *s, char *d, size_t num, float Wb, float Wr, float Wg, float Kb, float Kr)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	//if (2*idx>=num) return;
	int si = 6*idx, di=5*idx;
	float Y1, Cr1, Cb1, Y2, Cb2, Cr2;
	unsigned int Ya, Yb, Cb, Cr;
	for (int i=idx;i<num;i+=4096) {
		si = i * 6;
		di = i * 5;
		_RGB2YUV(s+si+0, &Y1, &Cb1, &Cr1,  Wb, Wr, Wg, Kb, Kr);
		_RGB2YUV(s+si+3, &Y2, &Cb2, &Cr2, Wb, Wr, Wg, Kb, Kr);
		Ya = luma(Y1);
		Yb = luma(Y2);
		Cr= chroma((Cr1+Cr2)/2.0f);
		Cb= chroma((Cb1+Cb2)/2.0f);
		
		d[di]=Ya&0xFF;
		d[di+1]=((Ya>>8)&0x03) | ((Cr<<2)&0xfc); 
		d[di+2]=((Cr>>6)&0x0F) | ((Yb<<4)&0xF0); 
		d[di+3]=((Yb>>4)&0x3F) | ((Cb<<6)&0xC0);
		d[di+4]=((Cb>>2)&0xFF);
	}
}
__device__ unsigned int chr(float v) 
{
	float val = v * 255.0f;
	return (unsigned int)(val>255?255:(val<0?0:val));
}
__global__ void YUV162RGB(char *s, char *d, size_t num, float Wb, float Wr, float Wg, float Kb, float Kr)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (2*idx>=num) return;
	int si = 6*idx, di=5*idx;
	float Y1, Y2, U, V;	
	for (int i=idx;i<num/2;i+=4096) {
		si = i * 4;
		di = i * 6;
		Y1 = (float)((unsigned char)(s[si+0]))/255.0f;
		Y2 = (float)((unsigned char)(s[si+2]))/255.0f;
		U = (float)((unsigned char)(s[si+1]))/255.0f - 0.5f;
		V = (float)((unsigned char)(s[si+3]))/255.0f - 0.5f;
		d[di+0] = chr(Y1 + 1.403*V);
		d[di+1] = chr(Y1 - 0.344*U - 0.714*V);
		d[di+2] = chr(Y1 + 1.770*U);
		d[di+3] = chr(Y2 + 1.403*V);
		d[di+4] = chr(Y2 - 0.344*U - 0.714*V);
		d[di+5] = chr(Y2 + 1.770*U);
	}
}

void *CudaAlloc(unsigned int size)
{
	void *x;
	cudaMalloc((void **) &x, size*sizeof(char));
	return x;

}
void CudaDealloc(void *mem)
{
	cudaFree(mem);
}

bool YuriConvertRGB24_YUV20(const char *src, char *dest, void *src_cuda, void *dest_cuda,unsigned int num, float Wb, float Wr)
{
	size_t size = num*3, out_size=num*5/2;
	cudaMemcpy(reinterpret_cast<char*>(src_cuda),reinterpret_cast<const char*>(src),
			size,cudaMemcpyHostToDevice);
	float Wg, Kb, Kr;
	Wg=1.0f-Wr-Wb;
	Kb = 0.5f / (1.0f - Wb);
	Kr = 0.5f / (1.0f - Wr);

//	for (int i=0;i<1000;++i) {
		RGB2YUV <<<512, 8 >>> (reinterpret_cast<char*>(src_cuda), 
				reinterpret_cast<char*>(dest_cuda), num, Wb,Wr,Wg,Kb,Kr);
//	}
	cudaMemcpy(reinterpret_cast<char*>(dest),reinterpret_cast<char*>(dest_cuda),
			out_size,cudaMemcpyDeviceToHost);
	return true;
}



bool YuriConvertYUV16_RGB24(const char *src, char *dest, void *src_cuda, void *dest_cuda,unsigned int num, float Wb, float Wr)
{
	size_t size = num*2, out_size=num*3;
	cudaMemcpy(src_cuda,src,size,cudaMemcpyHostToDevice);
	float Wg, Kb, Kr;
	Wg=1.0f-Wr-Wb;
	Kb = 0.5f / (1.0f - Wb);
	Kr = 0.5f / (1.0f - Wr);

	YUV162RGB <<<512, 8 >>> (reinterpret_cast<char*>(src_cuda), 
			reinterpret_cast<char*>(dest_cuda), num, Wb,Wr,Wg,Kb,Kr);
	cudaMemcpy(dest,dest_cuda,out_size,cudaMemcpyDeviceToHost);
	return true;
}

