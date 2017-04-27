#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "colour-convert.h"

#define THREADS_PER_BLOCK 700
#define BLOCK_SIZE 1000

//EXPERIMENTING WITH THREADS AND BLOCK-SIZES:

//THREADS_PER_BLOCK ARE 10 50 100 250 500 750 1000
//BLOCK_SIZE 100 - SMALL IMAGE
//BLOCK_SIZE 1000 - LARGE IMAGE


__host__ __device__ unsigned char clip_rgb(int x)
{
    if(x > 255)
        return 255;
    if(x < 0)
        return 0;

    return (unsigned char)x;
}

__global__ void rgb2yuvKernel(PPM_IMG ppm, YUV_IMG yuv)
{
    int blockId = blockIdx.y * gridDim.x + blockIdx.x;
    int threadId = blockId * blockDim.x + threadIdx.x;
    unsigned char r, g, b;
    unsigned char y, cb, cr;

    r = ppm.img_r[threadId];
    g = ppm.img_g[threadId];
    b = ppm.img_b[threadId];

    y  = (unsigned char)( 0.299*r + 0.587*g +  0.114*b);
    cb = (unsigned char)(-0.169*r - 0.331*g +  0.499*b + 128);
    cr = (unsigned char)( 0.499*r - 0.418*g - 0.0813*b + 128);

    yuv.img_y[threadId] = y;
    yuv.img_u[threadId] = cb;
    yuv.img_v[threadId] = cr;
}

__global__ void yuv2rgbKernel(YUV_IMG yuv, PPM_IMG ppm)
{
    int blockId = blockIdx.y * gridDim.x + blockIdx.x;
    int threadId = blockId * blockDim.x + threadIdx.x;
    int rt, gt, bt;
    int y, cb, cr;

    y  = (int)yuv.img_y[threadId];
    cb = (int)yuv.img_u[threadId] - 128;
    cr = (int)yuv.img_v[threadId] - 128;

    rt  = (int)( y + 1.402*cr);
    gt  = (int)( y - 0.344*cb - 0.714*cr);
    bt  = (int)( y + 1.772*cb);

    ppm.img_r[threadId] = clip_rgb(rt);
    ppm.img_g[threadId] = clip_rgb(gt);
    ppm.img_b[threadId] = clip_rgb(bt);
}

__global__ void emptyKernel(void) {}


void launchEmptyKernel(void)
{
    emptyKernel<<<1,1>>>();
}

YUV_IMG rgb2yuvGPU(PPM_IMG img_in)
{
    YUV_IMG img_out;

    PPM_IMG device_ppm;
    YUV_IMG device_yuv;

    int size = sizeof(unsigned char) * img_in.w * img_in.h;

    img_out.w = img_in.w;
    img_out.h = img_in.h;
    img_out.img_y = (unsigned char *)malloc(size);
    img_out.img_u = (unsigned char *)malloc(size);
    img_out.img_v = (unsigned char *)malloc(size);

    cudaMalloc((void **)&(device_ppm.img_r), size);
    cudaMalloc((void **)&(device_ppm.img_g), size);
    cudaMalloc((void **)&(device_ppm.img_b), size);

    cudaMemcpy(device_ppm.img_r, img_in.img_r, size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_ppm.img_g, img_in.img_g, size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_ppm.img_b, img_in.img_b, size, cudaMemcpyHostToDevice);

    cudaMalloc((void **)&(device_yuv.img_y), size);
    cudaMalloc((void **)&(device_yuv.img_u), size);
    cudaMalloc((void **)&(device_yuv.img_v), size);

    int N = img_in.w * img_in.h;
    int BLOCKS_IN_GRID = N / BLOCK_SIZE / THREADS_PER_BLOCK;
    dim3 BLOCKS_PER_GRID(BLOCK_SIZE, BLOCKS_IN_GRID);
    rgb2yuvKernel<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(device_ppm, device_yuv);

    cudaMemcpy(img_out.img_y, device_yuv.img_y, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(img_out.img_u, device_yuv.img_u, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(img_out.img_v, device_yuv.img_v, size, cudaMemcpyDeviceToHost);

    return img_out;
}

PPM_IMG yuv2rgbGPU(YUV_IMG img_in)
{
    PPM_IMG img_out;

    PPM_IMG device_ppm;
    YUV_IMG device_yuv;

    int size = sizeof(unsigned char) * img_in.w * img_in.h;

    img_out.w = img_in.w;
    img_out.h = img_in.h;
    img_out.img_r = (unsigned char *)malloc(size);
    img_out.img_g = (unsigned char *)malloc(size);
    img_out.img_b = (unsigned char *)malloc(size);

    cudaMalloc((void **)&(device_yuv.img_y), size);
    cudaMalloc((void **)&(device_yuv.img_u), size);
    cudaMalloc((void **)&(device_yuv.img_v), size);

    cudaMemcpy(device_yuv.img_y, img_in.img_y, size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_yuv.img_u, img_in.img_u, size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_yuv.img_v, img_in.img_v, size, cudaMemcpyHostToDevice);

    cudaMalloc((void **)&(device_ppm.img_r), size);
    cudaMalloc((void **)&(device_ppm.img_g), size);
    cudaMalloc((void **)&(device_ppm.img_b), size);

    int N = img_in.w * img_in.h;
    int BLOCKS_IN_GRID = N / BLOCK_SIZE / THREADS_PER_BLOCK;
    dim3 BLOCKS_PER_GRID(BLOCK_SIZE, BLOCKS_IN_GRID);
    yuv2rgbKernel<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(device_yuv, device_ppm);

    cudaMemcpy(img_out.img_r, device_ppm.img_r, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(img_out.img_g, device_ppm.img_g, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(img_out.img_b, device_ppm.img_b, size, cudaMemcpyDeviceToHost);

    return img_out;
}

YUV_IMG rgb2yuv(PPM_IMG img_in)
{
    YUV_IMG img_out;
    int i;
    unsigned char r, g, b;
    unsigned char y, cb, cr;
    int size = sizeof(unsigned char) * img_in.w * img_in.h;

    img_out.w = img_in.w;
    img_out.h = img_in.h;
    img_out.img_y = (unsigned char *)malloc(size);
    img_out.img_u = (unsigned char *)malloc(size);
    img_out.img_v = (unsigned char *)malloc(size);

    for(i = 0; i < img_out.w*img_out.h; i ++){
        r = img_in.img_r[i];
        g = img_in.img_g[i];
        b = img_in.img_b[i];

        y  = (unsigned char)( 0.299*r + 0.587*g +  0.114*b);
        cb = (unsigned char)(-0.169*r - 0.331*g +  0.499*b + 128);
        cr = (unsigned char)( 0.499*r - 0.418*g - 0.0813*b + 128);

        img_out.img_y[i] = y;
        img_out.img_u[i] = cb;
        img_out.img_v[i] = cr;
    }

    return img_out;
}

PPM_IMG yuv2rgb(YUV_IMG img_in)
{
    PPM_IMG img_out;
    int i;
    int rt, gt, bt;
    int y, cb, cr;
    int size = sizeof(unsigned char) * img_in.w * img_in.h;

    img_out.w = img_in.w;
    img_out.h = img_in.h;
    img_out.img_r = (unsigned char *)malloc(size);
    img_out.img_g = (unsigned char *)malloc(size);
    img_out.img_b = (unsigned char *)malloc(size);

    for(i = 0; i < img_out.w*img_out.h; i ++){
        y  = (int)img_in.img_y[i];
        cb = (int)img_in.img_u[i] - 128;
        cr = (int)img_in.img_v[i] - 128;

        rt  = (int)( y + 1.402*cr);
        gt  = (int)( y - 0.344*cb - 0.714*cr);
        bt  = (int)( y + 1.772*cb);

        img_out.img_r[i] = clip_rgb(rt);
        img_out.img_g[i] = clip_rgb(gt);
        img_out.img_b[i] = clip_rgb(bt);
    }

    return img_out;
}

                                                                 


