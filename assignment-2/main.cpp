#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <helper_timer.h>
#include "colour-convert.h"

void copy_image(PPM_IMG img_in);
void run_cpu_color_test(PPM_IMG img_in);
void run_gpu_color_test(PPM_IMG img_in);
bool confirm_gpu_rgb2yuv();
bool confirm_gpu_yuv2rgb();

PPM_IMG img_obuf_rgb_cpu, img_obuf_rgb_gpu;
YUV_IMG img_obuf_yuv_cpu, img_obuf_yuv_gpu;

int main()
{
    PPM_IMG img_ibuf_c;

    printf("Running colour space converter .\n \n");
    img_ibuf_c = read_ppm("in.ppm");
    copy_image(img_ibuf_c);
    run_cpu_color_test(img_ibuf_c);
    run_gpu_color_test(img_ibuf_c);
    printf("\n ** CPU  and GPU conversions ** \n");
    printf("RGB to YUV: %s\n", confirm_gpu_rgb2yuv() ? "CONVERTED SUCCESSFULLY :-)" : "FAILED :-(");
    printf("YUV to RGB: %s\n", confirm_gpu_yuv2rgb() ? "CONVERTED SUCCESSFULLY :-)" : "FAILED :-(");

    free_ppm(img_ibuf_c);

    free_ppm(img_obuf_rgb_cpu);
    free_yuv(img_obuf_yuv_cpu);
    free_ppm(img_obuf_rgb_gpu);
    free_yuv(img_obuf_yuv_gpu);

    return 0;
}

void copy_image(PPM_IMG img_in)
{
    StopWatchInterface *timer=NULL;

    PPM_IMG host_img;
    PPM_IMG device_img;

    int size = img_in.w * img_in.h * sizeof(unsigned char);

    host_img.w = img_in.w;
    host_img.h = img_in.h;
    host_img.img_r = (unsigned char *)malloc(size);
    host_img.img_g = (unsigned char *)malloc(size);
    host_img.img_b = (unsigned char *)malloc(size);

    device_img.w = img_in.w;
    device_img.h = img_in.h;
    cudaMalloc((void **)&(device_img.img_r), size);
    cudaMalloc((void **)&(device_img.img_g), size);
    cudaMalloc((void **)&(device_img.img_b), size);

    launchEmptyKernel();    // lauch an empty kernel


    // CPU to GPU
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);
    cudaMemcpy(device_img.img_r, img_in.img_r, size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_img.img_g, img_in.img_g, size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_img.img_b, img_in.img_b, size, cudaMemcpyHostToDevice);
    sdkStopTimer(&timer);
    printf("\n ** COPYING TIME(ms) ** \n");
    printf("CPU to GPU: %f (ms)\n", sdkGetTimerValue(&timer));
    sdkDeleteTimer(&timer);

    // GPU to CPU
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);
    cudaMemcpy(host_img.img_r, device_img.img_r, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_img.img_g, device_img.img_g, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_img.img_b, device_img.img_b, size, cudaMemcpyDeviceToHost);
    sdkStopTimer(&timer);
    printf("GPU to CPU: %f (ms)\n", sdkGetTimerValue(&timer));
    sdkDeleteTimer(&timer);

    cudaFree(device_img.img_r);
    cudaFree(device_img.img_g);
    cudaFree(device_img.img_b);

    free(host_img.img_r);
    free(host_img.img_g);
    free(host_img.img_b);
}

void run_gpu_color_test(PPM_IMG img_in)
{
    StopWatchInterface *timer=NULL;
    launchEmptyKernel();    // lauch an empty kernel
    printf("\nStarting GPU processing...\n");

    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);
    img_obuf_yuv_gpu = rgb2yuvGPU(img_in); //Start RGB 2 YUV
    sdkStopTimer(&timer);
    printf("RGB to YUV conversion time(GPU): %f (ms)\n", sdkGetTimerValue(&timer));
    sdkDeleteTimer(&timer);

    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);
    img_obuf_rgb_gpu = yuv2rgbGPU(img_obuf_yuv_gpu); //Start YUV 2 RGB
    sdkStopTimer(&timer);
    printf("YUV to RGB conversion time(GPU): %f (ms)\n", sdkGetTimerValue(&timer));
    sdkDeleteTimer(&timer);

    write_ppm(img_obuf_rgb_gpu, "out_rgb.ppm");
    write_yuv(img_obuf_yuv_gpu, "out_yuv.yuv");
}

void run_cpu_color_test(PPM_IMG img_in)
{
    StopWatchInterface *timer=NULL;
    printf("\nStarting CPU processing...\n");

    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);
    img_obuf_yuv_cpu = rgb2yuv(img_in); //Start RGB 2 YUV
    sdkStopTimer(&timer);
    printf("RGB to YUV conversion time: %f (ms)\n", sdkGetTimerValue(&timer));
    sdkDeleteTimer(&timer);

    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);
    img_obuf_rgb_cpu = yuv2rgb(img_obuf_yuv_cpu); //Start YUV 2 RGB
    sdkStopTimer(&timer);
    printf("YUV to RGB conversion time: %f (ms)\n", sdkGetTimerValue(&timer));
    sdkDeleteTimer(&timer);

    write_yuv(img_obuf_yuv_cpu, "out_yuv.yuv");
    write_ppm(img_obuf_rgb_cpu, "out_rgb.ppm");
}

bool confirm_gpu_rgb2yuv()
{
    int size = img_obuf_yuv_cpu.w * img_obuf_yuv_cpu.h;
    int i, dev_y, dev_u, dev_v;

    for(i = 0; i < size; i ++) {
        dev_y = (int)abs(img_obuf_yuv_cpu.img_y[i] - img_obuf_yuv_gpu.img_y[i]);
        dev_u = (int)abs(img_obuf_yuv_cpu.img_u[i] - img_obuf_yuv_gpu.img_u[i]);
        dev_v = (int)abs(img_obuf_yuv_cpu.img_v[i] - img_obuf_yuv_gpu.img_v[i]);
        if (dev_y > 2 || dev_u > 2 || dev_v > 2) {
            return false;
        }
    }
    return true;
}
bool confirm_gpu_yuv2rgb()
{
    int size = img_obuf_rgb_cpu.w * img_obuf_rgb_cpu.h;
    int i, dr, dg, db;

    for(i = 0; i < size; i ++) {
        dr = (int)abs(img_obuf_rgb_cpu.img_r[i] - img_obuf_rgb_gpu.img_r[i]);
        dg = (int)abs(img_obuf_rgb_cpu.img_g[i] - img_obuf_rgb_gpu.img_g[i]);
        db = (int)abs(img_obuf_rgb_cpu.img_b[i] - img_obuf_rgb_gpu.img_b[i]);
        if (dr > 2 || dg > 2 || db > 2) {
            return false;
        }
    }
    return true;
}

PPM_IMG read_ppm(const char * path){
    FILE * in_file;
    char sbuf[256];

    char *ibuf;
    PPM_IMG result;
    int v_max, i;
    in_file = fopen(path, "r");
    if (in_file == NULL){
        printf("Input file not found!\n");
        exit(1);
    }
    /*Skip the magic number*/
    fscanf(in_file, "%s", sbuf);
    fscanf(in_file, "%d", &result.w);
    fscanf(in_file, "%d", &result.h);
    fscanf(in_file, "%d\n", &v_max);
    printf("Image size: %d x %d\n", result.w, result.h);

    result.img_r = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    result.img_g = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    result.img_b = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    ibuf         = (char *)malloc(3 * result.w * result.h * sizeof(char));

    fread(ibuf,sizeof(unsigned char), 3 * result.w * result.h, in_file);

    for(i = 0; i < result.w * result.h; i ++) {
        result.img_r[i] = ibuf[3*i + 0];
        result.img_g[i] = ibuf[3*i + 1];
        result.img_b[i] = ibuf[3*i + 2];
    }

    fclose(in_file);
    free(ibuf);

    return result;
}

void write_yuv(YUV_IMG img, const char * path)
{
    FILE * out_file;
    int i;

    out_file = fopen(path, "wb");
    fwrite(img.img_y, sizeof(unsigned char), img.w*img.h, out_file);
    fwrite(img.img_u, sizeof(unsigned char), img.w*img.h, out_file);
    fwrite(img.img_v, sizeof(unsigned char), img.w*img.h, out_file);
    fclose(out_file);
}

void write_ppm(PPM_IMG img, const char * path)
{
    FILE * out_file;
    int i;
out_file = fopen(path, "wb");
    fwrite(img.img_y, sizeof(unsigned char), img.w*img.h, out_file);
    fwrite(img.img_u, sizeof(unsigned char), img.w*img.h, out_file);
    fwrite(img.img_v, sizeof(unsigned char), img.w*img.h, out_file);
    fclose(out_file);
}

void write_ppm(PPM_IMG img, const char * path)
{
    FILE * out_file;
    int i;

    char * obuf = (char *)malloc(3 * img.w * img.h * sizeof(char));
    for(i = 0; i < img.w*img.h; i ++){
        obuf[3*i + 0] = img.img_r[i];
        obuf[3*i + 1] = img.img_g[i];
        obuf[3*i + 2] = img.img_b[i];
    }
    out_file = fopen(path, "wb");
    fprintf(out_file, "P6\n");
    fprintf(out_file, "%d %d\n255\n",img.w, img.h);
    fwrite(obuf,sizeof(unsigned char), 3*img.w*img.h, out_file);
    fclose(out_file);
    free(obuf);
}

void free_yuv(YUV_IMG img)
{
    free(img.img_y);
    free(img.img_u);
    free(img.img_v);
}

void free_ppm(PPM_IMG img)
{
    free(img.img_r);
    free(img.img_g);
    free(img.img_b);
}

                                                                                                                    256,0-1       Bot

