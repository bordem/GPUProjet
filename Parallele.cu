#include <stdio.h>
#include <stdlib.h>
extern "C" {
#include "ppm_lib.h"
}

#define CREATOR "PARALLELISME2OPENMP"
#define DIVISIONFACTOR 4

extern "C" PPMImage *readPPM(const char *filename)
{
         char buff[16];
         PPMImage *img;
         FILE *fp;
         int c, rgb_comp_color;
         //open PPM file for reading
         fp = fopen(filename, "rb");
         if (!fp) {
              fprintf(stderr, "Unable to open file '%s'\n", filename);
              exit(1);
         }

         //read image format
         if (!fgets(buff, sizeof(buff), fp)) {
              perror(filename);
              exit(1);
         }

    //check the image format
    if (buff[0] != 'P' || buff[1] != '6') {
         fprintf(stderr, "Invalid image format (must be 'P6')\n");
         exit(1);
    }

    //alloc memory form image
    img = (PPMImage *)malloc(sizeof(PPMImage));
    if (!img) {
         fprintf(stderr, "Unable to allocate memory\n");
         exit(1);
    }

    //check for comments
    c = getc(fp);
    while (c == '#') {
    while (getc(fp) != '\n') ;
         c = getc(fp);
    }

    ungetc(c, fp);
    //read image size information
    if (fscanf(fp, "%d %d", &img->x, &img->y) != 2) {
         fprintf(stderr, "Invalid image size (error loading '%s')\n", filename);
         exit(1);
    }

    //read rgb component
    if (fscanf(fp, "%d", &rgb_comp_color) != 1) {
         fprintf(stderr, "Invalid rgb component (error loading '%s')\n", filename);
         exit(1);
    }

    //check rgb component depth
    if (rgb_comp_color!= RGB_COMPONENT_COLOR) {
         fprintf(stderr, "'%s' does not have 8-bits components\n", filename);
         exit(1);
    }

    while (fgetc(fp) != '\n') ;
    //memory allocation for pixel data
    img->data = (PPMPixel*)malloc(img->x * img->y * sizeof(PPMPixel));

    if (!img) {
         fprintf(stderr, "Unable to allocate memory\n");
         exit(1);
    }

    //read pixel data from file
    if (fread(img->data, sizeof(PPMPixel)*img->x, img->y, fp) != img->y) {
         fprintf(stderr, "Error loading image '%s'\n", filename);
         exit(1);
    }

    fclose(fp);
    return img;
}

extern "C" void writePPM(const char *filename, PPMImage *img)
{
    FILE *fp;
    //open file for output
    fp = fopen(filename, "wb");
    if (!fp) {
         fprintf(stderr, "Unable to open file '%s'\n", filename);
         exit(1);
    }

    //write the header file
    //image format
    fprintf(fp, "P6\n");

    //comments
    fprintf(fp, "# Created by %s\n",CREATOR);

    //image size
    fprintf(fp, "%d %d\n",img->x,img->y);

    // rgb component depth
    fprintf(fp, "%d\n",RGB_COMPONENT_COLOR);

    // pixel data
    fwrite(img->data, 3 * img->x, img->y, fp);
    fclose(fp);
}


//Version PARALLELE


__global__ void filtre(PPMPixel *g_idata,PPMPixel*g_odata,unsigned int width,unsigned int height){
    int filterShatter[25]={	1, 0, 0, 0, 1,
							0, 0, 0, 0, 0,
							0, 0, 0, 0, 0,
							0, 0, 0, 0, 0,
							1, 0, 0, 0, 1};
    __shared__ PPMPixel smem[20][20];

    int x = blockIdx.y * 16 + threadIdx.y;
    int y = blockIdx.x * 16 + threadIdx.x;
    unsigned int index = y*width + x;

if(x<0 || y<0 || x>=width || y>=height) {
       smem[threadIdx.x][threadIdx.y].red = 0;
       smem[threadIdx.x][threadIdx.y].blue = 0;
       smem[threadIdx.x][threadIdx.y].green = 0;
   }
   else{
       smem[threadIdx.x][threadIdx.y] = g_idata[index];
   }

__syncthreads();
int gridCounter=0;
int finalRED = 0;
int finalBLUE = 0;
int finalGREEN = 0;

if ((threadIdx.x >= 2) && (threadIdx.x < (18)) &&(threadIdx.y >= 2) && (threadIdx.y < (18)))
{
for (int dy=-2;dy<=2;dy++){
    for (int dx=-2;dx<=2;dx++){
        finalRED+=smem[threadIdx.x+dx][threadIdx.y+dy].red*filterShatter[gridCounter];
        finalBLUE+=smem[threadIdx.x+dx][threadIdx.y+dy].blue*filterShatter[gridCounter];
        finalGREEN+=smem[threadIdx.x+dx][threadIdx.y+dy].green*filterShatter[gridCounter];
        gridCounter++;
    }
}
finalRED 	= finalRED 	 / DIVISIONFACTOR;
finalBLUE 	= finalBLUE  / DIVISIONFACTOR;
finalGREEN 	= finalGREEN / DIVISIONFACTOR;
g_odata[index].red=finalRED;
g_odata[index].blue=finalBLUE;
g_odata[index].green=finalGREEN;




    }
}

int main(){
    PPMPixel  *data =NULL,*d_idata=NULL,*d_odata=NULL;
    PPMImage *image;
    image=readPPM("test2.ppm");
    size_t size=image->x*image->y*sizeof(PPMPixel);
    data=(PPMPixel *)malloc(size);
    cudaMalloc((void **)&d_idata, size);
    cudaMalloc((void **)&d_odata, size);

    PPMImage *outImage;
    outImage = (PPMImage *)malloc(sizeof(PPMImage));
    outImage->x = image->x;
    outImage->y = image->y;

    cudaMemcpy(d_idata, image->data, size, cudaMemcpyHostToDevice);
    int GRID_W = image->x/16 +1;
    int GRID_H = image->y/16 +1;
     dim3 threadsPerBlock(20,20);
     dim3 blocksPerGrid(GRID_W,GRID_H);
       filtre<<<blocksPerGrid, threadsPerBlock>>>(d_idata, d_odata, image->x,image->y);
       cudaDeviceSynchronize();
       cudaMemcpy(data, d_odata, size, cudaMemcpyDeviceToHost);
       outImage->data=data;
       //int x=0;
       //int y=0;
       /*for(int i=0;i<image->x*image->y;i++){
           if(x==500){
               x=0;
               y=y+1;
           }
           image->data[i].red	=	imageOut[i];
           image->data[i].green	=	imageOut[i];
           image->data[i].blue	=	imageOut[i];
           x=x+1;
       }*/
       writePPM("mon_image2.ppm",outImage);
       cudaFree(d_idata);
       cudaFree(d_odata);
       free(data);
       free(outImage);

       printf("TERMINE\n");

}
