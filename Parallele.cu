#include <stdio.h>
#include <stdlib.h>
extern "C" {
#include "ppm_lib.h"
}

#define CREATOR "PARALLELISME2OPENMP"


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


__global__ void appliquerfiltre(PPMPixel *input,PPMPixel*output,unsigned int width,unsigned int height){
    int filterShatter[25]={	1, 0, 0, 0, 1,
							0, 0, 0, 0, 0,
							0, 0, 0, 0, 0,
							0, 0, 0, 0, 0,
							1, 0, 0, 0, 1};
    int *filtre=filterShatter;
    int div=4;
    __shared__ PPMPixel destination[20][20];
    int y = blockIdx.y * 16 + threadIdx.y;
    int x = blockIdx.x * 16 + threadIdx.x;
    unsigned int index = y*width + x;

if(x<0 || y<0 || x>=width || y>=height) {
       destination[threadIdx.x][threadIdx.y].red = 0;
       destination[threadIdx.x][threadIdx.y].blue = 0;
       destination[threadIdx.x][threadIdx.y].green = 0;
   }
   else{
       destination[threadIdx.x][threadIdx.y] = input[index];
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
        finalRED+=destination[threadIdx.x+dx][threadIdx.y+dy].red*filtre[gridCounter];
        finalBLUE+=destination[threadIdx.x+dx][threadIdx.y+dy].blue*filtre[gridCounter];
        finalGREEN+=destination[threadIdx.x+dx][threadIdx.y+dy].green*filtre[gridCounter];
        gridCounter++;
    }
}
finalRED 	= finalRED 	 / div;
finalBLUE 	= finalBLUE  / div;
finalGREEN 	= finalGREEN / div;

output[index].red=finalRED;
output[index].blue=finalBLUE;
output[index].green=finalGREEN;




    }
}


void Filtre(PPMImage *image){

    PPMPixel  *data =NULL,*input_data=NULL,*output_data=NULL;
    size_t size=image->x*image->y*sizeof(PPMPixel);
    data=(PPMPixel *)malloc(size);
    cudaMalloc((void **)&input_data, size);
    cudaMalloc((void **)&output_data, size);

    PPMImage *outImage;
    outImage = (PPMImage *)malloc(sizeof(PPMImage));
    outImage->x = image->x;
    outImage->y = image->y;

    cudaMemcpy(input_data, image->data, size, cudaMemcpyHostToDevice);

    int GRID_W = (image->x-1)/16 +1;
    int GRID_H = (image->y-1)/16 +1;
     dim3 threadsPerBlock(20,20);
     dim3 blocksPerGrid(GRID_W,GRID_H);
       appliquerfiltre<<<blocksPerGrid, threadsPerBlock>>>(input_data, output_data, image->x,image->y);
       cudaDeviceSynchronize();
       cudaMemcpy(data, output_data, size, cudaMemcpyDeviceToHost);
       outImage->data=data;
	writePPM("mon_image2.ppm",outImage);
       cudaFree(input_data);
       cudaFree(output_data);
       free(data);
       free(outImage);


}

int main(){
PPMImage *image;
image=readPPM("test.ppm");
    for(int i=0;i<100;i++)
        Filtre(image);
    printf("TERMINE\n");

return 0;
}
