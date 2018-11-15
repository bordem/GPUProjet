#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "ppm_lib.h"

//Version de base
void changeColorPPM(PPMImage *img)
{   
	int i;
	if(img){
		for(i=0;i<img->x*img->y;i++){
			img->data[i].red=RGB_COMPONENT_COLOR-img->data[i].red;
			img->data[i].green=RGB_COMPONENT_COLOR-img->data[i].green;
			img->data[i].blue=RGB_COMPONENT_COLOR-img->data[i].blue;
		}
	}
}
int grayScale(int R,int G,int B){
	int Y = 0.2126*R+0.7152*G+0.0722*B;
	return Y;
}

//Version Martin qui marche
void applyFiltre(PPMImage *img)
{
	//printf("Debut de la fonction\n");
	
	int destinationRED[500][1000]= {0};
	int destinationBLUE[500][1000]= {0};
	int destinationGREEN[500][1000]= {0};
	
	int top=0;
	int bottom=1000;
	int left = 0;
	int right=500;
	
	int filterSofter[25]={	0, 0, 0, 0, 0,
							0, 1, 3, 1, 0,
							0, 3, 5, 3, 0,
							0, 1, 3, 1, 0,
							0, 0, 0, 0, 0};
	// Facteur de division = 21
							
	int filterSoften[25]= {	1, 1, 1, 1, 1,
							1, 1, 1, 1, 1,
							1, 1, 1, 1, 1,
							1, 1, 1, 1, 1,
							1, 1, 1, 1, 1};
	// Facteur de division = 25
	
	int filterShatter[25]={	1, 0, 0, 0, 1,
							0, 0, 0, 0, 0,
							0, 0, 0, 0, 0,
							0, 0, 0, 0, 0,
							1, 0, 0, 0, 1};
	// Facteur de division = 4
						
	int filterSobel[25]= {	1, 2, 0, -2, -1,
							4, 8, 0, -8, -4,
							6, 12, 0,-12,-6,
							4, 8, 0, -8, -4,
							1, 2, 0, -2, -1};
	// Facteur de division = 1
	
	int filterHorizontalBlur[25]= {	0, 0, 0, 0, 0,
									0, 0, 0, 0, 0,
									1, 2, 3, 2, 1,
									0, 0, 0, 0, 0,
									0, 0, 0, 0, 0};
	//Divide: 9
	
	int filterVerticalSobel[25]={	-1,-4,-6,-4,-1,
									-2,-8,-12,-8,-2,
									0, 0, 0, 0, 0,
									2, 8, 12, 8, 2,
									1, 4, 6, 4, 1};
	//Divide: 1
	int filterSharpenMedium[25]={	-1, -1, -1, -1, -1,
									-1, -1, -1, -1, -1,
									-1, -1, 49, -1, -1,
									-1, -1, -1, -1, -1,
									-1, -1, -1, -1, -1};
	//Divide: 25
	
	
	int filter[25]={0};
	for(int i=0;i<25;i++){
		filter[i]=filterShatter[i];
	}
	int divisionFactor = 4;
		
	
	//printf("Fin initialisation\n");
	#pragma omp parallel
	#pragma omp for
	for(int y=top; y<bottom; y++){
	// for each pixel in the image
		for(int x=left; x<right; x++){ 
			
			int gridCounter=0;// reset some values
			
			int finalRED = 0;
			int finalBLUE = 0;
			int finalGREEN = 0;
			
			for(int y2=-2; y2<=2; y2++)// and for each pixel around our
			{	
				for(int x2=-2; x2<=2; x2++)   //  "hot pixel"...
				{ 
				// Add to our running total
					if(y>=2 && x>=2 && y<=998 && x<=498){
						finalRED 	= finalRED 	+ img->data[(x+x2)+((y+y2)*right)].red 	 * filter[gridCounter];
						finalBLUE 	= finalBLUE + img->data[(x+x2)+((y+y2)*right)].blue  * filter[gridCounter];
						finalGREEN 	= finalGREEN+ img->data[(x+x2)+((y+y2)*right)].green * filter[gridCounter];
					}
					// Go to the next value on the filter grid
					gridCounter++;
				}
				// and put it back into the right range				
			}
			//int H = grayScale(finalRED,finalGREEN,finalBLUE);
			finalRED 	= finalRED 	 / divisionFactor;
			finalBLUE 	= finalBLUE  / divisionFactor;
			finalGREEN 	= finalGREEN / divisionFactor;
			
			destinationRED[x][y] 	=finalRED;
			destinationGREEN[x][y] 	=finalGREEN;
			destinationBLUE[x][y] 	=finalBLUE;
		}
	}
	
	//printf("Debut changement image\n");
	
	if(img){
		int x=0;
		int y=0;
		
		#pragma omp parallel for
		for(int i=0;i<img->x*img->y;i++){
			if(x==500){
				x=0;
				y=y+1;
			}
			img->data[i].red	=	destinationRED	[x][y];
			img->data[i].green	=	destinationGREEN[x][y];
			img->data[i].blue	=	destinationBLUE	[x][y];
			x=x+1;
		}
	}
}

//Version Clement qui marche pas 

int main(){
	PPMImage *image;
	image = readPPM("test100.ppm");
	for(int i=0;i<100;i++){		
		//image = readPPM("test1000.ppm");
		printf("%d\n",i);
		applyFiltre(image);
		writePPM("test100.ppm",image);
	}
}