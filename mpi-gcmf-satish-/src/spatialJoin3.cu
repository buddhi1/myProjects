#include <stdio.h>
/*#include "GPU_Manage.h"
#include "Types.h"
#include "Constants.h"
#include "GPU_Test.h"
#include "GPU_MBR.h"
#include "IO.h"
#include "SEQ_Overlay.h"
#include "GPU_Utility.h"
#include "Data_Visualization.h"
#include "Join.h"*/
#include "ST_Intersect.h"

cudaEvent_t start_GPU, stop_GPU;

int main(int argc, char* argv[]){  
    float Join_Total_Time_SEQ=0, Join_Total_Time_GPU=0;
    cudaError_t cudaMemError;
//------------------------ Console Input ---------------------------------- 
/*
First user input: dimSort
	1: Sorting just based on one dimension (default is X)
	0: Sorting based on both X and Y dimensions
Second user input: dimSelect
	If dimSort=1, this argument define which dimension should be picked for sorting (Values could be 'X' or 'Y')
*/
    int dimSort=1, dimSelect=1;
    if(argc<2){
       dimSort=1;
       dimSelect=0;
    }
    else if(argc<3){
      if(argv[1][0]=='2')dimSort=2;
      else dimSort=1;
      dimSelect=0;
    }
    else if(argc<4){
      if(argv[2][0]=='y')dimSelect=1;
      else dimSelect=0;
      if(argv[1][0]=='2'){dimSort=2;dimSelect=0;}
      else dimSort=1;
    }
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
//=============================== SEQUENTIAL RUN ===============================
//------------------------------------------------------------------------------    
//------------------------------------------------------------------------------    
    coord_t* baseCoords=(coord_t*)malloc(MAX_POLYS_BASE*4*AVG_VERTEX_PER_BASE_POL*sizeof(coord_t));
    coord_t* overlayCoords=(coord_t*)malloc(MAX_POLYS_OVERLAY*4*AVG_VERTEX_PER_OVERLAY_POL*sizeof(coord_t));
    int *bVNum=(int*)malloc(sizeof(int)*MAX_POLYS_BASE);
    int *oVNum=(int*)malloc(sizeof(int)*MAX_POLYS_OVERLAY);
    long *bVPSNum=(long*)malloc(sizeof(long)*MAX_POLYS_BASE);
    long *oVPSNum=(long*)malloc(sizeof(long)*MAX_POLYS_OVERLAY);
    long bPolNum, oPolNum, bVNumSum=0, oVNumSum=0;    
    mbr_t* seqMBR=(mbr_t*)malloc(MAX_POLYS_BASE*4*sizeof(mbr_t));
    mbr_t* seqOMBR=(mbr_t*)malloc(MAX_POLYS_OVERLAY*4*sizeof(mbr_t));
    coord_t* seqMBR2=(coord_t*)malloc(MAX_POLYS_BASE*4*sizeof(coord_t));
    coord_t* seqOMBR2=(coord_t*)malloc(MAX_POLYS_OVERLAY*4*sizeof(coord_t));
    //=================== Reading First(base) Polygon ==========================

    char baseFileName[100], overlayFileName[100];
    switch(DATASET){
       case 1:
         strcpy(baseFileName, "../admin_states.txt");
	 strcpy(overlayFileName, "../urban_areas.txt");
         break;
       case 2:
         strcpy(baseFileName, "bases_242.txt");
         strcpy(overlayFileName, "overlay_300.txt");
         break;
       case 3:
         strcpy(baseFileName, "block_boundaries.txt");
         strcpy(overlayFileName, "water_bodies.txt");
         break;
    }
    bPolNum=ReadTextFormatPolygon(baseFileName,bVNum, bVPSNum, seqMBR, seqMBR2, baseCoords, &bVNumSum, 1, MAX_POLYS_BASE);    
    printf("\n%lu Polygons with %lu vertices in total.\n",bPolNum,bVNumSum);
    oPolNum=ReadTextFormatPolygon(overlayFileName, oVNum, oVPSNum, seqOMBR, seqOMBR2, overlayCoords, &oVNumSum, 1, MAX_POLYS_OVERLAY); 
    printf("\n%lu Polygons with %lu vertices in total.\n",oPolNum,oVNumSum);
    //==========================================================================

//=========================== Reseting GPU device ==============================
    cudaError_t error_reset=cudaDeviceReset();    
    if(error_reset!=cudaSuccess)
    {
       fprintf(stderr,"ERROR: %s\n", cudaGetErrorString(error_reset) );
       exit(-1);
    }
    cudaThreadExit();
    
    //==================== Running Kernel (CreateMBR) =========================


    ST_Intersect(bPolNum, oPolNum, baseCoords, overlayCoords, bVNum, oVNum, bVPSNum, oVPSNum, seqMBR, seqOMBR, seqMBR2, seqOMBR2);

   cudaThreadExit();
   //==============================================================================
return 0;
}
