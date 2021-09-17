// 210 block changed
// trying to use only 2 arrays to pass data
// use asynch multiple kernes using streams

#include <stdio.h>
#include "GPU_Manage.h"
#include "Types.h"
#include "Constants.h"
#include "GPU_Test.h"
#include "GPU_MBR.h"
#include "IO.h"
#include "SEQ_Overlay.h"
#include "GPU_Utility-streams.h"
#include "Data_Visualization.h"
#include "Join-streams.h"
#include "cmbr_pre.h"

cudaEvent_t start_GPU, stop_GPU;

inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

int spatialJoin_ST_Intersect(int featureID1, int featureID2, int argc, char* argv[], cudaStream_t stream);

int main(int argc, char* argv[]) {
   

  // int devId = 0;

  // cudaDeviceProp prop;
  // checkCuda( cudaGetDeviceProperties(&prop, devId));
  // printf("Device : %s\n", prop.name);
  // checkCuda( cudaSetDevice(devId) );

  int nStreams = 3;
  cudaStream_t stream[nStreams];
  // cudaStream_t stream1;
  for (int i = 0; i < nStreams; ++i){
    checkCuda( cudaStreamCreate(&stream[i]) );
  }
  // cudaStreamCreate(&stream1);
  // spatialJoin_ST_Intersect(0, 1, argc, argv, stream1);

  //populate data from mbr code
  //read data into a table_row structure type 1D array
  struct table_row *dat;
  dat = createArray("data/Point_Of_Interest_modified.csv");

  getMBRList(dat);
  print_message("mbr array constructed");

  // preProcessAllMBRArray();
  

  int layer2 = 1;

  for (int i = 0; i < nStreams; ++i)
  {
    if (i != layer2)
    {
      spatialJoin_ST_Intersect(i, layer2, argc, argv, stream[i]);
    }    
  }
	

	return 0;
}

// int main(int argc, char* argv[]){ 
int spatialJoin_ST_Intersect(int featureID1, int featureID2, int argc, char* argv[], cudaStream_t stream){ 
// int spatialJoin_ST_Intersect(int argc, char* argv[]){  
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
    // if(argc<2){
    //    dimSort=1;
    //    dimSelect=0;
    // }
    // else if(argc<3){
    //   if(argv[1][0]=='2')dimSort=2;
    //   else dimSort=1;
    //   dimSelect=0;
    // }
    // else if(argc<4){
    //   if(argv[2][0]=='y')dimSelect=1;
    //   else dimSelect=0;
    //   if(argv[1][0]=='2'){dimSort=2;dimSelect=0;}
    //   else dimSort=1;
    // }
//------------------------------------------------------------------------------

//-----------------------------------------------------------------------------------------------------    

// ****************************************Print the xMBR1 and xMBR2 and yMBR1 and yMBR2 and verify the output.
	// printArray(x_MBR_all, featureID1, 10);
	// printArray(y_MBR_all, featureID1, 10);
	// printArray(x_MBR_all, featureID2, 10);
	// printArray(y_MBR_all, featureID2, 10);

	// printArray_coord_t(seq_bMBR2, 10);
	// printArray_coord_t(seq_oMBR2, 10);

  long bPolNum = MAX_COUNT;
  long oPolNum = MAX_COUNT;
  print_message("b size " + to_string(bPolNum));
  print_message("o size " + to_string(oPolNum));


//-----------------------------------------------------------------------------------------------------


//-------------------------------------------- Reading Input -----------------------------------------------    
  //   char baseFileName[100], overlayFileName[100];
  //   long bPolNum, oPolNum;
  //   switch(DATASET){
  //      case 1:
  //        bPolNum = 4646;
	 // oPolNum = 11878;
  //        //strcpy(baseFileName, "/pylon5/cc560kp/danialll/Text_Datasets/admin_states.txt");
  //        strcpy(baseFileName, "../admin_states.txt");
	 // //strcpy(overlayFileName, "/pylon5/cc560kp/danialll/Text_Datasets/urban_areas.txt");
	 // strcpy(overlayFileName, "../urban_areas.txt");
  //        printf("\nDataset: admin - urban\n");
  //        break;
  //      case 2:
  //        bPolNum = 15000;
	 // oPolNum = 15000;
  //        strcpy(baseFileName, "/pylon5/cc560kp/danialll/Text_Datasets/bases_242.txt");
  //        strcpy(overlayFileName, "/pylon5/cc560kp/danialll/Text_Datasets/overlay_300.txt");
  //        printf("\nDataset: bases - overlay\n");
  //        break;
  //      case 3:
  //        bPolNum = 15000;
	 // oPolNum = 15000;
  //        strcpy(baseFileName, "/pylon5/cc560kp/danialll/Text_Datasets/block_boundaries.txt");
  //        strcpy(overlayFileName, "/pylon5/cc560kp/danialll/Text_Datasets/water_bodies.txt");
  //        printf("\nDataset: boundaries - water\n");
  //        break;
  //      case 4:
  //        bPolNum = 15000;
	 // oPolNum = 15000;
  //        strcpy(baseFileName, "/pylon5/cc560kp/danialll/Text_Datasets/postal.txt");
  //        strcpy(overlayFileName, "/pylon5/cc560kp/danialll/Text_Datasets/sports.txt");
  //        printf("\nDataset: postal - sports\n");
  //        break;
  //   }
    //----------------------------------------- Memory Allocation -------------------------------------    
    long bVNumSum = 0, oVNumSum = 0;    
    int *bVNum=(int*)malloc(sizeof(int) * bPolNum);
    int *oVNum=(int*)malloc(sizeof(int) * oPolNum);
    long *bVPSNum=(long*)malloc(sizeof(long) * bPolNum);
    long *oVPSNum=(long*)malloc(sizeof(long) * oPolNum);

    coord_t* baseXCoords = (coord_t*) malloc( 2 * sizeof(coord_t) * (VERTEX_PER_BPOL * bPolNum + VERTEX_PER_OPOL * oPolNum) );
    coord_t* overlayXCoords = baseXCoords + VERTEX_PER_BPOL * bPolNum;
    coord_t* baseYCoords = overlayXCoords + VERTEX_PER_OPOL * oPolNum;
    coord_t* overlayYCoords = baseYCoords + VERTEX_PER_BPOL * bPolNum;

    mbr_t* seqXMBR, *seqOXMBR, *seqYMBR, *seqOYMBR;
    seqXMBR = (mbr_t*)malloc(4 * sizeof(mbr_t) * (bPolNum + oPolNum));
    seqOXMBR = seqXMBR + 2 * bPolNum;
    seqYMBR = seqOXMBR + 2 * oPolNum;
    seqOYMBR = seqYMBR + 2 * bPolNum;

    coord_t* seqXMBR2, *seqYMBR2, *seqOXMBR2, *seqOYMBR2;
    seqXMBR2 = (coord_t*)malloc(4 * sizeof(coord_t) * (bPolNum + oPolNum));
    seqOXMBR2= seqXMBR2 + 2 * bPolNum;
    seqYMBR2 = seqOXMBR2 + 2 * oPolNum;
    seqOYMBR2 = seqYMBR2 + 2 * bPolNum;

    //-------------------------------------------------------------------------------------------------    
    // ********************** cmbr edited *****************************

    // bPolNum=ReadTextFormatPolygon2(baseFileName,bVNum, bVPSNum, seqXMBR, seqYMBR, seqXMBR2, seqYMBR2, baseXCoords, baseYCoords, &bVNumSum, 1, bPolNum);    
    // printf("\n%lu Polygons with %lu vertices in total.\n",bPolNum,bVNumSum);
    // oPolNum=ReadTextFormatPolygon2(overlayFileName,oVNum, oVPSNum, seqOXMBR, seqOYMBR, seqOXMBR2, seqOYMBR2, overlayXCoords, overlayYCoords, &oVNumSum, 1, oPolNum);    
    // printf("\n%lu Polygons with %lu vertices in total.\n",oPolNum,oVNumSum);
//-----------------------------------------------------------------------------------------------------    

// //----------------------------------- Reseting GPU Device --------------------------------------------- 
//     cudaError_t error_reset=cudaDeviceReset();    
//     if(error_reset!=cudaSuccess)
//     {
//        fprintf(stderr,"ERROR: %s\n", cudaGetErrorString(error_reset) );
//        exit(-1);
//     }
//     cudaThreadExit();
//     size_t mem_free_0,mem_total_0;
//     cudaMemGetInfo  (&mem_free_0, &mem_total_0);
//     printf("\nFree: %lu  , Total: %lu\n",mem_free_0,mem_total_0);
// //-----------------------------------------------------------------------------------------------------    
    
//----------------------------------- Transfering data to GPU -----------------------------------------
    StartTimer(&start_GPU, &stop_GPU);
   
    int *dbVNum, *doVNum;
    coord_t *oXCoords, *oYCoords, *bXCoords, *bYCoords;
    mbr_t *doXMBR, *doYMBR, *dbXMBR, *dbYMBR;
    long *dbVPSNum, *doVPSNum;

    //----------- Transfering polygon number variables to GPU ---------------
    CopyToGPU_streams((void**)&dbVNum, bVNum, sizeof(int)*bPolNum, "dbVNum", 1, stream);
    CopyToGPU_streams((void**)&doVNum, oVNum, sizeof(int)*oPolNum, "doVNum", 1, stream);
    CopyToGPU_streams((void**)&dbVPSNum, bVPSNum, sizeof(long)*bPolNum, "dbVPSNum", 1, stream);
    CopyToGPU_streams((void**)&doVPSNum, oVPSNum, sizeof(long)*oPolNum, "doVPSNum", 1, stream);
    //-----------------------------------------------------------------------

    // //----------------------- Transfering MBRs to GPU -----------------------
    // cudaError_t memAlloc = cudaMalloc( (void**)&dbXMBR, 4 * sizeof(mbr_t) * (bPolNum + oPolNum) ); 
    // if(memAlloc != cudaSuccess){printf("\nError in device memory allocation!\n");return(0);}

    // CopyToGPU((void**)&dbXMBR, seqXMBR, 2 * sizeof(mbr_t) * bPolNum, "dbXMBR", 0);
    // doXMBR = dbXMBR + 2 * bPolNum;
    // CopyToGPU((void**)&doXMBR, seqOXMBR, 2 * sizeof(mbr_t) * oPolNum, "doXMBR", 0);
    // dbYMBR = doXMBR + 2 * oPolNum;
    // CopyToGPU((void**)&dbYMBR, seqYMBR, 2 * sizeof(mbr_t) * bPolNum, "dbYMBR", 0);
    // doYMBR = dbYMBR + 2 * bPolNum;
    // CopyToGPU((void**)&doYMBR, seqOYMBR, 2 * sizeof(mbr_t) * oPolNum, "doYMBR", 0);
    // //-----------------------------------------------------------------------

    //-----------------------**** Transfering MBRs to GPU - CMBR edited1 ****-----------------------
    preProcessMBRArray(featureID1, featureID2);

    print_message("Pre-process done");

    cudaError_t memAlloc = cudaMalloc( (void**)&dbXMBR, 4 * sizeof(mbr_t) * (bPolNum + oPolNum) ); 
    if(memAlloc != cudaSuccess){printf("\nError in device memory allocation!\n");return(0);}

    CopyToGPU_streams((void**)&dbXMBR, x_MBR1, 2 * sizeof(mbr_t) * bPolNum, "dbXMBR", 0, stream);
    doXMBR = dbXMBR + 2 * bPolNum;
    CopyToGPU_streams((void**)&doXMBR, x_MBR2, 2 * sizeof(mbr_t) * oPolNum, "doXMBR", 0, stream);
    dbYMBR = doXMBR + 2 * oPolNum;
    CopyToGPU_streams((void**)&dbYMBR, y_MBR1, 2 * sizeof(mbr_t) * bPolNum, "dbYMBR", 0, stream);
    doYMBR = dbYMBR + 2 * bPolNum;
    CopyToGPU_streams((void**)&doYMBR, y_MBR2, 2 * sizeof(mbr_t) * oPolNum, "doYMBR", 0, stream);
    //-----------------------------------------------------------------------

    //-----------------------**** Transfering MBRs to GPU - CMBR edited2 ****-----------------------
    // cudaError_t memAlloc = cudaMalloc( (void**)&dbXMBR, 4 * sizeof(mbr_t) * (bPolNum + oPolNum) ); 
    // if(memAlloc != cudaSuccess){printf("\nError in device memory allocation!\n");return(0);}

    // CopyToGPU((void**)&dbXMBR, x_MBR_all + featureID1*MAX_COUNT, 2 * sizeof(mbr_t) * bPolNum, "dbXMBR", 0);
    // doXMBR = dbXMBR + 2 * bPolNum;
    // CopyToGPU((void**)&doXMBR, x_MBR_all + featureID2*MAX_COUNT, 2 * sizeof(mbr_t) * oPolNum, "doXMBR", 0);
    // dbYMBR = doXMBR + 2 * oPolNum;
    // CopyToGPU((void**)&dbYMBR, y_MBR_all + featureID1*MAX_COUNT, 2 * sizeof(mbr_t) * bPolNum, "dbYMBR", 0);
    // doYMBR = dbYMBR + 2 * bPolNum;
    // CopyToGPU((void**)&doYMBR, y_MBR_all + featureID2*MAX_COUNT, 2 * sizeof(mbr_t) * oPolNum, "doYMBR", 0);
    //-----------------------------------------------------------------------



    //------------- Transfering polygon coordinates to GPU ------------------
    memAlloc = cudaMalloc( (void**)&bXCoords, 2 * sizeof(coord_t) * (bVNumSum + oVNumSum)); 
    if(memAlloc != cudaSuccess){printf("\nError in device memory allocation!\n");return(0);}

    CopyToGPU_streams((void**)&bXCoords, baseXCoords, sizeof(coord_t) * bVNumSum , "bXCoords", 0, stream);
    oXCoords = bXCoords + bVNumSum;
    CopyToGPU_streams((void**)&oXCoords, overlayXCoords, sizeof(coord_t) * oVNumSum, "oXCoords", 0, stream);
    bYCoords = oXCoords + oVNumSum;
    CopyToGPU_streams((void**)&bYCoords, baseYCoords, sizeof(coord_t) * bVNumSum , "bYCoords", 0, stream);
    oYCoords = bYCoords + bVNumSum;
    CopyToGPU_streams((void**)&oYCoords, overlayYCoords, sizeof(coord_t) * oVNumSum , "oYCoords", 0, stream);
    //-----------------------------------------------------------------------

    float runningTime_GPU_TransferData;
    Join_Total_Time_GPU+=StopTimer(&start_GPU, &stop_GPU, &runningTime_GPU_TransferData);
    printf("\n\nGPU running time for transfering data to GPU: %f (%f)\n",runningTime_GPU_TransferData, Join_Total_Time_GPU);
//-----------------------------------------------------------------------------------------------------    


//--------------------------- Find Overlaping MBRs (novel approach) -----------------------------------
    StartTimer(&start_GPU, &stop_GPU);

    int *djxyCounter, *djxyVector, polNum=bPolNum+oPolNum; 
    cudaMemError=cudaMalloc((void**)&djxyCounter,sizeof(int)*(polNum));

    // long pairNum=SortBaseMBROverlap(bPolNum, oPolNum, dbXMBR, dbYMBR, doXMBR, doYMBR, &djxyCounter, &djxyVector, dimSort, dimSelect);
    // long pairNum=SortBaseMBROverlap(bPolNum, oPolNum, dbXMBR, dbYMBR, doXMBR, doYMBR, &djxyCounter, &djxyVector, dimSort, dimSelect);
    long pairNum=SortBaseMBROverlap(bPolNum, oPolNum, dbXMBR, dbYMBR, doXMBR, doYMBR, &djxyCounter, &djxyVector, dimSort, dimSelect, stream);
    // long pairNum=SortBaseMBROverlap(bPolNum, dbXMBR, dbYMBR, &djxyCounter, &djxyVector, dimSort, dimSelect);
   
    printf("\n\n\tPolygon pairs candidate: %ld\n", pairNum);
    float runningTime_GPU_overlap2;
    Join_Total_Time_GPU+=StopTimer(&start_GPU, &stop_GPU, &runningTime_GPU_overlap2);
    printf("\nGPU Running Time For Computing MBR intersection (new approach %dD [dim:%c] ): %f (%f)\n",dimSort, 'X', runningTime_GPU_overlap2, Join_Total_Time_GPU);
    cudaFree(doXMBR);
    cudaFree(dbXMBR);
    cudaFree(djxyCounter);
//------------------------------------------------------------------------------------------------------

//---------------------------------- CMF filter for Polygon Test operation -----------------------------
    // StartTimer(&start_GPU, &stop_GPU);
    // int *djxy2IndexList, *djPiPIndexList, *dPiPFlag, *djoinFlag;
    // char* dPiPType;
    // long eiNum, pairNum3, pipNum, workLoadNum;
    // coord_t *dcMBR, *dbMBR2, *doMBR2;
    // // CopyToGPU((void**)&doMBR2, seqOMBR2, sizeof(coord_t)*oPolNum*4, "doMBR2", 1);
    // // CopyToGPU((void**)&dbMBR2, seqMBR2, sizeof(coord_t)*bPolNum*4, "dbMBR2", 1);
    // CopyToGPU((void**)&doMBR2, seq_oMBR2, sizeof(coord_t)*oPolNum*4, "doMBR2", 1);
    // CopyToGPU((void**)&dbMBR2, seq_bMBR2, sizeof(coord_t)*bPolNum*4, "dbMBR2", 1);

    // GetCMBR(pairNum, djxyVector, dbMBR2, doMBR2, &dcMBR, &djPiPIndexList, &dPiPFlag, &dPiPType, &djoinFlag, &pipNum);

    // float runningTime_GPU_PiPCMF;
    // Join_Total_Time_GPU+=StopTimer(&start_GPU, &stop_GPU, &runningTime_GPU_PiPCMF);
    // printf("\nGPU Running Time for CMF Filter for Point in Polygon Test: %f (%f)\n", runningTime_GPU_PiPCMF, Join_Total_Time_GPU);
//------------------------------------------------------------------------------------------------------
return(0);

/*
//---------------------------------- Point in Polygon Test operation -----------------------------------
    StartTimer(&start_GPU, &stop_GPU);
    long wNum;

    wNum=PointInPolygonTest(bCoords, oCoords, pairNum, pipNum, djxyVector, djPiPIndexList, dPiPType, dbVPSNum, doVPSNum, dPiPFlag, djoinFlag);
    
    printf("\n\tNumber of within pairs: %ld\n", wNum);

    float runningTime_GPU_PiP;
    Join_Total_Time_GPU+=StopTimer(&start_GPU, &stop_GPU, &runningTime_GPU_PiP);
    printf("\nGPU Running Time for Point in Polygon Test: %f (%f)\n", runningTime_GPU_PiP, Join_Total_Time_GPU);
 
//------------------------------------------------------------------------------------------------------

//--------------------------- Applying Common MBR Filtering (novel approach) ---------------------------
    StartTimer(&start_GPU, &stop_GPU);
    poly_size_t *dbEdgeList, *doEdgeList;
    long *dbEdgePSCounter, *doEdgePSCounter, *dWorkLoadPSCounter;


    CountCMF(bCoords, oCoords, pairNum, djxyVector, djoinFlag, dbVNum, doVNum, dbVPSNum, doVPSNum, dcMBR, &dbEdgePSCounter, &doEdgePSCounter, &dWorkLoadPSCounter, &djxy2IndexList, &dbEdgeList, &doEdgeList, &eiNum, &workLoadNum);

    float runningTime_GPU_CCMF;
    Join_Total_Time_GPU+=StopTimer(&start_GPU, &stop_GPU, &runningTime_GPU_CCMF);
    printf("\nGPU Running Time for Counting Common MBR Filter: %f (%f)\n", runningTime_GPU_CCMF, Join_Total_Time_GPU);

    StartTimer(&start_GPU, &stop_GPU);

    ApplyCMF(bCoords, oCoords, pairNum, djxyVector, eiNum, djxy2IndexList, dbVNum, doVNum, dbVPSNum, doVPSNum, dcMBR, dbEdgePSCounter, doEdgePSCounter, dbEdgeList, doEdgeList);

    float runningTime_GPU_ACMF;
    Join_Total_Time_GPU+=StopTimer(&start_GPU, &stop_GPU, &runningTime_GPU_ACMF);
    printf("\nGPU Running Time for Applying Common MBR Filter: %f (%f)\n", runningTime_GPU_ACMF, Join_Total_Time_GPU);
    cudaFree(dcMBR);
//------------------------------------------------------------------------------

//--------------------------- Join/Overlay operations --------------------------
    StartTimer(&start_GPU, &stop_GPU);
    int* dSegmentIntersectJoinFlag;
    pairNum3=SegmentIntersectJoin(bCoords, oCoords, eiNum, djxyVector, djxy2IndexList, dbVPSNum, doVPSNum, dbEdgePSCounter, doEdgePSCounter, dbEdgeList, doEdgeList, &dSegmentIntersectJoinFlag);

    printf("\n\tActual number of intersected polygon pairs: %ld\n", pairNum3);
    float runningTime_GPU_CEI;
    Join_Total_Time_GPU+=StopTimer(&start_GPU, &stop_GPU, &runningTime_GPU_CEI);
    printf("\nGPU Running Time for Counting Edge Intersecions: %f (%f)\n", runningTime_GPU_CEI, Join_Total_Time_GPU);
//------------------------------------------------------------------------------
*/


    cudaThreadExit();
    return 0;
}
