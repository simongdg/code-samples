
#include <cstdio>
#include "device_reduce_atomic.h"
#include "device_reduce_block_atomic.h"
#include "device_reduce_warp_atomic.h"
#include "device_reduce_stable.h"
#include "vector_functions.h"
#include "cub/cub/cub.cuh"

//Tangram generated code
//#include "Tangram_DistTile_DistTile_S_S_S.h"
//#include "Tangram_DistTile_DistTile_S_S_S.h"

#include "Tangram_DistTileAtomic_DistTileAtomic_S.h"
#include "Tangram_DistTileAtomic_Vs.h"
#include "Tangram_DistTileAtomic_Va1s.h"
#include "Tangram_DistTileAtomic_Va2.h"
#include "Tangram_DistTileAtomic_Va1.h"
#include "Tangram_DistTileAtomic_V.h"
#include "Tangram_DistTile_Va1s_Va1s.h"
#include "Tangram_DistTile_Va1s_Va1.h"
#include "Tangram_DistTile_Va1s_Va2.h"
#include "Tangram_DistTile_Va1s_Vs.h"
#include "Tangram_DistTile_Va1s_V.h"
#include "Tangram_DistTile_Va2_Va1s.h"
#include "Tangram_DistTile_Va2_Va1.h"
#include "Tangram_DistTile_Va2_Va2.h"
#include "Tangram_DistTile_Va2_Vs.h"
#include "Tangram_DistTile_Va2_V.h"
#include "Tangram_DistTileAtomic_DistStride_S_Va1s.h"
#include "Tangram_DistTileAtomic_DistStride_S_Va2.h"
#include "Tangram_DistTileAtomic_DistStride_S_Va1.h"
#include "Tangram_DistTileAtomic_DistStride_S_Vs.h"
#include "Tangram_DistTileAtomic_DistStride_S_V.h"

#define cudaCheckError() {                                          \
  cudaError_t e=cudaGetLastError();                                  \
  if(e!=cudaSuccess) {                                               \
  printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
  exit(0); \
  }                                                                  \
}

void RunTest(char* label, void (*fptr)(int* in, int* out, int N), int N, int REPEAT, int* src, int checksum) {
  int *in, *out;
  
  //allocate a buffer that is at least large enough that we can ensure it doesn't just sit in l2.
  int MIN_SIZE=4*1024*1024;
  int size=max(int(sizeof(int)*N),MIN_SIZE);
  
  //compute mod base for picking the correct buffer
  int mod=size/(N*sizeof(int));
  cudaEvent_t start,stop;
  cudaMalloc(&in,size);
  cudaMalloc(&out,sizeof(int)*1024);  //only stable version needs multiple elements, all others only need 1
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaCheckError();

  cudaMemcpy(in,src,N*sizeof(int),cudaMemcpyHostToDevice);
  
  //warm up
  fptr(in,out,N);

  cudaDeviceSynchronize();
  cudaCheckError();
  cudaEventRecord(start);

  for(int i=0;i<REPEAT;i++) {
    //iterate through different buffers
    int o=i%mod;
    fptr(in+o*N,out,N);
  }
  cudaEventRecord(stop);
  cudaDeviceSynchronize();
  cudaCheckError();

  float time_ms;
  cudaEventElapsedTime(&time_ms,start,stop);
  float time_s=time_ms/(float)1e3;

  float GB=(float)N*sizeof(int)*REPEAT;
  float GBs=GB/time_s/(float)1e9;

  int sum;
  cudaMemcpy(&sum,out,sizeof(int),cudaMemcpyDeviceToHost);
  cudaCheckError();

  char *valid;
  if(sum==checksum) 
    valid="CORRECT";
  else
    valid="INCORRECT";

  printf("%s: %s, Time: %f s, GB/s: %f\n", label, valid, time_s, GBs); 
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(in);
  cudaFree(out);
  cudaCheckError();
}

void RunTestCub(char* label, int N, int REPEAT, int* src, int checksum) {
  int *in, *out;
  cudaEvent_t start,stop;
  
  cudaMalloc(&in,sizeof(int)*N);
  cudaMalloc(&out,sizeof(int)*1024);  //only stable version needs multiple elements, all others only need 1
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaCheckError();

  cudaMemcpy(in,src,N*sizeof(int),cudaMemcpyHostToDevice);

  size_t temp_storage_bytes;
  int* temp_storage=NULL;
  cub::DeviceReduce::Reduce(temp_storage, temp_storage_bytes, in, out, N, cub::Sum(), 0);
  cudaMalloc(&temp_storage,temp_storage_bytes);

  cudaDeviceSynchronize();
  cudaCheckError();
  cudaEventRecord(start);

  for(int i=0;i<REPEAT;i++) {
    cub::DeviceReduce::Reduce(temp_storage, temp_storage_bytes, in, out, N, cub::Sum(), 0);
  }
  cudaEventRecord(stop);
  cudaDeviceSynchronize();
  cudaCheckError();

  float time_ms;
  cudaEventElapsedTime(&time_ms,start,stop);
  float time_s=time_ms/(float)1e3;

  float GB=(float)N*sizeof(int)*REPEAT;
  float GBs=GB/time_s/(float)1e9;

  int sum;
  cudaMemcpy(&sum,out,sizeof(int),cudaMemcpyDeviceToHost);
  cudaCheckError();

  char *valid;
  if(sum==checksum) 
    valid="CORRECT";
  else
    valid="INCORRECT";

  printf("%s: %s, Time: %f s, GB/s: %f, Sum: %d\n", label, valid, time_s, GBs, sum); 
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(in);
  cudaFree(out);
  cudaFree(temp_storage);
  cudaCheckError();
}


void RunTestTangram(char* label, void (*fptr)(int* in, int* out, int N, int blockNum, int threadNum), int N, int REPEAT, int* src, int checksum, int blockNum, int threadNum) {
  int *in, *out;
  //allocate a buffer that is at least large enough that we can ensure it doesn't just sit in l2.
  int MIN_SIZE=4*1024*1024;
  int size=max(int(sizeof(int)*N),MIN_SIZE);
  
  //compute mod base for picking the correct buffer
  int mod=size/(N*sizeof(int));
  cudaEvent_t start,stop;
  cudaMalloc(&in,size);
  cudaMalloc(&out,sizeof(int)*1024);  //only stable version needs multiple elements, all others only need 1
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaCheckError();

  cudaMemcpy(in,src,N*sizeof(int),cudaMemcpyHostToDevice);
  

  
  fptr(in,out,N, blockNum, threadNum);

  cudaMemset(out,0,sizeof(int));

  cudaDeviceSynchronize();
  cudaCheckError();
  cudaEventRecord(start);

  for(int i=0;i<REPEAT;i++) {
    //iterate through different buffers
    //int o=i%mod;
    fptr(in,out,N, blockNum, threadNum);
  }
  cudaEventRecord(stop);
  cudaDeviceSynchronize();
  cudaCheckError();

  float time_ms;
  cudaEventElapsedTime(&time_ms,start,stop);
  float time_s=time_ms/(float)1e3;

  float GB=(float)N*sizeof(int)*REPEAT;
  float GBs=GB/time_s/(float)1e9;

  int sum;
  cudaMemcpy(&sum,out,sizeof(int),cudaMemcpyDeviceToHost);
  cudaCheckError();

  char *valid;
  if(sum==checksum) 
    valid="CORRECT";
  else
    valid="INCORRECT";

  printf("%s: %s, Time: %f s, GB/s: %f, Sum: %d\n", label, valid, time_s, GBs, sum); 
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(in);
  cudaFree(out);
  cudaCheckError();
}



void RunTestTangram_2(char* label, void (*fptr)(int* in, int* out, int N, int blockNum), int N, int REPEAT, int* src, int checksum, int blockNum) {
  int *in, *out;
  //int sum = 0;  
  //allocate a buffer that is at least large enough that we can ensure it doesn't just sit in l2.
  int MIN_SIZE=4*1024*1024;
  int size=max(int(sizeof(int)*N),MIN_SIZE);
  
  //compute mod base for picking the correct buffer
  int mod=size/(N*sizeof(int));
  cudaEvent_t start,stop;
  cudaMalloc(&in,size);
  cudaMalloc(&out,sizeof(int)*1024);  //only stable version needs multiple elements, all others only need 1
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaCheckError();

  cudaMemcpy(in,src,N*sizeof(int),cudaMemcpyHostToDevice);
  

  //warm up
  fptr(in,out,N,blockNum);

  cudaMemset(out,0,sizeof(int));


  cudaDeviceSynchronize();
  cudaCheckError();
  cudaEventRecord(start);

  for(int i=0;i<REPEAT;i++) {
    //iterate through different buffers
    //int o=i%mod;
    fptr(in/*+o*N*/,out,N,blockNum);
  }
  cudaEventRecord(stop);
  cudaDeviceSynchronize();
  cudaCheckError();

  float time_ms;
  cudaEventElapsedTime(&time_ms,start,stop);
  float time_s=time_ms/(float)1e3;

  float GB=(float)N*sizeof(int)*REPEAT;
  float GBs=GB/time_s/(float)1e9;

  int sum;
  cudaMemcpy(&sum,out,sizeof(int),cudaMemcpyDeviceToHost);
  cudaCheckError();

  char *valid;
  if(sum==checksum) 
    valid="CORRECT";
  else
    valid="INCORRECT";

  printf("%s: %s, Time: %f s, GB/s: %f, Sum: %d\n", label, valid, time_s, GBs, sum); 
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(in);
  cudaFree(out);
  cudaCheckError();
}


int main(int argc, char** argv)
{
  if(argc<3) {
    printf("Usage: ./reduce num_elems repeat blockNum(optional) threadNum(optional)\n");
    exit(0);
  }
  int NUM_ELEMS=atoi(argv[1]);
  int REPEAT=atoi(argv[2]);

  int NUM_BLOCKS = 128;
  int NUM_THREADS = 512;
  int NUM_BLOCKS_VEC = 1024;
  if(argc == 5){
    NUM_BLOCKS = atoi(argv[3]);
    NUM_THREADS = atoi(argv[4]);
  }
  else if(argc == 6){
   NUM_BLOCKS = atoi(argv[3]);
   NUM_THREADS = atoi(argv[4]);
   NUM_BLOCKS_VEC = atoi(argv[5]);
  }

  printf("NUM_ELEMS: %d, REPEAT: %d, NUM_BLOCKS: %d, NUM_THREADS: %d, NUM_BLOCKS_VEC: %d\n", NUM_ELEMS, REPEAT, NUM_BLOCKS, NUM_THREADS, NUM_BLOCKS_VEC);

  int* vals=(int*)malloc(NUM_ELEMS*sizeof(int));
  int checksum =0;
  for(int i=0;i<NUM_ELEMS;i++) {
    vals[i]= 1;//rand()%4;
    checksum+=vals[i];
  }

  //RunTest("device_reduce_atomic", device_reduce_atomic,NUM_ELEMS,REPEAT,vals,checksum);
  //RunTest("device_reduce_atomic_vector2", device_reduce_atomic_vector2,NUM_ELEMS,REPEAT,vals,checksum);
  //RunTest("device_reduce_atomic_vector4", device_reduce_atomic_vector4,NUM_ELEMS,REPEAT,vals,checksum);
  
  //RunTest("device_reduce_warp_atomic",device_reduce_warp_atomic,NUM_ELEMS,REPEAT,vals,checksum);
  //RunTest("device_reduce_warp_atomic_vector2",device_reduce_warp_atomic_vector2,NUM_ELEMS,REPEAT,vals,checksum);
  //RunTest("device_reduce_warp_atomic_vector4",device_reduce_warp_atomic_vector4,NUM_ELEMS,REPEAT,vals,checksum);
  
  //RunTest("device_reduce_block_atomic",device_reduce_block_atomic,NUM_ELEMS,REPEAT,vals,checksum);
  
  //RunTest("device_reduce_stable",device_reduce_stable,NUM_ELEMS,REPEAT,vals,checksum);
  //RunTest("device_reduce_stable_vector2",device_reduce_stable_vector2,NUM_ELEMS,REPEAT,vals,checksum);
  //RunTest("device_reduce_stable_vector4",device_reduce_stable_vector4,NUM_ELEMS,REPEAT,vals,checksum);

  RunTestCub("device_reduce_cub",NUM_ELEMS,REPEAT,vals,checksum);

  //RunTestTangram("tangram_dTile_dTile_S_S_S", tangram_dTile_dTile_S_S_S, NUM_ELEMS,REPEAT,vals,checksum);
   
  RunTestTangram_2("tangram_dTileAtom_Va1s", tangram_dTileAtom_Va1s, NUM_ELEMS,REPEAT,vals,checksum, NUM_BLOCKS_VEC);
  RunTestTangram_2("tangram_dTileAtom_Va2", tangram_dTileAtom_Va2, NUM_ELEMS,REPEAT,vals,checksum, NUM_BLOCKS_VEC);
  RunTestTangram_2("tangram_dTileAtom_Va1", tangram_dTileAtom_Va1, NUM_ELEMS,REPEAT,vals,checksum, NUM_BLOCKS_VEC);
  RunTestTangram_2("tangram_dTileAtom_Vs", tangram_dTileAtom_Vs, NUM_ELEMS,REPEAT,vals,checksum, NUM_BLOCKS_VEC);
  RunTestTangram_2("tangram_dTileAtom_V", tangram_dTileAtom_V, NUM_ELEMS,REPEAT,vals,checksum, NUM_BLOCKS_VEC);
#if 0
  RunTestTangram_2("tangram_dTile_Va1s_Va1s", tangram_dTile_Va1s_Va1s, NUM_ELEMS,REPEAT,vals,checksum, NUM_BLOCKS_VEC);
  RunTestTangram_2("tangram_dTile_Va1s_Va1", tangram_dTile_Va1s_Va1, NUM_ELEMS,REPEAT,vals,checksum, NUM_BLOCKS_VEC);
  RunTestTangram_2("tangram_dTile_Va1s_Va2", tangram_dTile_Va1s_Va2, NUM_ELEMS,REPEAT,vals,checksum, NUM_BLOCKS_VEC);
  RunTestTangram_2("tangram_dTile_Va1s_Vs", tangram_dTile_Va1s_Vs, NUM_ELEMS,REPEAT,vals,checksum, NUM_BLOCKS_VEC);
  RunTestTangram_2("tangram_dTile_Va1s_V", tangram_dTile_Va1s_V, NUM_ELEMS,REPEAT,vals,checksum, NUM_BLOCKS_VEC);
  RunTestTangram_2("tangram_dTile_Va2_Va1s", tangram_dTile_Va2_Va1s, NUM_ELEMS,REPEAT,vals,checksum, NUM_BLOCKS_VEC);
  RunTestTangram_2("tangram_dTile_Va2_Va1", tangram_dTile_Va2_Va1, NUM_ELEMS,REPEAT,vals,checksum, NUM_BLOCKS_VEC);
  RunTestTangram_2("tangram_dTile_Va2_Va2", tangram_dTile_Va2_Va2, NUM_ELEMS,REPEAT,vals,checksum, NUM_BLOCKS_VEC);
  RunTestTangram_2("tangram_dTile_Va2_Vs", tangram_dTile_Va2_Vs, NUM_ELEMS,REPEAT,vals,checksum, NUM_BLOCKS_VEC);
  RunTestTangram_2("tangram_dTile_Va2_V", tangram_dTile_Va2_V, NUM_ELEMS,REPEAT,vals,checksum, NUM_BLOCKS_VEC);
#endif

  RunTestTangram("tangram_dTileAtom_dStride_S_Va1s", tangram_dTileAtom_dStride_S_Va1s, NUM_ELEMS,REPEAT,vals,checksum, NUM_BLOCKS, NUM_THREADS);
  RunTestTangram("tangram_dTileAtom_dStride_S_Va2", tangram_dTileAtom_dStride_S_Va2, NUM_ELEMS,REPEAT,vals,checksum, NUM_BLOCKS, NUM_THREADS);
  RunTestTangram("tangram_dTileAtom_dStride_S_Va1", tangram_dTileAtom_dStride_S_Va1, NUM_ELEMS,REPEAT,vals,checksum, NUM_BLOCKS, NUM_THREADS);
  RunTestTangram("tangram_dTileAtom_dStride_S_Vs", tangram_dTileAtom_dStride_S_Vs, NUM_ELEMS,REPEAT,vals,checksum, NUM_BLOCKS, NUM_THREADS);
  RunTestTangram("tangram_dTileAtom_dStride_S_V", tangram_dTileAtom_dStride_S_V, NUM_ELEMS,REPEAT,vals,checksum, NUM_BLOCKS, NUM_THREADS);
  RunTestTangram("tangram_dTileAtom_dStrideAtom_S", tangram_dTileAtom_dStrideAtom_S, NUM_ELEMS,REPEAT,vals,checksum, NUM_BLOCKS, NUM_THREADS);
  
  free(vals);

}
