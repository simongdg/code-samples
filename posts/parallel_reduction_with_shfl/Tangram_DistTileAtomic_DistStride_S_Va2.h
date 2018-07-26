__inline__ __device__ int Reduce_27(int *input_x_26, int SourceSize_35,
                                    int ObjectSize_36) {

  /*Vector*/;
  __shared__ int shared_28;
  if(threadIdx.x == 0)
    shared_28 = 0;

  __syncthreads();

  int val_29 = 0;
  val_29 = ((threadIdx.x < ObjectSize_36)) ? input_x_26[threadIdx.x] : 0;
  atomicAdd(&shared_28, val_29);
  __syncthreads();
  return shared_28;
}

__inline__ __device__ void Reduce_22(int *Return_41, int *input_x_21,
                                     int SourceSize_42, int OffsetEnd_43,
                                     int ObjectSize_44, int Stride_45) {

  unsigned int tid_46 = threadIdx.x;
  int result_22 = 0;
  for (int i_23 = 0; (i_23 < ObjectSize_44); i_23 += Stride_45) {
    if ((i_23 + threadIdx.x < SourceSize_42) &&
        (i_23 + (blockIdx.x * SourceSize_42 + threadIdx.x) < OffsetEnd_43)) {
      result_22 += input_x_21[i_23];
    }
  }

  Return_41[tid_46] = result_22;
}
__global__ void Reduce_15(int *Return_47, int *input_x_14, int ObjectSize_48,
                          int ObjectSize_49, int SourceSize_50) {

  unsigned int blockID_51 = blockIdx.x;
  int p_15 = blockDim.x;
  int x_size_16 = ObjectSize_49;
  int tile_17 = ((((x_size_16 + p_15) - 1)) / p_15);

  int *part_19 = input_x_14 + (blockIdx.x * ObjectSize_48);
  /*Map*/

  __shared__ int *map_return_4;
  if (threadIdx.x == 0) {
    map_return_4 = new int[p_15];
  }

  __syncthreads();

  Reduce_22(map_return_4, part_19 + (0 + (threadIdx.x * 1)), x_size_16,
            SourceSize_50, (p_15 * tile_17), p_15);

  __syncthreads();
  ;
  ;
  int result_24 = Reduce_27(map_return_4, p_15, p_15);

  if (threadIdx.x == 0) {
    atomicAdd_system(Return_47, result_24);
  }
}

void tangram_dTileAtom_dStride_S_Va2(int *input_x_7, int *out, int ObjectSize_52, int blockNum, int threadNum) {

  int p_8 = blockNum;
  int x_size_9 = ObjectSize_52;
  int tile_10 = ((((x_size_9 + p_8) - 1)) / p_8);

  dim3 dimBlock(threadNum);
  dim3 dimGrid(p_8);
  Reduce_15 << <dimGrid, dimBlock>>>
      (out, input_x_7, tile_10, tile_10, x_size_9);

}
