__inline__ __device__ void Reduce_20(int *Return_34, int *input_x_21,
                                     int SourceSize_35, int OffsetEnd_36,
                                     int ObjectSize_37, int Stride_38) {

  unsigned int tid_39 = threadIdx.x;
  int result_22 = 0;
  for (int i_23 = 0; (i_23 < ObjectSize_37); i_23 += Stride_38) {
    if ((i_23 + threadIdx.x < SourceSize_35) &&
        (i_23 + (blockIdx.x * SourceSize_35 + threadIdx.x) < OffsetEnd_36)) {
      result_22 += input_x_21[i_23];
    }
  }

  atomicAdd_system(Return_34, result_22);
}
__global__ void Reduce_13(int *Return_40, int *input_x_14, int ObjectSize_41,
                          int ObjectSize_42, int SourceSize_43) {

  unsigned int blockID_44 = blockIdx.x;
  int p_15 = blockDim.x;
  int x_size_16 = ObjectSize_42;
  int tile_17 = ((((x_size_16 + p_15) - 1)) / p_15);

  int *part_19 = input_x_14 + (blockIdx.x * ObjectSize_41);
  /*Map*/

  __shared__ int *map_return_4;
  if (threadIdx.x == 0) {
    map_return_4 = new int[1];
  }

  __syncthreads();

  Reduce_20(map_return_4, part_19 + (0 + (threadIdx.x * 1)), x_size_16,
            SourceSize_43, (p_15 * tile_17), p_15);

  __syncthreads();

  if (threadIdx.x == 0) {
    int result_24 = *map_return_4;

    atomicAdd_system(Return_40, result_24);
  }
}

void tangram_dTileAtom_dStrideAtom_S(int *input_x_7, int *out, int ObjectSize_45, int blockNum, int threadNum) {

  int p_8 = blockNum;
  int x_size_9 = ObjectSize_45;
  int tile_10 = ((((x_size_9 + p_8) - 1)) / p_8);

  dim3 dimBlock(threadNum);
  dim3 dimGrid(p_8);
  Reduce_13 << <dimGrid, dimBlock>>>
      (out, input_x_7, tile_10, tile_10, x_size_9);
}
