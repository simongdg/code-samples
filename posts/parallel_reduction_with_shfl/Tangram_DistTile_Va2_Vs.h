__global__ void Reduce_13_7(int *Return_26, int *input_x_14, int SourceSize_27,
                          int ObjectSize_28, int ObjectSize_29,
                          int ObjectSize_30, int ObjectSize_31) {

  unsigned int blockID_32 = blockIdx.x;
  /*Vector*/;
  __shared__ int partial_16[32];
  if(threadIdx.x < 32)
    partial_16[threadIdx.x] = 0;

  __syncthreads();

  int val_18 = 0;
  val_18 = (((threadIdx.x < ObjectSize_28)) &&
            ((blockIdx.x * blockDim.x + threadIdx.x) < SourceSize_27))
               ? input_x_14[blockIdx.x * blockDim.x + threadIdx.x]
               : 0;
  for (int offset_19 = (32 / 2); (offset_19 > 0); offset_19 /= 2) {
    val_18 += __shfl_down(val_18, offset_19, 32);
  }

  if (((ObjectSize_29 != 32) && ((ObjectSize_30 / 32) > 0))) {
    if ((threadIdx.x % warpSize == 0)) {
      partial_16[threadIdx.x / warpSize] = val_18;
      __syncthreads();
    }
    if ((threadIdx.x / warpSize == 0)) {
      val_18 = ((threadIdx.x <= ((ObjectSize_31 / 32))))
                   ? partial_16[threadIdx.x % warpSize]
                   : 0;
      for (int offset_20 = (32 / 2); (offset_20 > 0); offset_20 /= 2) {
        val_18 += __shfl_down(val_18, offset_20, 32);

      }
    }
  }

  if(threadIdx.x == 0)
  atomicAdd_system(Return_26, val_18);
}

__global__ void Reduce_137(int *Return_23, int *input_x_14, int SourceSize_24,
                          int ObjectSize_25) {

  unsigned int blockID_26 = blockIdx.x;
  /*Vector*/;
  __shared__ int shared_16;
  if(threadIdx.x == 0)
    shared_16 = 0;

  __syncthreads();

  int val_17 = 0;
  val_17 = (((threadIdx.x < ObjectSize_25)) &&
            ((blockIdx.x * blockDim.x + threadIdx.x) < SourceSize_24))
               ? input_x_14[blockIdx.x * blockDim.x + threadIdx.x]
               : 0;
  atomicAdd_block(&shared_16, val_17);
  __syncthreads();

  if(threadIdx.x == 0){
    atomicAdd(Return_23, shared_16);
  }
}


void tangram_dTile_Va2_Vs(int *input_x_7, int *out, int ObjectSize_35, int blockNum) {

  int p_8 = blockNum;
  int x_size_9 = ObjectSize_35;
  int tile_10 = ((((x_size_9 + p_8) - 1)) / p_8);

  dim3 dimBlock((((x_size_9 - 1) / p_8) + 1));
  dim3 dimGrid(p_8);
  Reduce_137 << <dimGrid, dimBlock>>>
      (out, input_x_7, x_size_9, tile_10);

  Reduce_13_7 << <1, p_8>>>
      (out, out, x_size_9, ObjectSize_35,ObjectSize_35,ObjectSize_35,ObjectSize_35);
}

