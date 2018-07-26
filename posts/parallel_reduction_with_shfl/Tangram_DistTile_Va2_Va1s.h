__global__ void Reduce_13_4(int *Return_29, int *input_x_14, int SourceSize_30,
                          int ObjectSize_31, int ObjectSize_32,
                          int ObjectSize_33) {

  unsigned int blockID_34 = blockIdx.x;
  /*Vector*/;
  __shared__ int partial_16;
  if(threadIdx.x == 0)
    partial_16 = 0;
  
  __syncthreads();

  int val_18 = 0;
  val_18 = (((threadIdx.x < ObjectSize_31)) &&
            ((blockIdx.x * blockDim.x + threadIdx.x) < SourceSize_30))
               ? input_x_14[blockIdx.x * blockDim.x + threadIdx.x]
               : 0;
  ;
  for (int offset_19 = (32 / 2); (offset_19 > 0); offset_19 /= 2) {
    val_18 += __shfl_down(val_18, offset_19, 32);
    ;
  }

  if (((ObjectSize_32 != 32) && ((ObjectSize_33 / 32) > 0))) {
    if ((threadIdx.x % warpSize == 0)) {
      atomicAdd(&partial_16, val_18);
      __syncthreads();
    };
    if ((threadIdx.x / warpSize == 0)) {
      val_18 = partial_16;
    };
  };

  if (threadIdx.x == 0) {
    Return_29[blockID_34] = val_18;
  }
}

__global__ void Reduce_134(int *Return_23, int *input_x_14, int SourceSize_24,
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


void tangram_dTile_Va2_Va1s(int *input_x_7, int *out, int ObjectSize_35, int blockNum) {

  int p_8 = blockNum;
  int x_size_9 = ObjectSize_35;
  int tile_10 = ((((x_size_9 + p_8) - 1)) / p_8);

  dim3 dimBlock((((x_size_9 - 1) / p_8) + 1));
  dim3 dimGrid(p_8);
  Reduce_134 << <dimGrid, dimBlock>>>
      (out, input_x_7, x_size_9, tile_10);

  Reduce_13_4 << <1, p_8>>>
      (out, out, x_size_9, ObjectSize_35, ObjectSize_35, ObjectSize_35);
}

