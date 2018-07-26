__global__ void Reduce_136(int *Return_23, int *input_x_14, int SourceSize_24,
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


void tangram_dTile_Va2_Va2(int *input_x_7, int *out, int ObjectSize_35, int blockNum) {

  int p_8 = blockNum;
  int x_size_9 = ObjectSize_35;
  int tile_10 = ((((x_size_9 + p_8) - 1)) / p_8);

  dim3 dimBlock((((x_size_9 - 1) / p_8) + 1));
  dim3 dimGrid(p_8);
  Reduce_136 << <dimGrid, dimBlock>>>
      (out, input_x_7, x_size_9, tile_10);

  Reduce_136 << <1, p_8>>>
      (out, out, x_size_9, ObjectSize_35);
}

