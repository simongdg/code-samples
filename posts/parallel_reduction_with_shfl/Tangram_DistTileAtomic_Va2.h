__global__ void Reduce_13(int *Return_23, int *input_x_14, int SourceSize_24,
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

void tangram_dTileAtom_Va2(int *input_x_7, int *out, int ObjectSize_27, int blockNum) {

  int p_8 = blockNum;
  int x_size_9 = ObjectSize_27;
  int tile_10 = ((((x_size_9 + p_8) - 1)) / p_8);

  dim3 dimBlock(tile_10);
  dim3 dimGrid(p_8);
  Reduce_13 << <dimGrid, dimBlock>>> (out, input_x_7, x_size_9, tile_10);

}
