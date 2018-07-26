
int Reduce_30(int *input_x_31, int SourceSize_34, int OffsetEnd_35,
              int ObjectSize_36, int Stride_37) {

  int result_32 = 0;
  for (int i_33 = 0; (i_33 < ObjectSize_36); i_33 += Stride_37) {
    result_32 += input_x_31[i_33];
  }

  return result_32;
}
__inline__ __device__ int Reduce_25(int *input_x_26, int SourceSize_38,
                                    int OffsetEnd_39, int ObjectSize_40,
                                    int Stride_41) {

  int result_27 = 0;
  for (int i_28 = 0; (i_28 < ObjectSize_40); i_28 += Stride_41) {
    if (i_28 + threadIdx.x < SourceSize_38) {
      result_27 += input_x_26[i_28];
    }
  }

  return result_27;
}
__inline__ __device__ void Reduce_20(int *Return_42, int *input_x_21,
                                     int SourceSize_43, int OffsetEnd_44,
                                     int ObjectSize_45, int Stride_46) {

  unsigned int tid_47 = threadIdx.x;
  int result_22 = 0;
  for (int i_23 = 0; (i_23 < ObjectSize_45); i_23 += Stride_46) {
    if ((i_23 + threadIdx.x * (ObjectSize_45 / Stride_46) < SourceSize_43) &&
        (i_23 + (blockIdx.x * SourceSize_43 + threadIdx.x) < OffsetEnd_44)) {
      result_22 += input_x_21[i_23];
    }
  }

  Return_42[tid_47] = result_22;
}
__global__ void Reduce_13(int *Return_48, int *input_x_14, int ObjectSize_49,
                          int ObjectSize_50, int SourceSize_51) {

  unsigned int blockID_52 = blockIdx.x;
  int p_15 = blockDim.x;
  int x_size_16 = ObjectSize_50;
  int tile_17 = ((((x_size_16 + p_15) - 1)) / p_15);

  int *part_19 = input_x_14 + (blockIdx.x * ObjectSize_49);
  /*Map*/

  __shared__ int *map_return_4;
  if (threadIdx.x == 0) {
    map_return_4 = new int[p_15];
  }

  __syncthreads();

  Reduce_20(map_return_4, part_19 + (0 + (threadIdx.x * tile_17)), x_size_16,
            SourceSize_51, tile_17, 1);

  __syncthreads();

  if (threadIdx.x == 0) {
    int result_24 = Reduce_25(map_return_4, p_15, p_15, p_15, 1);

    Return_48[blockID_52] = result_24;
  }
}

//template <unsigned int TGM_TEMPLATE_0, unsigned int TGM_TEMPLATE_1>
int tangram_dTile_dTile_S_S_S(int *input_x_7, int *out , int ObjectSize_53) {

  int p_8 = 128;//TGM_TEMPLATE_0;
  int x_size_9 = ObjectSize_53;
  int tile_10 = ((((x_size_9 + p_8) - 1)) / p_8);

  int *part_12 = input_x_7;
  /*Map*/

  int *map_return_h_2 = new int[p_8];
  //int *map_return_1;
  //cudaMalloc((void **)&map_return_1, (p_8) * sizeof(int));
  dim3 dimBlock(512/*TGM_TEMPLATE_1*/);
  dim3 dimGrid(p_8);
  Reduce_13 << <dimGrid, dimBlock>>>
      (out, part_12, tile_10, tile_10, x_size_9);

  cudaMemcpy(map_return_h_2, out, (p_8) * sizeof(int),
             cudaMemcpyDeviceToHost);
  int result_29 = Reduce_30(map_return_h_2, p_8, p_8, p_8, 1);

  return result_29;
}
