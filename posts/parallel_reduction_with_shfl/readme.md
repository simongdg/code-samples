To build you will need to download CUB.

To do this run the following command from the current directory:

%> rm -rf cub
%> git clone https://github.com/NVlabs/cub.git


Make file:
1) make sure you change to cuda flags based on the GPU architecture


Run:

./reduce (Array_size) (number of iterations) (Number of blocks) (Number of threads per block) (Number of blocks for Vector only codes)
  
example:
%> ./reduce 100000 1 128 512 1024

note: 
1) keep number of iterations to one, I do clean the arrays for tangram so they keep adding up incorrectly. 
2) The last argument is fore codes that Distribute --> Vector instead of Distribute --> Distribute --> S --> Vector --> etc



