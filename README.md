# CUDA-Sorting

These repo contains three files for GPU sorting.
thrust.cu used thrust::sort() function
singlethread.cu uses radix sort and multithread.cu uses the parallel GPU radix sort.

The programs accept three command line arguments while running, 
	- the size of the array
	- a seed value for generating the random number
	- an indicator of whether to print the sorted array or not (1/0)

the command for running the program should be in the following form:
	[executable file] [size of the array] [seed value] [1/0]
