all:
	nvcc thrust.cu -o thrust
	nvcc singlethread.cu -o singlethread
	nvcc multithread.cu -o multithread
