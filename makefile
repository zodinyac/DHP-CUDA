NVCC_LINK_FLAGS = -L/opt/cuda/lib64 -lcudart

all: dhp-cuda

dhp-cuda: dhp-cuda.o cuda.o
	mpicc -o dhp-cuda dhp-cuda.o cuda.o $(NVCC_LINK_FLAGS) -lstdc++

dhp-cuda.o: dhp-cuda.c
	mpicc -std=c99 -c dhp-cuda.c
cuda.o: cuda.cu
	nvcc -arch=sm_20 -c cuda.cu

clean:
	rm -f dhp-cuda dhp-cuda.o cuda.o
