all: build
build:
	nvcc main.cu --std=c++11 -O2 -o main -I ./lib
debug:
	nvcc main.cu --std=c++11 -g -G -o main -I ./lib
