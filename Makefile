all: build
build:
	nvcc main.cu --std=c++11 -O2 -o main -I ./lib
