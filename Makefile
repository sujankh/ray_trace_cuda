all: build
build:
	nvcc main.cu --std=c++11 -o main -I ./lib
