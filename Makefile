all: build
build:
	nvcc main.cu -o main -I ./lib
