stencil: stencil.c
	mpiicc -ipo -O3 -no-prec-div -fp-model fast=2 -xHost -std=c99 -Wall $^ -o $@