
HIT_COMPILE=/usr/bin/nvcc

EXE=SSSP

OPT=-O3

ARCH=-arch=sm_61


$(EXE): template.cu
	$(HIT_COMPILE) -w -o $(EXE) template.cu ${OPT} $(ARCH)

