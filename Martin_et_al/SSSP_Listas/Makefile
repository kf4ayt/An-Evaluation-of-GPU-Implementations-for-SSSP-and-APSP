
HIT_COMPILE=/usr/bin/nvcc

EXE=sssp-listas

OPT=-O3

ARCH=-arch=sm_61


$(EXE): template.cu
	$(HIT_COMPILE) -w -o $(EXE) template.cu ${OPT} $(ARCH)

debug: template.cu
	$(HIT_COMPILE) -w -o $(EXE) template.cu -DDEBUG ${OPT} $(ARCH)
	
clean:
	rm -f *.o && rm -f ${EXE}
