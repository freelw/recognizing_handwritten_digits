$(info "Makefile for mnist_cuda")
DIR_INC = -I./
DIR_INC += -I./tensor
DIR_INC += -I./graph

DIR_LIB = -L./
TARGET	= mnist_cuda
CFLAGS = $(DIR_INC) $(DIR_LIB) -g -G
#CFLAGS = $(DIR_INC) $(DIR_LIB)
SRCDIR := ./tensor \
          ./graph \
          ./backends/cpu \
          ./backends/gpu \
          ./backends/ \
          ../utils/dataloader
SRCS := $(wildcard *.cpp) $(wildcard $(addsuffix /*.cpp, $(SRCDIR)))
SRCS += $(wildcard *.cu) $(wildcard $(addsuffix /*.cu, $(SRCDIR)))
OBJECTS := $(patsubst %.c,%.o,$(patsubst %.cpp,%.o,$(patsubst %.cu,%.o,$(SRCS))))
NVCC = nvcc

$(TARGET) : $(OBJECTS)
	${NVCC} $(CFLAGS) $^ -o $@ $(LDFLAGS)
%.o : %.cu
	${NVCC} -c $(CFLAGS) $< -o $@
%.o : %.cpp
	${NVCC} -c $(CFLAGS) $< -o $@

clean:
	@rm -f ${OBJECTS} $(TARGET)

.PHONY: clean prepare_data
prepare_data:
	@cd ../../resources && \
	gunzip -c train-labels-idx1-ubyte.gz > train-labels-idx1-ubyte && \
	gunzip -c train-images-idx3-ubyte.gz > train-images-idx3-ubyte
dbg:
	@echo "CFLAGS is $(CFLAGS)"
	@echo "SRCS is $(SRCS)"
	@echo "OBJECTS is $(OBJECTS)"
	@echo "SRCDIR is $(SRCDIR)"
	@echo "DIR_INC is $(DIR_INC)"
	@echo "DIR_LIB is $(DIR_LIB)"
	@echo "TARGET is $(TARGET)"
	@echo "NVCC is $(NVCC)"