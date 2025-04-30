DIR_INC = -I./
DIR_INC += -I./tensor
DIR_INC += -I./graph

DIR_LIB = -L./
TARGET	= mnist_cuda
CFLAGS = $(DIR_INC) $(DIR_LIB) -g -G
#CFLAGS = $(DIR_INC) $(DIR_LIB)
SRCDIR := ./tensor
SRCDIR += ./graph
SRCDIR += ./backends/cpu
SRCDIR += ./backends/gpu
SRCDIR += ./backends/
SRCDIR += ../utils/dataloader
SRCS := $(wildcard *.cpp) $(wildcard $(addsuffix /*.cpp, $(SRCDIR)))
SRCS += $(wildcard *.cu) $(wildcard $(addsuffix /*.cu, $(SRCDIR)))
OBJECTS := $(patsubst %.c,%.o,$(SRCS))
NVCC = nvcc

$(TARGET) : $(OBJECTS)
	@echo $(RELEASE_MSG)
	${NVCC} $(CFLAGS) $^ -o $@ $(LDFLAGS)
%.o : %.cu
	${NVCC} -c $(CFLAGS) $< -o $@
clean:
	@rm -f *.o $(TARGET)
.PHONY:clean
prepare_data:
	cd ../../resources && \
	gunzip -c train-labels-idx1-ubyte.gz > train-labels-idx1-ubyte && \
	gunzip -c train-images-idx3-ubyte.gz > train-images-idx3-ubyte