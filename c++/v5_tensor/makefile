DIR_INC = -I./
DIR_INC += -I./tensor
DIR_INC += -I./graph

DIR_LIB = -L./
TARGET = mnist_cuda
TEST_TARGET = test
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
OBJECTS_MAIN := $(filter-out test.o,$(OBJECTS))
OBJECTS_TEST := $(filter-out main.o,$(OBJECTS))
NVCC = nvcc

$(TARGET) : $(OBJECTS_MAIN)
	${NVCC} $(CFLAGS) $^ -o $@ $(LDFLAGS)
$(TEST_TARGET) : $(OBJECTS_TEST)
	${NVCC} $(CFLAGS) $^ -o $@ $(LDFLAGS)
%.o : %.cu
	${NVCC} -c $(CFLAGS) $< -o $@
%.o : %.cpp
	${NVCC} -c $(CFLAGS) $< -o $@

clean:
	@rm -f ${OBJECTS} $(TARGET) $(TEST_TARGET)

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