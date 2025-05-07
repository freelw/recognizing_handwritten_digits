DIR_INC = -I./ \
		-I./tensor \
		-I./graph \
		-I../utils/

DIR_LIB = -L./
TARGET = mnist_cuda
TEST_TARGET = test


LDFLAGS = -lstdc++
SRCDIR := ./tensor \
          ./graph \
          ./backends/cpu \
          ./backends/gpu \
          ./backends/ \
		  ./optimizers \
		  ./model \
          ../utils/dataloader
SRCS := $(wildcard *.cpp) $(wildcard $(addsuffix /*.cpp, $(SRCDIR)))
SRCS += $(wildcard *.cu) $(wildcard $(addsuffix /*.cu, $(SRCDIR)))
OBJECTS := $(patsubst %.c,%.o,$(patsubst %.cpp,%.o,$(patsubst %.cu,%.o,$(SRCS))))
OBJECTS_MAIN := $(filter-out test.o,$(OBJECTS))
OBJECTS_TEST := $(filter-out main.o,$(OBJECTS))

ifeq ($(ASAN),1)
	NVCC = g++
	NVCC_CFLAGS = $(DIR_INC) $(DIR_LIB) -g -fsanitize=address -fno-omit-frame-pointer
else
	NVCC = nvcc
	NVCC_CFLAGS = $(DIR_INC) $(DIR_LIB) -g -G
endif

$(TARGET) : $(OBJECTS_MAIN)
	${NVCC} $(NVCC_CFLAGS) $^ -o $@ $(LDFLAGS)
$(TEST_TARGET) : $(OBJECTS_TEST)
	${NVCC} $(NVCC_CFLAGS) $^ -o $@ $(LDFLAGS)
%.o : %.cu
	${NVCC} -c $(NVCC_CFLAGS) $< -o $@
%.o : %.cpp
	${NVCC} -c $(NVCC_CFLAGS) $< -o $@

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