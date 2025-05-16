DIR_INC = -I./ \
    -I./tensor \
    -I./graph \
    -I../utils/

DIR_LIB = -L./
TARGET = curand
CUDA_TOOLKIT := $(shell dirname $$(command -v nvcc))/..
LIBS         := -L$(CUDA_TOOLKIT)/lib64 -lcudart -lcurand


LDFLAGS = -lstdc++ $(LIBS)
SRCDIR := ./tensor \
          ./graph \
          ./backends/cpu \
          ./backends/gpu \
          ./backends/ \
          ./optimizers \
          ./model \
		  ./module \
          ../utils/dataloader
SRCS := $(wildcard *.cpp) $(wildcard $(addsuffix /*.cpp, $(SRCDIR)))
ifeq ($(ASAN),1)
	OBJECTS := $(patsubst %.c,%.o,$(patsubst %.cpp,%.o,$(SRCS)))
else
	SRCS += $(wildcard *.cu) $(wildcard $(addsuffix /*.cu, $(SRCDIR)))
	OBJECTS := $(patsubst %.c,%.o,$(patsubst %.cpp,%.o,$(patsubst %.cu,%.o,$(SRCS))))
endif
OBJECTS_MAIN := $(filter-out test.o transformer.o,$(OBJECTS))
OBJECTS_TEST := $(filter-out main.o transformer.o,$(OBJECTS))
OBJECTS_TRANSFORMER := $(filter-out test.o main.o,$(OBJECTS))

ifeq ($(ASAN),1)
	NVCC = g++
	NVCC_CFLAGS = -DGCC_ASAN $(DIR_INC) $(DIR_LIB) -g -fsanitize=address -fno-omit-frame-pointer
else
	NVCC = nvcc
	NVCC_CFLAGS = $(DIR_INC) $(DIR_LIB) -g -G -O3
endif

ifeq ($(RELEASE),1)
	NVCC_CFLAGS += -DNDEBUG
	NVCC_CFLAGS := $(filter-out -G,$(NVCC_CFLAGS))
endif

all: $(TARGET)

$(TARGET) : $(OBJECTS_MAIN)
	${NVCC} $(NVCC_CFLAGS) $^ -o $@ $(LDFLAGS)


%.o : %.cu
	${NVCC} -c $(NVCC_CFLAGS) $< -o $@
%.o : %.cpp
	${NVCC} -c $(NVCC_CFLAGS) $< -o $@

clean:
	@rm -f ${OBJECTS} $(TARGET)

.PHONY: clean 
