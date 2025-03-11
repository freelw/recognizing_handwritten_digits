DIR_INC = -I./
DIR_INC += -I../utils/
DIR_LIB = -L./
TARGET	= autogradtest
CFLAGS = -g -Wall $(DIR_INC) $(DIR_LIB) -fsanitize=address -fopenmp
LDFLAGS += -lstdc++ -fopenmp
SRCDIR:= ../utils/dataloader
SRCDIR+= ../utils/matrix
SRCDIR+= ../utils/autograd
SRCDIR+= ../utils/optimizers
SRCDIR+= ../utils/parameters
SRCDIR+= ../utils/lmcommon
SRCS := $(wildcard *.cpp) $(wildcard $(addsuffix /*.cpp, $(SRCDIR)))
OBJECTS := $(patsubst %.c,%.o,$(SRCS))
$(TARGET) : $(OBJECTS)
	g++ $(CFLAGS) $^ -o $@ $(LDFLAGS)
%.o : %.c
	g++ -c $(CFLAGS) $< -o $@
clean:
	@rm -f *.o $(TARGET)
.PHONY:clean
prepare_data:
	cd ../../resources && \
	gunzip -c train-labels-idx1-ubyte.gz > train-labels-idx1-ubyte && \
	gunzip -c train-images-idx3-ubyte.gz > train-images-idx3-ubyte