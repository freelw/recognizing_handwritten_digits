DIR_INC = -I./
DIR_INC += -I../utils/
DIR_LIB = -L./
TARGET	= recognizing_handwritten_digits_v1
CFLAGS = -g -Wall $(DIR_INC) $(DIR_LIB)
LDFLAGS += -lstdc++
SRCDIR:= ../utils/dataloader
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