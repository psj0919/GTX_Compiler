#GTX ISS Make command
TMU_NUM ?= 4

MAIN_SIZE ?= 0x100000000

CXX := g++
CXXFLAGS := -DTMU_NUM=$(TMU_NUM) -DMAIN_SIZE=$(MAIN_SIZE) -Wunused-function -fdiagnostics-color=always -g -MMD -MP -fopenmp #-DUSE_VFIO_USER 

INC := -I/opt/systemc/include/ \
	   -I/usr/local/include/ \
	   -I/usr/local/include/openblas/ \
	   -I/usr/local/include/vfio-user
# 	   -I./c_test/common/ \

#STATIC_LIB := /ca/systemc-2.3.2-temp/libmulti-arch/libsystemc.a -lpthread -lrt
STATIC_LIB := /opt/systemc/lib/libsystemc.so /usr/local/lib/x86_64-linux-gnu/libvfio-user.so -lpthread -lrt -lopenblas 

#LIB := -L/ca/spdlog/lib64 -L/ca/systemc-2.3.2-temp/libmulti-arch/
LIB := -L/opt/systemc/lib/ -L/usr/local/lib/ -L/usr/local/lib/x86_64-linux-gnu/
LDLIBS := -lsystemc  -lopenblas

TARGET := ./GTX_ISS

SRCDIR := ./src
OBJDIR := ./obj


SRCS := $(wildcard $(SRCDIR)/*.cpp)
OBJS := $(patsubst $(SRCDIR)/%.cpp, $(OBJDIR)/%.o, $(SRCS))

#$(TARGET): $(OBJS)
#	$(CXX) $(CXXFLAGS) $(INC) $(LIB) -o $@ $^ $(LDLIBS)
all : static

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	@mkdir -p $(OBJDIR)
	$(CXX) $(CXXFLAGS) $(INC) -I./inc -c $< -o $@

static: $(OBJS)
	$(CXX) $(CXXFLAGS) $(INC) -I./inc -o $(TARGET) $(OBJS) $(STATIC_LIB)

dynamic: $(OBJS)
	$(CXX) $(CXXFLAGS) $(INC) -I./inc $(LIB) -o $(TARGET) $(OBJS) $(LDLIBS)

.PHONY: test
test:
	@make clean -C ./c_test/$(CODE)
	@make -C ./c_test/$(CODE)
	rm -rf L1_Memory*
	rm -rf L2_Memory*
	rm -rf MEM_DUMP.hex
	rm -rf COMMAND_DUMP.hex
	rm -rf REG_DUMP.txt
	./GTX_ISS -D ./c_test/$(CODE)/$(CODE).hex

clean:
	rm -rf $(OBJDIR) $(OBJDIR_DEV)

-include $(OBJS_DEV:.o=.d)
