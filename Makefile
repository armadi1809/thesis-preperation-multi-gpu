# Builds every .cu file in the repo into build/<same path without .cu>.
#
#   make                         build all programs
#   make run/map/map_explicit_memory_management
#                                build (if needed) and run one program
#   make run/<program> ARGS="..."
#                                pass command-line arguments to the program
#   make list                    show all buildable programs
#   make clean                   remove build/
#
# Override variables on the command line, e.g.
#   make ARCH=-arch=sm_80 NVCCFLAGS="-O0 -g -G"

NVCC      ?= nvcc
ARCH      ?= -arch=native
NVCCFLAGS ?= -O3 -std=c++17 -lineinfo
BUILD_DIR := build

SRCS := $(shell find . -name '*.cu' -not -path './$(BUILD_DIR)/*' | sed 's|^\./||')
BINS := $(SRCS:%.cu=$(BUILD_DIR)/%)
HDRS := $(wildcard common/*.h)

.PHONY: all clean list
.SECONDARY:

all: $(BINS)

$(BUILD_DIR)/%: %.cu $(HDRS)
	@mkdir -p $(dir $@)
	$(NVCC) $(ARCH) $(NVCCFLAGS) -o $@ $<

run/%: $(BUILD_DIR)/%
	./$< $(ARGS)

list:
	@printf '%s\n' $(SRCS:%.cu=%)

clean:
	rm -rf $(BUILD_DIR)
