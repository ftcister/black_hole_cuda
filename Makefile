BINS=black_hole
LDFLAGS=-L extern/lib -l SDL2
CFLAGS=-I extern/include

default: all

all: $(BINS)

%: %.cu
	nvcc -g -o  $@ $< $(CFLAGS) $(LDFLAGS)

clean:
	rm -f $(BINS) *.csv

.PHONY: default all clean
