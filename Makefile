CXX=g++
RM=rm -f
CPPFLAGS=-Wall -Wextra -O3 -g -fopenmp -march=native -std=c++17
LDFLAGS=-g $(shell root-config --ldflags)
LDLIBS=$(shell root-config --libs)

SRCS=mini-pi.cpp mini-pi_optimized_1_cached_twiddles.cpp mini-pi_optimized_2_SSE3.cpp mini-pi_optimized_3_OpenMP.cpp
BINS=$(subst .cpp,,$(SRCS))

all: $(BINS)

%: %.cpp
		$(CXX) $(CPPFLAGS) -o $@ $<

clean:
		$(RM) $(BINS)
