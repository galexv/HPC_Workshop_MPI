#!/usr/bin/make -f

CXX:=g++
CXXFLAGS:=-O3
AR:=ar

.PHONY: default
default: libmymath.a

libmymath.a:	mymath.cxx mymath.hpp
	$(CXX) $(CXXFLAGS) -o mymath.o -c mymath.cxx
	$(AR) rs $@ mymath.o

.PHONY: clean
clean:
	rm -f *.a *.o
