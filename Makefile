#!/usr/bin/make -f

CXX:=g++
MPICXX:=mpicxx
CXXFLAGS:=-O3 -g
AR:=ar

.PHONY: default
default: 
	@echo No default, choose a target!

.PHONY: clean
clean:
	rm -f *.x *.o

integral_seq.x: integral_seq.cxx
	$(CXX) $(CXXFLAGS) -o $@ integral_seq.cxx -L./mylib -lmymath

integral_par.x: integral_par.cxx
	$(MPICXX) $(CXXFLAGS) -o $@ integral_par.cxx -L./mylib -lmymath

integral_dyn.x: integral_dyn.cxx
	$(MPICXX) $(CXXFLAGS) -o $@ integral_dyn.cxx -L./mylib -lmymath

