# Indicate this is not a GNU project and that
# missing files should be automatically created
AUTOMAKE_OPTIONS=foreign

# Instruct aclocal to check the cfg directory for
# extra macros
ACLOCAL_AMFLAGS=-I cfg

bin_PROGRAMS=opConvolution

opConvolution_SOURCES = main.cpp timer.h opConvolutionFilter.cpp opConvolutionFilter.h statistics.h util.h StopWatch.cxx StopWatch.h

#CXXFLAGS=-fopenmp -O3 -msse -msse2 -msse3 -msse4 -msse4.1 -msse4.2 -mfpmath=sse -march=core2 -mtune=core2 -ffast-math -ftree-vectorize -fabi-version=4
CXXFLAGS=-fopenmp -O3 -msse -msse2 -msse3 -mfpmath=sse -ffast-math -ftree-vectorize -fabi-version=4
LDFLAGS=-lgomp -lboost_system -lboost_filesystem

#AUTOMAKE_OPTIONS = foreign
#EXTRA_DIST =  expansions.m4

#Warning: This is an automatically generated file, do not edit!
#if ENABLE_DEBUG
# SUBDIRS = src bin docs lib
#endif
#if ENABLE_RELEASE
# SUBDIRS = src bin docs lib
#endif
