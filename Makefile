CXX = g++
CXXFLAGS = -Wall -std=c++11 -pthread -I.

SRC = MapReduceFramework.cpp Barrier/Barrier.cpp
OBJ = $(SRC:.cpp=.o)
LIB = libMapReduceFramework.a

all: $(LIB)

$(LIB): $(OBJ)
	ar rcs $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(LIB)

CODESRC = MapReduceFramework.cpp Barrier/Barrier.cpp Barrier/Barrier.h
TAR = tar
TARFLAGS = -cvf
TARNAME = ex3.tar
TARSRCS = $(CODESRC) Makefile README
tar:
	$(TAR) $(TARFLAGS) $(TARNAME) $(TARSRCS)
