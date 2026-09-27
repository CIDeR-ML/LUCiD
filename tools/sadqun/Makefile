# SadQun needs ROOT and nothing else.
CXXFLAGS := -O2 -std=c++17 -Wall $(shell root-config --cflags)
LDLIBS   := $(shell root-config --libs)

sadqun: main.o sadqun.o
	$(CXX) $^ $(LDLIBS) -o $@

main.o sadqun.o: sadqun.h

clean:
	rm -f sadqun *.o
.PHONY: clean
