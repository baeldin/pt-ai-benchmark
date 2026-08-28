# POSIX build. Windows/MSVC: use build.bat instead.
CXX ?= g++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra
LDFLAGS ?= -pthread

pt: main.cpp
	$(CXX) $(CXXFLAGS) main.cpp -o $@ $(LDFLAGS)

run: pt
	./pt

clean:
	rm -f pt output.ppm

.PHONY: run clean
