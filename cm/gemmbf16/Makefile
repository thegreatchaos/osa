#OFLG=-O3
OFLG=-O0 -g

CFLG=${OFLG} -Isrc #-std=c++11

bin/bench:bin/bench.cpp.o bin/gemm.cpp.cm
	mkdir -p bin
	g++ $^ -o $@

bin/bench.cpp.o:src/bench.cpp
	mkdir -p bin
	g++ ${CFLG} -c $^ -o $@
bin/gemm.cpp.cm:src/gemm.cpp.o
	mkdir -p bin
	cmc -march=ptl ${CFLG} $^ -o $@
clean:
	-rm -rf bin
run:./bin/bench
	./bin/bench 10 20 30
