.PHONY: clean

main: housing.h housing.c main.c
	clang -Wall -g -O0 -o main housing.c main.c -lm -lpthread

clean:
	rm -rf main *.o *~ *dSYM