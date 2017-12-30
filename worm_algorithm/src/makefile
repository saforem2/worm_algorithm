CC:=g++ -std=c++11 -g -Wall -O3

worm_ising_2d: worm_ising_2d.o
	${CC} -o $@.x $^

%.o : %.cpp
	${CC} -c $<

clean:
	rm *.o *.x
