CC=gcc -Wall -g

all: screaming_computer

screaming_computer: screaming_computer.o
	$(CC) $^ -o $@ -lSDL2

%.o: %.c $(HEADERS)
	$(CC) $< -c -o $@

clean:
	rm -f screaming_computer *.o

