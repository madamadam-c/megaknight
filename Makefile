EXE ?= megaknight
CARGO ?= cargo

.PHONY: all clean

all:
	$(CARGO) build --release --locked
	cp target/release/chessbot "$(EXE)"

clean:
	$(CARGO) clean
	rm -f "$(EXE)"
