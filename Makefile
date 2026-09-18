EXE ?= megaknight
CARGO ?= cargo
RUSTFLAGS ?= -C target-cpu=native

.PHONY: all clean

all:
	$(CARGO) build --release --locked
	cp target/release/chessbot "$(EXE)"

clean:
	$(CARGO) clean
	rm -f "$(EXE)"
