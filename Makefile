EXE ?= megaknight
CARGO ?= cargo
RUSTFLAGS ?= -C target-cpu=native

.PHONY: all clean

all:
	rm -rf target/pgo-gen target/pgo-use target/pgo-data
	mkdir -p target/pgo-data
	sh profile_and_build.sh
	cp target/pgo-use/release/chessbot "$(EXE)"

clean:
	$(CARGO) clean
	rm -f "$(EXE)"
