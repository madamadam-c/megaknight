EXE ?= megaknight
CARGO ?= cargo
HOST ?= x86_64-unknown-linux-gnu
LLVM_PROFDATA ?= $(shell rustc --print sysroot)/lib/rustlib/$(HOST)/bin/llvm-profdata
PGO_DIR ?= target/pgo

.PHONY: all pgo clean

all:
	$(CARGO) build --release --locked
	cp target/release/chessbot "$(EXE)"

pgo:
	rm -rf "$(PGO_DIR)"
	mkdir -p "$(PGO_DIR)/data"
	CARGO_TARGET_DIR="$(PGO_DIR)/generate" RUSTFLAGS="-C target-cpu=native -C profile-generate=$(abspath $(PGO_DIR)/data)" $(CARGO) build --release --locked
	for run in 1 2 3 4 5; do "$(PGO_DIR)/generate/release/chessbot" bench > /dev/null; done
	"$(LLVM_PROFDATA)" merge -o "$(PGO_DIR)/merged.profdata" "$(PGO_DIR)"/data/*.profraw
	CARGO_TARGET_DIR="$(PGO_DIR)/use" RUSTFLAGS="-C target-cpu=native -C profile-use=$(abspath $(PGO_DIR)/merged.profdata)" $(CARGO) build --release --locked
	cp "$(PGO_DIR)/use/release/chessbot" "$(EXE)"

clean:
	$(CARGO) clean
	rm -f "$(EXE)"
