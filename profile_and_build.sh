CARGO_TARGET_DIR="$PWD/target/pgo-gen" \
RUSTFLAGS="-C target-cpu=native -C profile-generate=$PWD/target/pgo-data" \
cargo build --release --locked

target/pgo-gen/release/chessbot bench

LLVM_PROFDATA="$(rustc --print sysroot)/lib/rustlib/x86_64-unknown-linux-gnu/bin/llvm-profdata"

"$LLVM_PROFDATA" merge \
    -o target/pgo-data/merged.profdata \
    target/pgo-data/*.profraw

CARGO_TARGET_DIR="$PWD/target/pgo-use" \
RUSTFLAGS="-C target-cpu=native -C profile-use=$PWD/target/pgo-data/merged.profdata" \
cargo build --release --locked