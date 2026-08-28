# Chessbot Development Guide

This is a small Rust chess engine that speaks the UCI protocol. The project is
currently focused on getting the engine lifecycle, search limits, and protocol
behavior correct before adding stronger chess algorithms.

The most important boundary in the project is:

```text
UCI text -> Rust command/data -> Engine search -> Rust result/data -> UCI text
```

The search code should work with Rust data structures, not with strings such as
`"go depth 5"`. The UCI layer should be the only code that understands those
strings or writes protocol output.

## Quick Start

Run the checks:

```text
cargo check
cargo test
cargo fmt --check
```

Run the engine manually:

```text
cargo run
```

Then type UCI commands, one per line:

```text
uci
isready
position startpos
go depth 3
quit
```

The engine prints protocol responses such as:

```text
id name chessbot
id author me
uciok
readyok
info depth 1 score cp 0 nodes 41 time 0 nps 0 pv a2a3
bestmove a2a3
```

Do not put ordinary debugging output on stdout. A UCI GUI expects stdout to
contain only valid UCI messages. Use stderr for temporary debugging output.

## Repository Layout

```text
Cargo.toml          Rust package and dependency configuration
Cargo.lock          Locked dependency versions
src/main.rs         UCI protocol, board state, threads, and channels
src/engine.rs       Search limits, time control, iterative deepening, search
src/evaluate.rs     Static material evaluation
src/tests/           Unit tests
materials/          Offline reference material and external projects
```

The `materials/` directory is reference material. It is not part of the
engine's runtime code.

## Thread Architecture

There are three execution contexts.

```text
stdin
  |
  v
input reader thread
  |
  | mpsc::channel<String>
  v
main coordinator thread -------- stdout
  |
  | mpsc::channel<Command>
  v
search worker thread
  |
  | mpsc::channel<WorkerOutput>
  +--------------------------------> main coordinator
```

### Input reader thread

`spawn_input_reader` in `src/main.rs` reads stdin with the blocking
`BufRead::lines` iterator. Every line is sent to the main coordinator through
an `mpsc` channel.

This thread exists because reading stdin is blocking. If the main coordinator
read stdin directly while a search was running, it would not be able to process
search output or respond promptly to `stop` and `isready`.

### Main coordinator

The `main` function owns:

- The current UCI board.
- UCI command parsing.
- The currently active search token.
- The command and output channels.
- All stdout output.

The main loop first drains all worker output with `try_recv`. It then waits up
to 10 milliseconds for an input line with `recv_timeout`.

The timeout is important. A blocking receive would prevent the main thread
from printing `info` and `bestmove` while the worker is searching.

### Search worker

The worker creates one persistent `Engine` and owns it for its entire lifetime.
It receives `Command` values and runs one search at a time.

Keeping the engine inside the worker means future persistent state can be put
inside `Engine`, for example:

- Transposition tables.
- History heuristics.
- Killer moves.
- Search configuration.
- Evaluation caches.

The current `Engine` is empty, but the ownership model is already in place.

## Command Types

`Command` is the internal message from the main coordinator to the worker:

```rust
enum Command {
    Go { id: u64, request: SearchRequest },
    NewGame,
    Quit,
}
```

`WorkerOutput` is the message in the opposite direction:

```rust
enum WorkerOutput {
    Info { id: u64, info: SearchInfo },
    BestMove { id: u64, result: SearchResult },
}
```

The search ID identifies which search produced a message. The main thread
suppresses stale `info` messages from an older search after a newer `go` has
been received.

Final `bestmove` messages are still printed for older searches so that every
accepted search can finish its UCI lifecycle.

## Board Ownership

The main thread owns the current board:

```rust
let mut board = Board::default();
```

When `go` is received, the board is cloned into a `SearchRequest`. The worker
therefore searches a snapshot of the position. Later `position` commands do
not mutate the board that an already-running search is using.

This separation is intentional:

```text
main board       = current protocol position
request board    = immutable snapshot for one search
child board      = temporary clone for one candidate move
```

## Position Parsing

`parse_position` supports these forms:

```text
position startpos
position startpos moves e2e4 e7e5 g1f3
position fen <six FEN fields>
position fen <six FEN fields> moves e2e4 ...
```

The parser builds a temporary `next_board`. For a starting position it uses
`Board::default`. For a FEN position it uses `Board::from_fen`.

Every following UCI move is:

1. Parsed with `parse_uci_move`.
2. Checked and applied with `Board::try_play`.

The real current board is replaced only after the entire command succeeds.
This prevents an invalid FEN or move from leaving the board partially updated.

For engine-generated moves, `Board::generate_moves` has already produced legal
moves, so the search uses `Board::play`. For external text input, use
`try_play`, not `play`, because external input may be invalid.

## Search Request

`SearchRequest` in `src/engine.rs` describes one complete search:

```rust
pub struct SearchRequest {
    pub board: Board,
    pub limits: SearchLimits,
    pub stop: Arc<AtomicBool>,
}
```

The search does not receive a raw UCI command. It receives a board, structured
limits, and a search-specific cancellation flag.

## Search Limits

`SearchLimits` represents all the information from a UCI `go` command:

```rust
pub struct SearchLimits {
    pub depth: Option<i32>,
    pub nodes: Option<u64>,
    pub movetime: Option<u64>,
    pub wtime: Option<u64>,
    pub btime: Option<u64>,
    pub winc: Option<u64>,
    pub binc: Option<u64>,
    pub movestogo: Option<u64>,
    pub infinite: bool,
    pub searchmoves: Vec<Move>,
}
```

All time values are milliseconds.

The fields are optional because UCI can combine limits:

```text
go wtime 30000 btime 30000 winc 100 binc 100
go movetime 1000
go infinite
```

The search should honor every applicable limit and stop at the first one that
becomes active.

### Depth

`depth` is the maximum iterative-deepening depth. It is not a direct call to
one fixed-depth search anymore.

```text
go depth 4
```

searches depths 1, 2, 3, and 4.

### Nodes

`nodes` limits the number of visited search nodes. The node counter is in
`SearchContext` and is checked during recursive search.

### Movetime

`movetime` requests a direct time budget for the current move. The engine
reserves five milliseconds from the requested value for search shutdown and
printing the result.

### Clocks

`wtime` and `btime` are remaining clock times. `winc` and `binc` are increments.
The engine uses the clock for the side to move.

The current basic allocation is approximately:

```text
remaining time / moves remaining + 75% of the increment
```

If `movestogo` is absent, the engine assumes 30 moves remain. It also reserves
the larger of 1% of the remaining time or 5 milliseconds.

This is deliberately a simple first time manager, not a sophisticated chess
clock model.

### Infinite

With `go infinite`, the engine does not create a clock deadline. It continues
iterative deepening until `stop` is received.

### Search moves

`searchmoves` restricts the legal root moves considered by the engine:

```text
go searchmoves e2e4 d2d4
```

The search still generates all legal root moves, then retains only the listed
moves.

## Search Cancellation

Each `go` creates a fresh `Arc<AtomicBool>`:

```rust
let stop = Arc::new(AtomicBool::new(false));
```

The same flag is stored in the main coordinator and in the search request.

When UCI sends `stop`, the main thread sets that specific flag to `true`.
The stop command is not sent through the worker command channel. This matters
because the worker may be busy inside `Engine::search` and unable to receive
another queued command.

The search checks the flag through `SearchContext::should_stop` and returns
upward through recursion using `Option` and the `?` operator.

When a search is interrupted halfway through a depth, that incomplete depth is
discarded. The engine returns the best move from the most recently completed
depth. If no depth completed, it returns the first legal root move as a
fallback.

## Search Context

`SearchContext` contains state that belongs to exactly one search:

```rust
struct SearchContext {
    limits: SearchLimits,
    stop: Arc<AtomicBool>,
    start: Instant,
    soft_deadline: Option<Instant>,
    hard_deadline: Option<Instant>,
    nodes: u64,
}
```

Do not put per-search state in `Engine`. `Engine` survives across searches,
while `SearchContext` is created fresh for every `SearchRequest`.

### `should_stop`

`should_stop` checks:

1. The external atomic cancellation flag.
2. The node limit.
3. The hard time deadline.

The node counter is incremented every time the check is called. The clock is
checked on the first node and every 1024 nodes to avoid calling the system
clock too frequently.

This means the hard deadline is approximate by a small amount. A future engine
can adjust the check interval if needed.

## Time Deadlines

The time manager creates two deadlines:

```text
soft deadline = 80% of the calculated budget
hard deadline = 100% of the calculated budget
```

The hard deadline can interrupt recursive search immediately after it is
noticed.

The soft deadline is checked only after a complete iterative-deepening depth.
That gives the engine a chance to finish the current depth and return a stable
result rather than abandoning it halfway through.

## Iterative Deepening

`Engine::search` starts at depth 1 and repeatedly searches deeper positions.

The search flow is:

```text
generate legal root moves
choose the first root move as a fallback

for depth = 1, 2, 3, ...:
    check stop/time/node limits
    search every allowed root move at this depth

    if this depth was interrupted:
        discard this depth
        finish

    save the completed depth's best move
    report SearchInfo

    if soft deadline expired:
        finish
```

The maximum depth comes from `SearchLimits.depth`. If no depth was supplied,
the practical stopping condition is time, nodes, or external `stop`.

If there is no depth, time, node, or stop command, the search will continue
indefinitely.

## Root Search

`root_search` evaluates all permitted root moves for one depth.

For every root move it:

1. Clones the current board.
2. Plays the root move.
3. Searches the child position to `depth - 1`.
4. Negates the returned score.
5. Keeps the highest score.

The root result is either:

```text
Some(best_move, score)
```

if the complete depth finished, or `None` if cancellation interrupted it.

The `None` result is what prevents an incomplete depth from replacing the
previous completed result.

## Recursive Search

The current recursive search is a simple negamax form of minimax.

It stops at a leaf when:

```rust
depth <= 0
```

or when the game is no longer ongoing.

At a leaf it calls `evaluate::eval`.

Otherwise it generates legal moves and searches each child:

```text
score = max(-search(child) for every legal child)
```

The negation works because evaluation is from the side-to-move perspective.
After a move, the opponent is the side to move, so the child score must be
negated when viewed by the parent.

The current recursive search does not yet have alpha-beta pruning. It searches
every move at every level, which is correct but becomes expensive quickly.

## Evaluation

`src/evaluate.rs` currently evaluates only material.

Piece values are:

```text
Pawn    100
Knight  300
Bishop  300
Rook    500
Queen   900
King    20000
```

The evaluator adds the material of the side to move and subtracts the
opponent's material.

Therefore the score is positive when the side to move has more material.

Special cases:

- Drawn position: `0`.
- Won position: `-100000` from the current side's perspective.

The evaluator does not currently include positional bonuses, pawn structure,
king safety, mobility, passed pawns, or neural-network evaluation.

## Search Information

After every completed depth, the engine reports:

```rust
pub struct SearchInfo {
    pub depth: i32,
    pub score: i32,
    pub nodes: u64,
    pub elapsed: Duration,
    pub pv: Vec<Move>,
}
```

The worker sends this through the output channel. The main coordinator formats
it as a UCI `info` line.

The current `pv` contains only the best root move:

```rust
pv: vec![best_move]
```

It is not yet a complete principal variation.

Scores are always printed as centipawn scores:

```text
score cp N
```

Mate scores are not yet printed using UCI's `score mate N` form.

## UCI Commands

### `uci`

Prints the engine identity and `uciok`:

```text
id name chessbot
id author me
uciok
```

No configurable options are advertised yet.

### `isready`

Prints `readyok` immediately. The main coordinator remains responsive while
the worker searches, so this can be answered during a search.

### `position`

Updates the main coordinator's current board. It does not directly affect a
search that already received a cloned board.

### `go`

Parses search limits, creates a new search token, clones the current board, and
sends a `SearchRequest` to the worker.

If a previous search is active, its cancellation flag is set first.

### `stop`

Sets the current search's cancellation flag. The worker notices it through the
search context.

### `ucinewgame`

Stops the active search and queues `Engine::new_game`.

The current `new_game` method is empty because `Engine` has no persistent
state yet.

### `quit`

Stops the active search, sends `Quit` to the worker, exits the coordinator loop,
and joins the worker thread.

### `setoption`

Currently ignored. No options are advertised, so this is acceptable for the
current engine. Do not advertise an option until its behavior is actually
implemented.

## Safe Development Rules

### Adding a new UCI `go` parameter

Add it in three places:

1. Add a field to `SearchLimits`.
2. Parse it in `parse_go`.
3. Use it in `SearchContext`, time management, or root search.

Do not pass raw command strings into `Engine`.

### Adding per-search data

Put it in `SearchContext`, not `Engine`.

Examples:

- Node count.
- Search start time.
- Current limits.
- Cancellation state.
- Current iteration statistics.

### Adding persistent engine data

Put it in `Engine`.

Examples:

- Transposition table.
- History table.
- Killer move table.
- Persistent configuration.

Use `Engine::new_game` to reset state that belongs to a particular game.

### Printing output

Only the UCI layer should print to stdout. The engine should report structured
data using return values or callbacks.

Use `info string ...` for valid UCI debugging messages, or use stderr for
private debugging.

### Handling external moves

Use `parse_uci_move` and `try_play` for moves received from UCI. Never call
`play` on untrusted external input because `play` asserts that the move is
legal and can panic.

### Handling generated moves

Moves returned by `Board::generate_moves` are legal. It is appropriate to use
`play` for those moves inside search.

## Current Limitations

The protocol and search lifecycle are more complete than the chess strength.
The following are intentionally not implemented yet:

- Alpha-beta pruning.
- Move ordering.
- Quiescence search.
- Transposition tables.
- Full principal variation tracking.
- Mate-distance scores.
- Positional evaluation.
- NNUE evaluation.
- Pondering and `ponderhit`.
- MultiPV.
- Configurable UCI options.
- Chess960 support.
- Syzygy tablebases.

The first search-strength improvements should preserve the existing search
boundaries. For example, alpha-beta should be added inside the recursive
search, while `SearchRequest`, `SearchContext`, cancellation, and UCI output
should continue to work unchanged.

## Testing

The current tests are in `src/tests`.

`test_evaluate` verifies that the starting position evaluates to zero.

`test_uci` verifies that:

- Multiple `go` limits are parsed together.
- Complete move history is applied.
- Invalid position input does not replace the current board.

Before changing search or UCI behavior, test at least these manually:

```text
uci
isready
position startpos
go depth 2
```

```text
position startpos
go movetime 1000
```

```text
position startpos
go infinite
stop
```

For every `go`, exactly one `bestmove` should eventually be produced.

Useful future tests include:

- A legal best move for a known position.
- `go nodes N` stopping near the requested node count.
- Clock-based searches respecting the hard deadline.
- `isready` while searching.
- `ucinewgame` resetting future engine state.
- A stop during a deep search returning the last completed iteration's move.

## Recommended Next Development Order

When continuing development, a sensible order is:

1. Add alpha-beta pruning without changing the UCI boundary.
2. Add move ordering.
3. Track a full principal variation.
4. Add quiescence search.
5. Improve the static evaluation.
6. Add a transposition table to `Engine`.
7. Add UCI options only when they control real behavior.

Keep the following design intact while doing this:

```text
main.rs owns UCI and protocol I/O
Engine owns persistent search state
SearchContext owns one search's temporary state
SearchLimits describes requested limits
SearchInfo describes progress
SearchResult describes completion
```

before implementing lmr or rfp or qs see pruning or basically any search optimisations
except for nmp, the engine was at 2500 rating (compared to stash 18 at 10+0.1 tc) roughly