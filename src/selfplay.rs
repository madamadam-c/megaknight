use std::{
    collections::{HashMap, HashSet},
    env,
    error::Error,
    ffi::OsString,
    fs::{self, File, OpenOptions},
    io::{BufRead, BufReader, BufWriter, Write},
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

use cozy_chess::{Board, Color, GameStatus, Move, Piece};
use crossbeam_channel::{RecvTimeoutError, Sender, bounded};

use crate::engine::{Engine, SearchLimits, SearchRequest};

type AnyError = Box<dyn Error + Send + Sync>;

const DEFAULT_OUTPUT: &str = "datagen/selfplay.txt";
const DEFAULT_OPENINGS: &str =
    "materials/fastchess/books/UHO_Lichess_4852_v1_noob_4moves_dedup.epd";
const DEFAULT_NODES: u64 = 1_000;
const DEFAULT_HASH_MB: u64 = 4;
const DEFAULT_MAX_PLIES: usize = 400;
const DEFAULT_WARMUP_PLIES: usize = 8;
const DEFAULT_RANDOM_PLIES: usize = 2;
const DEFAULT_WIN_SCORE: i32 = 1_500;
const DEFAULT_WIN_PLIES: usize = 8;
const MAX_BULLET_SCORE: i32 = 30_000;

const USAGE: &str = "\
Usage: chessbot selfplay --positions N [options]

Options:
  --output PATH         Bullet text output (default: datagen/selfplay.txt)
  --openings PATH       Opening EPD file (default: UHO + noob 4-move dedup)
  --positions N         Number of output positions (required)
  --nodes N             Search nodes per move (default: 1000)
  --threads N           Parallel games (default: available CPUs)
  --hash-mb N           Hash size per worker (default: 4)
  --max-plies N         Discard games unresolved after N plies (default: 400)
  --warmup-plies N      Do not emit the first N self-play plies (default: 8)
  --random-plies N      Quiet random opening moves for diversity (default: 2)
  --win-score N         Win adjudication threshold in centipawns (default: 1500)
  --win-plies N         Consecutive plies required for a win (default: 8)
  --seed N              Reproducible game seed (default: 12345)
  --overwrite           Replace existing output and partial files
  -h, --help            Show this help
";

#[derive(Clone, Debug, PartialEq, Eq)]
struct SelfplayConfig {
    output: PathBuf,
    openings: PathBuf,
    positions: u64,
    nodes: u64,
    threads: usize,
    hash_mb: u64,
    max_plies: usize,
    warmup_plies: usize,
    random_plies: usize,
    win_score: i32,
    win_plies: usize,
    seed: u64,
    overwrite: bool,
}

#[derive(Clone, Copy)]
enum Outcome {
    WhiteWin,
    Draw,
    BlackWin,
}

#[derive(Clone, Copy)]
enum Termination {
    Checkmate,
    RulesDraw,
    Repetition,
    MaxPlies,
    ScoreAdjudication,
}

impl Outcome {
    fn text(self) -> &'static str {
        match self {
            Self::WhiteWin => "1.0",
            Self::Draw => "0.5",
            Self::BlackWin => "0.0",
        }
    }
}

struct PositionRecord {
    fen: String,
    white_score: i32,
}

struct GameResult {
    positions: Vec<PositionRecord>,
    outcome: Outcome,
    termination: Termination,
    plies: usize,
}

enum Event {
    Game(GameResult),
    Error(String),
    WorkerDone,
}

#[derive(Default)]
struct Counters {
    games: AtomicU64,
    plies: AtomicU64,
    candidates: AtomicU64,
    white_wins: AtomicU64,
    draws: AtomicU64,
    black_wins: AtomicU64,
    checkmates: AtomicU64,
    rules_draws: AtomicU64,
    repetitions: AtomicU64,
    max_plies: AtomicU64,
    score_adjudications: AtomicU64,
}

pub fn run_from_env() -> Result<(), AnyError> {
    let Some(config) = parse_args(env::args_os().skip(2))? else {
        print!("{USAGE}");
        return Ok(());
    };
    run(config)
}

fn parse_args<I>(args: I) -> Result<Option<SelfplayConfig>, String>
where
    I: IntoIterator<Item = OsString>,
{
    let mut config = SelfplayConfig {
        output: PathBuf::from(DEFAULT_OUTPUT),
        openings: PathBuf::from(DEFAULT_OPENINGS),
        positions: 0,
        nodes: DEFAULT_NODES,
        threads: thread::available_parallelism().map_or(1, usize::from),
        hash_mb: DEFAULT_HASH_MB,
        max_plies: DEFAULT_MAX_PLIES,
        warmup_plies: DEFAULT_WARMUP_PLIES,
        random_plies: DEFAULT_RANDOM_PLIES,
        win_score: DEFAULT_WIN_SCORE,
        win_plies: DEFAULT_WIN_PLIES,
        seed: 12_345,
        overwrite: false,
    };
    let mut args = args.into_iter();

    while let Some(argument) = args.next() {
        let argument = argument
            .into_string()
            .map_err(|_| "arguments must be valid UTF-8".to_string())?;
        let value = |args: &mut I::IntoIter, option: &str| -> Result<String, String> {
            args.next()
                .ok_or_else(|| format!("missing value for {option}"))?
                .into_string()
                .map_err(|_| format!("value for {option} must be valid UTF-8"))
        };

        match argument.as_str() {
            "-h" | "--help" => return Ok(None),
            "--output" => config.output = PathBuf::from(value(&mut args, "--output")?),
            "--openings" => config.openings = PathBuf::from(value(&mut args, "--openings")?),
            "--positions" => {
                config.positions = parse_positive(&value(&mut args, "--positions")?, "--positions")?
            }
            "--nodes" => config.nodes = parse_positive(&value(&mut args, "--nodes")?, "--nodes")?,
            "--threads" => {
                config.threads = parse_positive(&value(&mut args, "--threads")?, "--threads")?
            }
            "--hash-mb" => {
                config.hash_mb = parse_positive(&value(&mut args, "--hash-mb")?, "--hash-mb")?
            }
            "--max-plies" => {
                config.max_plies = parse_positive(&value(&mut args, "--max-plies")?, "--max-plies")?
            }
            "--warmup-plies" => {
                config.warmup_plies =
                    parse_nonnegative(&value(&mut args, "--warmup-plies")?, "--warmup-plies")?
            }
            "--random-plies" => {
                config.random_plies =
                    parse_nonnegative(&value(&mut args, "--random-plies")?, "--random-plies")?
            }
            "--win-score" => {
                config.win_score = parse_positive(&value(&mut args, "--win-score")?, "--win-score")?
            }
            "--win-plies" => {
                config.win_plies = parse_positive(&value(&mut args, "--win-plies")?, "--win-plies")?
            }
            "--seed" => {
                config.seed = value(&mut args, "--seed")?
                    .parse()
                    .map_err(|_| "invalid --seed".to_string())?
            }
            "--overwrite" => config.overwrite = true,
            _ => return Err(format!("unknown option: {argument}\n\n{USAGE}")),
        }
    }

    if config.positions == 0 {
        return Err(format!(
            "--positions is required and must be positive\n\n{USAGE}"
        ));
    }
    Ok(Some(config))
}

fn parse_positive<T>(value: &str, option: &str) -> Result<T, String>
where
    T: std::str::FromStr + Default + PartialEq,
{
    let parsed = value
        .parse::<T>()
        .map_err(|_| format!("invalid value for {option}: {value}"))?;
    if parsed == T::default() {
        return Err(format!("{option} must be positive"));
    }
    Ok(parsed)
}

fn parse_nonnegative<T>(value: &str, option: &str) -> Result<T, String>
where
    T: std::str::FromStr,
{
    value
        .parse::<T>()
        .map_err(|_| format!("invalid value for {option}: {value}"))
}

fn run(config: SelfplayConfig) -> Result<(), AnyError> {
    let mut openings = load_openings(&config.openings)?;
    shuffle(&mut openings, config.seed);
    let openings = Arc::new(openings);
    let partial = partial_path(&config.output);
    prepare_output(&config.output, &partial, config.overwrite)?;
    let output = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&partial)?;
    let mut output = BufWriter::with_capacity(4 * 1024 * 1024, output);

    let cancelled = Arc::new(AtomicBool::new(false));
    let next_game = Arc::new(AtomicU64::new(0));
    let counters = Arc::new(Counters::default());
    let (event_tx, event_rx) = bounded::<Event>(config.threads * 2);
    let mut workers = Vec::with_capacity(config.threads);

    for worker_id in 0..config.threads {
        let config = config.clone();
        let openings = Arc::clone(&openings);
        let event_tx = event_tx.clone();
        let cancelled = Arc::clone(&cancelled);
        let next_game = Arc::clone(&next_game);
        let counters = Arc::clone(&counters);
        workers.push(thread::spawn(move || {
            worker_loop(
                worker_id, &config, &openings, &event_tx, &cancelled, &next_game, &counters,
            )
        }));
    }
    drop(event_tx);

    eprintln!(
        "selfplay: {} positions, {} nodes/move, {} threads, {} openings",
        config.positions,
        config.nodes,
        config.threads,
        openings.len()
    );

    let started = Instant::now();
    let mut written = 0u64;
    let mut workers_done = 0usize;
    let mut first_error = None;
    while workers_done < config.threads {
        match event_rx.recv_timeout(Duration::from_secs(1)) {
            Ok(Event::Game(game)) => {
                let remaining = (config.positions - written) as usize;
                for position in game.positions.into_iter().take(remaining) {
                    writeln!(
                        output,
                        "{} | {} | {}",
                        position.fen,
                        position.white_score,
                        game.outcome.text()
                    )?;
                    written += 1;
                }
                if written == config.positions {
                    cancelled.store(true, Ordering::Relaxed);
                }
                if written.is_multiple_of(1_000) || written == config.positions {
                    print_progress(written, &config, &counters, started);
                }
            }
            Ok(Event::Error(error)) => {
                first_error.get_or_insert(error);
                cancelled.store(true, Ordering::Relaxed);
            }
            Ok(Event::WorkerDone) => workers_done += 1,
            Err(RecvTimeoutError::Timeout) => print_progress(written, &config, &counters, started),
            Err(RecvTimeoutError::Disconnected) => break,
        }
    }

    for worker in workers {
        let _ = worker.join();
    }
    output.flush()?;
    eprintln!();

    if let Some(error) = first_error {
        return Err(error.into());
    }
    if written != config.positions {
        return Err(format!(
            "expected {} positions but wrote {written}",
            config.positions
        )
        .into());
    }

    drop(output);
    fs::rename(&partial, &config.output)?;
    let elapsed = started.elapsed().as_secs_f64();
    eprintln!(
        "wrote {written} positions to {} in {:.2}s ({:.1} positions/s)",
        config.output.display(),
        elapsed,
        written as f64 / elapsed.max(f64::EPSILON)
    );
    Ok(())
}

fn load_openings(path: &Path) -> Result<Vec<Board>, AnyError> {
    let file = BufReader::new(File::open(path)?);
    let mut openings = Vec::new();
    for line in file.lines() {
        let line = line?;
        let fields = line.split_whitespace().take(4).collect::<Vec<_>>();
        if fields.len() != 4 {
            continue;
        }
        let fen = format!("{} 0 1", fields.join(" "));
        if let Ok(board) = Board::from_fen(&fen, false) {
            openings.push(board);
        }
    }
    if openings.is_empty() {
        return Err(format!(
            "opening file contained no valid positions: {}",
            path.display()
        )
        .into());
    }
    Ok(openings)
}

fn worker_loop(
    worker_id: usize,
    config: &SelfplayConfig,
    openings: &[Board],
    event_tx: &Sender<Event>,
    cancelled: &Arc<AtomicBool>,
    next_game: &AtomicU64,
    counters: &Counters,
) {
    let mut engine = Engine::new();
    engine.set_hash_size_mb(config.hash_mb);

    while !cancelled.load(Ordering::Relaxed) {
        let game_id = next_game.fetch_add(1, Ordering::Relaxed);
        let mut rng = SplitMix64::new(config.seed ^ game_id.wrapping_mul(0x9e37_79b9_7f4a_7c15));
        let opening_index = (game_id % openings.len() as u64) as usize;
        let mut board = openings[opening_index].clone();
        randomize_opening(&mut board, config.random_plies, &mut rng);

        match play_game(&mut engine, board, config, cancelled) {
            Some(game) => {
                counters.games.fetch_add(1, Ordering::Relaxed);
                counters
                    .plies
                    .fetch_add(game.plies as u64, Ordering::Relaxed);
                counters
                    .candidates
                    .fetch_add(game.positions.len() as u64, Ordering::Relaxed);
                match game.outcome {
                    Outcome::WhiteWin => &counters.white_wins,
                    Outcome::Draw => &counters.draws,
                    Outcome::BlackWin => &counters.black_wins,
                }
                .fetch_add(1, Ordering::Relaxed);
                match game.termination {
                    Termination::Checkmate => &counters.checkmates,
                    Termination::RulesDraw => &counters.rules_draws,
                    Termination::Repetition => &counters.repetitions,
                    Termination::MaxPlies => &counters.max_plies,
                    Termination::ScoreAdjudication => &counters.score_adjudications,
                }
                .fetch_add(1, Ordering::Relaxed);
                if event_tx.send(Event::Game(game)).is_err() {
                    break;
                }
            }
            None if cancelled.load(Ordering::Relaxed) => break,
            None => {
                let _ = event_tx.send(Event::Error(format!(
                    "worker {worker_id} could not complete a depth-one search"
                )));
                break;
            }
        }
    }

    let _ = event_tx.send(Event::WorkerDone);
}

fn play_game(
    engine: &mut Engine,
    mut board: Board,
    config: &SelfplayConfig,
    cancelled: &Arc<AtomicBool>,
) -> Option<GameResult> {
    engine.new_game();
    let mut positions = Vec::new();
    let mut emitted = HashSet::new();
    let mut history = vec![board.hash()];
    let mut repetitions = HashMap::from([(board.hash(), 1u8)]);
    let mut white_streak = 0;
    let mut black_streak = 0;
    let mut plies = 0;

    let (outcome, termination) = loop {
        if cancelled.load(Ordering::Relaxed) {
            return None;
        }
        match board.status() {
            GameStatus::Won => {
                let outcome = if board.side_to_move() == Color::White {
                    Outcome::BlackWin
                } else {
                    Outcome::WhiteWin
                };
                break (outcome, Termination::Checkmate);
            }
            GameStatus::Drawn => break (Outcome::Draw, Termination::RulesDraw),
            GameStatus::Ongoing => {}
        }
        if repetitions.get(&board.hash()).copied().unwrap_or(0) >= 3 {
            break (Outcome::Draw, Termination::Repetition);
        }
        if plies == config.max_plies {
            break (Outcome::Draw, Termination::MaxPlies);
        }

        let (mv, score) = search_position(engine, &board, &history, config.nodes, cancelled)?;
        let white_score = if board.side_to_move() == Color::White {
            score
        } else {
            -score
        };

        if plies >= config.warmup_plies
            && board.checkers().is_empty()
            && board.occupied().len() > 2
            && score.unsigned_abs() <= 10_000
            && is_quiet_move(&board, mv)
            && emitted.insert(board.hash())
        {
            positions.push(PositionRecord {
                fen: board.to_string(),
                white_score: white_score.clamp(-MAX_BULLET_SCORE, MAX_BULLET_SCORE),
            });
        }

        if white_score >= config.win_score {
            white_streak += 1;
            black_streak = 0;
        } else if white_score <= -config.win_score {
            black_streak += 1;
            white_streak = 0;
        } else {
            white_streak = 0;
            black_streak = 0;
        }
        if white_streak >= config.win_plies {
            break (Outcome::WhiteWin, Termination::ScoreAdjudication);
        }
        if black_streak >= config.win_plies {
            break (Outcome::BlackWin, Termination::ScoreAdjudication);
        }

        board.play_unchecked(mv);
        plies += 1;
        history.push(board.hash());
        *repetitions.entry(board.hash()).or_insert(0) += 1;
    };

    if matches!(termination, Termination::MaxPlies) {
        positions.clear();
    }

    Some(GameResult {
        positions,
        outcome,
        termination,
        plies,
    })
}

fn search_position(
    engine: &mut Engine,
    board: &Board,
    history: &[u64],
    nodes: u64,
    cancelled: &Arc<AtomicBool>,
) -> Option<(Move, i32)> {
    let mut request = SearchRequest {
        board: board.clone(),
        history: history.to_vec(),
        limits: SearchLimits {
            nodes: Some(nodes),
            ..SearchLimits::default()
        },
        stop: Arc::clone(cancelled),
    };
    let mut score = None;
    let mut result = engine.search(&request, |info| score = Some(info.score));
    if score.is_none() && !cancelled.load(Ordering::Relaxed) {
        request.limits.nodes = None;
        request.limits.depth = Some(1);
        result = engine.search(&request, |info| score = Some(info.score));
    }
    result.best_move.zip(score)
}

fn randomize_opening(board: &mut Board, plies: usize, rng: &mut SplitMix64) {
    for _ in 0..plies {
        let mut quiets = Vec::new();
        board.generate_moves(|moves| {
            quiets.extend(moves.into_iter().filter(|&mv| is_quiet_move(board, mv)));
            false
        });
        if quiets.is_empty() {
            return;
        }
        let mv = quiets[rng.next_u64() as usize % quiets.len()];
        board.play_unchecked(mv);
    }
}

fn shuffle<T>(values: &mut [T], seed: u64) {
    let mut rng = SplitMix64::new(seed);
    for end in (1..values.len()).rev() {
        let index = rng.next_u64() as usize % (end + 1);
        values.swap(end, index);
    }
}

fn is_quiet_move(board: &Board, mv: Move) -> bool {
    mv.promotion.is_none()
        && board.color_on(mv.to).is_none()
        && !(board.piece_on(mv.from) == Some(Piece::Pawn) && mv.from.file() != mv.to.file())
}

struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }
}

fn partial_path(output: &Path) -> PathBuf {
    let mut name = output.as_os_str().to_os_string();
    name.push(".part");
    PathBuf::from(name)
}

fn prepare_output(output: &Path, partial: &Path, overwrite: bool) -> Result<(), AnyError> {
    if let Some(parent) = output.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }
    for path in [output, partial] {
        if path.exists() {
            if overwrite {
                fs::remove_file(path)?;
            } else {
                return Err(
                    format!("output already exists: {}; use --overwrite", path.display()).into(),
                );
            }
        }
    }
    Ok(())
}

fn print_progress(written: u64, config: &SelfplayConfig, counters: &Counters, started: Instant) {
    let elapsed = started.elapsed().as_secs_f64();
    let rate = written as f64 / elapsed.max(f64::EPSILON);
    let remaining = config.positions.saturating_sub(written) as f64 / rate.max(f64::EPSILON);
    eprint!(
        "\rwrote {written}/{} ({:.1}%) | {:.1} pos/s | ETA {} | games {} | plies {} | W/D/L {}/{}/{} | term C/R/3/M/A {}/{}/{}/{}/{}    ",
        config.positions,
        written as f64 * 100.0 / config.positions as f64,
        rate,
        format_duration(remaining),
        counters.games.load(Ordering::Relaxed),
        counters.plies.load(Ordering::Relaxed),
        counters.white_wins.load(Ordering::Relaxed),
        counters.draws.load(Ordering::Relaxed),
        counters.black_wins.load(Ordering::Relaxed),
        counters.checkmates.load(Ordering::Relaxed),
        counters.rules_draws.load(Ordering::Relaxed),
        counters.repetitions.load(Ordering::Relaxed),
        counters.max_plies.load(Ordering::Relaxed),
        counters.score_adjudications.load(Ordering::Relaxed),
    );
    let _ = std::io::stderr().flush();
}

fn format_duration(seconds: f64) -> String {
    if !seconds.is_finite() {
        return "unknown".to_string();
    }
    let seconds = seconds.max(0.0) as u64;
    format!(
        "{:02}:{:02}:{:02}",
        seconds / 3600,
        seconds / 60 % 60,
        seconds % 60
    )
}

#[cfg(test)]
#[path = "tests/test_selfplay.rs"]
mod tests;
