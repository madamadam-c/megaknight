use std::{
    env,
    error::Error,
    ffi::OsString,
    fs::{self, File, OpenOptions},
    io::{BufWriter, Read, Write},
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

use cozy_chess::{Board, Color};
use crossbeam_channel::{Receiver, RecvTimeoutError, Sender, bounded, unbounded};
use flate2::read::GzDecoder;
use sfbinpack::{
    ChunkReader, TrainingDataEntry,
    chess::{color::Color as SfColor, r#move::MoveType, piecetype::PieceType},
};

use crate::engine::{Engine, SearchLimits, SearchRequest};

type AnyError = Box<dyn Error + Send + Sync>;

const DEFAULT_INPUT: &str = "datagen/positions.tar.zst";
const DEFAULT_OUTPUT: &str = "datagen/positions.txt";
const DEFAULT_NODES: u64 = 10_000;
const DEFAULT_HASH_MB: u64 = 4;
const MAX_BULLET_SCORE: i32 = 30_000;

const USAGE: &str = "\
Usage: chessbot bulk-eval --positions N [options]

Options:
  --input PATH         Input zstd/gzip/tar archive (default: datagen/positions.tar.zst)
  --output PATH        Bullet text output (default: datagen/positions.txt)
  --positions N        Number of output positions to evaluate (required)
  --nodes N            Search nodes per position (default: 10000)
  --threads N          Evaluation threads (default: available CPUs)
  --hash-mb N          Hash size per worker (default: 4)
  --include-tactical   Do not filter checks, captures, early plies, or extreme scores
  --overwrite          Replace existing output and partial files
  -h, --help           Show this help
";

#[derive(Debug, PartialEq, Eq)]
struct BulkConfig {
    input: PathBuf,
    output: PathBuf,
    positions: u64,
    nodes: u64,
    threads: usize,
    hash_mb: u64,
    include_tactical: bool,
    overwrite: bool,
}

struct WorkItem {
    fen: String,
    board: Board,
    result: &'static str,
}

enum BulkEvent {
    Evaluated(String),
    ReaderDone(Result<u64, String>),
    WorkerError(String),
    WorkerDone,
}

#[derive(Default)]
struct Counters {
    scanned: AtomicU64,
    filtered: AtomicU64,
    invalid: AtomicU64,
    queued: AtomicU64,
}

pub fn run_from_env() -> Result<(), AnyError> {
    let Some(config) = parse_args(env::args_os().skip(2))? else {
        print!("{USAGE}");
        return Ok(());
    };
    run(config)
}

fn parse_args<I>(args: I) -> Result<Option<BulkConfig>, String>
where
    I: IntoIterator<Item = OsString>,
{
    let mut config = BulkConfig {
        input: PathBuf::from(DEFAULT_INPUT),
        output: PathBuf::from(DEFAULT_OUTPUT),
        positions: 0,
        nodes: DEFAULT_NODES,
        threads: thread::available_parallelism().map_or(1, usize::from),
        hash_mb: DEFAULT_HASH_MB,
        include_tactical: false,
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
            "--input" => config.input = PathBuf::from(value(&mut args, "--input")?),
            "--output" => config.output = PathBuf::from(value(&mut args, "--output")?),
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
            "--include-tactical" => config.include_tactical = true,
            "--overwrite" => config.overwrite = true,
            _ => return Err(format!("unknown option: {argument}\n\n{USAGE}")),
        }
    }

    if config.positions == 0 {
        return Err(format!(
            "--positions is required and must be positive\n\n{USAGE}"
        ));
    }
    if config.input == config.output {
        return Err("input and output paths must differ".to_string());
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

fn run(config: BulkConfig) -> Result<(), AnyError> {
    if !config.input.is_file() {
        return Err(format!("input file does not exist: {}", config.input.display()).into());
    }

    let partial = partial_path(&config.output);
    prepare_output(&config.output, &partial, config.overwrite)?;
    let output = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&partial)?;
    let mut output = BufWriter::with_capacity(1024 * 1024, output);

    let cancelled = Arc::new(AtomicBool::new(false));
    let counters = Arc::new(Counters::default());
    let (work_tx, work_rx) = bounded::<WorkItem>(config.threads * 4);
    let (event_tx, event_rx) = unbounded::<BulkEvent>();

    let reader_handle = spawn_reader(
        config.input.clone(),
        config.positions,
        config.include_tactical,
        work_tx,
        event_tx.clone(),
        Arc::clone(&cancelled),
        Arc::clone(&counters),
    );

    let mut worker_handles = Vec::with_capacity(config.threads);
    for worker_id in 0..config.threads {
        worker_handles.push(spawn_worker(
            worker_id,
            config.nodes,
            config.hash_mb,
            work_rx.clone(),
            event_tx.clone(),
            Arc::clone(&cancelled),
        ));
    }
    drop(work_rx);
    drop(event_tx);

    eprintln!(
        "bulk-eval: {} positions, {} nodes, {} threads, {} MiB hash/thread",
        config.positions, config.nodes, config.threads, config.hash_mb
    );

    let started = Instant::now();
    let mut evaluated = 0u64;
    let mut workers_done = 0usize;
    let mut reader_done = false;
    let mut first_error = None;

    while !reader_done || workers_done < config.threads {
        match event_rx.recv_timeout(Duration::from_secs(1)) {
            Ok(BulkEvent::Evaluated(line)) => {
                writeln!(output, "{line}")?;
                evaluated += 1;
                if evaluated.is_multiple_of(1000) || evaluated == config.positions {
                    print_progress(evaluated, config.positions, &counters, started);
                }
            }
            Ok(BulkEvent::ReaderDone(result)) => {
                reader_done = true;
                if let Err(error) = result {
                    cancelled.store(true, Ordering::Relaxed);
                    first_error.get_or_insert(error);
                }
            }
            Ok(BulkEvent::WorkerError(error)) => {
                cancelled.store(true, Ordering::Relaxed);
                first_error.get_or_insert(error);
            }
            Ok(BulkEvent::WorkerDone) => workers_done += 1,
            Err(RecvTimeoutError::Timeout) => {
                print_progress(evaluated, config.positions, &counters, started)
            }
            Err(RecvTimeoutError::Disconnected) => break,
        }
    }

    let _ = reader_handle.join();
    for handle in worker_handles {
        let _ = handle.join();
    }
    output.flush()?;
    eprintln!();

    if let Some(error) = first_error {
        return Err(error.into());
    }
    if evaluated != config.positions {
        return Err(format!(
            "expected {} positions but evaluated {evaluated}",
            config.positions
        )
        .into());
    }

    drop(output);
    fs::rename(&partial, &config.output)?;
    let elapsed = started.elapsed().as_secs_f64();
    eprintln!(
        "wrote {} positions to {} in {:.1}s ({:.1} positions/s)",
        evaluated,
        config.output.display(),
        elapsed,
        evaluated as f64 / elapsed.max(f64::EPSILON)
    );
    Ok(())
}

fn spawn_reader(
    input: PathBuf,
    target: u64,
    include_tactical: bool,
    work_tx: Sender<WorkItem>,
    event_tx: Sender<BulkEvent>,
    cancelled: Arc<AtomicBool>,
    counters: Arc<Counters>,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        let result = read_positions(
            &input,
            target,
            include_tactical,
            &work_tx,
            &cancelled,
            &counters,
        )
        .map_err(|error| error.to_string());
        drop(work_tx);
        let _ = event_tx.send(BulkEvent::ReaderDone(result));
    })
}

fn read_positions(
    input: &Path,
    target: u64,
    include_tactical: bool,
    work_tx: &Sender<WorkItem>,
    cancelled: &AtomicBool,
    counters: &Counters,
) -> Result<u64, AnyError> {
    let file = File::open(input)?;
    let zstd = zstd::stream::read::Decoder::new(file)?;
    let gzip = GzDecoder::new(zstd);
    let mut archive = tar::Archive::new(gzip);
    let mut queued = 0u64;

    'archive: for entry in archive.entries()? {
        let mut entry = entry?;
        let is_binpack = entry.header().entry_type().is_file()
            && entry
                .path()?
                .extension()
                .is_some_and(|extension| extension == "binpack");
        if !is_binpack {
            continue;
        }

        let mut chunk = Vec::new();
        while read_binpack_chunk(&mut entry, &mut chunk)? {
            let mut reader = ChunkReader::default();
            while reader.has_next(&chunk) {
                if cancelled.load(Ordering::Relaxed) || queued == target {
                    break 'archive;
                }

                let source = reader.next(&chunk);
                counters.scanned.fetch_add(1, Ordering::Relaxed);
                if !include_tactical && !is_quiet(&source) {
                    counters.filtered.fetch_add(1, Ordering::Relaxed);
                    continue;
                }

                let Ok(fen) = source.pos.fen() else {
                    counters.invalid.fetch_add(1, Ordering::Relaxed);
                    continue;
                };
                let Ok(board) = Board::from_fen(&fen, false) else {
                    counters.invalid.fetch_add(1, Ordering::Relaxed);
                    continue;
                };
                let result =
                    white_result(source.result, source.pos.side_to_move() == SfColor::White)?;

                if work_tx.send(WorkItem { fen, board, result }).is_err() {
                    break 'archive;
                }
                queued += 1;
                counters.queued.store(queued, Ordering::Relaxed);
            }
        }
    }

    if queued != target && !cancelled.load(Ordering::Relaxed) {
        return Err(format!(
            "archive contained only {queued} usable positions; requested {target}"
        )
        .into());
    }
    Ok(queued)
}

fn read_binpack_chunk(reader: &mut impl Read, chunk: &mut Vec<u8>) -> Result<bool, AnyError> {
    let mut header = [0u8; 8];
    if reader.read(&mut header[..1])? == 0 {
        return Ok(false);
    }
    reader.read_exact(&mut header[1..])?;
    if &header[..4] != b"BINP" {
        return Err("invalid Stockfish binpack chunk header".into());
    }

    let size = u32::from_le_bytes(header[4..].try_into().unwrap()) as usize;
    if size > 100 * 1024 * 1024 {
        return Err(format!("invalid Stockfish binpack chunk size: {size}").into());
    }
    chunk.resize(size, 0);
    reader.read_exact(chunk)?;
    Ok(true)
}

fn is_quiet(entry: &TrainingDataEntry) -> bool {
    entry.ply >= 16
        && !entry.pos.is_checked(entry.pos.side_to_move())
        && entry.score.unsigned_abs() <= 10_000
        && entry.mv.mtype() == MoveType::Normal
        && entry.pos.piece_at(entry.mv.to()).piece_type() == PieceType::None
}

fn spawn_worker(
    worker_id: usize,
    nodes: u64,
    hash_mb: u64,
    work_rx: Receiver<WorkItem>,
    event_tx: Sender<BulkEvent>,
    cancelled: Arc<AtomicBool>,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        let mut engine = Engine::new();
        engine.set_hash_size_mb(hash_mb);

        while let Ok(item) = work_rx.recv() {
            if cancelled.load(Ordering::Relaxed) {
                break;
            }

            let (score, _) = search_score(&mut engine, &item.board, nodes, &cancelled);

            let Some(score) = score else {
                if cancelled.load(Ordering::Relaxed) {
                    break;
                }
                let _ = event_tx.send(BulkEvent::WorkerError(format!(
                    "worker {worker_id} did not complete depth 1 for FEN {}",
                    item.fen
                )));
                cancelled.store(true, Ordering::Relaxed);
                break;
            };
            let score = white_score(score, item.board.side_to_move() == Color::White);
            let line = format_record(&item.fen, score, item.result);
            if event_tx.send(BulkEvent::Evaluated(line)).is_err() {
                break;
            }
        }

        let _ = event_tx.send(BulkEvent::WorkerDone);
    })
}

fn search_score(
    engine: &mut Engine,
    board: &Board,
    nodes: u64,
    cancelled: &Arc<AtomicBool>,
) -> (Option<i32>, bool) {
    let mut request = SearchRequest {
        board: board.clone(),
        history: vec![board.hash()],
        limits: SearchLimits {
            nodes: Some(nodes),
            ..SearchLimits::default()
        },
        stop: Arc::clone(cancelled),
    };
    let mut score = None;
    engine.search(&request, |info| score = Some(info.score));

    let retry = score.is_none() && !cancelled.load(Ordering::Relaxed);
    if retry {
        request.limits.nodes = None;
        request.limits.depth = Some(1);
        engine.search(&request, |info| score = Some(info.score));
    }

    (score, retry)
}

fn white_score(side_to_move_score: i32, white_to_move: bool) -> i32 {
    let score = if white_to_move {
        side_to_move_score
    } else {
        -side_to_move_score
    };
    score.clamp(-MAX_BULLET_SCORE, MAX_BULLET_SCORE)
}

fn white_result(side_to_move_result: i16, white_to_move: bool) -> Result<&'static str, AnyError> {
    let white_result = if white_to_move {
        side_to_move_result
    } else {
        -side_to_move_result
    };
    match white_result {
        1 => Ok("1.0"),
        0 => Ok("0.5"),
        -1 => Ok("0.0"),
        _ => Err(format!("invalid binpack game result: {side_to_move_result}").into()),
    }
}

fn format_record(fen: &str, score: i32, result: &str) -> String {
    format!("{fen} | {score} | {result}")
}

fn print_progress(evaluated: u64, target: u64, counters: &Counters, started: Instant) {
    let elapsed = started.elapsed().as_secs_f64();
    let rate = evaluated as f64 / elapsed.max(f64::EPSILON);
    let remaining = target.saturating_sub(evaluated) as f64 / rate.max(f64::EPSILON);
    eprint!(
        "\revaluated {evaluated}/{target} ({:.1}%) | {:.1} pos/s | ETA {} | scanned {} | filtered {} | invalid {}    ",
        evaluated as f64 * 100.0 / target as f64,
        rate,
        format_duration(remaining),
        counters.scanned.load(Ordering::Relaxed),
        counters.filtered.load(Ordering::Relaxed),
        counters.invalid.load(Ordering::Relaxed),
    );
    let _ = std::io::stderr().flush();
}

fn format_duration(seconds: f64) -> String {
    if !seconds.is_finite() {
        return "--".to_string();
    }
    let seconds = seconds.max(0.0) as u64;
    let hours = seconds / 3600;
    let minutes = seconds % 3600 / 60;
    let seconds = seconds % 60;
    if hours > 0 {
        format!("{hours}h{minutes:02}m")
    } else if minutes > 0 {
        format!("{minutes}m{seconds:02}s")
    } else {
        format!("{seconds}s")
    }
}

fn partial_path(output: &Path) -> PathBuf {
    let mut path = output.as_os_str().to_os_string();
    path.push(".part");
    PathBuf::from(path)
}

fn prepare_output(output: &Path, partial: &Path, overwrite: bool) -> Result<(), AnyError> {
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

#[cfg(test)]
#[path = "tests/test_bulk.rs"]
mod tests;
