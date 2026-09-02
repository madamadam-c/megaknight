use std::{
    io::{self, BufRead, Write},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread::{self, JoinHandle},
};

use cozy_chess::{
    Board, Move,
    util::{display_uci_move, parse_uci_move},
};

use crate::engine::{Engine, SearchInfo, SearchLimits, SearchRequest, SearchResult};

mod bulk;
mod engine;
mod evaluate;
mod history;
mod nnue;
mod selfplay;
mod transposition;

#[cfg(test)]
#[path = "tests/test_evaluate.rs"]
mod tests;

enum Command {
    Go { id: u64, request: SearchRequest },
    SetHash(u64),
    NewGame,
    Quit,
}

enum WorkerOutput {
    Info {
        id: u64,
        board: Board,
        info: SearchInfo,
    },
    BestMove {
        id: u64,
        board: Board,
        result: SearchResult,
    },
}

enum MainEvent {
    Input(String),
    InputClosed,
    Worker(WorkerOutput),
}

fn parse_position(line: &str, board: &mut Board, history: &mut Vec<u64>) {
    let parts: Vec<&str> = line.split_whitespace().collect();
    let Some(position_type) = parts.get(1) else {
        return;
    };

    let mut next_board = match *position_type {
        "startpos" => Board::default(),
        "fen" => {
            let fen_end = parts
                .iter()
                .position(|part| *part == "moves")
                .unwrap_or(parts.len());
            let fen = parts.get(2..fen_end).unwrap_or_default().join(" ");
            match Board::from_fen(&fen, false) {
                Ok(board) => board,
                Err(_) => return,
            }
        }
        _ => return,
    };

    let mut next_history = vec![next_board.hash()];
    if let Some(moves_start) = parts.iter().position(|part| *part == "moves") {
        for move_text in &parts[moves_start + 1..] {
            let Ok(chess_move) = parse_uci_move(&next_board, move_text) else {
                return;
            };
            if next_board.try_play(chess_move).is_err() {
                return;
            }
            next_history.push(next_board.hash());
        }
    }

    *board = next_board;
    *history = next_history;
}

fn parse_go(line: &str, board: &Board) -> SearchLimits {
    let parts: Vec<&str> = line.split_whitespace().collect();
    let mut limits = SearchLimits::default();
    let mut index = 1;

    while index < parts.len() {
        match parts[index] {
            "depth" => {
                limits.depth = parse_next(&parts, &mut index);
            }
            "nodes" => {
                limits.nodes = parse_next(&parts, &mut index);
            }
            "movetime" => {
                limits.movetime = parse_next(&parts, &mut index);
            }
            "wtime" => {
                limits.wtime = parse_next(&parts, &mut index);
            }
            "btime" => {
                limits.btime = parse_next(&parts, &mut index);
            }
            "winc" => {
                limits.winc = parse_next(&parts, &mut index);
            }
            "binc" => {
                limits.binc = parse_next(&parts, &mut index);
            }
            "movestogo" => {
                limits.movestogo = parse_next(&parts, &mut index);
            }
            "infinite" => limits.infinite = true,
            "searchmoves" => {
                for move_text in &parts[index + 1..] {
                    if let Ok(chess_move) = parse_uci_move(board, move_text) {
                        limits.searchmoves.push(chess_move);
                    }
                }
                break;
            }
            _ => {}
        }
        index += 1;
    }

    limits
}

fn parse_next<T: std::str::FromStr>(parts: &[&str], index: &mut usize) -> Option<T> {
    *index += 1;
    parts.get(*index)?.parse().ok()
}

fn parse_hash_option(line: &str) -> Option<u64> {
    let parts: Vec<&str> = line.split_whitespace().collect();
    let name_start = parts.iter().position(|part| *part == "name")? + 1;
    let value_start = parts.iter().position(|part| *part == "value")?;
    let name = parts.get(name_start..value_start)?.join(" ");

    if !name.eq_ignore_ascii_case("hash") {
        return None;
    }

    parts
        .get(value_start + 1)?
        .parse::<u64>()
        .ok()
        .map(|megabytes| megabytes.clamp(1, 65_536))
}

fn print_output(output: &str) {
    println!("{output}");
    io::stdout().flush().unwrap();
}

fn format_uci_move(board: &Board, chess_move: Move) -> String {
    display_uci_move(board, chess_move).to_string()
}

fn print_info(id: u64, active_id: Option<u64>, board: &Board, info: SearchInfo) {
    if active_id != Some(id) {
        return;
    }

    let elapsed_ms = info.elapsed.as_millis() as u64;
    let nps = info
        .nodes
        .saturating_mul(1000)
        .checked_div(elapsed_ms)
        .unwrap_or(0);

    let pv = info
        .pv
        .iter()
        .map(|&chess_move| format_uci_move(board, chess_move))
        .collect::<Vec<_>>()
        .join(" ");

    print_output(&format!(
        "info depth {} score cp {} nodes {} time {} nps {} pv {}",
        info.depth, info.score, info.nodes, elapsed_ms, nps, pv
    ));
}

fn print_best_move(
    id: u64,
    active_search: &mut Option<(u64, Arc<AtomicBool>)>,
    board: &Board,
    result: SearchResult,
) {
    let best_move = result
        .best_move
        .map(|chess_move| format_uci_move(board, chess_move))
        .unwrap_or_else(|| "0000".to_string());
    print_output(&format!("bestmove {best_move}"));

    if active_search.as_ref().map(|(active_id, _)| *active_id) == Some(id) {
        *active_search = None;
    }
}

fn spawn_input_reader(event_tx: mpsc::Sender<MainEvent>) -> JoinHandle<()> {
    let input_handle = thread::spawn(move || {
        for line in io::stdin().lock().lines() {
            let Ok(line) = line else {
                break;
            };
            if event_tx.send(MainEvent::Input(line)).is_err() {
                break;
            }
        }

        let _ = event_tx.send(MainEvent::InputClosed);
    });

    input_handle
}

fn main() {
    match std::env::args().nth(1).as_deref() {
        Some("bench") => {
            run_bench();
            return;
        }
        Some("bulk-eval") => {
            if let Err(error) = bulk::run_from_env() {
                eprintln!("bulk-eval failed: {error}");
                std::process::exit(1);
            }
            return;
        }
        Some("selfplay") => {
            if let Err(error) = selfplay::run_from_env() {
                eprintln!("selfplay failed: {error}");
                std::process::exit(1);
            }
            return;
        }
        _ => {}
    }

    run_uci();
}

fn run_bench() {
    const POSITIONS: [(&str, i32); 8] = [
        (
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            12,
        ),
        (
            "r3k2r/p1ppqpb1/bn2pnp1/2pP4/1p2P3/2N2N2/PPQBBPPP/R3K2R w KQkq - 0 1",
            11,
        ),
        ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 13),
        (
            "rnbqkb1r/pp2pppp/3p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R b KQkq - 0 1",
            12,
        ),
        (
            "rnbq1rk1/pp2bppp/2p1pn2/3p4/3P4/2NBPN2/PPQ2PPP/R1B1K2R w KQ - 2 9",
            12,
        ),
        (
            "2r2rk1/1bqnbppp/p2ppn2/1p6/3NP3/P1N1B3/1PQ1BPPP/2RR2K1 w - - 3 16",
            12,
        ),
        (
            "4rrk1/1pp2ppp/p1npb3/8/2P1P3/1PN1B3/P4PPP/2RR2K1 w - - 1 20",
            13,
        ),
        ("8/5pk1/6p1/3pP3/3P1P2/6P1/5K2/8 w - - 0 40", 15),
    ];

    let mut engine = Engine::new();
    let started = std::time::Instant::now();
    let mut total_nodes = 0u64;

    for (fen, depth) in POSITIONS {
        let board = Board::from_fen(fen, false)
            .unwrap_or_else(|error| panic!("invalid benchmark FEN {fen}: {error:?}"));
        engine.new_game();
        let mut nodes = 0;
        let request = SearchRequest {
            board: board.clone(),
            history: vec![board.hash()],
            limits: SearchLimits {
                depth: Some(depth),
                ..SearchLimits::default()
            },
            stop: Arc::new(AtomicBool::new(false)),
        };

        engine.search(&request, |info| nodes = info.nodes);
        total_nodes = total_nodes.saturating_add(nodes);
    }

    let elapsed_nanos = started.elapsed().as_nanos().max(1);
    let nps = (u128::from(total_nodes) * 1_000_000_000 / elapsed_nanos) as u64;
    println!("{total_nodes} nodes {nps} nps");
}

fn run_uci() {
    let (event_tx, event_rx) = mpsc::channel::<MainEvent>();
    let _input_handle = spawn_input_reader(event_tx.clone());
    let (command_tx, command_rx) = mpsc::channel::<Command>();

    let worker_handle = thread::spawn(move || {
        let mut engine = Engine::new();

        while let Ok(command) = command_rx.recv() {
            match command {
                Command::Go { id, request } => {
                    let event_tx_for_info = &event_tx;
                    let search_board = request.board.clone();
                    let result = engine.search(&request, |info| {
                        let _ = event_tx_for_info.send(MainEvent::Worker(WorkerOutput::Info {
                            id,
                            board: search_board.clone(),
                            info,
                        }));
                    });
                    let _ = event_tx.send(MainEvent::Worker(WorkerOutput::BestMove {
                        id,
                        board: search_board,
                        result,
                    }));
                }
                Command::SetHash(megabytes) => engine.set_hash_size_mb(megabytes),
                Command::NewGame => engine.new_game(),
                Command::Quit => break,
            }
        }
    });

    let mut board = Board::default();
    let mut history = vec![board.hash()];
    let mut active_search: Option<(u64, Arc<AtomicBool>)> = None;
    let mut next_search_id = 0;
    let mut quitting = false;

    while !quitting {
        let line = match event_rx.recv() {
            Ok(MainEvent::Worker(output)) => {
                match output {
                    WorkerOutput::Info { id, board, info } => {
                        print_info(id, active_search.as_ref().map(|(id, _)| *id), &board, info);
                    }
                    WorkerOutput::BestMove { id, board, result } => {
                        print_best_move(id, &mut active_search, &board, result);
                    }
                }
                continue;
            }
            Ok(MainEvent::Input(line)) => line,
            Ok(MainEvent::InputClosed) | Err(_) => {
                if let Some((_, stop)) = &active_search {
                    stop.store(true, Ordering::Relaxed);
                }
                let _ = command_tx.send(Command::Quit);
                break;
            }
        };
        let command = line.split_whitespace().next();

        match command {
            Some("uci") => {
                print_output("id name chessbot");
                print_output("id author me");
                print_output("option name Threads type spin default 1 min 1 max 1");
                print_output("option name Hash type spin default 16 min 1 max 65536");
                print_output("uciok");
            }
            Some("isready") => print_output("readyok"),
            Some("position") => parse_position(&line, &mut board, &mut history),
            Some("go") => {
                if let Some((_, stop)) = active_search.take() {
                    stop.store(true, Ordering::Relaxed);
                }

                let stop = Arc::new(AtomicBool::new(false));
                let id = next_search_id;
                next_search_id += 1;
                let request = SearchRequest {
                    board: board.clone(),
                    history: history.clone(),
                    limits: parse_go(&line, &board),
                    stop: Arc::clone(&stop),
                };

                active_search = Some((id, stop));
                let _ = command_tx.send(Command::Go { id, request });
            }
            Some("stop") => {
                if let Some((_, stop)) = &active_search {
                    stop.store(true, Ordering::Relaxed);
                }
            }
            Some("ucinewgame") => {
                if let Some((_, stop)) = &active_search {
                    stop.store(true, Ordering::Relaxed);
                }
                history = vec![board.hash()];
                let _ = command_tx.send(Command::NewGame);
            }
            Some("quit") => {
                if let Some((_, stop)) = &active_search {
                    stop.store(true, Ordering::Relaxed);
                }
                let _ = command_tx.send(Command::Quit);
                quitting = true;
            }
            Some("setoption") => {
                if let Some(megabytes) = parse_hash_option(&line) {
                    if let Some((_, stop)) = &active_search {
                        stop.store(true, Ordering::Relaxed);
                    }
                    let _ = command_tx.send(Command::SetHash(megabytes));
                }
            }
            _ => {}
        }
    }

    let _ = worker_handle.join();
}
