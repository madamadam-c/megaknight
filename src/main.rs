use std::{
    io::{self, BufRead, Write},
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc, Arc,
    },
    thread::{self, JoinHandle},
    time::Duration,
};

use cozy_chess::{util::parse_uci_move, Board};

use crate::engine::{Engine, SearchInfo, SearchLimits, SearchRequest, SearchResult};

mod engine;
mod evaluate;

#[cfg(test)]
mod tests;

enum Command {
    Go { id: u64, request: SearchRequest },
    NewGame,
    Quit,
}

enum WorkerOutput {
    Info { id: u64, info: SearchInfo },
    BestMove { id: u64, result: SearchResult },
}

fn parse_position(line: &str, board: &mut Board) {
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

    if let Some(moves_start) = parts.iter().position(|part| *part == "moves") {
        for move_text in &parts[moves_start + 1..] {
            let Ok(chess_move) = parse_uci_move(&next_board, move_text) else {
                return;
            };
            if next_board.try_play(chess_move).is_err() {
                return;
            }
        }
    }

    *board = next_board;
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

fn print_output(output: &str) {
    println!("{output}");
    io::stdout().flush().unwrap();
}

fn print_info(id: u64, active_id: Option<u64>, info: SearchInfo) {
    if active_id != Some(id) {
        return;
    }

    let elapsed_ms = info.elapsed.as_millis() as u64;
    let nps = info.nodes.saturating_mul(1000).checked_div(elapsed_ms).unwrap_or(0);
    
    let pv = info
        .pv
        .iter()
        .map(ToString::to_string)
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
    result: SearchResult,
) {
    let best_move = result
        .best_move
        .map(|chess_move| chess_move.to_string())
        .unwrap_or_else(|| "0000".to_string());
    print_output(&format!("bestmove {best_move}"));

    if active_search.as_ref().map(|(active_id, _)| *active_id) == Some(id) {
        *active_search = None;
    }
}

fn spawn_input_reader() -> (mpsc::Receiver<String>, JoinHandle<()>) {
    let (input_tx, input_rx) = mpsc::channel::<String>();
    let input_handle = thread::spawn(move || {
        for line in io::stdin().lock().lines() {
            let Ok(line) = line else {
                break;
            };
            if input_tx.send(line).is_err() {
                break;
            }
        }
    });

    (input_rx, input_handle)
}

fn main() {
    let (input_rx, _input_handle) = spawn_input_reader();
    let (command_tx, command_rx) = mpsc::channel::<Command>();
    let (output_tx, output_rx) = mpsc::channel::<WorkerOutput>();

    let worker_handle = thread::spawn(move || {
        let mut engine = Engine::new();

        while let Ok(command) = command_rx.recv() {
            match command {
                Command::Go { id, request } => {
                    let output_tx_for_info = &output_tx;
                    let result = engine.search(&request, |info| {
                        let _ = output_tx_for_info.send(WorkerOutput::Info { id, info });
                    });
                    let _ = output_tx.send(WorkerOutput::BestMove { id, result });
                }
                Command::NewGame => engine.new_game(),
                Command::Quit => break,
            }
        }
    });

    let mut board = Board::default();
    let mut active_search: Option<(u64, Arc<AtomicBool>)> = None;
    let mut next_search_id = 0;
    let mut quitting = false;

    while !quitting {
        while let Ok(output) = output_rx.try_recv() {
            match output {
                WorkerOutput::Info { id, info } => {
                    print_info(id, active_search.as_ref().map(|(id, _)| *id), info);
                }
                WorkerOutput::BestMove { id, result } => {
                    print_best_move(id, &mut active_search, result);
                }
            }
        }

        let line = match input_rx.recv_timeout(Duration::from_millis(10)) {
            Ok(line) => line,
            Err(mpsc::RecvTimeoutError::Timeout) => continue,
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        };
        let command = line.split_whitespace().next();

        match command {
            Some("uci") => {
                print_output("id name chessbot");
                print_output("id author me");
                print_output("uciok");
            }
            Some("isready") => print_output("readyok"),
            Some("position") => parse_position(&line, &mut board),
            Some("go") => {
                if let Some((_, stop)) = active_search.take() {
                    stop.store(true, Ordering::Relaxed);
                }

                let stop = Arc::new(AtomicBool::new(false));
                let id = next_search_id;
                next_search_id += 1;
                let request = SearchRequest {
                    board: board.clone(),
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
                let _ = command_tx.send(Command::NewGame);
            }
            Some("quit") => {
                if let Some((_, stop)) = &active_search {
                    stop.store(true, Ordering::Relaxed);
                }
                let _ = command_tx.send(Command::Quit);
                quitting = true;
            }
            Some("setoption") => {}
            _ => {}
        }
    }

    let _ = worker_handle.join();
}
