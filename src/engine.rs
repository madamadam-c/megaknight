use std::{
    cmp::{max, min},
    ops::Neg,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

use cozy_chess::{Board, Color, GameStatus, Move};

use crate::{
    evaluate::eval,
    transposition::{
        NodeType::{EXACT, LOWER, UPPER},
        Table, TableEntry,
    },
};

#[derive(Clone, Default)]
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

pub struct SearchRequest {
    pub board: Board,
    pub limits: SearchLimits,
    pub stop: Arc<AtomicBool>,
}

pub struct SearchInfo {
    pub depth: i32,
    pub score: i32,
    pub nodes: u64,
    pub elapsed: Duration,
    pub pv: Vec<Move>,
}

pub struct SearchResult {
    pub best_move: Option<Move>,
}

struct SearchContext {
    limits: SearchLimits,
    stop: Arc<AtomicBool>,
    start: Instant,
    soft_deadline: Option<Instant>,
    hard_deadline: Option<Instant>,
    nodes: u64,
}

#[derive(Clone, Copy)]
struct SearchBounds {
    alpha: i32,
    beta: i32,
}

impl SearchContext {
    fn new(board: &Board, limits: SearchLimits, stop: Arc<AtomicBool>) -> Self {
        let start = Instant::now();
        let (soft_deadline, hard_deadline) = time_deadlines(board, &limits, start);

        Self {
            limits,
            stop,
            start,
            soft_deadline,
            hard_deadline,
            nodes: 0,
        }
    }

    fn should_stop(&mut self) -> bool {
        if self.stop.load(Ordering::Relaxed) {
            return true;
        }

        self.nodes += 1;

        if self.limits.nodes.is_some_and(|limit| self.nodes >= limit) {
            return true;
        }

        if self.nodes == 1 || self.nodes.is_multiple_of(1024) {
            return self
                .hard_deadline
                .is_some_and(|deadline| Instant::now() >= deadline);
        }

        false
    }

    fn soft_expired(&self) -> bool {
        self.soft_deadline
            .is_some_and(|deadline| Instant::now() >= deadline)
    }

    fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }
}

impl Neg for SearchBounds {
    type Output = Self;

    fn neg(self) -> Self::Output {
        SearchBounds {
            alpha: -self.beta,
            beta: -self.alpha,
        }
    }
}

fn time_deadlines(
    board: &Board,
    limits: &SearchLimits,
    start: Instant,
) -> (Option<Instant>, Option<Instant>) {
    let Some(budget_ms) = time_budget_ms(board, limits) else {
        return (None, None);
    };

    let hard = Duration::from_millis(budget_ms.max(1));
    let soft = Duration::from_millis((budget_ms * 4 / 5).max(1));
    (Some(start + soft), Some(start + hard))
}

fn time_budget_ms(board: &Board, limits: &SearchLimits) -> Option<u64> {
    if let Some(movetime) = limits.movetime {
        return Some(movetime.saturating_sub(5).max(1));
    }

    if limits.infinite {
        return None;
    }

    let (remaining, increment) = if board.side_to_move() == Color::White {
        (limits.wtime?, limits.winc.unwrap_or(0))
    } else {
        (limits.btime?, limits.binc.unwrap_or(0))
    };

    let moves_to_go = limits.movestogo.unwrap_or(30).max(1);
    let reserve = (remaining / 100).max(5);
    let available = remaining.saturating_sub(reserve).max(1);
    let allocation = remaining / moves_to_go + increment * 3 / 4;

    Some(allocation.min(available).max(1))
}

pub struct Engine {
    tt: Table,
}

impl Engine {
    pub fn new() -> Self {
        Self {
            tt: Table::new_for_mb(16),
        }
    }

    pub fn new_game(&mut self) {}

    pub fn set_hash_size_mb(&mut self, megabytes: u64) {
        self.tt = Table::new_for_mb(megabytes);
    }

    fn minimax(
        &mut self,
        board: &Board,
        depth: i32,
        mut bounds: SearchBounds,
        context: &mut SearchContext,
    ) -> Option<i32> {
        if context.should_stop() {
            return None;
        }

        if depth <= 0 || board.status() != GameStatus::Ongoing {
            return Some(eval(board));
        }

        let original_bounds = bounds;

        let mut moves = Vec::new();

        let key = board.hash();
        if let Some(entry) = self.tt.get(key) {
            if let Some(mv) = entry.best_move
                && board.is_legal(mv)
            {
                moves.push(mv);
            }
            if entry.depth >= depth {
                match entry.node_type {
                    EXACT => return Some(entry.score),
                    UPPER => bounds.beta = min(bounds.beta, entry.score),
                    LOWER => bounds.alpha = max(bounds.alpha, entry.score),
                }

                if bounds.alpha >= bounds.beta {
                    return Some(entry.score);
                }
            }
        }

        board.generate_moves(|moves_for_piece| {
            moves.extend(moves_for_piece);
            false
        });

        let mut result = -1_000_000_000;
        let mut best = moves[0];

        for mv in moves {
            let mut next_board = board.clone();
            next_board.play(mv);

            let x = -self.minimax(&next_board, depth - 1, -bounds, context)?;

            if x > result {
                result = x;
                best = mv;
                if x > bounds.alpha {
                    bounds.alpha = x;
                }
            }

            if x >= bounds.beta {
                break;
            }
        }

        let node_type = {
            if result >= original_bounds.beta {
                LOWER
            } else if result <= original_bounds.alpha {
                UPPER
            } else {
                EXACT
            }
        };

        self.tt.insert(
            key,
            TableEntry {
                key: key,
                score: result,
                depth: depth,
                node_type: node_type,
                best_move: Some(best),
            },
        );
        Some(result)
    }

    fn root_search(
        &mut self,
        board: &Board,
        depth: i32,
        root_moves: &[Move],
        context: &mut SearchContext,
    ) -> Option<(Move, i32)> {
        let mut best_move = None;
        let mut best_score = -1_000_000_000;
        let mut bounds = SearchBounds {
            alpha: -1_000_000_000,
            beta: 1_000_000_000,
        };

        for &mv in root_moves {
            if context.should_stop() {
                return None;
            }

            let mut next_board = board.clone();
            next_board.play(mv);

            let score = -self.minimax(&next_board, depth - 1, -bounds, context)?;
            if score > best_score {
                best_score = score;
                if score > bounds.alpha {
                    bounds.alpha = score;
                }
                best_move = Some(mv);
            }

            if score >= bounds.beta {
                break;
            }
        }

        best_move.map(|mv| (mv, best_score))
    }

    pub fn search<F>(&mut self, request: &SearchRequest, mut report: F) -> SearchResult
    where
        F: FnMut(SearchInfo),
    {
        let mut root_moves = Vec::new();
        request.board.generate_moves(|moves_for_piece| {
            root_moves.extend(moves_for_piece);
            false
        });

        if !request.limits.searchmoves.is_empty() {
            root_moves.retain(|mv| request.limits.searchmoves.contains(mv));
        }

        if root_moves.is_empty() {
            return SearchResult { best_move: None };
        }

        let fallback_move = root_moves[0];
        let max_depth = request.limits.depth.unwrap_or(i32::MAX).max(1);
        let mut context = SearchContext::new(
            &request.board,
            request.limits.clone(),
            Arc::clone(&request.stop),
        );
        let mut completed_move = None;
        let mut depth = 1;

        while depth <= max_depth {
            if context.should_stop() {
                break;
            }

            let Some((best_move, score)) =
                self.root_search(&request.board, depth, &root_moves, &mut context)
            else {
                break;
            };

            completed_move = Some(best_move);

            report(SearchInfo {
                depth,
                score,
                nodes: context.nodes,
                elapsed: context.elapsed(),
                pv: vec![best_move],
            });

            if context.soft_expired() || depth == max_depth {
                break;
            }

            depth += 1;
        }

        SearchResult {
            best_move: Some(completed_move.unwrap_or(fallback_move)),
        }
    }
}
