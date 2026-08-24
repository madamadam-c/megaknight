use std::{
    cmp::{max, min}, ops::Neg, sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    }, time::{Duration, Instant},
};

use cozy_chess::{Board, Color::{self, White}, Move, Piece::{self, Pawn}};

use crate::{
    evaluate::{eval, static_exchange_evaluation, value}, transposition::{
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
    pub history: Vec<u64>,
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
    history: Vec<u64>,
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
    fn new(
        board: &Board,
        history: Vec<u64>,
        limits: SearchLimits,
        stop: Arc<AtomicBool>,
    ) -> Self {
        let start = Instant::now();
        let (soft_deadline, hard_deadline) = time_deadlines(board, &limits, start);

        Self {
            limits,
            stop,
            history,
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

const MATE_SCORE: i32 = 100_000;
const MATE_THRESHOLD: i32 = 99_000;

fn score_to_tt(score: i32, ply: i32) -> i32 {
    if score >= MATE_THRESHOLD {
        score + ply
    } else if score <= -MATE_THRESHOLD {
        score - ply
    } else {
        score
    }
}

fn score_from_tt(score: i32, ply: i32) -> i32 {
    if score >= MATE_THRESHOLD {
        score - ply
    } else if score <= -MATE_THRESHOLD {
        score + ply
    } else {
        score
    }
}

fn is_repetition(board: &Board, history: &[u64]) -> bool {
    let key = board.hash();
    let reversible_positions = board.halfmove_clock() as usize + 1;

    history
        .iter()
        .rev()
        .take(reversible_positions)
        .filter(|&&previous| previous == key)
        .take(3)
        .count()
        >= 3
}

fn terminal_score(board: &Board, ply: i32) -> i32 {
    if board.checkers().is_empty() {
        0
    } else {
        -MATE_SCORE + ply
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


#[derive(Clone, Copy)]
pub struct EngineMove {
    pub mv: Move,
    pub material_value: i32,
    pub see_score: i16,
    pub is_capture: bool,
    pub is_ep: bool,
    pub is_tt: bool, 
    pub promotion: bool,
    pub piece_type: Piece,
    pub target_type: Option<Piece>
}

impl EngineMove {
    #[inline(always)]
    pub fn new(board: &Board, mv: Move, piece: Piece, is_tt: bool) -> Self {
        let mut material_value = 0;
        let mut capture = false;
        let mut ep = false;
        let mut promotion = false;
        let mut target_type = None;

        if mv.promotion.is_some() {
            material_value += value(mv.promotion.unwrap()) - value(piece);
            promotion = true;
        }

        if board.en_passant().is_some() && piece == Pawn && mv.from.file() != mv.to.file() && board.piece_on(mv.to).is_none() {
            material_value += value(Pawn);
            capture = true; ep = true;
            target_type = Some(Pawn);
        } else if let Some(target) = board.piece_on(mv.to) && board.color_on(mv.to) != board.color_on(mv.from) {
            material_value += value(target);
            capture = true;
            target_type = Some(target);
        }
        
        Self {
            mv: mv,
            material_value: material_value,
            see_score: 0,
            is_capture: capture,
            is_ep: ep,
            is_tt: is_tt,
            promotion: promotion,
            piece_type: piece,
            target_type: target_type
        }
    }
}
#[derive(Clone)]
struct MoveList {
    pub good_captures: Vec<EngineMove>,
    pub bad_captures: Vec<EngineMove>,
    pub quiets: Vec<EngineMove>,
}

impl MoveList {
    #[inline(always)]
    pub fn new(qsearch: bool) -> Self {
        let g = Vec::new();
        let b = Vec::new();
        let q = Vec::with_capacity(if qsearch {0} else {64});

        Self {
            good_captures: g,
            bad_captures: b,
            quiets: q
        }
    }

    pub fn is_empty(&self) -> bool {
        return self.good_captures.is_empty() && self.bad_captures.is_empty() && self.quiets.is_empty();
    }

    pub fn len(&self) -> usize {
        self.good_captures.len() + self.quiets.len() + self.bad_captures.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = &EngineMove> {
        self.good_captures
            .iter()
            .chain(self.quiets.iter())
            .chain(self.bad_captures.iter())
    }
}

impl IntoIterator for MoveList {
    type Item = EngineMove;

    type IntoIter = std::iter::Flatten<
        std::array::IntoIter<Vec<EngineMove>, 3>,
    >;

    fn into_iter(self) -> Self::IntoIter {
        [
            self.good_captures,
            self.quiets,
            self.bad_captures,
        ]
        .into_iter()
        .flatten()
    }
}

impl std::ops::Index<usize> for MoveList {
    type Output = EngineMove;

    fn index(&self, mut index: usize) -> &Self::Output {
        if index < self.good_captures.len() {
            return &self.good_captures[index];
        }

        index -= self.good_captures.len();

        if index < self.quiets.len() {
            return &self.quiets[index];
        }

        index -= self.quiets.len();
        &self.bad_captures[index]
    }
}

const MAX_HISTORY: i32 = 16_384;
fn update_history(value: &mut i32, bonus: i32) {
    let bonus = bonus.clamp(-MAX_HISTORY, MAX_HISTORY);
    *value += bonus - *value * bonus.abs() / MAX_HISTORY;
}

fn history_bonus(depth: i32) -> i32 {
    depth.saturating_mul(depth)
}

pub struct Engine {
    tt: Table,
    eval: i32,
    history: [[[i32; 64]; 64]; 2],
}

impl Engine {
    pub fn new() -> Self {
        Self {
            tt: Table::new_for_mb(16),
            eval: 0,
            history: [[[0; 64]; 64]; 2]
        }
    }

    pub fn new_game(&mut self) {}

    pub fn set_hash_size_mb(&mut self, megabytes: u64) {
        self.tt = Table::new_for_mb(megabytes);
    }

    fn generate_moves(&self, board: &Board, tt_move: Option<Move>) -> MoveList {
        let mut moves = MoveList::new(false);

        board.generate_moves(|moves_for_piece| {
            for mv in moves_for_piece {
                let mut emv = EngineMove::new(
                    board,
                    mv,
                    moves_for_piece.piece,
                    tt_move == Some(mv),
                );

                if emv.is_capture {
                    emv.see_score = static_exchange_evaluation(board, &emv);
                }

                if emv.is_capture || emv.is_tt {
                    if emv.see_score >= 0 || emv.is_tt {
                        moves.good_captures.push(emv);
                    } else {
                        moves.bad_captures.push(emv);
                    }
                } else {
                    moves.quiets.push(emv);
                }
            }
            false
        });

        moves.good_captures.sort_unstable_by_key(|mv| {
            if mv.is_tt {
                -1_000_000
            } else {
                // value(mv.piece_type) - value(mv.target_type.unwrap())
                -mv.see_score as i32
            }
        });

        moves.quiets.sort_unstable_by_key(|mv| {
            if mv.promotion {
                -mv.material_value - MAX_HISTORY
            } else {
                -self.history[if board.side_to_move() == White {0} else {1}][mv.mv.from as usize][mv.mv.to as usize]
            }
        });

        moves.bad_captures.sort_unstable_by_key(|mv| {
            -mv.see_score
        });

        return moves;
    }

    fn generate_qsearch_moves(&self, board: &Board, empty_flag: &mut bool) -> MoveList {
        let mut moves = MoveList::new(true);

        let enemy_pieces = board.colors(!board.side_to_move());
        board.generate_moves(|mut moves_for_piece| {
            if !moves_for_piece.is_empty() {
                *empty_flag = false;
            }
            if moves_for_piece.piece != Pawn {
                moves_for_piece.to &= enemy_pieces;
            }
            for mv in moves_for_piece {
                if moves_for_piece.piece == Pawn && mv.to.file() == mv.from.file() && mv.promotion.is_none() {
                    continue;
                }
                let mut emv = EngineMove::new(
                    board,
                    mv,
                    moves_for_piece.piece,
                    false,
                );

                if emv.is_capture {
                    emv.see_score = static_exchange_evaluation(board, &emv);
                    if emv.see_score >= 0 {
                        moves.good_captures.push(emv);
                    } else {
                        moves.bad_captures.push(emv);
                    }
                    
                } else if emv.promotion {
                    moves.quiets.push(emv);
                }
            }
            false
        });

        moves.good_captures.sort_unstable_by_key(|mv| {
            // value(mv.piece_type) - value(mv.target_type.unwrap())
            -mv.see_score
        });

        moves.bad_captures.sort_unstable_by_key(|mv| {
            -mv.see_score
        });

        return moves;
    }

    fn quiesce(
        &mut self,
        board: &Board,
        ply: i32,
        mut bounds: SearchBounds,
        context: &mut SearchContext,
    ) -> Option<i32> {
        if context.should_stop() {
            return None;
        }

        if is_repetition(board, &context.history) {
            return Some(0);
        }

        if board.halfmove_clock() >= 100 {
            if board.generate_moves(|_| true) {
                return Some(0);
            } else {
                return Some(terminal_score(board, ply));
            }
        }

        let in_check = !board.checkers().is_empty();

        let static_eval = self.eval;
        if !in_check {
            if static_eval >= bounds.beta {
                if board.generate_moves(|_| true) {
                    return Some(static_eval);
                }
                return Some(terminal_score(board, ply));
            }
            if static_eval > bounds.alpha {
                bounds.alpha = static_eval;
            }
        }

        let mut empty_flag = true;
        let moves;
        if in_check {
            moves = self.generate_moves(board, None);
            empty_flag = moves.is_empty();
        } else {
            moves = self.generate_qsearch_moves(board, &mut empty_flag);
        }

        if empty_flag {
            return Some(terminal_score(board, ply));
        }

        let mut result;
        if in_check {
            result = -MATE_SCORE;
        } else {
            result = self.eval;
        }
        for mv in moves {
            let mut next_board = board.clone();
            next_board.play_unchecked(mv.mv);

            self.eval += mv.material_value;
            self.eval = -self.eval;

            context.history.push(next_board.hash());
            let child = self.quiesce(&next_board, ply + 1, -bounds, context);
            context.history.pop();

            self.eval = -self.eval;
            self.eval -= mv.material_value;

            let x = -child?;

            if x > result {
                result = x;
                if x > bounds.alpha {
                    bounds.alpha = x;
                }
            }

            if x >= bounds.beta {
                break;
            }
        }

        Some(result)
    }

    fn minimax(
        &mut self,
        board: &Board,
        depth: i32,
        ply: i32,
        mut bounds: SearchBounds,
        context: &mut SearchContext,
    ) -> Option<i32> {
        if context.should_stop() {
            return None;
        }

        if is_repetition(board, &context.history) {
            return Some(0);
        }

        if board.halfmove_clock() >= 100 {
            if board.generate_moves(|_| true) {
                return Some(0);
            } else {
                return Some(terminal_score(board, ply));
            }
        }

        if depth <= 0 {
            return self.quiesce(board, ply, bounds, context);
        }

        let original_bounds = bounds;

        let key = board.hash();
        let mut tt_move = None;
        if let Some(entry) = self.tt.get(key) {
            let entry_score = score_from_tt(entry.score, ply);
            if let Some(mv) = entry.best_move
                && board.is_legal(mv)
            {
                tt_move = Some(mv);
            }
            if entry.depth >= depth {
                match entry.node_type {
                    EXACT => return Some(entry_score),
                    UPPER => bounds.beta = min(bounds.beta, entry_score),
                    LOWER => bounds.alpha = max(bounds.alpha, entry_score),
                }

                if bounds.alpha >= bounds.beta {
                    return Some(entry_score);
                }
            }
        }
        
        let moves = self.generate_moves(board, tt_move);

        if moves.is_empty() {
            return Some(terminal_score(board, ply));
        }

        let mut result = -1_000_000_000;
        let mut best = moves[0];
        for mv in moves.iter().copied() {
            let mut next_board = board.clone();
            next_board.play_unchecked(mv.mv);

            self.eval += mv.material_value;
            self.eval = -self.eval;
            
            context.history.push(next_board.hash());
            let child = self.minimax(&next_board, depth - 1, ply + 1, -bounds, context);
            context.history.pop();

            self.eval = -self.eval;
            self.eval -= mv.material_value;

            let x = -child?;

            if x > result {
                result = x;
                best = mv;
                if x > bounds.alpha {
                    bounds.alpha = x;
                }
            }

            if x >= bounds.beta {
                if !mv.is_capture && !mv.promotion {
                    update_history(&mut self.history[if board.side_to_move() == White {0} else {1}][mv.mv.from as usize][mv.mv.to as usize], history_bonus(depth));
                    for mv2 in moves.iter().copied() {
                        if mv2.mv == mv.mv {
                            break;
                        }
                        update_history(&mut self.history[if board.side_to_move() == White {0} else {1}][mv2.mv.from as usize][mv2.mv.to as usize], 
                            -history_bonus(depth)/3);
                    }
                }
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
                score: score_to_tt(result, ply),
                depth: depth,
                node_type: node_type,
                best_move: Some(best.mv),
            },
        );
        Some(result)
    }

    fn root_search(
        &mut self,
        board: &Board,
        depth: i32,
        root_moves: MoveList,
        context: &mut SearchContext,
    ) -> Option<(Move, i32)> {
        let mut best_move = None;
        let mut best_score = -1_000_000_000;
        let mut bounds = SearchBounds {
            alpha: -1_000_000_000,
            beta: 1_000_000_000,
        };

        self.eval = eval(board);
        for mv in root_moves {
            if context.should_stop() {
                return None;
            }

            let mut next_board = board.clone();
            next_board.play_unchecked(mv.mv);

            self.eval += mv.material_value;
            self.eval = -self.eval;

            context.history.push(next_board.hash());
            let child = self.minimax(&next_board, depth - 1, 1, -bounds, context);
            context.history.pop();

            self.eval = -self.eval;
            self.eval -= mv.material_value;

            let score = -child?;

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

        best_move.map(|mv| (mv.mv, best_score))
    }

    pub fn search<F>(&mut self, request: &SearchRequest, mut report: F) -> SearchResult
    where
        F: FnMut(SearchInfo),
    {
        let mut root_moves = self.generate_moves(&request.board, None);

        self.history = [[[0; 64] ; 64] ; 2];

        if root_moves.is_empty() {
            return SearchResult { best_move: None };
        }

        let fallback_move = root_moves[0];
        let max_depth = request.limits.depth.unwrap_or(i32::MAX).max(1);
        let mut context = SearchContext::new(
            &request.board,
            request.history.clone(),
            request.limits.clone(),
            Arc::clone(&request.stop),
        );
        let mut completed_move = None;
        let mut depth = 1;

        while depth <= max_depth {
            if context.should_stop() {
                break;
            }

            root_moves = self.generate_moves(&request.board, completed_move);
            if !request.limits.searchmoves.is_empty() {
                root_moves.good_captures.retain(|mv| request.limits.searchmoves.contains(&mv.mv));
                root_moves.bad_captures.retain(|mv| request.limits.searchmoves.contains(&mv.mv));
                root_moves.quiets.retain(|mv| request.limits.searchmoves.contains(&mv.mv));
            }

            let Some((best_move, score)) =
                self.root_search(&request.board, depth, root_moves.clone(), &mut context)
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
            best_move: Some(completed_move.unwrap_or(fallback_move.mv)),
        }
    }
}

#[cfg(test)]
#[path = "tests/test_engine.rs"]
mod tests;
