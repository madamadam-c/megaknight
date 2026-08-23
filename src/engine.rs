use std::{
    cmp::{max, min}, ops::Neg, sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    }, time::{Duration, Instant},
};

use cozy_chess::{Board, Color, Move, Piece::{self, Pawn}};

use crate::{
    evaluate::{eval, value}, transposition::{
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


#[derive(Clone, Copy)]
struct EngineMove {
    mv: Move,
    material_value: i32,
    is_capture: bool,
    is_ep: bool,
    is_tt: bool, 
    promotion: bool,
    piece_type: Piece,
    target_type: Option<Piece>
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
            is_capture: capture,
            is_ep: ep,
            is_tt: is_tt,
            promotion: promotion,
            piece_type: piece,
            target_type: target_type
        }
    }
}

pub struct Engine {
    tt: Table,
    eval: i32,
}

impl Engine {
    pub fn new() -> Self {
        Self {
            tt: Table::new_for_mb(16),
            eval: 0,
        }
    }

    pub fn new_game(&mut self) {}

    pub fn set_hash_size_mb(&mut self, megabytes: u64) {
        self.tt = Table::new_for_mb(megabytes);
    }

    fn generate_moves(&self, board: &Board, tt_move: Option<Move>) -> Vec<EngineMove> {
        let mut moves = Vec::with_capacity(64);
        let mut captures_len = 0;

        board.generate_moves(|moves_for_piece| {
            for mv in moves_for_piece {
                let emv = EngineMove::new(
                    board,
                    mv,
                    moves_for_piece.piece,
                    tt_move == Some(mv),
                );

                if emv.is_capture || emv.is_tt {
                    moves.insert(captures_len, emv);
                    captures_len += 1;
                } else {
                    moves.push(emv);
                }
            }
            false
        });

        moves[..captures_len].sort_by_key(|mv| {
            if mv.is_tt {
                -1_000_000
            } else {
                value(mv.piece_type) - value(mv.target_type.unwrap())
            }
        });

        return moves;
    }

    fn generate_qsearch_moves(&self, board: &Board, empty_flag: &mut bool) -> Vec<EngineMove> {
        let mut moves = Vec::new();
        let mut captures_len = 0;

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
                let emv = EngineMove::new(
                    board,
                    mv,
                    moves_for_piece.piece,
                    false,
                );

                if emv.is_capture {
                    moves.insert(captures_len, emv);
                    captures_len += 1;
                } else if emv.promotion {
                    moves.push(emv);
                }
            }
            false
        });

        moves[..captures_len].sort_by_key(|mv| {
            value(mv.piece_type) - value(mv.target_type.unwrap())
        });

        return moves;
    }

    fn quiesce(
        &mut self,
        board: &Board,
        mut bounds: SearchBounds,
        context: &mut SearchContext,
    ) -> Option<i32> {
        if context.should_stop() {
            return None;
        }

        if board.halfmove_clock() >= 100 {
            if board.generate_moves(|_| true) {
                return Some(0);
            } else {
                return Some(eval(board));
            }
        }

        let in_check = !board.checkers().is_empty();

        let mut empty_flag = true;
        let moves;
        if in_check {
            moves = self.generate_moves(board, None);
            empty_flag = moves.is_empty();
        } else {
            moves = self.generate_qsearch_moves(board, &mut empty_flag);
        }

        if empty_flag {
            return Some(eval(board));
        }

        
        let static_eval = self.eval;
        if !in_check {
            if static_eval >= bounds.beta {
                return Some(static_eval);
            }
            if static_eval > bounds.alpha {
                bounds.alpha = static_eval;
            }
        }

        let mut result;
        if in_check {
            result = -100_000;
        } else {
            result = self.eval;
        }
        for mv in moves {
            let mut next_board = board.clone();
            next_board.play_unchecked(mv.mv);

            self.eval += mv.material_value;
            self.eval = -self.eval;

            let x = -self.quiesce(&next_board, -bounds, context)?;

            self.eval = -self.eval;
            self.eval -= mv.material_value;

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
        mut bounds: SearchBounds,
        context: &mut SearchContext,
    ) -> Option<i32> {
        if context.should_stop() {
            return None;
        }

        if board.halfmove_clock() >= 100 {
            if board.generate_moves(|_| true) {
                return Some(0);
            } else {
                return Some(eval(board));
            }
        }

        if depth <= 0 {
            return self.quiesce(board, bounds, context);
        }

        let original_bounds = bounds;

        let key = board.hash();
        let mut tt_move = None;
        if let Some(entry) = self.tt.get(key) {
            if let Some(mv) = entry.best_move
                && board.is_legal(mv)
            {
                tt_move = Some(mv);
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
        
        let moves = self.generate_moves(board, tt_move);

        if moves.is_empty() {
            return Some(eval(board));
        }

        let mut result = -1_000_000_000;
        let mut best = moves[0];
        for mv in moves {
            let mut next_board = board.clone();
            next_board.play_unchecked(mv.mv);

            self.eval += mv.material_value;
            self.eval = -self.eval;
            
            let x = -self.minimax(&next_board, depth - 1, -bounds, context)?;
            
            self.eval = -self.eval;
            self.eval -= mv.material_value;

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
                best_move: Some(best.mv),
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

        self.eval = eval(board);
        for &emv in root_moves {
            let mv = EngineMove::new(board, emv, board.piece_on(emv.from).unwrap(), false);
            if context.should_stop() {
                return None;
            }

            let mut next_board = board.clone();
            next_board.play_unchecked(mv.mv);

            self.eval += mv.material_value;
            self.eval = -self.eval;

            let score = -self.minimax(&next_board, depth - 1, -bounds, context)?;

            self.eval = -self.eval;
            self.eval -= mv.material_value;

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

#[cfg(test)]
mod tests {
    use super::*;
    use cozy_chess::Square;

    fn board(fen: &str) -> Board {
        Board::from_fen(fen, false).unwrap()
    }

    fn run_quiesce(board: &Board) -> (i32, u64) {
        let mut engine = Engine {
            tt: Table::new(1),
            eval: eval(board),
        };
        let mut context = SearchContext::new(
            board,
            SearchLimits {
                infinite: true,
                ..SearchLimits::default()
            },
            Arc::new(AtomicBool::new(false)),
        );
        let score = engine
            .quiesce(
                board,
                SearchBounds {
                    alpha: -1_000_000_000,
                    beta: 1_000_000_000,
                },
                &mut context,
            )
            .unwrap();

        (score, context.nodes)
    }

    #[test]
    fn qsearch_generates_captures_but_not_quiet_moves() {
        let board = board("7k/8/8/8/8/8/p7/R6K w - - 0 1");
        let engine = Engine::new();
        let mut empty = true;
        let moves = engine.generate_qsearch_moves(&board, &mut empty);

        assert!(!empty);
        assert_eq!(moves.len(), 1);
        assert_eq!(
            moves[0].mv,
            Move {
                from: Square::A1,
                to: Square::A2,
                promotion: None,
            }
        );
        assert!(moves[0].is_capture);
        assert_eq!(run_quiesce(&board).0, 500);
    }

    #[test]
    fn qsearch_generates_en_passant() {
        let board = board("7k/8/8/3pP3/8/8/8/7K w - d6 0 1");
        let engine = Engine::new();
        let mut empty = true;
        let moves = engine.generate_qsearch_moves(&board, &mut empty);

        assert!(!empty);
        assert_eq!(moves.len(), 1);
        assert_eq!(
            moves[0].mv,
            Move {
                from: Square::E5,
                to: Square::D6,
                promotion: None,
            }
        );
        assert!(moves[0].is_ep);
    }

    #[test]
    fn qsearch_generates_all_quiet_promotions() {
        let board = board("7k/P7/8/8/8/8/8/7K w - - 0 1");
        let engine = Engine::new();
        let mut empty = true;
        let moves = engine.generate_qsearch_moves(&board, &mut empty);
        let promotions: Vec<_> = moves.iter().map(|mv| mv.mv.promotion.unwrap()).collect();

        assert!(!empty);
        assert_eq!(moves.len(), 4);
        assert!(moves.iter().all(|mv| {
            mv.mv.from == Square::A7 && mv.mv.to == Square::A8 && mv.promotion
        }));
        assert_eq!(
            promotions,
            vec![Piece::Knight, Piece::Bishop, Piece::Rook, Piece::Queen]
        );
    }

    #[test]
    fn qsearch_searches_quiet_evasions_while_in_check() {
        let board = board("k3r3/8/8/8/8/8/8/4K3 w - - 0 1");
        let mut legal_moves = Vec::new();
        board.generate_moves(|moves| {
            legal_moves.extend(moves);
            false
        });

        assert!(!board.checkers().is_empty());
        assert!(legal_moves.iter().all(|mv| board.piece_on(mv.to).is_none()));

        let (score, nodes) = run_quiesce(&board);
        assert_eq!(score, -500);
        assert_eq!(nodes, 1 + legal_moves.len() as u64);
    }

    #[test]
    fn qsearch_scores_checkmate_and_stalemate() {
        let checkmate = board("7k/6Q1/5K2/8/8/8/8/8 b - - 0 1");
        let stalemate = board("7k/5K2/6Q1/8/8/8/8/8 b - - 0 1");

        assert_eq!(run_quiesce(&checkmate).0, -100_000);
        assert_eq!(run_quiesce(&stalemate).0, 0);
    }
}
