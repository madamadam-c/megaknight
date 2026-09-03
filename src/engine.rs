use std::{
    cmp::{max, min}, ops::Neg, sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    }, time::{Duration, Instant},
};

use cozy_chess::{
    Board, Color::{self, White}, Move, Piece::{self, Pawn},
};

use crate::{
    engine::NodeType::CUT, evaluate::{static_exchange_evaluation, value}, history::{CONTHIST_PLY, CaptureHistory, ContinuationHistory, CorrectionHistory, MAX_QUIET_HISTORY, PawnHistory, QuietHistory, history_bonus}, nnue::NnueState, transposition::{
        TTNodeType::{self, EXACT, LOWER, UPPER}, Table, TableEntry,
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
#[derive(Copy, Clone)]
struct StackMove {
    mv: Move,
    piece: Piece,
    is_null: bool,
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
    fn new(board: &Board, history: Vec<u64>, limits: SearchLimits, stop: Arc<AtomicBool>) -> Self {
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
    let Some((soft_ms, hard_ms)) = time_budget_ms(board, limits) else {
        return (None, None);
    };

    (Some(start + Duration::from_millis(soft_ms)), Some(start + Duration::from_millis(hard_ms)))
}

fn time_budget_ms(board: &Board, limits: &SearchLimits) -> Option<(u64, u64)> {
    if let Some(movetime) = limits.movetime {
        let budget = movetime.saturating_sub(5).max(1);
        return Some((budget, budget));
    }

    if limits.infinite {
        return None;
    }

    let (remaining, increment) = if board.side_to_move() == Color::White {
        (limits.wtime?, limits.winc.unwrap_or(0))
    } else {
        (limits.btime?, limits.binc.unwrap_or(0))
    };

    let safe = remaining.saturating_sub(100).max(1);
    let moves_to_go = limits.movestogo.unwrap_or(30).clamp(1, 52);
    let increment_share = increment.saturating_mul(3) / 4;

    let soft = (safe / moves_to_go)
        .saturating_add(increment_share)
        .min(safe)
        .max(1);

    let hard = (safe / moves_to_go)
        .saturating_mul(7)
        .saturating_add(increment_share)
        .min(safe)
        .max(soft);

    Some((soft, hard))
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
    pub target_type: Option<Piece>,
    pub is_castle: bool,
    pub history: i32,
}

impl EngineMove {
    #[inline(always)]
    pub fn new(board: &Board, mv: Move, piece: Piece, is_tt: bool) -> Self {
        let mut material_value = 0;
        let mut capture = false;
        let mut ep = false;
        let mut promotion = false;
        let mut target_type = None;
        let is_castle = piece == Piece::King && board.color_on(mv.to) == Some(board.side_to_move());

        if mv.promotion.is_some() {
            material_value += value(mv.promotion.unwrap()) - value(piece);
            promotion = true;
        }

        if board.en_passant().is_some()
            && piece == Pawn
            && mv.from.file() != mv.to.file()
            && board.piece_on(mv.to).is_none()
        {
            material_value += value(Pawn);
            capture = true;
            ep = true;
            target_type = Some(Pawn);
        } else if let Some(target) = board.piece_on(mv.to)
            && board.color_on(mv.to) != board.color_on(mv.from)
        {
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
            target_type: target_type,
            is_castle,
            history: 0,
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
        let q = Vec::with_capacity(if qsearch { 0 } else { 64 });

        Self {
            good_captures: g,
            bad_captures: b,
            quiets: q,
        }
    }

    pub fn is_empty(&self) -> bool {
        return self.good_captures.is_empty()
            && self.bad_captures.is_empty()
            && self.quiets.is_empty();
    }

    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.good_captures.len() + self.quiets.len() + self.bad_captures.len()
    }

    #[cfg(test)]
    pub fn iter(&self) -> impl Iterator<Item = &EngineMove> {
        self.good_captures
            .iter()
            .chain(self.quiets.iter())
            .chain(self.bad_captures.iter())
    }
}

impl IntoIterator for MoveList {
    type Item = EngineMove;

    type IntoIter = std::iter::Flatten<std::array::IntoIter<Vec<EngineMove>, 3>>;

    fn into_iter(self) -> Self::IntoIter {
        [self.good_captures, self.quiets, self.bad_captures]
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

#[derive(Clone, Copy, PartialEq)]
enum Stage {
    TtMove,
    GoodCaptures,
    Quiets,
    BadCaptures,
    Done,
}

struct MovePicker {
    stage: Stage,
    tt_move: Option<EngineMove>,
    good_captures: Vec<EngineMove>,
    quiets: Vec<EngineMove>,
    bad_captures: Vec<EngineMove>,
    tried_captures: Vec<EngineMove>,
    quiet_remaining: usize,
}

fn pick_max(list: &mut Vec<EngineMove>, key: impl Fn(&EngineMove) -> i32) -> Option<EngineMove> {
    let len = list.len();
    if len == 0 {
        return None;
    }
    let mut best = 0;
    let mut best_key = key(&list[0]);
    for i in 1..len {
        let k = key(&list[i]);
        if k > best_key {
            best = i;
            best_key = k;
        }
    }
    Some(list.swap_remove(best))
}

fn pick_max_prefix(
    list: &mut [EngineMove],
    len: usize,
    key: impl Fn(&EngineMove) -> i32,
) -> Option<EngineMove> {
    if len == 0 {
        return None;
    }
    let mut best = 0;
    let mut best_key = key(&list[0]);
    for (i, mv) in list.iter().enumerate().take(len).skip(1) {
        let k = key(mv);
        if k > best_key {
            best = i;
            best_key = k;
        }
    }
    list.swap(best, len - 1);
    Some(list[len - 1])
}

impl MovePicker {
    fn new(mut moves: MoveList) -> Self {
        let tt_move = moves
            .good_captures
            .iter()
            .position(|mv| mv.is_tt)
            .map(|i| moves.good_captures.swap_remove(i));
        let stage = if tt_move.is_some() {
            Stage::TtMove
        } else {
            Stage::GoodCaptures
        };

        let quiet_remaining = moves.quiets.len();
        Self {
            stage,
            tt_move,
            good_captures: moves.good_captures,
            quiets: moves.quiets,
            bad_captures: moves.bad_captures,
            tried_captures: Vec::with_capacity(8),
            quiet_remaining,
        }
    }

    fn next(&mut self) -> Option<EngineMove> {
        loop {
            match self.stage {
                Stage::TtMove => {
                    self.stage = Stage::GoodCaptures;
                    if let Some(mv) = self.tt_move.take() {
                        return Some(mv);
                    }
                }
                Stage::GoodCaptures => {
                    if let Some(mv) = pick_max(&mut self.good_captures, |mv| {
                        if mv.is_tt {
                            i32::MAX
                        } else {
                            // mv.history
                            mv.see_score as i32
                            // 0
                        }
                    }) {
                        self.tried_captures.push(mv);
                        return Some(mv);
                    }
                    self.stage = Stage::Quiets;
                }
                Stage::Quiets => {
                    let mv = pick_max_prefix(&mut self.quiets, self.quiet_remaining, |mv| {
                        if mv.promotion {
                            mv.material_value + 8*MAX_QUIET_HISTORY
                        } else {
                            mv.history
                        }
                    });
                    self.quiet_remaining = self.quiet_remaining.saturating_sub(1);
                    if let Some(mv) = mv {
                        return Some(mv);
                    }
                    self.stage = Stage::BadCaptures;
                }
                Stage::BadCaptures => {
                    if let Some(mv) = pick_max(&mut self.bad_captures, |mv| 
                        mv.see_score as i32
                        // mv.history
                        // 0
                        ) {
                        self.tried_captures.push(mv);
                        return Some(mv);
                    }
                    self.stage = Stage::Done;
                }
                Stage::Done => return None,
            }
        }
    }

    fn tried_quiets(&self) -> &[EngineMove] {
        &self.quiets[self.quiet_remaining..]
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum NodeType {
    CUT,
    ALL, 
    PV,
}

impl NodeType {
    fn is_pv(self) -> bool {
        self == Self::PV
    }

    fn first_child(self) -> Self {
        match self {
            Self::PV => Self::PV,
            Self::CUT => Self::ALL,
            Self::ALL => Self::CUT
        }
    }
}

const DUMMY_NULL: [StackMove; 2] = [
    StackMove{mv: Move{from: cozy_chess::Square::A2, to: cozy_chess::Square::A1, promotion: None}, piece: Pawn, is_null: true}, 
    StackMove{mv: Move{from: cozy_chess::Square::A7, to: cozy_chess::Square::A8, promotion: None}, piece: Pawn, is_null: true}
];

pub struct Engine {
    tt: Table,
    nnue: NnueState,
    quiet_history: QuietHistory,
    continuation_history: ContinuationHistory,
    capture_history: CaptureHistory,
    pawn_correction_history: CorrectionHistory,
    stm_non_pawn_correction_history: CorrectionHistory,
    nstm_non_pawn_correction_history: CorrectionHistory,
    minor_correction_history: CorrectionHistory,
    major_correction_history: CorrectionHistory,
    pawn_history: PawnHistory,
    move_stack: Vec<StackMove>,
}

impl Engine {
    pub fn new() -> Self {
        Self {
            tt: Table::new_for_mb(16),
            nnue: NnueState::default(),
            quiet_history: QuietHistory::new(),
            continuation_history: ContinuationHistory::new(),
            capture_history: CaptureHistory::new(),
            pawn_correction_history: CorrectionHistory::new(),
            stm_non_pawn_correction_history: CorrectionHistory::new(),
            nstm_non_pawn_correction_history: CorrectionHistory::new(),
            minor_correction_history: CorrectionHistory::new(),
            major_correction_history: CorrectionHistory::new(),
            pawn_history: PawnHistory::new(),
            move_stack: Vec::with_capacity(256),
        }
    }

    pub fn new_game(&mut self) {
        self.pawn_correction_history.clear();
        self.stm_non_pawn_correction_history.clear();
        self.nstm_non_pawn_correction_history.clear();
        self.minor_correction_history.clear();
        self.major_correction_history.clear();
        self.continuation_history.clear();
        self.pawn_history.clear();
        self.tt.clear();
    }

    pub fn set_hash_size_mb(&mut self, megabytes: u64) {
        self.tt = Table::new_for_mb(megabytes);
    }

    pub fn get_correction_value(&self, board: &Board) -> i32 {
        let pawn_hash = board.pawn_hash(board.side_to_move()) ^ board.pawn_hash(!board.side_to_move());
        let minor_hash = board.minor_piece_hash(board.side_to_move()) ^ board.minor_piece_hash(!board.side_to_move());
        let major_hash = board.major_piece_hash(board.side_to_move()) ^ board.major_piece_hash(!board.side_to_move());
        let stm_non_pawn_hash = board.non_pawn_hash(board.side_to_move());
        let nstm_non_pawn_hash = board.non_pawn_hash(!board.side_to_move());

        let correction = (
            0 // just for formatting reasons
            + 0*self.pawn_correction_history.get(board.side_to_move() as usize, pawn_hash)
            + 0*self.minor_correction_history.get(board.side_to_move() as usize, minor_hash)
            + 0*self.major_correction_history.get(board.side_to_move() as usize, major_hash)
            + 0*self.stm_non_pawn_correction_history.get(board.side_to_move() as usize, stm_non_pawn_hash)
            + 0*self.nstm_non_pawn_correction_history.get(board.side_to_move() as usize, nstm_non_pawn_hash)
        ) / 64;
        return correction;
    }

    fn generate_moves(&self, board: &Board, tt_move: Option<Move>) -> MoveList {
        let mut moves = MoveList::new(false);
        let pawn_hash = board.pawn_hash(board.side_to_move()) ^ board.pawn_hash(!board.side_to_move());

        board.generate_moves(|moves_for_piece| {
            for mv in moves_for_piece {
                let mut emv = EngineMove::new(board, mv, moves_for_piece.piece, tt_move == Some(mv));

                if emv.is_capture {
                    emv.see_score = static_exchange_evaluation(board, &emv);
                    emv.history = self.capture_history.get(board.side_to_move() as usize, mv.to, emv.piece_type, emv.target_type.unwrap());
                } else {
                    let quiet_history = self.quiet_history.get(board.side_to_move() as usize, mv.from, mv.to);
                    let pawn_history = self.pawn_history.get(board.side_to_move() as usize, pawn_hash, mv.to, emv.piece_type);
                    let mut continuation_history = 0;

                    // for ply in 0..min(self.move_stack.len(), CONTHIST_PLY) {
                    //     let prev = &self.move_stack[self.move_stack.len()-ply-1];
                    //     continuation_history += self.continuation_history.get(
                    //         ply, board.side_to_move() as usize, prev.mv.to, prev.piece, mv.to, emv.piece_type
                    //     );
                    // }
                    
                    emv.history = quiet_history + pawn_history + continuation_history;
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
                if moves_for_piece.piece == Pawn
                    && mv.to.file() == mv.from.file()
                    && mv.promotion.is_none()
                {
                    continue;
                }
                let mut emv = EngineMove::new(board, mv, moves_for_piece.piece, false);

                if emv.is_capture {
                    emv.history = self.capture_history.get(board.side_to_move() as usize, mv.to, emv.piece_type, emv.target_type.unwrap());
                    emv.see_score = static_exchange_evaluation(board, &emv);
                    if emv.see_score >= 0 {
                        moves.good_captures.push(emv);
                    } else {
                        moves.bad_captures.push(emv);
                    }
                } else if emv.promotion {
                    moves.quiets.push(emv);
                    emv.history = self.quiet_history.get(board.side_to_move() as usize, mv.from, mv.to);
                }
            }
            false
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

        let mut static_eval = self.nnue.evaluate(board.side_to_move());
        let correction = self.get_correction_value(board);
        static_eval += correction;

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
            result = static_eval;
        }
        let mut picker = MovePicker::new(moves);
        while let Some(mv) = picker.next() {
            if !in_check && mv.see_score < -100 {
                continue;
            }
            let mut next_board = board.clone();
            let previous_nnue = self.nnue;
            self.nnue.play_move(board.side_to_move(), &mv);
            next_board.play_unchecked(mv.mv);

            context.history.push(next_board.hash());
            self.move_stack.push(StackMove {mv: mv.mv, piece: mv.piece_type, is_null: false});
            let child = self.quiesce(&next_board, ply + 1, -bounds, context);
            context.history.pop();
            self.move_stack.pop();
            self.nnue = previous_nnue;

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

    fn search_move(
        &mut self,
        board: &Board,
        mv: EngineMove,
        depth: i32,
        ply: i32,
        expected: NodeType,
        bounds: SearchBounds,
        allow_research: bool,
        first_move: bool,
        context: &mut SearchContext,
    ) -> Option<i32> {
        let mut next_board = board.clone();
        let previous_nnue = self.nnue;
        self.nnue.play_move(board.side_to_move(), &mv);
        next_board.play_unchecked(mv.mv);

        context.history.push(next_board.hash());
        self.move_stack.push(StackMove { mv: mv.mv, piece: mv.piece_type, is_null: false });

        let child = if first_move {
            self.minimax(&next_board, depth, ply, expected.first_child(), false, -bounds, context)
        } else {
            let null_bounds = SearchBounds {
                alpha: bounds.alpha,
                beta: bounds.alpha + 1,
            };

            match self.minimax(&next_board, depth, ply, NodeType::CUT, false, -null_bounds, context) {
                Some(score) if -score > bounds.alpha && -score < bounds.beta && allow_research => {
                    self.minimax(&next_board, depth, ply, expected.first_child(), false, -bounds, context)
                }
                child => child,
            }
        };

        self.move_stack.pop();
        context.history.pop();
        self.nnue = previous_nnue;

        child.map(|score| -score)
    }

    fn minimax(
        &mut self,
        board: &Board,
        mut depth: i32,
        ply: i32,
        expected: NodeType,
        null_position: bool,
        mut bounds: SearchBounds,
        context: &mut SearchContext,
    ) -> Option<i32> {
        let pv = expected.is_pv();

        if depth <= 0 {
            return self.quiesce(board, ply, bounds, context);
        }

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

        if !board.generate_moves(|_| true) {
            return Some(terminal_score(board, ply));
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

        let in_check = !board.checkers().is_empty();
        let mut static_eval = self.nnue.evaluate(board.side_to_move());
        let mut correction = self.get_correction_value(board);
        static_eval += correction;

        // let razoring_margin = 163 + 41 * depth * depth;
        // if !in_check && !pv && static_eval + razoring_margin < bounds.alpha {
        //     return Some(static_eval);
        // }

        let rfp_margin = 123 * depth; // + correction.abs() / 2;
        if !in_check && !pv && depth <= 5 && static_eval - rfp_margin >= bounds.beta {
            return Some(static_eval);
        }

        if !in_check && 
           !pv && 
           !null_position && 
           static_eval >= bounds.beta && 
           depth >= 3 && 
           board.halfmove_clock() <= 95 && 
           (board.colors(board.side_to_move()) & !(board.pieces(Piece::Pawn) | board.pieces(Piece::King))).len() >= 1
        {
            let r = 3 + depth / 4 + min(3, (static_eval - bounds.beta) / 200); // depth reduction apparently
            let mut new_board = board.null_move().unwrap();
            new_board.set_halfmove_clock(new_board.halfmove_clock() - 1);

            let null_bounds = SearchBounds {
                alpha: -bounds.beta,
                beta: -bounds.beta + 1,
            };
            
            self.move_stack.push(DUMMY_NULL[board.side_to_move() as usize]);
            let score = -self.minimax(
                &new_board,
                depth - r,
                ply+1,
                NodeType::ALL,
                true,
                null_bounds,
                context,
            )?;
            self.move_stack.pop();

            if score >= bounds.beta {
                return Some(score);
            };
        }

        // iir
        // if !in_check && expected != NodeType::ALL && depth >= 6 && tt_move.is_none() {
        //     depth -= 1;
        // }

        let moves = self.generate_moves(board, tt_move);
        let stm_index = if board.side_to_move() == White { 0 } else { 1 };
        let mut picker = MovePicker::new(moves);

        let mut result = -1_000_000_000;
        let mut best_move: Option<EngineMove> = None;
        let mut first_move = true;

        let mut moves_played = 0;
        let lmp_cap = 5 + 3*depth*depth;

        // let mut actual = NodeType::ALL;
        while let Some(mv) = picker.next() {
            moves_played += 1;

            // lmp
            if !pv && !in_check && moves_played >= lmp_cap && !mv.is_capture && !mv.promotion {
                picker.stage = Stage::BadCaptures;
                continue;
            }   

            // lmr
            let mut lmr_depth = 0;
            if depth >= 2 && moves_played >= 2 && !mv.is_capture && !mv.promotion && !in_check {
                // base formula
                lmr_depth += (1024.0 * (0.99 + (depth as f32).ln() * (moves_played as f32).ln() / 3.14)) as i32;

                // reduce more on a cutnode
                // lmr_depth += 500 * expected == CUT as i32;

                // reduce less for a check
                // lmr_depth -= 800 * in_check as i32;

                // reduce less for pv nodes
                // lmr_depth -= 500 * pv as i32;
                
                lmr_depth = lmr_depth.max(0) / 1024;
            }

            let mut x = self.search_move(
                board,
                mv,
                depth - 1 - lmr_depth,
                ply + 1,
                expected,
                bounds,
                lmr_depth == 0,
                first_move,
                context,
            )?;

            if lmr_depth > 0 && x > bounds.alpha {
                x = self.search_move(
                    board,
                    mv,
                    depth - 1,
                    ply + 1,
                    expected,
                    bounds,
                    true,
                    first_move,
                    context,
                )?;
            }

            if x > result {
                result = x;
                best_move = Some(mv);
                if x > bounds.alpha {
                    bounds.alpha = x;
                    // actual = NodeType::PV;
                }
            }

            if x >= bounds.beta {
                // actual = NodeType::CUT;
                if !mv.is_capture && !mv.promotion {
                    let pawn_hash = board.pawn_hash(board.side_to_move()) ^ board.pawn_hash(!board.side_to_move());
                    
                    self.quiet_history.update(stm_index, mv.mv.from, mv.mv.to, history_bonus(depth));
                    self.pawn_history.update(stm_index, pawn_hash, mv.mv.to, mv.piece_type, history_bonus(depth));

                    for mv2 in picker.tried_quiets() {
                        if mv2.mv != mv.mv {
                            self.quiet_history.update(stm_index,mv2.mv.from, mv2.mv.to, -history_bonus(depth));
                            self.pawn_history.update(stm_index, pawn_hash, mv2.mv.to, mv2.piece_type, -history_bonus(depth));
                        }
                    }

                    for ply in 0..min(self.move_stack.len(), CONTHIST_PLY) {
                        let prev = &self.move_stack[self.move_stack.len()-ply-1];
                        if prev.is_null {continue;}

                        self.continuation_history.update(ply, stm_index, prev.mv.to, prev.piece, mv.mv.to, mv.piece_type, history_bonus(depth));
                    
                        for mv2 in picker.tried_quiets() {
                            if mv2.mv == mv.mv {continue;}
                            self.continuation_history.update(ply, stm_index, prev.mv.to, prev.piece, mv2.mv.to, mv2.piece_type, -history_bonus(depth));
                        }
                    }
                } else if mv.is_capture {
                    self.capture_history.update(stm_index, mv.mv.to, mv.piece_type, mv.target_type.unwrap(), history_bonus(depth));
                    for mv2 in picker.tried_captures {
                        if mv2.mv == mv.mv {continue;}
                        self.capture_history.update(stm_index, mv2.mv.to, mv2.piece_type, mv2.target_type.unwrap(), -history_bonus(depth)/4);
                    }
                }
                break;
            }

            first_move = false;
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

        // update correction histories
        static_eval -= correction;
        correction = self.get_correction_value(board);
        static_eval += correction;

        if !in_check && best_move.is_none_or(|mv| !mv.is_capture && !mv.promotion) &&
            !(node_type == TTNodeType::LOWER && result <= static_eval) && !(node_type == TTNodeType::UPPER && result >= static_eval)
            && result.abs() <= 95_000
        {
            let bonus = (result - static_eval) * depth / 4;
            let pawn_hash = board.pawn_hash(board.side_to_move()) ^ board.pawn_hash(!board.side_to_move());
            let minor_hash = board.minor_piece_hash(board.side_to_move()) ^ board.minor_piece_hash(!board.side_to_move());
            let major_hash = board.major_piece_hash(board.side_to_move()) ^ board.major_piece_hash(!board.side_to_move());
            let stm_non_pawn_hash = board.non_pawn_hash(board.side_to_move());
            let nstm_non_pawn_hash = board.non_pawn_hash(!board.side_to_move());

            self.pawn_correction_history.update(stm_index, pawn_hash, bonus);
            self.minor_correction_history.update(stm_index, minor_hash, bonus);
            self.major_correction_history.update(stm_index, major_hash, bonus);
            self.stm_non_pawn_correction_history.update(stm_index, stm_non_pawn_hash, bonus);
            self.nstm_non_pawn_correction_history.update(stm_index, nstm_non_pawn_hash, bonus);
        }

        self.tt.insert(
            key,
            TableEntry {
                key: key,
                score: score_to_tt(result, ply),
                depth: depth,
                node_type: node_type,
                best_move: best_move.map(|mv| mv.mv),
            },
        );
        Some(result)
    }

    fn root_search(
        &mut self,
        board: &Board,
        depth: i32,
        root_moves: &MoveList,
        mut bounds: SearchBounds,
        context: &mut SearchContext,
    ) -> Option<(Move, i32)> {
        let mut best_move = None;
        let mut best_score = -1_000_000_000;

        let mut picker = MovePicker::new(root_moves.clone());
        let mut first_move = true;
        while let Some(mv) = picker.next() {
            let score = self.search_move(
                board,
                mv,
                depth - 1,
                1,
                NodeType::PV,
                bounds,
                true,
                first_move,
                context,
            )?;

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

            first_move = false;
        }

        best_move.map(|mv| (mv.mv, best_score))
    }

    pub fn search<F>(&mut self, request: &SearchRequest, mut report: F) -> SearchResult
    where
        F: FnMut(SearchInfo),
    {
        let root_key = request.board.hash();
        let root_tt_move = self
            .tt
            .get(root_key)
            .and_then(|entry| entry.best_move)
            .filter(|&mv| request.board.is_legal(mv));
        let mut root_moves: MoveList = self.generate_moves(&request.board, root_tt_move);

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
        let mut completed_move = root_tt_move;
        let mut depth = 1;

        self.move_stack.clear();
        self.quiet_history.clear();
        self.capture_history.clear();
        self.nnue = NnueState::from_board(&request.board);

        let mut previous_score = 0;
        while depth <= max_depth {
            if context.should_stop() {
                break;
            }

            root_moves = self.generate_moves(&request.board, completed_move);
            if !request.limits.searchmoves.is_empty() {
                root_moves
                    .good_captures
                    .retain(|mv| request.limits.searchmoves.contains(&mv.mv));
                root_moves
                    .bad_captures
                    .retain(|mv| request.limits.searchmoves.contains(&mv.mv));
                root_moves
                    .quiets
                    .retain(|mv| request.limits.searchmoves.contains(&mv.mv));
            }

            let mut alpha_delta = 20;
            let mut beta_delta = 20;

            loop {
                // aspiration windows
                let bounds = SearchBounds {
                    alpha: if depth <= 4 {-1_000_000_000} else {previous_score - alpha_delta},
                    beta: if depth <= 4 {1_000_000_000} else {previous_score + beta_delta},
                };

                let Some((best_move, score)) =
                    self.root_search(&request.board, depth, &root_moves, bounds, &mut context)
                else {
                    break;
                };

                if score <= bounds.alpha {
                    alpha_delta *= 2;
                } else if score >= bounds.beta {
                    beta_delta *= 2;
                } else {
                    previous_score = score;
                    completed_move = Some(best_move);
                    if request.limits.searchmoves.is_empty() {
                        self.tt.insert(
                            root_key,
                            TableEntry {
                                key: root_key,
                                score: score_to_tt(score, 0),
                                depth,
                                node_type: EXACT,
                                best_move: Some(best_move),
                            },
                        );
                    }

                    report(SearchInfo {
                        depth,
                        score,
                        nodes: context.nodes,
                        elapsed: context.elapsed(),
                        pv: vec![best_move],
                    });
                    break;
                }
            }

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
