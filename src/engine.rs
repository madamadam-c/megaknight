use std::{
    cmp::{max, min},
    ops::Neg,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

use cozy_chess::{
    Board,
    Color::{self, White},
    Move,
    Piece::{self, Pawn},
};

use crate::{
    evaluate::{static_exchange_evaluation, value},
    nnue::NnueState,
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
    pub target_type: Option<Piece>,
    pub is_castle: bool,
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

const MAX_HISTORY: i32 = 16_384;
fn update_history(value: &mut i32, bonus: i32) {
    let bonus = bonus.clamp(-MAX_HISTORY, MAX_HISTORY);
    *value += bonus - *value * bonus.abs() / MAX_HISTORY;
}

fn history_bonus(depth: i32) -> i32 {
    depth.saturating_mul(depth)
}

#[derive(Clone, Copy, PartialEq)]
enum Stage {
    TtMove,
    GoodCaptures,
    Quiets,
    BadCaptures,
    Done,
}

/// Orders moves lazily: instead of fully sorting every list up front, each
/// `next()` call selects the highest-priority remaining move of the current
/// stage. Nodes usually cut off after a handful of moves, so most ordering
/// work never happens.
struct MovePicker {
    stage: Stage,
    tt_move: Option<EngineMove>,
    good_captures: Vec<EngineMove>,
    quiets: Vec<EngineMove>,
    bad_captures: Vec<EngineMove>,
    stm_index: usize,
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
    fn new(mut moves: MoveList, stm_index: usize) -> Self {
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
            stm_index,
            quiet_remaining,
        }
    }

    fn next(&mut self, history: &[[[i32; 64]; 64]; 2]) -> Option<EngineMove> {
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
                            mv.see_score as i32
                        }
                    }) {
                        return Some(mv);
                    }
                    self.stage = Stage::Quiets;
                }
                Stage::Quiets => {
                    let stm = self.stm_index;
                    let mv = pick_max_prefix(&mut self.quiets, self.quiet_remaining, |mv| {
                        if mv.promotion {
                            mv.material_value + MAX_HISTORY
                        } else {
                            history[stm][mv.mv.from as usize][mv.mv.to as usize]
                        }
                    });
                    self.quiet_remaining = self.quiet_remaining.saturating_sub(1);
                    if let Some(mv) = mv {
                        return Some(mv);
                    }
                    self.stage = Stage::BadCaptures;
                }
                Stage::BadCaptures => {
                    if let Some(mv) = pick_max(&mut self.bad_captures, |mv| mv.see_score as i32) {
                        return Some(mv);
                    }
                    self.stage = Stage::Done;
                }
                Stage::Done => return None,
            }
        }
    }

    /// All quiet moves handed out so far.
    fn tried_quiets(&self) -> &[EngineMove] {
        &self.quiets[self.quiet_remaining..]
    }
}

pub struct Engine {
    tt: Table,
    nnue: NnueState,
    history: [[[i32; 64]; 64]; 2],
}

impl Engine {
    pub fn new() -> Self {
        Self {
            tt: Table::new_for_mb(16),
            nnue: NnueState::default(),
            history: [[[0; 64]; 64]; 2],
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
                let mut emv =
                    EngineMove::new(board, mv, moves_for_piece.piece, tt_move == Some(mv));

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

        let static_eval = self.nnue.evaluate(board.side_to_move());
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
        let mut picker = MovePicker::new(moves, if board.side_to_move() == White { 0 } else { 1 });
        while let Some(mv) = picker.next(&self.history) {
            if !in_check && mv.see_score < -100 {
                continue;
            }
            let mut next_board = board.clone();
            let previous_nnue = self.nnue;
            self.nnue.play_move(board.side_to_move(), &mv);
            next_board.play_unchecked(mv.mv);

            context.history.push(next_board.hash());
            let child = self.quiesce(&next_board, ply + 1, -bounds, context);
            context.history.pop();
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
        pv: bool,
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
        let child = if first_move {
            self.minimax(&next_board, depth, ply, pv, false, -bounds, context)
        } else {
            let null_bounds = SearchBounds {
                alpha: bounds.alpha,
                beta: bounds.alpha + 1,
            };

            match self.minimax(&next_board, depth, ply, false, false, -null_bounds, context) {
                Some(score) if -score > bounds.alpha && -score < bounds.beta && allow_research => {
                    self.minimax(&next_board, depth, ply, pv, false, -bounds, context)
                }
                child => child,
            }
        };
        context.history.pop();
        self.nnue = previous_nnue;

        child.map(|score| -score)
    }

    fn minimax(
        &mut self,
        board: &Board,
        depth: i32,
        ply: i32,
        pv: bool,
        null_position: bool,
        mut bounds: SearchBounds,
        context: &mut SearchContext,
    ) -> Option<i32> {
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
        let static_eval = self.nnue.evaluate(board.side_to_move());

        // let rfp_margin = 350 * depth;
        // if !in_check && !pv && depth <= 5 && static_eval - rfp_margin >= bounds.beta {
        //     return Some(static_eval);
        // }

        if !in_check && 
           !pv && 
           !null_position && 
           static_eval >= bounds.beta && 
           depth >= 3 && 
           board.halfmove_clock() <= 95 && 
           (board.colors(board.side_to_move()) & !(board.pieces(Piece::Pawn) | board.pieces(Piece::King))).len() >= 1
        {
            let r = 3; // depth reduction apparently
            let mut new_board = board.null_move().unwrap();
            new_board.set_halfmove_clock(new_board.halfmove_clock() - 1);

            let null_bounds = SearchBounds {
                alpha: -bounds.beta,
                beta: -bounds.beta + 1,
            };
            let score = -self.minimax(
                &new_board,
                depth - r,
                ply,
                false,
                true,
                null_bounds,
                context,
            )?;

            if score >= bounds.beta {
                return Some(score);
            };
        }

        let moves = self.generate_moves(board, tt_move);
        let stm_index = if board.side_to_move() == White { 0 } else { 1 };
        let mut picker = MovePicker::new(moves, stm_index);

        let mut result = -1_000_000_000;
        let mut best_move: Option<EngineMove> = None;
        let mut first_move = true;

        // let mut moves_played = 0;
        while let Some(mv) = picker.next(&self.history) {
            // moves_played += 1;

            let lmr_depth = 0;
            // let lmr_depth = if depth <= 3 || first_move || mv.is_capture || mv.promotion || in_check {
            //     0
            // } else {
            //     (0.99 + (depth as f32).ln() * (moves_played as f32).ln() / 3.14) as i32
            // };

            let mut x = self.search_move(
                board,
                mv,
                depth - 1 - lmr_depth,
                ply + 1,
                pv,
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
                    pv,
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
                }
            }

            if x >= bounds.beta {
                if !mv.is_capture && !mv.promotion {
                    update_history(
                        &mut self.history[if board.side_to_move() == White { 0 } else { 1 }]
                            [mv.mv.from as usize][mv.mv.to as usize],
                        history_bonus(depth),
                    );
                    for mv2 in picker.tried_quiets() {
                        if mv2.mv != mv.mv {
                            update_history(
                                &mut self.history[stm_index][mv2.mv.from as usize]
                                    [mv2.mv.to as usize],
                                -history_bonus(depth) / 4,
                            );
                        }
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
        context: &mut SearchContext,
    ) -> Option<(Move, i32)> {
        let mut best_move = None;
        let mut best_score = -1_000_000_000;
        let mut bounds = SearchBounds {
            alpha: -1_000_000_000,
            beta: 1_000_000_000,
        };

        let stm_index = if board.side_to_move() == White { 0 } else { 1 };
        let mut picker = MovePicker::new(root_moves.clone(), stm_index);
        let mut first_move = true;
        while let Some(mv) = picker.next(&self.history) {
            let score = self.search_move(
                board,
                mv,
                depth - 1,
                1,
                true,
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
        let mut root_moves: MoveList = self.generate_moves(&request.board, None);

        self.history = [[[0; 64]; 64]; 2];
        self.tt.clear();
        self.nnue = NnueState::from_board(&request.board);

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
            best_move: Some(completed_move.unwrap_or(fallback_move.mv)),
        }
    }
}

#[cfg(test)]
#[path = "tests/test_engine.rs"]
mod tests;
