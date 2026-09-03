use std::array;

use cozy_chess::{Piece, Square};

pub const MAX_QUIET_HISTORY: i32 = 16_384;
pub const MAX_CORRECTION_HISTORY: i32 = 8192;


fn update_history(value: &mut i16, bonus: i32, scale: i32) {
    let bonus = bonus.clamp(-scale, scale);
    let v = *value as i32;
    *value += (bonus - v * bonus.abs() / scale) as i16;
}

pub fn history_bonus(depth: i32) -> i32 {
    depth.saturating_mul(depth)
}

pub const CONTHIST_PLY: usize = 1;
pub const CORRHIST_SIZE: usize = 8192;
pub const PAWNHIST_SIZE: usize = 4096;

type QHEntry = [[[i16; 64]; 64]; 2];
type ContHistEntry = [[[[[i16; 6]; 64]; 6]; 64]; 2];
type CaptureHistEntry = [[[[i16; 6]; 6]; 64]; 2];
type CorrHistEntry = [[i16; CORRHIST_SIZE]; 2];
type PawnHistEntry = [[[[i16; 64]; 6]; PAWNHIST_SIZE]; 2];

pub struct QuietHistory {
    history: Box<QHEntry>
}

impl QuietHistory {
    pub fn new() -> Self {
        Self {
            history: Box::new([[[0i16; 64]; 64]; 2])
        }
    }

    pub fn clear(&mut self) {
        self.history = Box::new([[[0i16; 64]; 64]; 2]);
    }

    pub fn get(&self, stm: usize, from: Square, to: Square) -> i32 {
        self.history[stm][from as usize][to as usize] as i32
    }

    pub fn update(&mut self, stm: usize, from: Square, to: Square, bonus: i32) {
        update_history(
            &mut self.history[stm][from as usize][to as usize], 
            bonus,
            MAX_QUIET_HISTORY
        );
    }
}

pub struct ContinuationHistory {
    history: [Box<ContHistEntry>; CONTHIST_PLY]
}

impl ContinuationHistory {
    pub fn new() -> Self {
        Self {
            history: array::from_fn(|_| {Box::new([[[[[0i16; 6]; 64]; 6]; 64]; 2])})
        }
    }

    pub fn clear(&mut self) {
        self.history = array::from_fn(|_| {Box::new([[[[[0i16; 6]; 64]; 6]; 64]; 2])})
    }

    pub fn get(&self, ply: usize, stm: usize, prev_to: Square, prev_piece: Piece, to: Square, piece: Piece) -> i32 {
        self.history[ply][stm][prev_to as usize][prev_piece as usize][to as usize][piece as usize] as i32
    }

    pub fn update(&mut self, ply: usize, stm: usize, prev_to: Square, prev_piece: Piece, to: Square, piece: Piece, bonus: i32) {
        update_history(
            &mut self.history[ply][stm][prev_to as usize][prev_piece as usize][to as usize][piece as usize], 
            bonus,
            MAX_QUIET_HISTORY
        );
    }
}

pub struct CaptureHistory {
    history: Box<CaptureHistEntry>
}

impl CaptureHistory {
    pub fn new() -> Self {
        Self {
            history: Box::new([[[[0i16; 6]; 6]; 64]; 2])
        }
    }

    pub fn clear(&mut self) {
        self.history = Box::new([[[[0i16; 6]; 6]; 64]; 2])
    }

    pub fn get(&self, stm: usize, to: Square, piece: Piece, target: Piece) -> i32 {
        self.history[stm][to as usize][piece as usize][target as usize] as i32
    }

    pub fn update(&mut self, stm: usize, to: Square, piece: Piece, target: Piece, bonus: i32) {
        update_history(
            &mut self.history[stm][to as usize][piece as usize][target as usize], 
            bonus,
            MAX_QUIET_HISTORY
        );
    }
}

pub struct CorrectionHistory {
    history: Box<CorrHistEntry>
}

impl CorrectionHistory {
    pub fn new() -> Self {
        Self {
            history: Box::new([[0i16; CORRHIST_SIZE]; 2])
        }
    }

    pub fn clear(&mut self) {
        self.history = Box::new([[0i16; CORRHIST_SIZE]; 2]);
    }

    pub fn get(&self, stm: usize, pawn_hash: u64) -> i32 {
        self.history[stm][(pawn_hash as usize) & (CORRHIST_SIZE - 1)] as i32
    }

    pub fn update(&mut self, stm: usize, hash: u64, bonus: i32) {
        update_history(
            &mut self.history[stm][(hash as usize) & (CORRHIST_SIZE - 1)], 
            bonus.clamp(-MAX_CORRECTION_HISTORY/4, MAX_CORRECTION_HISTORY/4),
            MAX_CORRECTION_HISTORY
        );
    }
}

pub struct PawnHistory {
    history: Box<PawnHistEntry>
}

impl PawnHistory {
    pub fn new() -> Self {
        Self {
            history: Box::new([[[[0i16; 64]; 6]; PAWNHIST_SIZE]; 2])
        }
    }

    pub fn clear(&mut self) {
        self.history = Box::new([[[[0i16; 64]; 6]; PAWNHIST_SIZE]; 2])
    }

    pub fn get(&self, stm: usize, pawn_hash: u64, to: Square, piece_type: Piece) -> i32 {
        self.history[stm][(pawn_hash as usize) & (PAWNHIST_SIZE - 1)][piece_type as usize][to as usize] as i32
    }

    pub fn update(&mut self, stm: usize, pawn_hash: u64, to: Square, piece_type: Piece, bonus: i32) {
        update_history(
            &mut self.history[stm][(pawn_hash as usize) & (PAWNHIST_SIZE - 1)][piece_type as usize][to as usize],
            bonus,
            MAX_QUIET_HISTORY
        );
    }
}   