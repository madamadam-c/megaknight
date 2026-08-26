use cozy_chess::{
    Board,
    Color::{self, Black, White},
    File,
    Piece::{self, King, Rook},
    Rank, Square,
};

use crate::engine::EngineMove;

const INPUT_SIZE: usize = 768;
const HIDDEN_SIZE: usize = 16;
const QA: i32 = 255;
const QB: i32 = 64;
const EVAL_SCALE: i32 = 400;
const NETWORK_PAYLOAD_SIZE: usize = (INPUT_SIZE * HIDDEN_SIZE + HIDDEN_SIZE + HIDDEN_SIZE + 1) * 2;
const NETWORK_FILE_SIZE: usize = 24_704;

const NETWORK_BYTES: &[u8; NETWORK_FILE_SIZE] = include_bytes!(
    "../datagen/chessbot-selfplay-100m-10k-100m-checkpoints/chessbot-768-16-1-selfplay-100m-10k-100m-100/quantised.bin"
);
const NETWORK: Network = Network::from_bytes(NETWORK_BYTES);

#[derive(Clone, Copy)]
struct Network {
    feature_weights: [[i16; HIDDEN_SIZE]; INPUT_SIZE],
    feature_bias: [i16; HIDDEN_SIZE],
    output_weights: [i16; HIDDEN_SIZE],
    output_bias: i16,
}

impl Network {
    const fn from_bytes(bytes: &[u8; NETWORK_FILE_SIZE]) -> Self {
        assert!(NETWORK_PAYLOAD_SIZE <= NETWORK_FILE_SIZE);

        let mut offset = 0;
        let mut feature_weights = [[0; HIDDEN_SIZE]; INPUT_SIZE];
        let mut feature = 0;
        while feature < INPUT_SIZE {
            let mut hidden = 0;
            while hidden < HIDDEN_SIZE {
                feature_weights[feature][hidden] = read_i16(bytes, offset);
                offset += 2;
                hidden += 1;
            }
            feature += 1;
        }

        let mut feature_bias = [0; HIDDEN_SIZE];
        let mut hidden = 0;
        while hidden < HIDDEN_SIZE {
            feature_bias[hidden] = read_i16(bytes, offset);
            offset += 2;
            hidden += 1;
        }

        let mut output_weights = [0; HIDDEN_SIZE];
        hidden = 0;
        while hidden < HIDDEN_SIZE {
            output_weights[hidden] = read_i16(bytes, offset);
            offset += 2;
            hidden += 1;
        }

        let output_bias = read_i16(bytes, offset);
        offset += 2;
        assert!(offset == NETWORK_PAYLOAD_SIZE);

        Self {
            feature_weights,
            feature_bias,
            output_weights,
            output_bias,
        }
    }

    #[inline(always)]
    fn evaluate(&self, accumulator: &Accumulator) -> i32 {
        let mut output = 0;
        for hidden in 0..HIDDEN_SIZE {
            let value = i32::from(accumulator.values[hidden]).clamp(0, QA);
            output += value * value * i32::from(self.output_weights[hidden]);
        }

        output /= QA;
        output += i32::from(self.output_bias);
        output * EVAL_SCALE / (QA * QB)
    }
}

const fn read_i16(bytes: &[u8; NETWORK_FILE_SIZE], offset: usize) -> i16 {
    i16::from_le_bytes([bytes[offset], bytes[offset + 1]])
}

const fn output_abs_sum(weights: &[i16; HIDDEN_SIZE]) -> i64 {
    let mut sum = 0;
    let mut hidden = 0;
    while hidden < HIDDEN_SIZE {
        sum += weights[hidden].unsigned_abs() as i64;
        hidden += 1;
    }
    sum
}

const OUTPUT_RAW_BOUND: i64 = output_abs_sum(&NETWORK.output_weights) * QA as i64 * QA as i64;
const OUTPUT_SCALED_BOUND: i64 =
    (OUTPUT_RAW_BOUND / QA as i64 + NETWORK.output_bias.unsigned_abs() as i64) * EVAL_SCALE as i64;
const _: () = assert!(OUTPUT_RAW_BOUND <= i32::MAX as i64);
const _: () = assert!(OUTPUT_SCALED_BOUND <= i32::MAX as i64);

#[repr(align(32))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Accumulator {
    values: [i16; HIDDEN_SIZE],
}

impl Accumulator {
    #[inline(always)]
    fn add(&mut self, feature: usize) {
        for hidden in 0..HIDDEN_SIZE {
            self.values[hidden] =
                self.values[hidden].wrapping_add(NETWORK.feature_weights[feature][hidden]);
        }
    }

    #[inline(always)]
    fn remove(&mut self, feature: usize) {
        for hidden in 0..HIDDEN_SIZE {
            self.values[hidden] =
                self.values[hidden].wrapping_sub(NETWORK.feature_weights[feature][hidden]);
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NnueState {
    accumulators: [Accumulator; 2],
}

impl Default for NnueState {
    fn default() -> Self {
        let accumulator = Accumulator {
            values: NETWORK.feature_bias,
        };
        Self {
            accumulators: [accumulator; 2],
        }
    }
}

impl NnueState {
    pub fn from_board(board: &Board) -> Self {
        let mut state = Self::default();
        for piece in Piece::ALL {
            for color in [White, Black] {
                for square in board.colored_pieces(color, piece) {
                    state.add_piece(color, piece, square);
                }
            }
        }
        state
    }

    #[inline(always)]
    pub fn evaluate(&self, side_to_move: Color) -> i32 {
        NETWORK.evaluate(&self.accumulators[color_index(side_to_move)])
    }

    #[inline(always)]
    pub fn play_move(&mut self, color: Color, mv: &EngineMove) {
        if mv.is_castle {
            self.play_castle(color, mv);
            return;
        }

        self.remove_piece(color, mv.piece_type, mv.mv.from);

        if let Some(victim) = mv.target_type {
            let victim_square = if mv.is_ep {
                Square::new(mv.mv.to.file(), Rank::Fifth.relative_to(color))
            } else {
                mv.mv.to
            };
            self.remove_piece(!color, victim, victim_square);
        }

        self.add_piece(color, mv.mv.promotion.unwrap_or(mv.piece_type), mv.mv.to);
    }

    #[inline(always)]
    fn play_castle(&mut self, color: Color, mv: &EngineMove) {
        let back_rank = Rank::First.relative_to(color);
        let (king_file, rook_file) = if mv.mv.from.file() < mv.mv.to.file() {
            (File::G, File::F)
        } else {
            (File::C, File::D)
        };

        self.remove_piece(color, King, mv.mv.from);
        self.remove_piece(color, Rook, mv.mv.to);
        self.add_piece(color, King, Square::new(king_file, back_rank));
        self.add_piece(color, Rook, Square::new(rook_file, back_rank));
    }

    #[inline(always)]
    fn add_piece(&mut self, color: Color, piece: Piece, square: Square) {
        for perspective in [White, Black] {
            let feature = feature_index(perspective, color, piece, square);
            self.accumulators[color_index(perspective)].add(feature);
        }
    }

    #[inline(always)]
    fn remove_piece(&mut self, color: Color, piece: Piece, square: Square) {
        for perspective in [White, Black] {
            let feature = feature_index(perspective, color, piece, square);
            self.accumulators[color_index(perspective)].remove(feature);
        }
    }
}

#[inline(always)]
const fn color_index(color: Color) -> usize {
    match color {
        White => 0,
        Black => 1,
    }
}

#[inline(always)]
const fn feature_index(perspective: Color, color: Color, piece: Piece, square: Square) -> usize {
    let color_offset = if color_index(perspective) == color_index(color) {
        0
    } else {
        384
    };
    let square = if color_index(perspective) == color_index(White) {
        square as usize
    } else {
        square as usize ^ 56
    };
    color_offset + piece as usize * 64 + square
}

#[cfg(test)]
#[path = "tests/test_nnue.rs"]
mod tests;
