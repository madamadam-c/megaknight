use std::arch::x86_64::{__m256i, _mm256_add_epi32, _mm256_and_si256, _mm256_cmpgt_epi16, _mm256_loadu_si256, _mm256_madd_epi16, _mm256_max_epi16, _mm256_min_epi16, _mm256_mullo_epi16, _mm256_set1_epi16, _mm256_setzero_si256, _mm256_slli_epi32, _mm256_storeu_si256};

use cozy_chess::{
    Board,
    Color::{self, Black, White},
    File,
    Piece::{self, King, Rook},
    Rank, Square,
};

use crate::engine::EngineMove;

const INPUT_SIZE: usize = 768;
const QA: i32 = 255;
const QB: i32 = 64;
const EVAL_SCALE: i32 = 400;
const NETWORK_ALIGNMENT: usize = 64;

const NETWORK_BYTES: &[u8] = include_bytes!("../networks/17_09_26-2.bin");
const NETWORK_FILE_SIZE: usize = NETWORK_BYTES.len();
const HIDDEN_SIZE: usize = hidden_size_for_file_size(NETWORK_FILE_SIZE);
const OUTPUT_INPUT_SIZE: usize = 2 * HIDDEN_SIZE;
const NETWORK_PAYLOAD_SIZE: usize = network_payload_size(HIDDEN_SIZE);
const NETWORK: Network = Network::from_bytes(NETWORK_BYTES);

#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct Network {
    feature_weights: [[i16; HIDDEN_SIZE]; INPUT_SIZE],
    feature_bias: [i16; HIDDEN_SIZE],
    output_weights: [i16; OUTPUT_INPUT_SIZE],
    output_bias: i16,
}

impl Network {
    const fn from_bytes(bytes: &[u8]) -> Self {
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

        let mut output_weights = [0; OUTPUT_INPUT_SIZE];
        hidden = 0;
        while hidden < OUTPUT_INPUT_SIZE {
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
    fn evaluate(&self, us: &Accumulator, them: &Accumulator) -> i32 {
        if is_x86_feature_detected!("avx2") {
            unsafe {return self.avx2_evaluate(us, them);}
        } else {
            return self.save_evaluate(us, them);
        }
    }

    /*
    simd is weird
    */

    #[target_feature(enable = "avx2")]
    unsafe fn avx2_evaluate(&self, us: &Accumulator, them: &Accumulator) -> i32 {
        let lower = _mm256_setzero_si256();
        let upper = _mm256_set1_epi16(QA as i16);
        let ones  = _mm256_set1_epi16(1);

        let mut output_store = _mm256_setzero_si256();

        let mut idx = 0;
        while idx < HIDDEN_SIZE {
            let mut us_chunk   = unsafe {_mm256_loadu_si256(us.values.as_ptr().add(idx) as *const __m256i)};
            let mut them_chunk = unsafe {_mm256_loadu_si256(them.values.as_ptr().add(idx) as *const __m256i)};
            
            // clamp
            us_chunk   = _mm256_max_epi16(_mm256_min_epi16(us_chunk, upper), lower);
            them_chunk = _mm256_max_epi16(_mm256_min_epi16(them_chunk, upper), lower);

            // square
            us_chunk   = _mm256_mullo_epi16(us_chunk, us_chunk);
            them_chunk = _mm256_mullo_epi16(them_chunk, them_chunk);

            // some might appear negative because of the squaring, find which ones it is
            let mut us_neg   = _mm256_cmpgt_epi16(lower, us_chunk); // determines the indices where 0 > value
            let mut them_neg = _mm256_cmpgt_epi16(lower, them_chunk); // same here

            // load weights
            let us_weights   = unsafe {_mm256_loadu_si256(self.output_weights.as_ptr().add(idx) as *const __m256i)};
            let them_weights = unsafe {_mm256_loadu_si256(self.output_weights.as_ptr().add(idx+HIDDEN_SIZE) as *const __m256i)};

            // this instruction multiplies by the weights, and turns it from 16x i16s into 8x i32s by adding adjacent pairs
            // eg. [a,b] x [c,d] = [a*c+b*d]
            us_chunk   = _mm256_madd_epi16(us_chunk, us_weights);
            them_chunk = _mm256_madd_epi16(them_chunk, them_weights);
            
            // determine which weights we need to add again
            us_neg   = _mm256_and_si256(us_neg, us_weights);
            them_neg = _mm256_and_si256(them_neg, them_weights);

            // do the same pair sum business to get it into the same form, using an array of ones as a dummy
            us_neg   = _mm256_madd_epi16(us_neg, ones);
            them_neg = _mm256_madd_epi16(them_neg, ones);

            // multiply by 65536 because its the high bit
            us_neg   = _mm256_slli_epi32::<16>(us_neg);
            them_neg = _mm256_slli_epi32::<16>(them_neg);

            // add the correction terms back
            us_chunk   = _mm256_add_epi32(us_chunk, us_neg);
            them_chunk = _mm256_add_epi32(them_chunk, them_neg);

            // add to the output buffer
            output_store = _mm256_add_epi32(output_store, us_chunk);
            output_store = _mm256_add_epi32(output_store, them_chunk);

            idx += 16;
        }

        let mut values = [0i32; 8];
        unsafe {_mm256_storeu_si256(values.as_mut_ptr().cast::<__m256i>(), output_store) };

        let mut output: i32 = values.into_iter().sum();

        output /= QA;
        output += i32::from(self.output_bias);

        return (i64::from(output) * i64::from(EVAL_SCALE) / i64::from(QA * QB)) as i32;
    }

    #[inline(always)]
    fn save_evaluate(&self, us: &Accumulator, them: &Accumulator) -> i32 {
        let mut output = 0;
        for hidden in 0..HIDDEN_SIZE {
            let us_value = i32::from(us.values[hidden]).clamp(0, QA);
            let them_value = i32::from(them.values[hidden]).clamp(0, QA);
            output += us_value * us_value * i32::from(self.output_weights[hidden]);
            output +=
                them_value * them_value * i32::from(self.output_weights[HIDDEN_SIZE + hidden]);
        }

        output /= QA;
        output += i32::from(self.output_bias);
        (i64::from(output) * i64::from(EVAL_SCALE) / i64::from(QA * QB)) as i32
    }
}

const fn read_i16(bytes: &[u8], offset: usize) -> i16 {
    i16::from_le_bytes([bytes[offset], bytes[offset + 1]])
}

const fn network_payload_size(hidden_size: usize) -> usize {
    (INPUT_SIZE * hidden_size + hidden_size + 2 * hidden_size + 1) * 2
}

const fn padded_network_file_size(hidden_size: usize) -> usize {
    let payload_size = network_payload_size(hidden_size);
    (payload_size + NETWORK_ALIGNMENT - 1) / NETWORK_ALIGNMENT * NETWORK_ALIGNMENT
}

const fn hidden_size_for_file_size(file_size: usize) -> usize {
    let mut hidden_size = 1;
    while padded_network_file_size(hidden_size) <= file_size {
        if padded_network_file_size(hidden_size) == file_size {
            return hidden_size;
        }
        hidden_size += 1;
    }
    panic!("NNUE file size does not match a padded 768->N->N->1 network");
}

const fn output_abs_sum(weights: &[i16; OUTPUT_INPUT_SIZE]) -> i64 {
    let mut sum = 0;
    let mut hidden = 0;
    while hidden < OUTPUT_INPUT_SIZE {
        sum += weights[hidden].unsigned_abs() as i64;
        hidden += 1;
    }
    sum
}

const OUTPUT_RAW_BOUND: i64 = output_abs_sum(&NETWORK.output_weights) * QA as i64 * QA as i64;
const OUTPUT_SCALED_BOUND: i64 =
    (OUTPUT_RAW_BOUND / QA as i64 + NETWORK.output_bias.unsigned_abs() as i64) * EVAL_SCALE as i64;
const _: () = assert!(OUTPUT_RAW_BOUND <= i32::MAX as i64);
const _: () = assert!(OUTPUT_SCALED_BOUND / (QA * QB) as i64 <= i32::MAX as i64);

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
    square_xors: [u8; 2],
}

impl Default for NnueState {
    fn default() -> Self {
        let accumulator = Accumulator {
            values: NETWORK.feature_bias,
        };
        Self {
            accumulators: [accumulator; 2],
            square_xors: [0, 56],
        }
    }
}

impl NnueState {
    pub fn from_board(board: &Board) -> Self {
        let mut state = Self::default();
        state.square_xors = [perspective_xor(board, White), perspective_xor(board, Black)];
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
        NETWORK.evaluate(
            &self.accumulators[color_index(side_to_move)],
            &self.accumulators[color_index(!side_to_move)],
        )
    }

    #[inline(always)]
    pub fn play_move(&mut self, board_after: &Board, color: Color, mv: &EngineMove) {
        if mv.is_castle {
            self.play_castle(color, mv);
        } else {
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

        if mv.piece_type == King {
            let perspective = color_index(color);
            let new_xor = perspective_xor(board_after, color);
            if self.square_xors[perspective] != new_xor {
                self.rebuild_accumulator(board_after, color, new_xor);
            }
        }
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
            let index = color_index(perspective);
            let feature = feature_index(perspective, color, piece, square, self.square_xors[index]);
            self.accumulators[index].add(feature);
        }
    }

    #[inline(always)]
    fn remove_piece(&mut self, color: Color, piece: Piece, square: Square) {
        for perspective in [White, Black] {
            let index = color_index(perspective);
            let feature = feature_index(perspective, color, piece, square, self.square_xors[index]);
            self.accumulators[index].remove(feature);
        }
    }

    fn rebuild_accumulator(&mut self, board: &Board, perspective: Color, square_xor: u8) {
        let index = color_index(perspective);
        self.square_xors[index] = square_xor;
        self.accumulators[index].values = NETWORK.feature_bias;

        for piece in Piece::ALL {
            for color in [White, Black] {
                for square in board.colored_pieces(color, piece) {
                    let feature = feature_index(perspective, color, piece, square, square_xor);
                    self.accumulators[index].add(feature);
                }
            }
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
fn perspective_xor(board: &Board, perspective: Color) -> u8 {
    let king = board
        .colored_pieces(perspective, King)
        .into_iter()
        .next()
        .expect("position must contain both kings");
    let rank_xor = if perspective == Black { 56 } else { 0 };
    let file_xor = if king.file() >= File::E { 7 } else { 0 };
    rank_xor ^ file_xor
}

#[inline(always)]
const fn feature_index(
    perspective: Color,
    color: Color,
    piece: Piece,
    square: Square,
    square_xor: u8,
) -> usize {
    let color_offset = if color_index(perspective) == color_index(color) {
        0
    } else {
        384
    };
    let square = square as usize ^ square_xor as usize;
    color_offset + piece as usize * 64 + square
}

#[cfg(test)]
#[path = "tests/test_nnue.rs"]
mod tests;
