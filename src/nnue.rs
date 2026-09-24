use std::arch::x86_64::{__m256i, _mm256_add_epi16, _mm256_add_epi32, _mm256_and_si256, _mm256_cmpgt_epi16, _mm256_loadu_si256, _mm256_madd_epi16, _mm256_max_epi16, _mm256_min_epi16, _mm256_mullo_epi16, _mm256_set1_epi16, _mm256_setzero_si256, _mm256_slli_epi32, _mm256_storeu_si256, _mm256_sub_epi16};

use cozy_chess::{
    Board,
    Color::{self, Black, White},
    File,
    Piece::{self, King, Rook},
    Rank, Square,
};

use crate::engine::{EngineMove, FLAG_CASTLE, FLAG_EN_PASSANT};

const INPUT_SIZE: usize = 768;
const QA: i32 = 255;
const QB: i32 = 64;
const EVAL_SCALE: i32 = 400;
const NETWORK_ALIGNMENT: usize = 64;

const NETWORK_BYTES: &[u8] = include_bytes!("../networks/24_09_26-2.bin");
const NETWORK_FILE_SIZE: usize = NETWORK_BYTES.len();
const HIDDEN_SIZE: usize = hidden_size_for_file_size(NETWORK_FILE_SIZE);
const OUTPUT_INPUT_SIZE: usize = 2 * HIDDEN_SIZE;
const NETWORK_PAYLOAD_SIZE: usize = network_payload_size(HIDDEN_SIZE);
#[allow(long_running_const_eval)]
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

        let mut output_store = _mm256_setzero_si256();

        let mut idx = 0;
        while idx < HIDDEN_SIZE {
            let mut us_chunk   = unsafe {_mm256_loadu_si256(us.values.as_ptr().add(idx) as *const __m256i)};
            let mut them_chunk = unsafe {_mm256_loadu_si256(them.values.as_ptr().add(idx) as *const __m256i)};
            
            // clamp
            us_chunk   = _mm256_max_epi16(_mm256_min_epi16(us_chunk, upper), lower);
            them_chunk = _mm256_max_epi16(_mm256_min_epi16(them_chunk, upper), lower);

            // store the clamped values for later
            let us_clamp   = us_chunk;
            let them_clamp = them_chunk;

            // load weights
            let us_weights   = unsafe {_mm256_loadu_si256(self.output_weights.as_ptr().add(idx) as *const __m256i)};
            let them_weights = unsafe {_mm256_loadu_si256(self.output_weights.as_ptr().add(idx+HIDDEN_SIZE) as *const __m256i)};

            // this instruction multiplies by the weights, and turns it from 16x i16s into 8x i32s by adding adjacent pairs
            // eg. [a,b] x [c,d] = [a*c+b*d]
            us_chunk   = _mm256_mullo_epi16(us_chunk, us_weights);
            them_chunk = _mm256_mullo_epi16(them_chunk, them_weights);

            // square
            us_chunk   = _mm256_madd_epi16(us_chunk, us_clamp);
            them_chunk = _mm256_madd_epi16(them_chunk, them_clamp);

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
// const _: () = assert!(OUTPUT_RAW_BOUND <= i32::MAX as i64);
const _: () = assert!(OUTPUT_SCALED_BOUND / (QA * QB) as i64 <= i32::MAX as i64);
const _: () = assert!(HIDDEN_SIZE % 16 == 0);

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
    #[inline(always)]
    fn replace_features(&mut self, removed: [usize; 2], added: [usize; 2]) {
        unsafe { self.replace_features_avx2(removed, added) }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn replace_features_avx2(&mut self, removed: [usize; 2], added: [usize; 2]) {
        for perspective in 0..2 {
            let mut hidden = 0;
            while hidden < HIDDEN_SIZE {
                let accumulator = unsafe {
                    _mm256_loadu_si256(
                        self.accumulators[perspective].values.as_ptr().add(hidden).cast(),
                    )
                };
                let removed = unsafe {
                    _mm256_loadu_si256(
                        NETWORK.feature_weights[removed[perspective]].as_ptr().add(hidden).cast(),
                    )
                };
                let added = unsafe {
                    _mm256_loadu_si256(
                        NETWORK.feature_weights[added[perspective]].as_ptr().add(hidden).cast(),
                    )
                };
                let accumulator = _mm256_add_epi16(_mm256_sub_epi16(accumulator, removed), added);
                unsafe {
                    _mm256_storeu_si256(
                        self.accumulators[perspective].values.as_mut_ptr().add(hidden).cast(),
                        accumulator,
                    )
                };
                hidden += 16;
            }
        }
    }

    #[inline(always)]
    fn capture_features(
        &mut self,
        moved: [usize; 2],
        captured: [usize; 2],
        added: [usize; 2],
    ) {
        unsafe { self.capture_features_avx2(moved, captured, added) }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn capture_features_avx2(
        &mut self,
        moved: [usize; 2],
        captured: [usize; 2],
        added: [usize; 2],
    ) {
        for perspective in 0..2 {
            let mut hidden = 0;
            while hidden < HIDDEN_SIZE {
                let accumulator = unsafe {
                    _mm256_loadu_si256(
                        self.accumulators[perspective].values.as_ptr().add(hidden).cast(),
                    )
                };
                let moved = unsafe {
                    _mm256_loadu_si256(
                        NETWORK.feature_weights[moved[perspective]].as_ptr().add(hidden).cast(),
                    )
                };
                let captured = unsafe {
                    _mm256_loadu_si256(
                        NETWORK.feature_weights[captured[perspective]].as_ptr().add(hidden).cast(),
                    )
                };
                let added = unsafe {
                    _mm256_loadu_si256(
                        NETWORK.feature_weights[added[perspective]].as_ptr().add(hidden).cast(),
                    )
                };
                let accumulator = _mm256_add_epi16(
                    _mm256_sub_epi16(_mm256_sub_epi16(accumulator, moved), captured),
                    added,
                );
                unsafe {
                    _mm256_storeu_si256(
                        self.accumulators[perspective].values.as_mut_ptr().add(hidden).cast(),
                        accumulator,
                    )
                };
                hidden += 16;
            }
        }
    }

    #[inline(always)]
    fn castle_features(
        &mut self,
        old_king: [usize; 2],
        old_rook: [usize; 2],
        new_king: [usize; 2],
        new_rook: [usize; 2],
    ) {
        unsafe { self.castle_features_avx2(old_king, old_rook, new_king, new_rook) }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn castle_features_avx2(
        &mut self,
        old_king: [usize; 2],
        old_rook: [usize; 2],
        new_king: [usize; 2],
        new_rook: [usize; 2],
    ) {
        for perspective in 0..2 {
            let mut hidden = 0;
            while hidden < HIDDEN_SIZE {
                let accumulator = unsafe {
                    _mm256_loadu_si256(
                        self.accumulators[perspective].values.as_ptr().add(hidden).cast(),
                    )
                };
                let old_king = unsafe {
                    _mm256_loadu_si256(
                        NETWORK.feature_weights[old_king[perspective]].as_ptr().add(hidden).cast(),
                    )
                };
                let old_rook = unsafe {
                    _mm256_loadu_si256(
                        NETWORK.feature_weights[old_rook[perspective]].as_ptr().add(hidden).cast(),
                    )
                };
                let new_king = unsafe {
                    _mm256_loadu_si256(
                        NETWORK.feature_weights[new_king[perspective]].as_ptr().add(hidden).cast(),
                    )
                };
                let new_rook = unsafe {
                    _mm256_loadu_si256(
                        NETWORK.feature_weights[new_rook[perspective]].as_ptr().add(hidden).cast(),
                    )
                };
                let accumulator = _mm256_add_epi16(
                    _mm256_add_epi16(
                        _mm256_sub_epi16(_mm256_sub_epi16(accumulator, old_king), old_rook),
                        new_king,
                    ),
                    new_rook,
                );
                unsafe {
                    _mm256_storeu_si256(
                        self.accumulators[perspective].values.as_mut_ptr().add(hidden).cast(),
                        accumulator,
                    )
                };
                hidden += 16;
            }
        }
    }

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
        if mv.flags & FLAG_CASTLE != 0 {
            self.play_castle(color, mv);
        } else {
            let added_piece = mv.mv.promotion.unwrap_or(mv.piece_type);
            let victim = mv.target_type.map(|piece| {
                let square = if mv.flags & FLAG_EN_PASSANT != 0 {
                    Square::new(mv.mv.to.file(), Rank::Fifth.relative_to(color))
                } else {
                    mv.mv.to
                };
                (piece, square)
            });

            let mut moved = [0; 2];
            let mut added = [0; 2];
            let mut captured = [0; 2];
            for perspective in [White, Black] {
                let index = color_index(perspective);
                let square_xor = self.square_xors[index];
                moved[index] = feature_index(
                    perspective,
                    color,
                    mv.piece_type,
                    mv.mv.from,
                    square_xor,
                );
                added[index] =
                    feature_index(perspective, color, added_piece, mv.mv.to, square_xor);
                if let Some((victim, square)) = victim {
                    captured[index] =
                        feature_index(perspective, !color, victim, square, square_xor);
                }
            }
            if victim.is_some() {
                self.capture_features(moved, captured, added);
            } else {
                self.replace_features(moved, added);
            }
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

        let king_square = Square::new(king_file, back_rank);
        let rook_square = Square::new(rook_file, back_rank);
        let mut old_king = [0; 2];
        let mut old_rook = [0; 2];
        let mut new_king = [0; 2];
        let mut new_rook = [0; 2];
        for perspective in [White, Black] {
            let index = color_index(perspective);
            let square_xor = self.square_xors[index];
            old_king[index] = feature_index(perspective, color, King, mv.mv.from, square_xor);
            old_rook[index] = feature_index(perspective, color, Rook, mv.mv.to, square_xor);
            new_king[index] = feature_index(perspective, color, King, king_square, square_xor);
            new_rook[index] = feature_index(perspective, color, Rook, rook_square, square_xor);
        }
        self.castle_features(old_king, old_rook, new_king, new_rook);
    }

    #[inline(always)]
    fn add_piece(&mut self, color: Color, piece: Piece, square: Square) {
        for perspective in [White, Black] {
            let index = color_index(perspective);
            let feature = feature_index(perspective, color, piece, square, self.square_xors[index]);
            self.accumulators[index].add(feature);
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
