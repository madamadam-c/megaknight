use std::arch::x86_64::{__m256i, _mm256_add_epi16, _mm256_add_epi32, _mm256_loadu_si256, _mm256_madd_epi16, _mm256_max_epi16, _mm256_min_epi16, _mm256_mullo_epi16, _mm256_set1_epi16, _mm256_setzero_si256, _mm256_storeu_si256, _mm256_sub_epi16};

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
const OUTPUT_BUCKETS: usize = 8;
const INPUT_BUCKETS: usize = 10;

// Change this path alone to select a network; all dimensions and branches are compile-time constants.
macro_rules! network_file {
    // () => { "../networks/24_09_26.bin" };
    () => { "../networks/25_09_26/512_OB.bin" };
}

const NETWORK_BYTES: &[u8] = include_bytes!(network_file!());
const NETWORK_FILE_SIZE: usize = NETWORK_BYTES.len();
const LAYOUT: (Architecture, usize) = layout_for_file_size(NETWORK_FILE_SIZE);
const ARCHITECTURE: Architecture = LAYOUT.0;
const HIDDEN_SIZE: usize = LAYOUT.1;
const HAS_INPUT_BUCKETS: bool = matches!(ARCHITECTURE, Architecture::InputBucketed | Architecture::Multi);
const HAS_OUTPUT_BUCKETS: bool = !matches!(ARCHITECTURE, Architecture::Plain);
const IS_MULTI: bool = matches!(ARCHITECTURE, Architecture::Multi);
const NUM_INPUT_BUCKETS: usize = if HAS_INPUT_BUCKETS { INPUT_BUCKETS } else { 1 };
const NUM_OUTPUT_BUCKETS: usize = if HAS_OUTPUT_BUCKETS { OUTPUT_BUCKETS } else { 1 };
const OUTPUT_INPUT_SIZE: usize = 2 * HIDDEN_SIZE;
const L0_B_OFFSET: usize = INPUT_SIZE * NUM_INPUT_BUCKETS * HIDDEN_SIZE * 2;
const L1_W_OFFSET: usize = L0_B_OFFSET + HIDDEN_SIZE * 2;
const L1_OUTPUTS: usize = if IS_MULTI { OUTPUT_BUCKETS * 16 } else { NUM_OUTPUT_BUCKETS };
const L1_B_OFFSET: usize = L1_W_OFFSET + OUTPUT_INPUT_SIZE * L1_OUTPUTS * 2;
const L2_W_OFFSET: usize = L1_B_OFFSET + L1_OUTPUTS * 2;
const L2_B_OFFSET: usize = L2_W_OFFSET + 16 * OUTPUT_BUCKETS * 32 * 2;
const L3_W_OFFSET: usize = L2_B_OFFSET + OUTPUT_BUCKETS * 32 * 2;
const L3_B_OFFSET: usize = L3_W_OFFSET + 32 * OUTPUT_BUCKETS * 2;
const NETWORK_PAYLOAD_SIZE: usize = if IS_MULTI {
    L3_B_OFFSET + OUTPUT_BUCKETS * 2
} else {
    L1_B_OFFSET + NUM_OUTPUT_BUCKETS * 2
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Architecture {
    Plain,
    OutputBucketed,
    InputBucketed,
    Multi,
}

#[repr(align(64))]
struct AlignedBytes<const N: usize>([u8; N]);

// The large transformer is embedded, aligned and read directly: no multi-million-step
// compile-time conversion and no per-worker allocation or runtime weight decoding.
static NETWORK_DATA: AlignedBytes<NETWORK_FILE_SIZE> = AlignedBytes(*include_bytes!(network_file!()));
const NETWORK: Network = Network;

#[cfg(not(target_endian = "little"))]
compile_error!("embedded NNUE weights require little-endian i16 storage");

const fn output_weights() -> [[i16; OUTPUT_INPUT_SIZE]; NUM_OUTPUT_BUCKETS] {
    let mut weights = [[0; OUTPUT_INPUT_SIZE]; NUM_OUTPUT_BUCKETS];
    if !IS_MULTI {
        let mut bucket = 0;
        while bucket < NUM_OUTPUT_BUCKETS {
            let mut input = 0;
            while input < OUTPUT_INPUT_SIZE {
                weights[bucket][input] = read_i16(NETWORK_BYTES, L1_W_OFFSET + 2 * (bucket * OUTPUT_INPUT_SIZE + input));
                input += 1;
            }
            bucket += 1;
        }
    }
    weights
}

#[repr(align(64))]
struct AlignedOutputWeights([[i16; OUTPUT_INPUT_SIZE]; NUM_OUTPUT_BUCKETS]);
static OUTPUT_WEIGHTS: AlignedOutputWeights = AlignedOutputWeights(output_weights());

// Bullet exports the dense layers output-major via SavedFormat::transpose().
// Keep each selected neuron contiguous for AVX2 dot products.
const fn multi_l1_weights() -> [[[i16; OUTPUT_INPUT_SIZE]; 16]; OUTPUT_BUCKETS] {
    let mut weights = [[[0; OUTPUT_INPUT_SIZE]; 16]; OUTPUT_BUCKETS];
    if IS_MULTI {
        let mut bucket = 0;
        while bucket < OUTPUT_BUCKETS {
            let mut output = 0;
            while output < 16 {
                let mut input = 0;
                while input < OUTPUT_INPUT_SIZE {
                    weights[bucket][output][input] = read_i16(
                        NETWORK_BYTES,
                        L1_W_OFFSET + 2 * ((bucket * 16 + output) * OUTPUT_INPUT_SIZE + input),
                    );
                    input += 1;
                }
                output += 1;
            }
            bucket += 1;
        }
    }
    weights
}

#[repr(align(64))]
struct AlignedMultiWeights([[[i16; OUTPUT_INPUT_SIZE]; 16]; OUTPUT_BUCKETS]);
#[allow(long_running_const_eval)]
static MULTI_L1_WEIGHTS: AlignedMultiWeights = AlignedMultiWeights(multi_l1_weights());

const fn multi_l2_weights() -> [[[i16; 16]; 32]; OUTPUT_BUCKETS] {
    let mut weights = [[[0; 16]; 32]; OUTPUT_BUCKETS];
    if IS_MULTI {
        let mut bucket = 0;
        while bucket < OUTPUT_BUCKETS {
            let mut output = 0;
            while output < 32 {
                let mut input = 0;
                while input < 16 {
                    weights[bucket][output][input] = read_i16(
                        NETWORK_BYTES,
                        L2_W_OFFSET + 2 * ((bucket * 32 + output) * 16 + input),
                    );
                    input += 1;
                }
                output += 1;
            }
            bucket += 1;
        }
    }
    weights
}

#[repr(align(64))]
struct AlignedMultiL2Weights([[[i16; 16]; 32]; OUTPUT_BUCKETS]);
static MULTI_L2_WEIGHTS: AlignedMultiL2Weights = AlignedMultiL2Weights(multi_l2_weights());

#[derive(Clone, Copy)]
struct Network;

impl Network {
    #[inline(always)]
    fn feature_weights(&self, feature: usize) -> &'static [i16; HIDDEN_SIZE] {
        debug_assert!(feature < INPUT_SIZE * NUM_INPUT_BUCKETS);
        unsafe { &*NETWORK_DATA.0.as_ptr().add(feature * HIDDEN_SIZE * 2).cast() }
    }

    #[inline(always)]
    fn feature_bias(&self) -> &'static [i16; HIDDEN_SIZE] {
        unsafe { &*NETWORK_DATA.0.as_ptr().add(L0_B_OFFSET).cast() }
    }

    #[inline(always)]
    fn layer_weights(&self, offset: usize) -> &'static [i16] {
        unsafe { std::slice::from_raw_parts(NETWORK_DATA.0.as_ptr().add(offset).cast(), NETWORK_FILE_SIZE / 2 - offset / 2) }
    }

    #[inline(always)]
    fn evaluate(&self, us: &Accumulator, them: &Accumulator, bucket: usize) -> i32 {
        if is_x86_feature_detected!("avx2") {
            unsafe { self.avx2_evaluate(us, them, bucket) }
        } else {
            self.scalar_evaluate(us, them, bucket)
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn avx2_evaluate(&self, us: &Accumulator, them: &Accumulator, bucket: usize) -> i32 {
        if IS_MULTI {
            return unsafe { self.multi_avx2_evaluate(us, them, bucket) };
        }

        let lower = _mm256_setzero_si256();
        let upper = _mm256_set1_epi16(QA as i16);
        let mut output_store = _mm256_setzero_si256();
        let weights = &OUTPUT_WEIGHTS.0[bucket];
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
            let us_weights   = unsafe {_mm256_loadu_si256(weights.as_ptr().add(idx) as *const __m256i)};
            let them_weights = unsafe {_mm256_loadu_si256(weights.as_ptr().add(idx+HIDDEN_SIZE) as *const __m256i)};

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
        output += i32::from(read_i16(NETWORK_BYTES, L1_B_OFFSET + bucket * 2));

        return (i64::from(output) * i64::from(EVAL_SCALE) / i64::from(QA * QB)) as i32;
    }

    #[inline(always)]
    fn scalar_evaluate(&self, us: &Accumulator, them: &Accumulator, bucket: usize) -> i32 {
        if IS_MULTI {
            return self.multi_scalar_evaluate(us, them, bucket);
        }
        let weights = &OUTPUT_WEIGHTS.0[bucket];
        let mut output = 0;
        for hidden in 0..HIDDEN_SIZE {
            let us_value = i32::from(us.values[hidden]).clamp(0, QA);
            let them_value = i32::from(them.values[hidden]).clamp(0, QA);
            output += us_value * us_value * i32::from(weights[hidden]);
            output +=
                them_value * them_value * i32::from(weights[HIDDEN_SIZE + hidden]);
        }

        output /= QA;
        output += i32::from(read_i16(NETWORK_BYTES, L1_B_OFFSET + bucket * 2));
        (i64::from(output) * i64::from(EVAL_SCALE) / i64::from(QA * QB)) as i32
    }

    fn multi_scalar_evaluate(&self, us: &Accumulator, them: &Accumulator, bucket: usize) -> i32 {
        let l1 = self.layer_weights(L1_W_OFFSET);
        let l1_bias = self.layer_weights(L1_B_OFFSET);
        let mut hidden16 = [0i16; 16];
        for out in 0..16 {
            let mut sum = 0i32;
            for input in 0..OUTPUT_INPUT_SIZE {
                let value = if input < HIDDEN_SIZE { us.values[input] } else { them.values[input - HIDDEN_SIZE] };
                let value = i32::from(value).clamp(0, QA);
                sum += value * value * i32::from(l1[(bucket * 16 + out) * OUTPUT_INPUT_SIZE + input]);
            }
            hidden16[out] = ((sum / QA + i32::from(l1_bias[bucket * 16 + out])) / QB).clamp(0, QA) as i16;
        }

        let l2 = self.layer_weights(L2_W_OFFSET);
        let l2_bias = self.layer_weights(L2_B_OFFSET);
        let mut hidden32 = [0i16; 32];
        for out in 0..32 {
            let mut sum = 0i32;
            for input in 0..16 {
                let value = i32::from(hidden16[input]);
                sum += value * value * i32::from(l2[(bucket * 32 + out) * 16 + input]);
            }
            hidden32[out] = ((sum / QA + i32::from(l2_bias[bucket * 32 + out])) / QB).clamp(0, QA) as i16;
        }

        self.multi_output(&hidden32, bucket)
    }

    #[inline(always)]
    fn multi_output(&self, hidden32: &[i16; 32], bucket: usize) -> i32 {
        let l3 = self.layer_weights(L3_W_OFFSET);
        let l3_bias = self.layer_weights(L3_B_OFFSET);
        let mut sum = 0i32;
        for input in 0..32 {
            let value = i32::from(hidden32[input]);
            sum += value * value * i32::from(l3[bucket * 32 + input]);
        }
        let output = sum / QA + i32::from(l3_bias[bucket]);
        (i64::from(output) * i64::from(EVAL_SCALE) / i64::from(QA * QB)) as i32
    }

    // Selected bucket/neuron rows are contiguous in the exported matrices.
    #[target_feature(enable = "avx2")]
    unsafe fn multi_avx2_evaluate(&self, us: &Accumulator, them: &Accumulator, bucket: usize) -> i32 {
        let l1_bias = self.layer_weights(L1_B_OFFSET);
        let mut hidden16 = [0i16; 16];
        for out in 0..16 {
            let weights = &MULTI_L1_WEIGHTS.0[bucket][out];
            let mut acc = _mm256_setzero_si256();
            for input in (0..OUTPUT_INPUT_SIZE).step_by(16) {
                let values = if input < HIDDEN_SIZE {
                    unsafe { _mm256_loadu_si256(us.values.as_ptr().add(input).cast()) }
                } else {
                    unsafe { _mm256_loadu_si256(them.values.as_ptr().add(input - HIDDEN_SIZE).cast()) }
                };
                let values = _mm256_max_epi16(_mm256_min_epi16(values, _mm256_set1_epi16(QA as i16)), _mm256_setzero_si256());
                let weights = unsafe { _mm256_loadu_si256(weights.as_ptr().add(input).cast()) };
                let product = _mm256_mullo_epi16(values, weights);
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(product, values));
            }
            let mut lanes = [0i32; 8];
            unsafe { _mm256_storeu_si256(lanes.as_mut_ptr().cast(), acc) };
            let sum: i32 = lanes.into_iter().sum();
            hidden16[out] = ((sum / QA + i32::from(l1_bias[bucket * 16 + out])) / QB).clamp(0, QA) as i16;
        }

        let l2_bias = self.layer_weights(L2_B_OFFSET);
        let mut hidden32 = [0i16; 32];
        for out in 0..32 {
            let values = unsafe { _mm256_loadu_si256(hidden16.as_ptr().cast()) };
            let weights = unsafe { _mm256_loadu_si256(MULTI_L2_WEIGHTS.0[bucket][out].as_ptr().cast()) };
            let product = _mm256_mullo_epi16(values, weights);
            let pairs = _mm256_madd_epi16(product, values);
            let mut lanes = [0i32; 8];
            unsafe { _mm256_storeu_si256(lanes.as_mut_ptr().cast(), pairs) };
            let sum: i32 = lanes.into_iter().sum();
            hidden32[out] = ((sum / QA + i32::from(l2_bias[bucket * 32 + out])) / QB).clamp(0, QA) as i16;
        }
        self.multi_output(&hidden32, bucket)
    }
}

const fn read_i16(bytes: &[u8], offset: usize) -> i16 {
    i16::from_le_bytes([bytes[offset], bytes[offset + 1]])
}

const fn network_payload_size(arch: Architecture, hidden: usize) -> usize {
    let inputs = if matches!(arch, Architecture::Plain | Architecture::OutputBucketed) { 1 } else { INPUT_BUCKETS };
    let outputs = if matches!(arch, Architecture::Plain) { 1 } else { OUTPUT_BUCKETS };
    let first = (INPUT_SIZE * inputs * hidden + hidden) * 2;
    if matches!(arch, Architecture::Multi) {
        first + (2 * hidden * outputs * 16 + outputs * 16
            + 16 * outputs * 32 + outputs * 32 + 32 * outputs + outputs) * 2
    } else {
        first + (2 * hidden * outputs + outputs) * 2
    }
}

const fn padded_network_file_size(arch: Architecture, hidden_size: usize) -> usize {
    let payload_size = network_payload_size(arch, hidden_size);
    (payload_size + NETWORK_ALIGNMENT - 1) / NETWORK_ALIGNMENT * NETWORK_ALIGNMENT
}

const fn layout_for_file_size(file_size: usize) -> (Architecture, usize) {
    let mut found = None;
    let mut hidden = 16;
    while hidden <= 2048 {
        let mut kind = 0;
        while kind < 4 {
            let arch = match kind {
                0 => Architecture::Plain,
                1 => Architecture::OutputBucketed,
                2 => Architecture::InputBucketed,
                _ => Architecture::Multi,
            };
            if padded_network_file_size(arch, hidden) == file_size {
                assert!(found.is_none(), "ambiguous NNUE file size; add a file header");
                found = Some((arch, hidden));
            }
            kind += 1;
        }
        hidden += 16;
    }
    match found {
        Some(layout) => layout,
        None => panic!("NNUE file size does not match any supported architecture"),
    }
}

const fn max_signed_weight_sum(weights: &[[i16; OUTPUT_INPUT_SIZE]; NUM_OUTPUT_BUCKETS]) -> i64 {
    let mut max = 0;
    let mut bucket = 0;
    while bucket < NUM_OUTPUT_BUCKETS {
        let mut positive = 0;
        let mut negative = 0;
        let mut hidden = 0;
        while hidden < OUTPUT_INPUT_SIZE {
            let weight = weights[bucket][hidden] as i64;
            if weight > 0 { positive += weight; } else { negative -= weight; }
            hidden += 1;
        }
        if positive > max { max = positive; }
        if negative > max { max = negative; }
        bucket += 1;
    }
    max
}

const OUTPUT_RAW_BOUND: i64 = max_signed_weight_sum(&output_weights()) * QA as i64 * QA as i64;
const OUTPUT_SCALED_BOUND: i64 =
    (OUTPUT_RAW_BOUND / QA as i64 + i16::MAX as i64) * EVAL_SCALE as i64;
const _: () = assert!(IS_MULTI || OUTPUT_RAW_BOUND <= i32::MAX as i64);
const _: () = assert!(IS_MULTI || OUTPUT_SCALED_BOUND / (QA * QB) as i64 <= i32::MAX as i64);
const _: () = assert!(NETWORK_PAYLOAD_SIZE <= NETWORK_FILE_SIZE);
const _: () = assert!(padded_network_file_size(ARCHITECTURE, HIDDEN_SIZE) == NETWORK_FILE_SIZE);
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
                self.values[hidden].wrapping_add(NETWORK.feature_weights(feature)[hidden]);
        }
    }

}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NnueState {
    accumulators: [Accumulator; 2],
    square_xors: [u8; 2],
    input_buckets: [u8; 2],
    output_bucket: u8,
}

impl Default for NnueState {
    fn default() -> Self {
        let accumulator = Accumulator {
            values: *NETWORK.feature_bias(),
        };
        Self {
            accumulators: [accumulator; 2],
            square_xors: [0, 56],
            input_buckets: [0; 2],
            output_bucket: 0,
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
                        NETWORK.feature_weights(removed[perspective]).as_ptr().add(hidden).cast(),
                    )
                };
                let added = unsafe {
                    _mm256_loadu_si256(
                        NETWORK.feature_weights(added[perspective]).as_ptr().add(hidden).cast(),
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
                        NETWORK.feature_weights(moved[perspective]).as_ptr().add(hidden).cast(),
                    )
                };
                let captured = unsafe {
                    _mm256_loadu_si256(
                        NETWORK.feature_weights(captured[perspective]).as_ptr().add(hidden).cast(),
                    )
                };
                let added = unsafe {
                    _mm256_loadu_si256(
                        NETWORK.feature_weights(added[perspective]).as_ptr().add(hidden).cast(),
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
                        NETWORK.feature_weights(old_king[perspective]).as_ptr().add(hidden).cast(),
                    )
                };
                let old_rook = unsafe {
                    _mm256_loadu_si256(
                        NETWORK.feature_weights(old_rook[perspective]).as_ptr().add(hidden).cast(),
                    )
                };
                let new_king = unsafe {
                    _mm256_loadu_si256(
                        NETWORK.feature_weights(new_king[perspective]).as_ptr().add(hidden).cast(),
                    )
                };
                let new_rook = unsafe {
                    _mm256_loadu_si256(
                        NETWORK.feature_weights(new_rook[perspective]).as_ptr().add(hidden).cast(),
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
        if HAS_INPUT_BUCKETS {
            state.input_buckets = [king_bucket(board, White), king_bucket(board, Black)];
        }
        if HAS_OUTPUT_BUCKETS {
            state.output_bucket = output_bucket(board);
        }
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
            self.output_bucket as usize,
        )
    }

    #[inline(always)]
    pub fn play_move(&mut self, board_after: &Board, color: Color, mv: &EngineMove) {
        if HAS_OUTPUT_BUCKETS && mv.target_type.is_some() {
            self.output_bucket = output_bucket(board_after);
        }
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
                moved[index] = self.feature_index(
                    perspective,
                    color,
                    mv.piece_type,
                    mv.mv.from,
                    square_xor,
                );
                added[index] = self.feature_index(perspective, color, added_piece, mv.mv.to, square_xor);
                if let Some((victim, square)) = victim {
                    captured[index] = self.feature_index(perspective, !color, victim, square, square_xor);
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
            let new_bucket = if HAS_INPUT_BUCKETS { king_bucket(board_after, color) } else { 0 };
            if self.square_xors[perspective] != new_xor || self.input_buckets[perspective] != new_bucket {
                self.rebuild_accumulator(board_after, color, new_xor, new_bucket);
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
            old_king[index] = self.feature_index(perspective, color, King, mv.mv.from, square_xor);
            old_rook[index] = self.feature_index(perspective, color, Rook, mv.mv.to, square_xor);
            new_king[index] = self.feature_index(perspective, color, King, king_square, square_xor);
            new_rook[index] = self.feature_index(perspective, color, Rook, rook_square, square_xor);
        }
        self.castle_features(old_king, old_rook, new_king, new_rook);
    }

    #[inline(always)]
    fn add_piece(&mut self, color: Color, piece: Piece, square: Square) {
        for perspective in [White, Black] {
            let index = color_index(perspective);
            let feature = self.feature_index(perspective, color, piece, square, self.square_xors[index]);
            self.accumulators[index].add(feature);
        }
    }

    fn rebuild_accumulator(&mut self, board: &Board, perspective: Color, square_xor: u8, bucket: u8) {
        let index = color_index(perspective);
        self.square_xors[index] = square_xor;
        self.input_buckets[index] = bucket;
        self.accumulators[index].values = *NETWORK.feature_bias();

        for piece in Piece::ALL {
            for color in [White, Black] {
                for square in board.colored_pieces(color, piece) {
                    let feature = self.feature_index(perspective, color, piece, square, square_xor);
                    self.accumulators[index].add(feature);
                }
            }
        }
    }

    #[inline(always)]
    fn feature_index(&self, perspective: Color, color: Color, piece: Piece, square: Square, square_xor: u8) -> usize {
        let feature = feature_index(perspective, color, piece, square, square_xor);
        if HAS_INPUT_BUCKETS {
            INPUT_SIZE * self.input_buckets[color_index(perspective)] as usize + feature
        } else {
            feature
        }
    }
}

#[rustfmt::skip]
const KING_BUCKET_LAYOUT: [u8; 32] = [
    0, 1, 2, 3, 4, 4, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7,
    8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9,
];

#[inline(always)]
fn king_bucket(board: &Board, perspective: Color) -> u8 {
    let king = board.colored_pieces(perspective, King).into_iter().next().expect("position must contain both kings");
    let square = king as usize ^ if perspective == Black { 56 } else { 0 };
    KING_BUCKET_LAYOUT[(square / 8) * 4 + (square % 8).min(7 - square % 8)]
}

#[inline(always)]
fn output_bucket(board: &Board) -> u8 {
    ((board.occupied().len() as u8 - 2) / 4).min(7)
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
