use super::*;
use cozy_chess::{Piece::Pawn, util::parse_uci_move};

fn assert_incremental_move(fen: &str, move_text: &str) {
    let mut board = Board::from_fen(fen, false).unwrap();
    let mv = parse_uci_move(&board, move_text).unwrap();
    let piece = board.piece_on(mv.from).unwrap();
    let engine_move = EngineMove::new(&board, mv, piece, false);
    let mut incremental = NnueState::from_board(&board);

    let side_to_move = board.side_to_move();
    board.play_unchecked(mv);
    incremental.play_move(&board, side_to_move, &engine_move);

    let rebuilt = NnueState::from_board(&board);
    assert_eq!(incremental, rebuilt);
    assert_eq!(
        incremental.evaluate(board.side_to_move()),
        rebuilt.evaluate(board.side_to_move())
    );
}

#[test]
fn state_stores_two_accumulators_and_their_orientation() {
    assert_eq!(
        std::mem::size_of::<NnueState>(),
        2 * std::mem::size_of::<Accumulator>() + std::mem::align_of::<Accumulator>()
    );
    assert_eq!(std::mem::align_of::<NnueState>(), 32);
}

#[test]
fn network_file_matches_bucketed_layout() {
    assert_eq!(HIDDEN_SIZE, 512);
    assert_eq!(NETWORK_FILE_SIZE, 7_881_792);
    assert_eq!((NETWORK_PAYLOAD_SIZE + 63) / 64 * 64, NETWORK_FILE_SIZE);
    for bucket in [0, 3, 7] {
        for input in [0, HIDDEN_SIZE, OUTPUT_INPUT_SIZE - 1] {
            assert_eq!(NETWORK.output_weights(bucket)[input], read_i16(NETWORK_BYTES, OUTPUT_WEIGHTS_OFFSET + 2 * (bucket * OUTPUT_INPUT_SIZE + input)));
        }
    }
}

#[test]
fn chess768_features_match_bullet_perspectives() {
    assert_eq!(feature_index(White, White, Pawn, Square::A2, 0), 8);
    assert_eq!(feature_index(White, White, Pawn, Square::A2, 7), 15);
    assert_eq!(feature_index(White, Black, Pawn, Square::A7, 7), 439);
    assert_eq!(feature_index(Black, Black, Pawn, Square::A7, 56), 8);
    assert_eq!(feature_index(Black, Black, Pawn, Square::A7, 63), 15);
    assert_eq!(feature_index(Black, White, Pawn, Square::A2, 63), 439);
}

#[test]
fn incrementally_updates_quiet_moves_and_captures() {
    assert_incremental_move(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "e2e4",
    );
    assert_incremental_move("7k/8/8/3p4/4P3/8/8/K7 w - - 0 1", "e4d5");
}

#[test]
fn incrementally_updates_en_passant() {
    assert_incremental_move("7k/8/8/3pP3/8/8/8/7K w - d6 0 1", "e5d6");
    assert_incremental_move("7k/8/8/8/3Pp3/8/8/7K b - d3 0 1", "e4d3");
}

#[test]
fn incrementally_updates_promotions() {
    assert_incremental_move("7k/P7/8/8/8/8/8/7K w - - 0 1", "a7a8q");
    assert_incremental_move("7k/8/8/8/8/8/p7/7K b - - 0 1", "a2a1n");
    assert_incremental_move("1r5k/P7/8/8/8/8/8/7K w - - 0 1", "a7b8q");
}

#[test]
fn incrementally_updates_castling() {
    assert_incremental_move("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1");
    assert_incremental_move("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1c1");
    assert_incremental_move("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", "e8g8");
    assert_incremental_move("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", "e8c8");
}

#[test]
fn refreshes_when_a_king_crosses_the_de_boundary_or_changes_bucket() {
    assert_incremental_move("7k/8/8/8/8/3K4/8/8 w - - 0 1", "d3e3");
    assert_incremental_move("8/8/3k4/8/8/8/8/K7 b - - 0 1", "d6e6");
    assert_incremental_move("7k/8/8/8/8/4K3/8/8 w - - 0 1", "e3d3");
    assert_incremental_move("8/8/4k3/8/8/8/8/K7 b - - 0 1", "e6d6");
    assert_incremental_move("7k/8/8/8/8/8/8/3K4 w - - 0 1", "d1c1");
    assert_incremental_move("7k/8/8/8/8/8/8/4K3 w - - 0 1", "e1f1");
    assert_incremental_move("8/3k4/8/8/8/8/8/K7 b - - 0 1", "d7c7");
}

#[test]
fn capture_crosses_output_bucket_boundary() {
    let mut board = Board::from_fen("7k/8/8/3p4/4P3/8/4PN2/K7 w - - 0 1", false).unwrap();
    let before = NnueState::from_board(&board);
    let mv = parse_uci_move(&board, "e4d5").unwrap();
    let engine_move = EngineMove::new(&board, mv, Pawn, false);
    board.play_unchecked(mv);
    let mut incremental = before;
    incremental.play_move(&board, White, &engine_move);
    assert_eq!(incremental, NnueState::from_board(&board));
    assert_eq!(before.output_bucket, 1);
    assert_eq!(incremental.output_bucket, 0);
}

#[test]
fn incremental_state_matches_recomputation_for_a_move_sequence() {
    let mut board = Board::default();
    let mut state = NnueState::from_board(&board);

    for move_text in [
        "e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "a7a6", "c1e3",
        "e7e5", "d4b3", "c8e6", "f2f3", "b8d7", "d1d2", "b7b5", "e1c1",
    ] {
        let mv = parse_uci_move(&board, move_text).unwrap();
        let piece = board.piece_on(mv.from).unwrap();
        let engine_move = EngineMove::new(&board, mv, piece, false);
        let side_to_move = board.side_to_move();
        board.play_unchecked(mv);
        state.play_move(&board, side_to_move, &engine_move);
        assert_eq!(state, NnueState::from_board(&board), "after {move_text}");
    }
}

#[test]
fn state_keeps_a_distinct_accumulator_for_each_perspective() {
    let board = Board::from_fen("7k/8/8/8/4P3/8/8/K7 w - - 0 1", false).unwrap();
    let state = NnueState::from_board(&board);

    assert_ne!(state.accumulators[0], state.accumulators[1]);
    assert_eq!(
        state.evaluate(White),
        NETWORK.evaluate(&state.accumulators[0], &state.accumulators[1], state.output_bucket as usize)
    );
    assert_eq!(
        state.evaluate(Black),
        NETWORK.evaluate(&state.accumulators[1], &state.accumulators[0], state.output_bucket as usize)
    );
}

#[test]
fn scalar_and_simd_inference_agree_across_material_buckets() {
    for fen in [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "7k/8/8/3p4/4P3/8/4PN2/K7 w - - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/2pP4/1p2P3/2N2N2/PPQBBPPP/R3K2R w KQkq - 0 1",
        "8/5pk1/6p1/3pP3/3P1P2/6P1/5K2/8 w - - 0 40",
    ] {
        let board = Board::from_fen(fen, false).unwrap();
        let state = NnueState::from_board(&board);
        for side in [White, Black] {
            let us = &state.accumulators[color_index(side)];
            let them = &state.accumulators[color_index(!side)];
            let bucket = state.output_bucket as usize;
            assert_eq!(NETWORK.save_evaluate(us, them, bucket), unsafe { NETWORK.avx2_evaluate(us, them, bucket) }, "{fen}");
        }
    }
}

#[test]
fn randomized_games_match_rebuilt_state_and_scalar_inference() {
    let mut rng = 0x9E3779B97F4A7C15u64;
    for _ in 0..4 {
        let mut board = Board::default();
        let mut state = NnueState::from_board(&board);
        for _ in 0..120 {
            let mut moves = Vec::new();
            board.generate_moves(|piece_moves| {
                moves.extend(piece_moves);
                false
            });
            if moves.is_empty() { break; }
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let mv = moves[(rng as usize) % moves.len()];
            let piece = board.piece_on(mv.from).unwrap();
            let engine_move = EngineMove::new(&board, mv, piece, false);
            let color = board.side_to_move();
            board.play_unchecked(mv);
            state.play_move(&board, color, &engine_move);
            assert_eq!(state, NnueState::from_board(&board), "after {mv} in {board}");

            let us = &state.accumulators[color_index(board.side_to_move())];
            let them = &state.accumulators[color_index(!board.side_to_move())];
            assert_eq!(state.evaluate(board.side_to_move()), NETWORK.save_evaluate(us, them, state.output_bucket as usize), "{board}");
        }
    }
}

#[test]
fn startpos_evaluation_is_symmetric() {
    let state = NnueState::from_board(&Board::default());
    assert_eq!(state.evaluate(White), state.evaluate(Black));
}
