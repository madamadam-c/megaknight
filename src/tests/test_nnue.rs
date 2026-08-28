use super::*;
use cozy_chess::{Piece::Pawn, util::parse_uci_move};

fn assert_incremental_move(fen: &str, move_text: &str) {
    let mut board = Board::from_fen(fen, false).unwrap();
    let mv = parse_uci_move(&board, move_text).unwrap();
    let piece = board.piece_on(mv.from).unwrap();
    let engine_move = EngineMove::new(&board, mv, piece, false);
    let mut incremental = NnueState::from_board(&board);

    incremental.play_move(board.side_to_move(), &engine_move);
    board.play_unchecked(mv);

    let rebuilt = NnueState::from_board(&board);
    assert_eq!(incremental, rebuilt);
    assert_eq!(
        incremental.evaluate(board.side_to_move()),
        rebuilt.evaluate(board.side_to_move())
    );
}

#[test]
fn state_is_two_cache_line_sized_accumulators() {
    assert_eq!(std::mem::size_of::<NnueState>(), 256);
    assert_eq!(std::mem::align_of::<NnueState>(), 32);
}

#[test]
fn chess768_features_match_bullet_perspectives() {
    assert_eq!(feature_index(White, White, Pawn, Square::A2), 8);
    assert_eq!(feature_index(White, Black, Pawn, Square::A7), 432);
    assert_eq!(feature_index(Black, Black, Pawn, Square::A7), 8);
    assert_eq!(feature_index(Black, White, Pawn, Square::A2), 432);
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
        state.play_move(board.side_to_move(), &engine_move);
        board.play_unchecked(mv);
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
        NETWORK.evaluate(&state.accumulators[0], &state.accumulators[1])
    );
    assert_eq!(
        state.evaluate(Black),
        NETWORK.evaluate(&state.accumulators[1], &state.accumulators[0])
    );
}

#[test]
fn startpos_evaluation_is_bit_stable() {
    let state = NnueState::from_board(&Board::default());
    assert_eq!(state.evaluate(White), 44);
    assert_eq!(state.evaluate(Black), 44);
}
