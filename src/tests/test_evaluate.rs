use crate::{
    engine::EngineMove,
    evaluate::{eval, static_exchange_evaluation},
};
use cozy_chess::{Board, util::parse_uci_move};

fn see(fen: &str, move_text: &str) -> i16 {
    let board = Board::from_fen(fen, false).unwrap();
    let mv = parse_uci_move(&board, move_text).unwrap();
    let piece = board.piece_on(mv.from).unwrap();

    static_exchange_evaluation(&board, &EngineMove::new(&board, mv, piece, false))
}

#[test]
fn test_eval() {
    let board = Board::default();
    assert_eq!(0, eval(&board));
}

#[test]
fn see_scores_undefended_and_defended_pawn_captures() {
    assert_eq!(
        see("7k/8/8/3p4/4P3/8/8/K7 w - - 0 1", "e4d5"),
        100
    );
    assert_eq!(
        see("7k/8/2p5/3p4/4P3/8/8/K7 w - - 0 1", "e4d5"),
        0
    );
}

#[test]
fn see_scores_a_losing_queen_capture() {
    assert_eq!(
        see("7k/8/2p5/3p4/4Q3/8/8/K7 w - - 0 1", "e4d5"),
        -800
    );
}

#[test]
fn see_reveals_sliding_attackers() {
    assert_eq!(
        see("3r3k/8/8/3p4/8/3R4/8/K2Q4 w - - 0 1", "d3d5"),
        100
    );
}

#[test]
fn see_removes_the_captured_en_passant_pawn_from_occupancy() {
    assert_eq!(
        see("3r3k/8/8/3pP3/8/8/8/K2R4 w - d6 0 1", "e5d6"),
        100
    );
}

#[test]
fn see_includes_initial_promotion_material() {
    assert_eq!(
        see("k6r/6P1/8/8/8/8/8/K7 w - - 0 1", "g7h8q"),
        1300
    );
}

#[test]
fn see_tracks_a_recapturing_pawn_as_its_promoted_piece() {
    assert_eq!(
        see("Rr6/1Pn4k/8/8/8/8/8/K7 b - - 0 1", "b8a8"),
        100
    );
}

#[test]
fn see_rejects_a_king_capture_on_a_defended_square() {
    assert_eq!(
        see("5k2/4p3/4Q3/8/1B6/8/8/K7 w - - 0 1", "e6e7"),
        100
    );
}
