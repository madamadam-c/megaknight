use cozy_chess::{util::parse_uci_move, Board, Color};

use crate::{format_uci_move, parse_go, parse_position};

#[test]
fn parse_go_collects_multiple_limits() {
    let board = Board::default();
    let limits = parse_go(
        "go depth 8 nodes 100000 wtime 5000 btime 6000 winc 100 binc 200 movestogo 30",
        &board,
    );

    assert_eq!(limits.depth, Some(8));
    assert_eq!(limits.nodes, Some(100_000));
    assert_eq!(limits.wtime, Some(5_000));
    assert_eq!(limits.btime, Some(6_000));
    assert_eq!(limits.winc, Some(100));
    assert_eq!(limits.binc, Some(200));
    assert_eq!(limits.movestogo, Some(30));
}

#[test]
fn parse_position_applies_the_complete_move_history() {
    let mut board = Board::default();
    parse_position("position startpos moves e2e4 e7e5", &mut board);

    assert_eq!(board.side_to_move(), Color::White);
}

#[test]
fn invalid_position_does_not_replace_the_current_board() {
    let mut board = Board::default();
    parse_position("position startpos moves e2e5", &mut board);

    assert_eq!(board.side_to_move(), Color::White);
}

#[test]
fn format_uci_move_uses_standard_castling_notation() {
    let white_board = Board::from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", false).unwrap();
    let black_board = Board::from_fen("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", false).unwrap();

    let white_short = parse_uci_move(&white_board, "e1g1").unwrap();
    let white_long = parse_uci_move(&white_board, "e1c1").unwrap();
    let black_short = parse_uci_move(&black_board, "e8g8").unwrap();
    let black_long = parse_uci_move(&black_board, "e8c8").unwrap();

    assert_eq!(format_uci_move(&white_board, white_short), "e1g1");
    assert_eq!(format_uci_move(&white_board, white_long), "e1c1");
    assert_eq!(format_uci_move(&black_board, black_short), "e8g8");
    assert_eq!(format_uci_move(&black_board, black_long), "e8c8");
}
