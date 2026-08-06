use crate::evaluate::eval;
use cozy_chess::Board;

#[test]
fn test_eval() {
    let board = Board::default();
    assert_eq!(0, eval(&board));
}
