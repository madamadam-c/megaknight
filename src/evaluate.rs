use cozy_chess::{Board, GameStatus, Piece};

pub fn value(piece: Piece) -> i32 {
    match piece {
        Piece::Pawn => 100,
        Piece::Knight => 300,
        Piece::Bishop => 300,
        Piece::Rook => 500,
        Piece::Queen => 900,
        Piece::King => 20000,
    }
}

pub fn eval(board: &Board) -> i32 {
    if board.status() == GameStatus::Drawn {
        return 0;
    }
    if board.status() == GameStatus::Won {
        return -100_000;
    }

    let mut res: i32 = 0;
    for piece in Piece::ALL {
        res += value(piece) * board.colored_pieces(board.side_to_move(), piece).len() as i32;
        res -= value(piece) * board.colored_pieces(!board.side_to_move(), piece).len() as i32;
    }

    res
}
