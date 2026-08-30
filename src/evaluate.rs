use std::cmp::max;

use cozy_chess::{
    Board,
    Color::{Black, White},
    Piece::{self, *},
    Rank, Square, get_bishop_moves, get_king_moves, get_knight_moves, get_line_rays,
    get_pawn_attacks, get_rook_moves,
};

use crate::engine::EngineMove;

pub fn value(piece: Piece) -> i32 {
    match piece {
        Pawn => 100,
        Knight => 300,
        Bishop => 300,
        Rook => 500,
        Queen => 900,
        King => 20000,
    }
}

/*
choose the least valuable piece that is attacking the square at each point in time
and take the current piece on the square, and record the material gain in an array
go backwards in the array to choose the best time for the side-to-move to stop the sequence
and return the best value from the start
*/
pub fn static_exchange_evaluation(board: &Board, mv: &EngineMove) -> i16 {
    let square = mv.mv.to;
    let mut piece;
    let mut colour = !board.side_to_move();

    let mut blocked_squares = board.occupied();
    blocked_squares ^= mv.mv.from.bitboard();

    if !mv.is_capture {
        blocked_squares ^= mv.mv.to.bitboard();
    }

    if mv.is_ep {
        if colour == White {
            blocked_squares ^= Square::new(square.file(), Rank::Fourth).bitboard();
        } else {
            blocked_squares ^= Square::new(square.file(), Rank::Fifth).bitboard();
        }
    }

    let mut attacking_pieces = (get_king_moves(square) & board.pieces(King) & blocked_squares)
        | (get_knight_moves(square) & board.pieces(Knight) & blocked_squares)
        | (get_bishop_moves(square, blocked_squares)
            & blocked_squares
            & (board.pieces(Bishop) | board.pieces(Queen)))
        | (get_rook_moves(square, blocked_squares)
            & blocked_squares
            & (board.pieces(Rook) | board.pieces(Queen)))
        | (get_pawn_attacks(square, Black) & blocked_squares & board.colored_pieces(White, Pawn))
        | (get_pawn_attacks(square, White) & blocked_squares & board.colored_pieces(Black, Pawn));

    let mut gain: [i32; 32] = [0; 32];
    let mut d = 0;

    gain[0] = mv.material_value;
    piece = if mv.promotion {
        mv.mv.promotion
    } else {
        Some(mv.piece_type)
    };

    let rays = [
        get_line_rays(board.king(White), square),
        get_line_rays(board.king(Black), square),
    ];

    loop {
        let mut found = false;
        let other = board.colors(!colour) & attacking_pieces;

        for piece_type in Piece::ALL {
            if piece_type == King && !other.is_empty() {
                break;
            }

            let pinned = board.pinned_for(colour);
            let bad = pinned & !rays[colour as usize];

            let ours = board.colored_pieces(colour, piece_type) & attacking_pieces & !bad;
            if let Some(sq) = ours.next_square() {
                found = true;

                d += 1;
                gain[d] = value(piece.unwrap()) - gain[d - 1];

                blocked_squares ^= sq.bitboard();
                attacking_pieces ^= sq.bitboard();
                piece = Some(piece_type);

                if piece_type == Pawn
                    && ((colour == White && square.rank() == Rank::Eighth)
                        || (colour == Black && square.rank() == Rank::First))
                {
                    gain[d] += value(Queen) - value(Pawn);
                    piece = Some(Queen);
                }

                if matches!(piece_type, Rook | Queen) {
                    attacking_pieces |= get_rook_moves(square, blocked_squares)
                        & blocked_squares
                        & (board.pieces(Rook) | board.pieces(Queen));
                }
                if matches!(piece_type, Pawn | Bishop | Queen) {
                    attacking_pieces |= get_bishop_moves(square, blocked_squares)
                        & blocked_squares
                        & (board.pieces(Bishop) | board.pieces(Queen));
                }
            }

            if found {
                break;
            }
        }
        if !found {
            break;
        }
        colour = !colour;
    }

    while d > 0 {
        d -= 1;
        gain[d] = -max(-gain[d], gain[d + 1]);
    }
    return gain[0] as i16;
}
