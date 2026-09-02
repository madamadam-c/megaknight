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
        let pinned = board.pinned_for(colour);

        for piece_type in Piece::ALL {
            if piece_type == King && !other.is_empty() {
                break;
            }

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

pub fn static_exchange_evaluation_ge(board: &Board, mv: &EngineMove, threshold: i32) -> bool {
    let square = mv.mv.to;

    if mv.is_ep || mv.promotion || matches!(square.rank(), Rank::First | Rank::Eighth) {
        return i32::from(static_exchange_evaluation(board, mv)) >= threshold;
    }

    let mut swap = mv.material_value - threshold;
    if swap < 0 {
        return false;
    }

    swap = value(mv.piece_type) - swap;
    if swap <= 0 {
        return true;
    }

    let mut occupied = board.occupied() ^ mv.mv.from.bitboard();
    if !mv.is_capture {
        occupied ^= square.bitboard();
    }

    let mut attackers = (get_king_moves(square) & board.pieces(King) & occupied)
        | (get_knight_moves(square) & board.pieces(Knight) & occupied)
        | (get_bishop_moves(square, occupied)
            & occupied
            & (board.pieces(Bishop) | board.pieces(Queen)))
        | (get_rook_moves(square, occupied)
            & occupied
            & (board.pieces(Rook) | board.pieces(Queen)))
        | (get_pawn_attacks(square, Black) & occupied & board.colored_pieces(White, Pawn))
        | (get_pawn_attacks(square, White) & occupied & board.colored_pieces(Black, Pawn));

    let rays = [
        get_line_rays(board.king(White), square),
        get_line_rays(board.king(Black), square),
    ];
    let mut colour = !board.side_to_move();
    let mut result = true;

    loop {
        attackers &= occupied;

        let pinned = board.pinned_for(colour);
        let legal_attackers = attackers & board.colors(colour) & !(pinned & !rays[colour as usize]);
        let defended = !(attackers & board.colors(!colour)).is_empty();
        let mut selected = None;

        for piece_type in Piece::ALL {
            if piece_type == King && defended {
                break;
            }

            if let Some(attacker) = (legal_attackers & board.pieces(piece_type)).next_square() {
                selected = Some((attacker, piece_type));
                break;
            }
        }

        let Some((attacker, piece_type)) = selected else {
            break;
        };

        result = !result;
        swap = value(piece_type) - swap;
        if swap < i32::from(result) {
            break;
        }

        occupied ^= attacker.bitboard();
        if matches!(piece_type, Rook | Queen) {
            attackers |= get_rook_moves(square, occupied)
                & occupied
                & (board.pieces(Rook) | board.pieces(Queen));
        }
        if matches!(piece_type, Pawn | Bishop | Queen) {
            attackers |= get_bishop_moves(square, occupied)
                & occupied
                & (board.pieces(Bishop) | board.pieces(Queen));
        }

        colour = !colour;
    }

    result
}

const SEE_NEG_INF: i32 = -1_000_000;

pub fn see_bucket(board: &Board, mv: &EngineMove) -> i32 {
    let square = mv.mv.to;

    if mv.is_ep || mv.promotion || matches!(square.rank(), Rank::First | Rank::Eighth) {
        return match i32::from(static_exchange_evaluation(board, mv)) {
            0.. => 0,
            -100..=-1 => -100,
            _ => SEE_NEG_INF,
        };
    }

    let moved_value = value(mv.piece_type);
    let mut swap_0 = moved_value - mv.material_value;
    let mut swap_100 = moved_value - (mv.material_value + 100);
    let mut zero_result = (swap_0 <= 0).then_some(true);
    let mut minus_100_result = (swap_100 <= 0).then_some(true);

    if zero_result == Some(true) {
        return 0;
    }

    let mut occupied = board.occupied() ^ mv.mv.from.bitboard();
    if !mv.is_capture {
        occupied ^= square.bitboard();
    }

    let mut attackers = (get_king_moves(square) & board.pieces(King) & occupied)
        | (get_knight_moves(square) & board.pieces(Knight) & occupied)
        | (get_bishop_moves(square, occupied)
            & occupied
            & (board.pieces(Bishop) | board.pieces(Queen)))
        | (get_rook_moves(square, occupied)
            & occupied
            & (board.pieces(Rook) | board.pieces(Queen)))
        | (get_pawn_attacks(square, Black) & occupied & board.colored_pieces(White, Pawn))
        | (get_pawn_attacks(square, White) & occupied & board.colored_pieces(Black, Pawn));

    let rays = [
        get_line_rays(board.king(White), square),
        get_line_rays(board.king(Black), square),
    ];
    let mut colour = !board.side_to_move();
    let mut result = true;

    loop {
        attackers &= occupied;

        let pinned = board.pinned_for(colour);
        let legal_attackers = attackers & board.colors(colour) & !(pinned & !rays[colour as usize]);
        let defended = !(attackers & board.colors(!colour)).is_empty();
        let mut selected = None;

        for piece_type in Piece::ALL {
            if piece_type == King && defended {
                break;
            }

            if let Some(attacker) = (legal_attackers & board.pieces(piece_type)).next_square() {
                selected = Some((attacker, piece_type));
                break;
            }
        }

        let Some((attacker, piece_type)) = selected else {
            zero_result.get_or_insert(result);
            minus_100_result.get_or_insert(result);
            break;
        };

        result = !result;
        let attacker_value = value(piece_type);

        if zero_result.is_none() {
            swap_0 = attacker_value - swap_0;
            if swap_0 < result as i32 {
                zero_result = Some(result);
            }
        }

        if minus_100_result.is_none() {
            swap_100 = attacker_value - swap_100;
            if swap_100 < result as i32 {
                minus_100_result = Some(result);
            }
        }

        if zero_result == Some(true) {
            return 0;
        }
        if minus_100_result == Some(false) {
            return SEE_NEG_INF;
        }
        if zero_result == Some(false) && minus_100_result == Some(true) {
            return -100;
        }

        occupied ^= attacker.bitboard();
        if matches!(piece_type, Rook | Queen) {
            attackers |= get_rook_moves(square, occupied)
                & occupied
                & (board.pieces(Rook) | board.pieces(Queen));
        }
        if matches!(piece_type, Pawn | Bishop | Queen) {
            attackers |= get_bishop_moves(square, occupied)
                & occupied
                & (board.pieces(Bishop) | board.pieces(Queen));
        }

        colour = !colour;
    }

    if zero_result == Some(true) {
        0
    } else if minus_100_result == Some(true) {
        -100
    } else {
        SEE_NEG_INF
    }
}
