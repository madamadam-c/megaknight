use crate::*;

#[derive(Debug)]
struct ColorZobristConstants {
    pieces: [[u64; Square::NUM]; Piece::NUM],
    castle_rights: [u64; File::NUM]
}

#[derive(Debug)]
struct ZobristConstants {
    color: [ColorZobristConstants; Color::NUM],
    en_passant: [u64; File::NUM],
    black_to_move: u64
}

const ZOBRIST: ZobristConstants = {
    // Simple Pcg64Mcg impl
    let mut state = 0x7369787465656E2062797465206E756Du128 | 1;
    macro_rules! rand {
        () => {{
            state = state.wrapping_mul(0x2360ED051FC65DA44385DF649FCCF645);
            let rot = (state >> 122) as u32;
            let xsl = (state >> 64) as u64 ^ state as u64;
            xsl.rotate_right(rot)
        }};
    }

    macro_rules! fill_array {
        ($array:ident: $expr:expr) => {{
            let mut i = 0;
            while i < $array.len() {
                $array[i] = $expr;
                i += 1;
            }
        }};
    }

    macro_rules! color_zobrist_constant {
        () => {{
            let mut castle_rights = [0; File::NUM];
            fill_array!(castle_rights: rand!());

            let mut pieces = [[0; Square::NUM]; Piece::NUM];
            fill_array!(pieces: {
                let mut squares = [0; Square::NUM];
                fill_array!(squares: rand!());
                squares
            });
            
            ColorZobristConstants {
                pieces,
                castle_rights
            }
        }};
    }

    let mut en_passant = [0; File::NUM];
    fill_array!(en_passant: rand!());

    let white = color_zobrist_constant!();
    let black = color_zobrist_constant!();

    let black_to_move = rand!();

    ZobristConstants {
        color: [white, black],
        en_passant,
        black_to_move
    }
};

// This is Copy for performance reasons, since Copy guarantees a bit-for-bit copy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ZobristBoard {
    pieces: [BitBoard; Piece::NUM],
    colors: [BitBoard; Color::NUM],
    side_to_move: Color,
    castle_rights: [CastleRights; Color::NUM],
    en_passant: Option<File>,
    kings: [Square; Color::NUM],
    pawn_hashes: [u64; Color::NUM],
    non_pawn_hashes: [u64; Color::NUM],
    minor_piece_hashes: [u64; Color::NUM],
    major_piece_hashes: [u64; Color::NUM],
    hash: u64,
}

impl ZobristBoard {
    #[inline(always)]
    pub fn empty() -> Self {
        Self {
            pieces: [BitBoard::EMPTY; Piece::NUM],
            colors: [BitBoard::EMPTY; Color::NUM],
            side_to_move: Color::White,
            castle_rights: [CastleRights {
                short: None,
                long: None
            }; 2],
            en_passant: None,
            kings: [Square::A1; Color::NUM],
            pawn_hashes: [0; Color::NUM],
            non_pawn_hashes: [0; Color::NUM],
            minor_piece_hashes: [0; Color::NUM],
            major_piece_hashes: [0; Color::NUM],
            hash: 0,
        }
    }

    #[inline(always)]
    pub fn pieces(&self, piece: Piece) -> BitBoard {
        self.pieces[piece as usize]
    }

    #[inline(always)]
    pub fn colors(&self, color: Color) -> BitBoard {
        self.colors[color as usize]
    }

    #[inline(always)]
    pub fn side_to_move(&self) -> Color {
        self.side_to_move
    }

    #[inline(always)]
    pub fn castle_rights(&self, color: Color) -> &CastleRights {
        &self.castle_rights[color as usize]
    }

    #[inline(always)]
    pub fn en_passant(&self) -> Option<File> {
        self.en_passant
    }

    #[inline(always)]
    pub fn king(&self, color: Color) -> Square {
        self.kings[color as usize]
    }

    #[inline(always)]
    pub fn pawn_hash(&self, color: Color) -> u64 {
        self.pawn_hashes[color as usize]
    }

    #[inline(always)]
    pub fn non_pawn_hash(&self, color: Color) -> u64 {
        self.non_pawn_hashes[color as usize]
    }

    #[inline(always)]
    pub fn minor_piece_hash(&self, color: Color) -> u64 {
        self.minor_piece_hashes[color as usize]
    }

    #[inline(always)]
    pub fn major_piece_hash(&self, color: Color) -> u64 {
        self.major_piece_hashes[color as usize]
    }

    #[inline(always)]
    pub fn hash(&self) -> u64 {
        self.hash
    }

    #[inline(always)]
    pub fn hash_without_ep(&self) -> u64 {
        let mut hash = self.hash;
        if let Some(file) = self.en_passant {
            hash ^= ZOBRIST.en_passant[file as usize];
        }
        hash
    }

    pub fn board_is_equal(&self, other: &Self) -> bool {
        self.pieces == other.pieces
            && self.colors == other.colors
            && self.side_to_move == other.side_to_move
            && self.castle_rights == other.castle_rights
    }

    #[inline(always)]
    pub fn xor_square(&mut self, piece: Piece, color: Color, square: Square) {
        let square_bb = square.bitboard();
        self.pieces[piece as usize] ^= square_bb;
        self.colors[color as usize] ^= square_bb;
        if piece == Piece::King {
            self.kings[color as usize] = square;
        }
        let key = ZOBRIST
            .color[color as usize]
            .pieces[piece as usize]
            [square as usize];
        self.hash ^= key;
        if piece == Piece::Pawn {
            self.pawn_hashes[color as usize] ^= key;
        } else {
            self.non_pawn_hashes[color as usize] ^= key;
        }
        match piece {
            Piece::Knight | Piece::Bishop => self.minor_piece_hashes[color as usize] ^= key,
            Piece::Rook | Piece::Queen => self.major_piece_hashes[color as usize] ^= key,
            _ => {}
        }
    }

    #[inline(always)]
    pub fn move_square(&mut self, piece: Piece, color: Color, from: Square, to: Square) {
        let squares_bb = from.bitboard() | to.bitboard();
        self.pieces[piece as usize] ^= squares_bb;
        self.colors[color as usize] ^= squares_bb;
        if piece == Piece::King {
            self.kings[color as usize] = to;
        }
        let key = ZOBRIST
            .color[color as usize]
            .pieces[piece as usize]
            [from as usize]
            ^ ZOBRIST
                .color[color as usize]
                .pieces[piece as usize]
                [to as usize];
        self.hash ^= key;
        if piece == Piece::Pawn {
            self.pawn_hashes[color as usize] ^= key;
        } else {
            self.non_pawn_hashes[color as usize] ^= key;
        }
        match piece {
            Piece::Knight | Piece::Bishop => self.minor_piece_hashes[color as usize] ^= key,
            Piece::Rook | Piece::Queen => self.major_piece_hashes[color as usize] ^= key,
            _ => {}
        }
    }

    #[inline(always)]
    pub fn set_castle_right(&mut self, color: Color, short: bool, file: Option<File>)  {
        let rights = &mut self.castle_rights[color as usize];
        let right = if short {
            &mut rights.short
        } else {
            &mut rights.long
        };
        if let Some(prev) = core::mem::replace(right, file) {
            self.hash ^= ZOBRIST.color[color as usize].castle_rights[prev as usize];
        }
        if let Some(file) = file {
            self.hash ^= ZOBRIST.color[color as usize].castle_rights[file as usize];
        }
    }

    #[inline(always)]
    pub fn set_en_passant(&mut self, new_en_passant: Option<File>) {
        if let Some(file) = core::mem::replace(&mut self.en_passant, new_en_passant) {
            self.hash ^= ZOBRIST.en_passant[file as usize];
        }
        if let Some(file) = self.en_passant {
            self.hash ^= ZOBRIST.en_passant[file as usize];
        }
    }

    #[inline(always)]
    pub fn toggle_side_to_move(&mut self) {
        self.side_to_move = !self.side_to_move;
        self.hash ^= ZOBRIST.black_to_move;
    }
}

#[cfg(test)]
mod tests {
    use super::ZOBRIST;
    use crate::{Board, Color, Piece};

    fn recompute_piece_hash(board: &Board, color: Color, pieces: &[Piece]) -> u64 {
        pieces.iter().fold(0, |hash, &piece| {
            board
                .colored_pieces(color, piece)
                .into_iter()
                .fold(hash, |hash, square| {
                    hash ^ ZOBRIST.color[color as usize].pieces[piece as usize][square as usize]
                })
        })
    }

    fn assert_piece_hashes(board: &Board) {
        for color in Color::ALL {
            assert_eq!(
                board.pawn_hash(color),
                recompute_piece_hash(board, color, &[Piece::Pawn])
            );
            assert_eq!(
                board.non_pawn_hash(color),
                recompute_piece_hash(
                    board,
                    color,
                    &[
                        Piece::Knight,
                        Piece::Bishop,
                        Piece::Rook,
                        Piece::Queen,
                        Piece::King,
                    ]
                )
            );
            assert_eq!(
                board.minor_piece_hash(color),
                recompute_piece_hash(board, color, &[Piece::Knight, Piece::Bishop])
            );
            assert_eq!(
                board.major_piece_hash(color),
                recompute_piece_hash(board, color, &[Piece::Rook, Piece::Queen])
            );
        }
    }

    #[test]
    fn piece_hashes_track_construction_and_moves() {
        let mut board = Board::default();
        assert_piece_hashes(&board);

        for mv in ["e2e4", "d7d5", "e4d5", "d8d5"] {
            board.play_unchecked(mv.parse().unwrap());
            assert_piece_hashes(&board);
        }

        let mut en_passant: Board = "7k/8/8/3pP3/8/8/8/7K w - d6 0 1".parse().unwrap();
        en_passant.play_unchecked("e5d6".parse().unwrap());
        assert_piece_hashes(&en_passant);

        let mut promotion: Board = "7k/P7/8/8/8/8/8/7K w - - 0 1".parse().unwrap();
        promotion.play_unchecked("a7a8q".parse().unwrap());
        assert_piece_hashes(&promotion);

        let mut minor_promotion: Board = "7k/P7/8/8/8/8/8/7K w - - 0 1".parse().unwrap();
        minor_promotion.play_unchecked("a7a8n".parse().unwrap());
        assert_piece_hashes(&minor_promotion);

        let mut castling: Board = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1".parse().unwrap();
        castling.play_unchecked("e1h1".parse().unwrap());
        assert_piece_hashes(&castling);

        let after_null = board.null_move().unwrap();
        assert_piece_hashes(&after_null);
        for color in Color::ALL {
            assert_eq!(board.pawn_hash(color), after_null.pawn_hash(color));
            assert_eq!(board.non_pawn_hash(color), after_null.non_pawn_hash(color));
            assert_eq!(
                board.minor_piece_hash(color),
                after_null.minor_piece_hash(color)
            );
            assert_eq!(
                board.major_piece_hash(color),
                after_null.major_piece_hash(color)
            );
        }
    }

    #[test]
    fn zobrist_transpositions() {
        let board = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
            .parse::<Board>().unwrap();
        const MOVES: &[[[&str; 4]; 2]] = &[
            [["e2c4", "h8f8", "d2h6", "b4b3"], ["e2c4", "b4b3", "d2h6", "h8f8"]],
            [["c3a4", "f6g8", "e1d1", "a8c8"], ["c3a4", "a8c8", "e1d1", "f6g8"]],
            [["h1g1", "f6g4", "d2h6", "b4b3"], ["h1g1", "b4b3", "d2h6", "f6g4"]],
            [["a1c1", "c7c5", "c3a4", "a6e2"], ["c3a4", "c7c5", "a1c1", "a6e2"]],
            [["e2c4", "h8h5", "f3f5", "e7d8"], ["f3f5", "h8h5", "e2c4", "e7d8"]],
            [["d5d6", "e8h8", "f3f6", "a6c4"], ["f3f6", "a6c4", "d5d6", "e8h8"]],
            [["f3e3", "e8h8", "a2a4", "a8c8"], ["a2a4", "a8c8", "f3e3", "e8h8"]],
            [["e1d1", "f6d5", "b2b3", "a8c8"], ["e1d1", "a8c8", "b2b3", "f6d5"]],
            [["e1d1", "e8f8", "e5c6", "h8h5"], ["e1d1", "h8h5", "e5c6", "e8f8"]],
            [["e2d3", "c7c6", "g2g4", "h8h6"], ["e2d3", "h8h6", "g2g4", "c7c6"]],
            [["f3h5", "f6h7", "c3b1", "g7f6"], ["c3b1", "f6h7", "f3h5", "g7f6"]],
            [["e2d3", "g6g5", "d2f4", "b6d5"], ["d2f4", "g6g5", "e2d3", "b6d5"]],
            [["a2a3", "h8h5", "c3b1", "a8d8"], ["a2a3", "a8d8", "c3b1", "h8h5"]],
            [["a2a4", "e8h8", "e1h1", "e7d8"], ["e1h1", "e8h8", "a2a4", "e7d8"]],
            [["b2b3", "e8f8", "g2g3", "a6b7"], ["b2b3", "a6b7", "g2g3", "e8f8"]],
            [["e5g4", "e8d8", "d2e3", "a6d3"], ["d2e3", "a6d3", "e5g4", "e8d8"]],
            [["g2h3", "e7d8", "e5g4", "b6c8"], ["e5g4", "b6c8", "g2h3", "e7d8"]],
            [["e5d3", "a6b7", "g2g3", "h8h6"], ["e5d3", "h8h6", "g2g3", "a6b7"]],
            [["e5g4", "h8h5", "f3f5", "e6f5"], ["f3f5", "e6f5", "e5g4", "h8h5"]],
            [["g2g3", "a8c8", "e5d3", "e7f8"], ["e5d3", "a8c8", "g2g3", "e7f8"]]
        ];
        for (i, [moves_a, moves_b]) in MOVES.iter().enumerate() {
            let mut board_a = board.clone();
            let mut board_b = board.clone();
            for mv in moves_a {
                board_a.play_unchecked(mv.parse().unwrap());
            }
            for mv in moves_b {
                board_b.play_unchecked(mv.parse().unwrap());
            }
            assert_eq!(board_a.hash(), board_b.hash(), "Test {}", i + 1);
        }
    }
}
