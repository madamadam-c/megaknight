use std::{
    fs::File,
    io::{self, BufRead, BufReader, Seek, SeekFrom, Write},
    path::Path,
};

use cozy_chess::Board;

const MAX_SAMPLE_ATTEMPTS: usize = 128;
const DEFAULT_RANDOM_PLIES: usize = 6;

pub fn run(command: &str) -> Result<(), String> {
    let mut arguments = command.split_whitespace();
    if arguments.next() != Some("genfens") {
        return Err("genfens command must start with genfens".to_string());
    }

    let count = parse_positive(arguments.next(), "opening count")?;
    if arguments.next() != Some("seed") {
        return Err("genfens requires seed <u64>".to_string());
    }
    let seed = arguments
        .next()
        .ok_or_else(|| "genfens requires a seed".to_string())?
        .parse::<u64>()
        .map_err(|_| "invalid genfens seed".to_string())?;
    if arguments.next() != Some("book") {
        return Err("genfens requires book <path|None>".to_string());
    }

    let book = arguments
        .next()
        .ok_or_else(|| "genfens requires book <path|None>".to_string())?;
    let extra = arguments.collect::<Vec<_>>();
    let random_plies = parse_random_plies(&extra)?;
    let mut rng = SplitMix64::new(seed);
    let mut sampler = if book.eq_ignore_ascii_case("none") {
        None
    } else {
        Some(BookSampler::open(book)?)
    };

    let stdout = io::stdout();
    let mut stdout = io::LineWriter::new(stdout.lock());
    for _ in 0..count {
        let fen = match sampler.as_mut() {
            Some(sampler) => sampler.sample_board(&mut rng, random_plies)?.to_string(),
            None => random_position(&mut rng, random_plies),
        };
        writeln!(stdout, "info string genfens {fen}")
            .map_err(|error| format!("failed to write genfens output: {error}"))?;
    }
    stdout
        .flush()
        .map_err(|error| format!("failed to flush genfens output: {error}"))?;
    Ok(())
}

fn parse_random_plies(arguments: &[&str]) -> Result<usize, String> {
    let mut random_plies = DEFAULT_RANDOM_PLIES;
    let mut index = 0;
    while index < arguments.len() {
        let argument = arguments[index];
        let (name, inline_value) = argument.split_once('=').unwrap_or((argument, ""));
        if !matches!(
            name,
            "--random-plies" | "random-plies" | "--plies" | "plies"
        ) {
            index += 1;
            continue;
        }

        let value = if inline_value.is_empty() {
            index += 1;
            arguments
                .get(index)
                .ok_or_else(|| "random plies option requires a value".to_string())?
        } else {
            inline_value
        };
        random_plies = value
            .parse::<usize>()
            .map_err(|_| "invalid random plies value".to_string())?;
        index += 1;
    }
    Ok(random_plies)
}

fn parse_positive(value: Option<&str>, name: &str) -> Result<usize, String> {
    let value = value.ok_or_else(|| format!("genfens requires an {name}"))?;
    let value = value
        .parse::<usize>()
        .map_err(|_| format!("invalid {name}"))?;
    if value == 0 {
        return Err(format!("{name} must be positive"));
    }
    Ok(value)
}

pub struct BookSampler {
    reader: BufReader<File>,
    length: u64,
}

impl BookSampler {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let file = File::open(path)
            .map_err(|error| format!("failed to open book {}: {error}", path.display()))?;
        let length = file
            .metadata()
            .map_err(|error| format!("failed to stat book {}: {error}", path.display()))?
            .len();
        if length == 0 {
            return Err(format!("book is empty: {}", path.display()));
        }
        Ok(Self {
            reader: BufReader::with_capacity(64 * 1024, file),
            length,
        })
    }

    pub fn sample_board(
        &mut self,
        rng: &mut SplitMix64,
        random_plies: usize,
    ) -> Result<Board, String> {
        for _ in 0..MAX_SAMPLE_ATTEMPTS {
            let offset = rng.next_u64() % self.length;
            self.reader
                .seek(SeekFrom::Start(offset))
                .map_err(|error| format!("failed to seek book: {error}"))?;

            if offset != 0 {
                let mut discarded = Vec::new();
                self.reader
                    .read_until(b'\n', &mut discarded)
                    .map_err(|error| format!("failed to seek to a book line: {error}"))?;
            }

            let mut line = String::new();
            if self
                .reader
                .read_line(&mut line)
                .map_err(|error| format!("failed to read book: {error}"))?
                == 0
            {
                continue;
            }
            if let Some(fen) = normalize_book_line(&line) {
                let mut board = fen
                    .parse::<Board>()
                    .map_err(|_| "book produced an invalid FEN".to_string())?;
                randomize_position(&mut board, random_plies, rng);
                return Ok(board);
            }
        }

        Err("unable to sample a valid FEN from the opening book".to_string())
    }
}

fn normalize_book_line(line: &str) -> Option<String> {
    let fields = line.split_whitespace().collect::<Vec<_>>();
    if fields.len() < 4 {
        return None;
    }

    let halfmove = fields
        .get(4)
        .and_then(|value| value.parse::<u8>().ok())
        .or_else(|| epd_value(&fields[4..], "hmvc").and_then(|value| u8::try_from(value).ok()))
        .unwrap_or(0);
    let fullmove = fields
        .get(5)
        .and_then(|value| value.parse::<u16>().ok())
        .or_else(|| epd_value(&fields[4..], "fmvn"))
        .unwrap_or(1);

    let fen = format!(
        "{} {} {} {} {} {}",
        fields[0], fields[1], fields[2], fields[3], halfmove, fullmove
    );
    fen.parse::<Board>().ok().map(|board| board.to_string())
}

fn epd_value(fields: &[&str], name: &str) -> Option<u16> {
    fields
        .windows(2)
        .find(|pair| pair[0].eq_ignore_ascii_case(name))?
        .get(1)?
        .trim_end_matches(';')
        .parse()
        .ok()
}

fn random_position(rng: &mut SplitMix64, random_plies: usize) -> String {
    let mut board = Board::default();
    randomize_position(&mut board, random_plies, rng);
    board.to_string()
}

fn randomize_position(board: &mut Board, plies: usize, rng: &mut SplitMix64) {
    for _ in 0..plies {
        let mut moves = Vec::new();
        board.generate_moves(|piece_moves| {
            moves.extend(piece_moves.into_iter().filter(|&chess_move| {
                chess_move.promotion.is_none()
                    && board.color_on(chess_move.to).is_none()
                    && !(board
                        .piece_on(chess_move.from)
                        .is_some_and(|piece| piece == cozy_chess::Piece::Pawn)
                        && chess_move.from.file() != chess_move.to.file())
            }));
            false
        });
        if moves.is_empty() {
            break;
        }
        board.play_unchecked(moves[rng.next_u64() as usize % moves.len()]);
    }
}

pub(crate) struct SplitMix64(u64);

impl SplitMix64 {
    pub(crate) const fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub(crate) fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_four_field_epd_to_full_fen() {
        assert_eq!(
            normalize_book_line("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -"),
            Some("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1".to_string())
        );
    }

    #[test]
    fn normalizes_epd_clock_operations() {
        assert_eq!(
            normalize_book_line("8/8/8/8/8/8/8/K6k w - - hmvc 12; fmvn 34;"),
            Some("8/8/8/8/8/8/8/K6k w - - 12 34".to_string())
        );
    }

    #[test]
    fn random_positions_are_valid_full_fens() {
        let mut rng = SplitMix64::new(123);
        for _ in 0..32 {
            assert!(random_position(&mut rng, 6).parse::<Board>().is_ok());
        }
    }

    #[test]
    fn parses_random_plies_options() {
        assert_eq!(parse_random_plies(&[]).unwrap(), 6);
        assert_eq!(parse_random_plies(&["--random-plies", "9"]).unwrap(), 9);
        assert_eq!(parse_random_plies(&["--plies=3"]).unwrap(), 3);
    }
}
