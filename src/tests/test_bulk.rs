use std::{
    ffi::OsString,
    io::Cursor,
    sync::{Arc, atomic::AtomicBool},
};

use super::*;

#[test]
fn parses_bulk_options_and_defaults() {
    let config = parse_args(
        [
            "--positions",
            "123",
            "--nodes",
            "456",
            "--threads",
            "3",
            "--hash-mb",
            "2",
            "--input",
            "input.tar.zst",
            "--output",
            "output.txt",
            "--include-tactical",
            "--overwrite",
        ]
        .map(OsString::from),
    )
    .unwrap()
    .unwrap();

    assert_eq!(config.positions, 123);
    assert_eq!(config.nodes, 456);
    assert_eq!(config.threads, 3);
    assert_eq!(config.hash_mb, 2);
    assert_eq!(config.input, PathBuf::from("input.tar.zst"));
    assert_eq!(config.output, PathBuf::from("output.txt"));
    assert!(config.include_tactical);
    assert!(config.overwrite);
}

#[test]
fn requires_a_positive_position_count() {
    assert!(parse_args(Vec::<OsString>::new()).is_err());
    assert!(parse_args([OsString::from("--positions"), OsString::from("0")]).is_err());
}

#[test]
fn converts_scores_and_results_to_white_perspective() {
    assert_eq!(white_score(125, true), 125);
    assert_eq!(white_score(125, false), -125);
    assert_eq!(white_score(100_000, true), MAX_BULLET_SCORE);
    assert_eq!(white_score(-100_000, false), MAX_BULLET_SCORE);

    assert_eq!(white_result(1, true).unwrap(), "1.0");
    assert_eq!(white_result(-1, true).unwrap(), "0.0");
    assert_eq!(white_result(1, false).unwrap(), "0.0");
    assert_eq!(white_result(-1, false).unwrap(), "1.0");
    assert_eq!(white_result(0, false).unwrap(), "0.5");
}

#[test]
fn formats_the_bullet_text_record() {
    let fen = "8/8/8/8/8/8/4K3/7k w - - 0 1";
    assert_eq!(
        format_record(fen, -42, "0.5"),
        "8/8/8/8/8/8/4K3/7k w - - 0 1 | -42 | 0.5"
    );
}

#[test]
fn reads_streaming_binpack_chunks() {
    let mut data = b"BINP".to_vec();
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&[7, 8, 9]);
    let mut reader = Cursor::new(data);
    let mut chunk = Vec::new();

    assert!(read_binpack_chunk(&mut reader, &mut chunk).unwrap());
    assert_eq!(chunk, [7, 8, 9]);
    assert!(!read_binpack_chunk(&mut reader, &mut chunk).unwrap());
}

#[test]
fn retries_depth_one_when_the_node_limit_is_too_small() {
    let board = Board::from_fen(
        "r2qk2r/pp3p2/2p1p2b/3PPb1p/1P1PB1p1/2N1P2P/P5P1/2RQ1RK1 w kq - 2 20",
        false,
    )
    .unwrap();
    let mut engine = Engine::new();

    // A tiny node budget: lazy move picking makes depth 1 fit well under the
    // old 10,000-node limit, so use a bound that is genuinely too small.
    let (score, retried) =
        search_score(&mut engine, &board, 100, &Arc::new(AtomicBool::new(false)));

    assert!(retried);
    assert!(score.is_some());
}
