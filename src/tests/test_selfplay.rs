use super::*;
use cozy_chess::{Square, util::parse_uci_move};

#[test]
fn parses_selfplay_options_and_defaults() {
    let config = parse_args(
        [
            "--positions",
            "100",
            "--nodes",
            "2000",
            "--threads",
            "3",
            "--warmup-plies",
            "0",
        ]
        .map(OsString::from),
    )
    .unwrap()
    .unwrap();

    assert_eq!(config.positions, 100);
    assert_eq!(config.nodes, 2_000);
    assert_eq!(config.threads, 3);
    assert_eq!(config.warmup_plies, 0);
    assert_eq!(config.random_plies, DEFAULT_RANDOM_PLIES);
}

#[test]
fn requires_a_positive_position_count() {
    assert!(parse_args(Vec::<OsString>::new()).is_err());
    assert!(parse_args([OsString::from("--positions"), OsString::from("0")]).is_err());
}

#[test]
fn quiet_filter_rejects_captures_en_passant_and_promotions() {
    let start = Board::default();
    assert!(is_quiet_move(
        &start,
        parse_uci_move(&start, "e2e4").unwrap()
    ));

    let capture = Board::from_fen("7k/8/8/3p4/4P3/8/8/K7 w - - 0 1", false).unwrap();
    assert!(!is_quiet_move(
        &capture,
        parse_uci_move(&capture, "e4d5").unwrap()
    ));

    let en_passant = Board::from_fen("7k/8/8/3pP3/8/8/8/7K w - d6 0 1", false).unwrap();
    assert!(!is_quiet_move(
        &en_passant,
        parse_uci_move(&en_passant, "e5d6").unwrap()
    ));

    let promotion = Board::from_fen("7k/P7/8/8/8/8/8/7K w - - 0 1", false).unwrap();
    assert!(!is_quiet_move(
        &promotion,
        parse_uci_move(&promotion, "a7a8q").unwrap()
    ));
}

#[test]
fn random_opening_moves_are_reproducible_and_legal() {
    let mut first = Board::default();
    let mut second = Board::default();
    randomize_opening(&mut first, 8, &mut SplitMix64::new(42));
    randomize_opening(&mut second, 8, &mut SplitMix64::new(42));

    assert_eq!(first, second);
    assert_ne!(first, Board::default());
    assert!(first.piece_on(Square::E1).is_some());
}

#[test]
fn nnue_selfplay_produces_valid_bullet_records() {
    let mut engine = Engine::new();
    let config = SelfplayConfig {
        output: PathBuf::new(),
        openings: PathBuf::new(),
        positions: 1,
        nodes: 100,
        threads: 1,
        hash_mb: 1,
        max_plies: 24,
        warmup_plies: 0,
        random_plies: 0,
        win_score: DEFAULT_WIN_SCORE,
        win_plies: DEFAULT_WIN_PLIES,
        seed: 1,
        overwrite: false,
    };
    let game = play_game(
        &mut engine,
        Board::default(),
        &config,
        &Arc::new(AtomicBool::new(false)),
    )
    .unwrap();

    assert!(game.plies <= config.max_plies);
    for position in game.positions {
        let board = Board::from_fen(&position.fen, false).unwrap();
        assert!(board.occupied().len() > 2);
        assert!(position.white_score.abs() <= MAX_BULLET_SCORE);
    }
}
