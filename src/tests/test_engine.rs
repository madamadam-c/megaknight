use super::*;
use cozy_chess::Square;

fn board(fen: &str) -> Board {
    Board::from_fen(fen, false).unwrap()
}

fn run_quiesce(board: &Board) -> (i32, u64) {
    let mut engine = Engine {
        tt: Table::new(1),
        nnue: NnueState::from_board(board),
        history: [[[0; 64]; 64]; 2],
    };
    let mut context = SearchContext::new(
        board,
        vec![board.hash()],
        SearchLimits {
            infinite: true,
            ..SearchLimits::default()
        },
        Arc::new(AtomicBool::new(false)),
    );
    let score = engine
        .quiesce(
            board,
            0,
            SearchBounds {
                alpha: -1_000_000_000,
                beta: 1_000_000_000,
            },
            &mut context,
        )
        .unwrap();

    (score, context.nodes)
}

#[test]
fn qsearch_generates_captures_but_not_quiet_moves() {
    let board = board("7k/8/8/8/8/8/p7/R6K w - - 0 1");
    let engine = Engine::new();
    let mut empty = true;
    let moves = engine.generate_qsearch_moves(&board, &mut empty);

    assert!(!empty);
    assert_eq!(moves.len(), 1);
    assert_eq!(
        moves[0].mv,
        Move {
            from: Square::A1,
            to: Square::A2,
            promotion: None,
        }
    );
    assert!(moves[0].is_capture);
    let mut after = board.clone();
    after.play_unchecked(moves[0].mv);
    let capture_score = -NnueState::from_board(&after).evaluate(after.side_to_move());
    let stand_pat = NnueState::from_board(&board).evaluate(board.side_to_move());
    assert_eq!(run_quiesce(&board).0, stand_pat.max(capture_score));
}

#[test]
fn qsearch_generates_en_passant() {
    let board = board("7k/8/8/3pP3/8/8/8/7K w - d6 0 1");
    let engine = Engine::new();
    let mut empty = true;
    let moves = engine.generate_qsearch_moves(&board, &mut empty);

    assert!(!empty);
    assert_eq!(moves.len(), 1);
    assert_eq!(
        moves[0].mv,
        Move {
            from: Square::E5,
            to: Square::D6,
            promotion: None,
        }
    );
    assert!(moves[0].is_ep);
}

#[test]
fn qsearch_generates_all_quiet_promotions() {
    let board = board("7k/P7/8/8/8/8/8/7K w - - 0 1");
    let engine = Engine::new();
    let mut empty = true;
    let moves = engine.generate_qsearch_moves(&board, &mut empty);
    let promotions: Vec<_> = moves.iter().map(|mv| mv.mv.promotion.unwrap()).collect();

    assert!(!empty);
    assert_eq!(moves.len(), 4);
    assert!(
        moves
            .iter()
            .all(|mv| { mv.mv.from == Square::A7 && mv.mv.to == Square::A8 && mv.promotion })
    );
    assert_eq!(
        promotions,
        vec![Piece::Knight, Piece::Bishop, Piece::Rook, Piece::Queen]
    );
}

#[test]
fn qsearch_searches_quiet_evasions_while_in_check() {
    let board = board("k3r3/8/8/8/8/8/8/4K3 w - - 0 1");
    let mut legal_moves = Vec::new();
    board.generate_moves(|moves| {
        legal_moves.extend(moves);
        false
    });

    assert!(!board.checkers().is_empty());
    assert!(legal_moves.iter().all(|mv| board.piece_on(mv.to).is_none()));

    let (score, nodes) = run_quiesce(&board);
    assert!(score > -MATE_THRESHOLD);
    assert_eq!(nodes, 1 + legal_moves.len() as u64);
}

#[test]
fn qsearch_scores_checkmate_and_stalemate() {
    let checkmate = board("7k/6Q1/5K2/8/8/8/8/8 b - - 0 1");
    let stalemate = board("7k/5K2/6Q1/8/8/8/8/8 b - - 0 1");

    assert_eq!(run_quiesce(&checkmate).0, -100_000);
    assert_eq!(run_quiesce(&stalemate).0, 0);
}

#[test]
fn mate_scores_prefer_shorter_mates_and_round_trip_through_tt() {
    let checkmate = board("7k/6Q1/5K2/8/8/8/8/8 b - - 0 1");
    let score = terminal_score(&checkmate, 3);

    assert_eq!(score, -99_997);
    assert_eq!(score_from_tt(score_to_tt(score, 3), 3), score);
    assert!(MATE_SCORE - 3 > MATE_SCORE - 5);
    assert!(-MATE_SCORE + 5 > -MATE_SCORE + 3);
}

#[test]
fn repetition_requires_three_matching_positions() {
    let board = board("7k/8/8/8/8/8/8/K7 w - - 8 5");
    let key = board.hash();

    assert!(!is_repetition(&board, &[key, key]));
    assert!(is_repetition(&board, &[key, key, key]));
}

#[test]
fn history_bonus_scales_with_depth_without_an_early_cap() {
    assert_eq!(history_bonus(1), 1);
    assert_eq!(history_bonus(8), 64);
    assert_eq!(history_bonus(16), 256);
    assert!(history_bonus(16) > history_bonus(8));
    assert_eq!(history_bonus(i32::MAX), i32::MAX);
}

#[test]
fn history_update_applies_gravity_and_stays_bounded() {
    let mut value = 0;
    update_history(&mut value, MAX_HISTORY / 2);
    assert_eq!(value, MAX_HISTORY / 2);

    update_history(&mut value, MAX_HISTORY / 2);
    assert_eq!(value, MAX_HISTORY * 3 / 4);

    update_history(&mut value, i32::MAX);
    assert_eq!(value, MAX_HISTORY);

    let mut negative = 0;
    update_history(&mut negative, i32::MIN);
    assert_eq!(negative, -MAX_HISTORY);
}

#[test]
fn history_orders_quiets_for_the_side_to_move() {
    let board = Board::default();
    let mut engine = Engine::new();
    let preferred = Move {
        from: Square::E2,
        to: Square::E4,
        promotion: None,
    };

    engine.history[0][Square::E2 as usize][Square::E4 as usize] = 1;
    engine.history[1][Square::D2 as usize][Square::D4 as usize] = MAX_HISTORY;

    let moves = engine.generate_moves(&board, None);
    let mut picker = MovePicker::new(moves, 0);
    let first = picker.next(&engine.history).unwrap();
    assert_eq!(first.mv, preferred);
}

#[test]
fn quiet_promotions_are_ordered_before_maximum_history_quiets() {
    let board = board("7k/P7/8/8/8/8/8/7K w - - 0 1");
    let mut engine = Engine::new();
    engine.history[0] = [[MAX_HISTORY; 64]; 64];

    let moves = engine.generate_moves(&board, None);
    let mut picker = MovePicker::new(moves, 0);
    let mut picked = Vec::new();
    while let Some(mv) = picker.next(&engine.history) {
        picked.push(mv);
    }

    assert!(picked[..4].iter().all(|mv| mv.promotion));
    assert_eq!(picked[0].mv.promotion, Some(Piece::Queen));
    assert!(picked[4..].iter().all(|mv| !mv.promotion));
}

#[test]
fn a_new_search_resets_history() {
    let board = Board::default();
    let mut engine = Engine::new();
    engine.history = [[[123; 64]; 64]; 2];
    let request = SearchRequest {
        history: vec![board.hash()],
        board,
        limits: SearchLimits {
            depth: Some(1),
            ..SearchLimits::default()
        },
        stop: Arc::new(AtomicBool::new(false)),
    };

    engine.search(&request, |_| {});

    assert!(
        engine
            .history
            .iter()
            .flatten()
            .flatten()
            .all(|&value| value == 0)
    );
}

#[test]
fn interrupted_pvs_probe_restores_search_state() {
    let board = Board::default();
    let mut engine = Engine::new();
    engine.nnue = NnueState::from_board(&board);
    let initial_nnue = engine.nnue;
    let initial_history = vec![board.hash()];
    let mut context = SearchContext::new(
        &board,
        initial_history.clone(),
        SearchLimits {
            infinite: true,
            ..SearchLimits::default()
        },
        Arc::new(AtomicBool::new(true)),
    );
    let mv = Move {
        from: Square::E2,
        to: Square::E4,
        promotion: None,
    };
    let engine_move = EngineMove::new(&board, mv, Piece::Pawn, false);

    let result = engine.search_move(
        &board,
        engine_move,
        1,
        1,
        SearchBounds {
            alpha: -100,
            beta: 100,
        },
        false,
        &mut context,
    );

    assert_eq!(result, None);
    assert_eq!(engine.nnue, initial_nnue);
    assert_eq!(context.history, initial_history);
}

#[test]
fn root_pvs_matches_full_window_root_search() {
    let board = board("r3k2r/p1ppqpb1/bn2pnp1/2pP4/1p2P3/2N2N2/PPQBBPPP/R3K2R w KQkq - 0 1");
    let depth = 4;
    let limits = SearchLimits {
        infinite: true,
        ..SearchLimits::default()
    };

    let mut pvs_engine = Engine::new();
    pvs_engine.nnue = NnueState::from_board(&board);
    let root_moves = pvs_engine.generate_moves(&board, None);
    let mut pvs_context = SearchContext::new(
        &board,
        vec![board.hash()],
        limits.clone(),
        Arc::new(AtomicBool::new(false)),
    );
    let pvs_result = pvs_engine
        .root_search(&board, depth, &root_moves, &mut pvs_context)
        .unwrap();

    let mut full_engine = Engine::new();
    let full_moves = full_engine.generate_moves(&board, None);
    let mut full_context = SearchContext::new(
        &board,
        vec![board.hash()],
        limits,
        Arc::new(AtomicBool::new(false)),
    );
    let mut bounds = SearchBounds {
        alpha: -1_000_000_000,
        beta: 1_000_000_000,
    };
    let mut full_result = None;
    full_engine.nnue = NnueState::from_board(&board);

    for mv in full_moves.iter().copied() {
        let score = full_engine
            .search_move(&board, mv, depth - 1, 1, bounds, true, &mut full_context)
            .unwrap();

        if full_result.is_none_or(|(_, best_score)| score > best_score) {
            full_result = Some((mv.mv, score));
            bounds.alpha = score;
        }
    }

    assert_eq!(pvs_result, full_result.unwrap());
}
