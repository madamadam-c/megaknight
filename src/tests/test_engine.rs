use super::*;
use cozy_chess::Square;

fn board(fen: &str) -> Board {
    Board::from_fen(fen, false).unwrap()
}

fn run_quiesce(board: &Board) -> (i32, u64) {
    let mut engine = Engine {
        tt: Table::new(1),
        eval: eval(board),
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
    assert_eq!(run_quiesce(&board).0, 500);
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
    assert!(moves.iter().all(|mv| {
        mv.mv.from == Square::A7 && mv.mv.to == Square::A8 && mv.promotion
    }));
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
    assert_eq!(score, -500);
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
