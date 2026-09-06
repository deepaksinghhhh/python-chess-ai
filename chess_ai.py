"""
Chess AI Engine
----------------
Minimax algorithm with Alpha-Beta pruning + piece-square table evaluation.
Uses python-chess for legal move generation and board rules (checkmate,
stalemate, castling, en-passant, promotion, etc.) so the AI logic can focus
purely on search + evaluation.
"""

import chess

# ---------------------------------------------------------------------------
# 1. Material values
# ---------------------------------------------------------------------------
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000,
}

# ---------------------------------------------------------------------------
# 2. Piece-Square Tables (encourage good positional play)
#    Tables are defined for White from White's perspective (a1 = index 0).
#    For Black we mirror the table vertically.
# ---------------------------------------------------------------------------
PAWN_TABLE = [
      0,   0,   0,   0,   0,   0,   0,   0,
      5,  10,  10, -20, -20,  10,  10,   5,
      5,  -5, -10,   0,   0, -10,  -5,   5,
      0,   0,   0,  20,  20,   0,   0,   0,
      5,   5,  10,  25,  25,  10,   5,   5,
     10,  10,  20,  30,  30,  20,  10,  10,
     50,  50,  50,  50,  50,  50,  50,  50,
      0,   0,   0,   0,   0,   0,   0,   0,
]

KNIGHT_TABLE = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20,   0,   5,   5,   0, -20, -40,
    -30,   5,  10,  15,  15,  10,   5, -30,
    -30,   0,  15,  20,  20,  15,   0, -30,
    -30,   5,  15,  20,  20,  15,   5, -30,
    -30,   0,  10,  15,  15,  10,   0, -30,
    -40, -20,   0,   0,   0,   0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50,
]

BISHOP_TABLE = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10,   5,   0,   0,   0,   0,   5, -10,
    -10,  10,  10,  10,  10,  10,  10, -10,
    -10,   0,  10,  10,  10,  10,   0, -10,
    -10,   5,   5,  10,  10,   5,   5, -10,
    -10,   0,   5,  10,  10,   5,   0, -10,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -20, -10, -10, -10, -10, -10, -10, -20,
]

ROOK_TABLE = [
      0,   0,   0,   5,   5,   0,   0,   0,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
      5,  10,  10,  10,  10,  10,  10,   5,
      0,   0,   0,   0,   0,   0,   0,   0,
]

QUEEN_TABLE = [
    -20, -10, -10,  -5,  -5, -10, -10, -20,
    -10,   0,   5,   0,   0,   0,   0, -10,
    -10,   5,   5,   5,   5,   5,   0, -10,
      0,   0,   5,   5,   5,   5,   0,  -5,
     -5,   0,   5,   5,   5,   5,   0,  -5,
    -10,   0,   5,   5,   5,   5,   0, -10,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -20, -10, -10,  -5,  -5, -10, -10, -20,
]

KING_TABLE = [
     20,  30,  10,   0,   0,  10,  30,  20,
     20,  20,   0,   0,   0,   0,  20,  20,
    -10, -20, -20, -20, -20, -20, -20, -10,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
]

PIECE_TABLES = {
    chess.PAWN: PAWN_TABLE,
    chess.KNIGHT: KNIGHT_TABLE,
    chess.BISHOP: BISHOP_TABLE,
    chess.ROOK: ROOK_TABLE,
    chess.QUEEN: QUEEN_TABLE,
    chess.KING: KING_TABLE,
}


def _square_score(piece: chess.Piece, square: int) -> int:
    """Positional bonus for a piece sitting on a given square."""
    table = PIECE_TABLES[piece.piece_type]
    if piece.color == chess.WHITE:
        return table[square]
    # mirror the square vertically for black
    mirrored = chess.square_mirror(square)
    return table[mirrored]


def evaluate_board(board: chess.Board) -> int:
    """
    Static evaluation of a position from White's perspective.
    Positive = good for White, Negative = good for Black.
    """
    if board.is_checkmate():
        # side to move is checkmated -> very bad for side to move
        return -99999 if board.turn == chess.WHITE else 99999
    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    score = 0
    for square, piece in board.piece_map().items():
        value = PIECE_VALUES[piece.piece_type] + _square_score(piece, square)
        score += value if piece.color == chess.WHITE else -value

    # small bonus for mobility (more legal moves = more flexible position)
    score += 2 * (len(list(board.legal_moves)) if board.turn == chess.WHITE else 0)

    return score


def order_moves(board: chess.Board, moves):
    """Search captures/promotions first -> much better alpha-beta pruning."""
    def score(move):
        s = 0
        if board.is_capture(move):
            s += 10
        if move.promotion:
            s += 9
        return s
    return sorted(moves, key=score, reverse=True)


def minimax(board: chess.Board, depth: int, alpha: float, beta: float,
            maximizing: bool, nodes: list) -> float:
    """Minimax search with alpha-beta pruning. `nodes` is a 1-item list used
    as a mutable counter so we can report search stats."""
    nodes[0] += 1

    if depth == 0 or board.is_game_over():
        return evaluate_board(board)

    legal_moves = order_moves(board, list(board.legal_moves))

    if maximizing:
        best = -float("inf")
        for move in legal_moves:
            board.push(move)
            best = max(best, minimax(board, depth - 1, alpha, beta, False, nodes))
            board.pop()
            alpha = max(alpha, best)
            if beta <= alpha:
                break  # beta cut-off
        return best
    else:
        best = float("inf")
        for move in legal_moves:
            board.push(move)
            best = min(best, minimax(board, depth - 1, alpha, beta, True, nodes))
            board.pop()
            beta = min(beta, best)
            if beta <= alpha:
                break  # alpha cut-off
        return best


def find_best_move(board: chess.Board, depth: int = 3):
    """
    Returns (best_move, evaluation, nodes_searched) for the position.
    AI always searches assuming it wants to maximize score for its own color.
    """
    maximizing = board.turn == chess.WHITE
    best_move = None
    best_value = -float("inf") if maximizing else float("inf")
    nodes = [0]

    legal_moves = order_moves(board, list(board.legal_moves))

    for move in legal_moves:
        board.push(move)
        value = minimax(board, depth - 1, -float("inf"), float("inf"), not maximizing, nodes)
        board.pop()

        if maximizing and value > best_value:
            best_value, best_move = value, move
        elif not maximizing and value < best_value:
            best_value, best_move = value, move

    return best_move, best_value, nodes[0]


if __name__ == "__main__":
    # Quick sanity test: play the AI against itself for a few moves.
    b = chess.Board()
    print(b)
    print()
    for i in range(4):
        move, val, nodes = find_best_move(b, depth=2)
        b.push(move)
        print(f"Move {i+1}: {move}  (eval={val}, nodes={nodes})")
    print()
    print(b)
