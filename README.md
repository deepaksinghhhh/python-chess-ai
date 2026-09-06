# Python Chess AI ♟️

A fully playable chess game built in Python, featuring an AI opponent powered
by the **Minimax algorithm with Alpha-Beta pruning**.

## Features

- ✅ Full chess rules — legal moves, castling, en passant, pawn promotion,
  check/checkmate/stalemate detection (via the `python-chess` library)
- ✅ AI opponent using **Minimax + Alpha-Beta pruning** with a custom
  evaluation function (material value + piece-square tables + mobility)
- ✅ 3 difficulty levels (Easy / Medium / Hard → search depth 1 / 2 / 3)
- ✅ Clean Pygame GUI — click-to-move, legal-move highlighting, check
  highlighting, move history sidebar, node-count stats
- ✅ Undo move / Restart game
- ✅ Clean, modular code: game engine (`chess_ai.py`) separated from GUI
  (`chess_game.py`)

## Requirements

```
pip install python-chess pygame --break-system-packages
```

## How to Run

```
python3 chess_game.py
```

## Controls

| Key / Action        | Effect                          |
|----------------------|----------------------------------|
| Click a piece         | Select it (shows legal moves)   |
| Click a highlighted square | Move there                 |
| `U`                   | Undo last move (yours + AI's)   |
| `R`                   | Restart the game                |
| `1` / `2` / `3`       | Set difficulty Easy/Medium/Hard |

## How the AI Works

1. **Move generation & rules** are handled by `python-chess` (a well-tested
   library), so the project logic focuses on decision-making, not chess-rule
   edge cases.
2. **Evaluation function** (`evaluate_board`) scores a position using:
   - Material value (pawn=100, knight=320, bishop=330, rook=500, queen=900)
   - Piece-square tables — positional bonuses (e.g. knights are worth more
     in the center, pawns are worth more advanced)
   - Mobility bonus
3. **Minimax search** explores future move sequences up to a fixed depth,
   assuming both sides play optimally.
4. **Alpha-Beta pruning** cuts off branches of the search tree that can't
   possibly affect the final decision, making the search much faster
   (this is why "Hard" mode at depth 3 is still responsive).
5. **Move ordering** (captures/promotions searched first) further improves
   pruning efficiency.

## Project Structure

```
chess_ai.py     # AI engine: evaluation function + minimax + alpha-beta pruning
chess_game.py   # Pygame GUI: board rendering, input handling, game loop
```

## Possible Extensions (great for making this project even more impressive)

- Opening book (predefined strong opening moves)
- Iterative deepening + time-limited search
- Transposition tables (caching evaluated positions)
- Quiescence search (avoid evaluating mid-capture positions)
- Save/load games in PGN format
- Online multiplayer via sockets
