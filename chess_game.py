"""
Professional Python Chess Game - Player vs AI
===============================================
Play against an AI opponent that uses Minimax + Alpha-Beta pruning
(see chess_ai.py). Built with Pygame for the GUI and python-chess for
rules/legal-move generation.

Controls
--------
- Click a piece to select it (legal destination squares are highlighted).
- Click a highlighted square to move there.
- Press U to undo your last move (and the AI's reply).
- Press R to restart the game.
- Press 1 / 2 / 3 to change AI difficulty (Easy / Medium / Hard).

Run
---
    python3 chess_game.py
"""

import sys
import pygame
import chess

from chess_ai import find_best_move, evaluate_board

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BOARD_SIZE = 640
SQUARE = BOARD_SIZE // 8
SIDEBAR_WIDTH = 260
WINDOW_W = BOARD_SIZE + SIDEBAR_WIDTH
WINDOW_H = BOARD_SIZE

LIGHT = (240, 217, 181)
DARK = (181, 136, 99)
HIGHLIGHT = (246, 246, 105)
MOVE_DOT = (60, 60, 60)
SELECTED = (186, 202, 68)
CHECK_RED = (220, 90, 90)
SIDEBAR_BG = (35, 33, 30)
TEXT_COLOR = (235, 235, 235)
ACCENT = (100, 180, 255)

PIECE_UNICODE = {
    "P": "\u2659", "N": "\u2658", "B": "\u2657", "R": "\u2656", "Q": "\u2655", "K": "\u2654",
    "p": "\u265F", "n": "\u265E", "b": "\u265D", "r": "\u265C", "q": "\u265B", "k": "\u265A",
}

DIFFICULTY = {1: ("Easy", 1), 2: ("Medium", 2), 3: ("Hard", 3)}


class ChessGame:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Python Chess — Player vs AI")
        self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        self.clock = pygame.time.Clock()

        self.piece_font = self._load_piece_font(int(SQUARE * 0.7))
        self.ui_font = pygame.font.SysFont("arial", 20) or pygame.font.Font(None, 20)
        self.ui_font_bold = pygame.font.SysFont("arial", 22, bold=True) or pygame.font.Font(None, 22)
        self.small_font = pygame.font.SysFont("arial", 16) or pygame.font.Font(None, 16)

        self.board = chess.Board()
        self.selected_square = None
        self.legal_targets = []
        self.human_color = chess.WHITE
        self.difficulty = 2  # Medium by default
        self.status_message = "Your move (White)"
        self.ai_thinking = False
        self.last_ai_nodes = 0
        self.move_log = []

    # ------------------------------------------------------------------
    # Font loading (cross-platform: Windows / macOS / Linux)
    # ------------------------------------------------------------------
    def _load_piece_font(self, size):
        """
        Chess unicode glyphs (♔♕♖♗♘♙) are only rendered correctly by fonts
        that include that symbol block. Different OSes ship different
        fonts, so we try several known-good candidates in order and use
        the first one that's actually installed on this machine.
        """
        candidates = [
            "segoeuisymbol",      # Windows
            "applesymbols",       # macOS
            "arialunicodems",     # macOS / Office
            "notosanssymbols2",   # Linux (Noto)
            "notosanssymbols",    # Linux (Noto, older)
            "dejavusans",         # Linux (has decent glyph coverage)
            "freeserif",          # Linux
        ]

        for name in candidates:
            path = pygame.font.match_font(name)
            if path:
                try:
                    font = pygame.font.Font(path, size)
                    # sanity check: does this font actually draw a
                    # non-empty glyph for a king symbol?
                    test = font.render("\u2654", True, (255, 255, 255))
                    if test.get_width() > 2:
                        return font
                except Exception:
                    continue

        # Last resort: pygame's bundled default font (still shows *something*
        # rather than crashing, though glyph coverage may be limited).
        return pygame.font.Font(None, size)

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------
    def draw_board(self):
        for rank in range(8):
            for file in range(8):
                square = chess.square(file, 7 - rank)
                color = LIGHT if (rank + file) % 2 == 0 else DARK

                if square == self.selected_square:
                    color = SELECTED
                elif (self.board.is_check() and
                      self.board.piece_at(square) and
                      self.board.piece_at(square).piece_type == chess.KING and
                      self.board.piece_at(square).color == self.board.turn):
                    color = CHECK_RED

                rect = pygame.Rect(file * SQUARE, rank * SQUARE, SQUARE, SQUARE)
                pygame.draw.rect(self.screen, color, rect)

                if square in self.legal_targets:
                    center = rect.center
                    if self.board.piece_at(square):
                        pygame.draw.circle(self.screen, MOVE_DOT, center, SQUARE // 2 - 4, 4)
                    else:
                        pygame.draw.circle(self.screen, MOVE_DOT, center, 10)

        # coordinates
        for i in range(8):
            label = self.small_font.render(str(8 - i), True, (120, 120, 120))
            self.screen.blit(label, (4, i * SQUARE + 2))
            label2 = self.small_font.render(chr(ord('a') + i), True, (120, 120, 120))
            self.screen.blit(label2, (i * SQUARE + SQUARE - 14, BOARD_SIZE - 18))

    def draw_pieces(self):
        for rank in range(8):
            for file in range(8):
                square = chess.square(file, 7 - rank)
                piece = self.board.piece_at(square)
                if piece:
                    symbol = PIECE_UNICODE[piece.symbol()]
                    text_color = (250, 250, 250) if piece.color == chess.WHITE else (20, 20, 20)
                    shadow_color = (20, 20, 20) if piece.color == chess.WHITE else (0, 0, 0)
                    x = file * SQUARE + SQUARE // 2
                    y = rank * SQUARE + SQUARE // 2

                    shadow = self.piece_font.render(symbol, True, shadow_color)
                    srect = shadow.get_rect(center=(x + 2, y + 2))
                    self.screen.blit(shadow, srect)

                    text = self.piece_font.render(symbol, True, text_color)
                    trect = text.get_rect(center=(x, y))
                    self.screen.blit(text, trect)

    def draw_sidebar(self):
        rect = pygame.Rect(BOARD_SIZE, 0, SIDEBAR_WIDTH, WINDOW_H)
        pygame.draw.rect(self.screen, SIDEBAR_BG, rect)

        y = 20
        title = self.ui_font_bold.render("Python Chess AI", True, ACCENT)
        self.screen.blit(title, (BOARD_SIZE + 20, y))
        y += 40

        diff_name = DIFFICULTY[self.difficulty][0]
        diff_text = self.ui_font.render(f"Difficulty: {diff_name}", True, TEXT_COLOR)
        self.screen.blit(diff_text, (BOARD_SIZE + 20, y))
        y += 30

        turn_str = "White" if self.board.turn == chess.WHITE else "Black"
        turn_text = self.ui_font.render(f"Turn: {turn_str}", True, TEXT_COLOR)
        self.screen.blit(turn_text, (BOARD_SIZE + 20, y))
        y += 30

        status_lines = self._wrap_text(self.status_message, 26)
        for line in status_lines:
            stext = self.small_font.render(line, True, (200, 200, 100))
            self.screen.blit(stext, (BOARD_SIZE + 20, y))
            y += 20
        y += 10

        if self.last_ai_nodes:
            nodes_text = self.small_font.render(
                f"AI searched {self.last_ai_nodes:,} nodes", True, (150, 150, 150))
            self.screen.blit(nodes_text, (BOARD_SIZE + 20, y))
            y += 30

        # move log
        y += 10
        log_title = self.ui_font.render("Move History", True, ACCENT)
        self.screen.blit(log_title, (BOARD_SIZE + 20, y))
        y += 25
        recent = self.move_log[-14:]
        for i, mv in enumerate(recent):
            mtext = self.small_font.render(mv, True, TEXT_COLOR)
            self.screen.blit(mtext, (BOARD_SIZE + 20, y))
            y += 18

        # controls at bottom
        controls = [
            "U - Undo move",
            "R - Restart game",
            "1/2/3 - Difficulty",
        ]
        y = WINDOW_H - 90
        for c in controls:
            ctext = self.small_font.render(c, True, (140, 140, 140))
            self.screen.blit(ctext, (BOARD_SIZE + 20, y))
            y += 20

    def _wrap_text(self, text, width):
        words = text.split(" ")
        lines, cur = [], ""
        for w in words:
            if len(cur) + len(w) + 1 <= width:
                cur = (cur + " " + w).strip()
            else:
                lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        return lines

    def render(self):
        self.screen.fill((0, 0, 0))
        self.draw_board()
        self.draw_pieces()
        self.draw_sidebar()
        pygame.display.flip()

    # ------------------------------------------------------------------
    # Game logic
    # ------------------------------------------------------------------
    def square_from_pos(self, pos):
        x, y = pos
        if x >= BOARD_SIZE:
            return None
        file = x // SQUARE
        rank = 7 - (y // SQUARE)
        return chess.square(file, rank)

    def handle_click(self, pos):
        if self.board.turn != self.human_color or self.board.is_game_over():
            return

        square = self.square_from_pos(pos)
        if square is None:
            return

        if self.selected_square is None:
            piece = self.board.piece_at(square)
            if piece and piece.color == self.human_color:
                self.selected_square = square
                self.legal_targets = [
                    m.to_square for m in self.board.legal_moves
                    if m.from_square == square
                ]
        else:
            move = self._build_move(self.selected_square, square)
            if move in self.board.legal_moves:
                self._push_move(move)
                self.selected_square = None
                self.legal_targets = []
                self.check_game_over()
                if not self.board.is_game_over():
                    self.ai_turn()
            else:
                # reselect if clicking own piece
                piece = self.board.piece_at(square)
                if piece and piece.color == self.human_color:
                    self.selected_square = square
                    self.legal_targets = [
                        m.to_square for m in self.board.legal_moves
                        if m.from_square == square
                    ]
                else:
                    self.selected_square = None
                    self.legal_targets = []

    def _build_move(self, frm, to):
        """Build a move, auto-promoting to Queen if it's a pawn reaching the last rank."""
        piece = self.board.piece_at(frm)
        promotion = None
        if piece and piece.piece_type == chess.PAWN:
            to_rank = chess.square_rank(to)
            if to_rank in (0, 7):
                promotion = chess.QUEEN
        return chess.Move(frm, to, promotion=promotion)

    def _push_move(self, move):
        san = self.board.san(move)
        self.board.push(move)
        move_number = (len(self.move_log) // 2) + 1
        if self.board.turn == chess.BLACK:  # white just moved
            self.move_log.append(f"{move_number}. {san}")
        else:
            self.move_log.append(f"    ...{san}")

    def ai_turn(self):
        self.status_message = "AI is thinking..."
        self.render()
        depth = DIFFICULTY[self.difficulty][1]
        move, value, nodes = find_best_move(self.board, depth=depth)
        self.last_ai_nodes = nodes
        if move is not None:
            self._push_move(move)
        self.check_game_over()
        if not self.board.is_game_over():
            self.status_message = "Your move"

    def check_game_over(self):
        if self.board.is_checkmate():
            winner = "Black" if self.board.turn == chess.WHITE else "White"
            self.status_message = f"Checkmate! {winner} wins."
        elif self.board.is_stalemate():
            self.status_message = "Draw by stalemate."
        elif self.board.is_insufficient_material():
            self.status_message = "Draw - insufficient material."
        elif self.board.is_check():
            self.status_message = "Check!"

    def undo(self):
        if len(self.board.move_stack) >= 2:
            self.board.pop()
            self.board.pop()
            self.move_log = self.move_log[:-2]
            self.selected_square = None
            self.legal_targets = []
            self.status_message = "Move undone. Your move."

    def restart(self):
        self.board = chess.Board()
        self.selected_square = None
        self.legal_targets = []
        self.move_log = []
        self.status_message = "New game. Your move (White)"
        self.last_ai_nodes = 0

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_click(event.pos)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_u:
                        self.undo()
                    elif event.key == pygame.K_r:
                        self.restart()
                    elif event.key == pygame.K_1:
                        self.difficulty = 1
                    elif event.key == pygame.K_2:
                        self.difficulty = 2
                    elif event.key == pygame.K_3:
                        self.difficulty = 3

            self.render()
            self.clock.tick(30)

        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    game = ChessGame()
    game.run()