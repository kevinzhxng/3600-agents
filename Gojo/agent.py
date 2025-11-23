from collections.abc import Callable
from time import sleep
from typing import List, Set, Tuple
import numpy as np
from game import *

"""
Minimax agent with alpha-beta pruning for optimal gameplay.
"""


class PlayerAgent:
    """
    /you may add functions, however, __init__ and play are the entry points for
    your program and should not be changed.
    """

    def __init__(self, board: board.Board, time_left: Callable):
        pass

    def play(
        self,
        board: board.Board,
        sensor_data: List[Tuple[bool, bool]],
        time_left: Callable,
    ):
        location = board.chicken_player.get_location()
        print(f"I'm at {location}.")
        print(f"Trapdoor A: heard? {sensor_data[0][0]}, felt? {sensor_data[0][1]}")
        print(f"Trapdoor B: heard? {sensor_data[1][0]}, felt? {sensor_data[1][1]}")
        print(f"Starting to think with {time_left()} seconds left.")

        moves = board.get_valid_moves()

        # Use minimax with alpha-beta pruning to find the best move
        best_move = self.get_best_move(board, time_left)
        result = best_move if best_move is not None else moves[0]

        print(f"I have {time_left()} seconds left. Playing {result}.")
        return result

    def get_best_move(self, board: board.Board, time_left: Callable):
        """
        Find the best move using iterative deepening minimax with alpha-beta pruning.
        """
        moves = board.get_valid_moves()

        if not moves:
            return None

        best_move = moves[0]
        max_depth = 1

        # Budget time per move: use up to 10% of remaining time, but at least 0.5s
        # With 360s total and ~40 moves, this allows ~9s early and scales down
        remaining = time_left()
        time_budget = max(0.5, min(remaining * 0.10, 15.0))  # Cap at 15s per move
        stop_time = remaining - time_budget

        # Iterative deepening to handle time constraints
        while time_left() > stop_time and max_depth < 20:
            try:
                move, score = self.minimax_root(board, max_depth, time_left, stop_time)
                if move is not None:
                    best_move = move
                max_depth += 1
            except TimeoutError:
                break

        return best_move

    def minimax_root(self, board: board.Board, depth: int, time_left: Callable, stop_time: float):
        """
        Root call for minimax with alpha-beta pruning.
        Returns the best move and its score.
        Uses negamax style: after reversing perspective, we negate the returned score.
        """
        moves = board.get_valid_moves()

        best_move = None
        best_score = float('-inf')
        alpha = float('-inf')
        beta = float('inf')

        # Order moves by quick heuristic (explore promising moves first)
        moves = self.order_moves(board, moves)

        for move in moves:
            if time_left() < stop_time:
                raise TimeoutError()

            # Apply move, recurse, then undo (avoids expensive deepcopy)
            direction, move_type = move
            new_board = board.forecast_move(direction, move_type)

            if new_board is None:
                continue

            new_board.reverse_perspective()

            # Negamax: negate the score since we reversed perspective
            score = -self.negamax(new_board, depth - 1, -beta, -alpha, time_left, stop_time)

            if score > best_score:
                best_score = score
                best_move = move

            alpha = max(alpha, best_score)

        return best_move, best_score

    def negamax(self, board: board.Board, depth: int, alpha: float, beta: float,
                time_left: Callable, stop_time: float):
        """
        Negamax algorithm with alpha-beta pruning.
        Always maximizes from the current player's perspective.
        After reversing perspective, the returned score is negated by the caller.
        """
        if time_left() < stop_time:
            raise TimeoutError()

        # Terminal conditions
        if depth == 0 or self.is_terminal(board):
            return self.evaluate(board)

        moves = board.get_valid_moves()
        
        max_eval = float('-inf')
        for move in moves:
            direction, move_type = move
            new_board = board.forecast_move(direction, move_type)

            if new_board is None:
                continue

            new_board.reverse_perspective()
            # Negamax: negate score and swap alpha/beta
            eval_score = -self.negamax(new_board, depth - 1, -beta, -alpha, time_left, stop_time)
            max_eval = max(max_eval, eval_score)
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break
        return max_eval

    def order_moves(self, board: board.Board, moves):
        """
        Order moves to improve alpha-beta pruning efficiency.
        Prioritize egg-laying moves first, then turd moves, then plain moves.
        """
        from game.enums import MoveType
        
        def move_score(move):
            direction, move_type = move
            score = 0
            # Strongly prefer egg-laying moves
            if move_type == MoveType.EGG:
                score += 100
            # Prefer turd moves over plain moves
            elif move_type == MoveType.TURD:
                score += 50
            return score

        return sorted(moves, key=move_score, reverse=True)

    def is_terminal(self, board: board.Board):
        """
        Check if the board state is terminal (game over).
        """
        try:
            if hasattr(board, 'is_game_over'):
                return board.is_game_over()
            elif hasattr(board, 'game_over'):
                return board.game_over()
            else:
                # Check if there are valid moves
                moves = board.get_valid_moves()
                return len(moves) == 0
        except:
            return False

    def evaluate(self, board: board.Board):
        """
        Evaluation function to score the board state.
        Higher scores are better for the player.
        """
        score = 0

        # Primary: egg differential (player eggs - enemy eggs)
        # Eggs are the main win condition, so weight them heavily
        player_eggs = len(board.eggs_player)
        enemy_eggs = len(board.eggs_enemy)
        score += (player_eggs - enemy_eggs) * 100

        # Value turds: more player turds on board is good (blocks opponent)
        player_turds = len(board.turds_player)
        enemy_turds = len(board.turds_enemy)
        score += (player_turds - enemy_turds) * 3

        location = board.chicken_player.get_location()
        parity = board.chicken_player.even_chicken
        
        # Bonus for being on a square where we can lay an egg
        if (location[0] + location[1]) % 2 == parity:
            score += 5  # Can lay egg here

        # Corner proximity bonus - reward being closer to any corner
        corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
        min_corner_distance = min(
            abs(location[0] - cx) + abs(location[1] - cy) 
            for cx, cy in corners
        )
        # Give up to 7 points for being close to a corner (reduced from 14)
        score += (14 - min_corner_distance) // 2
        
        # Extra bonus for actually being on a corner with egg-laying potential
        if location in corners and (location[0] + location[1]) % 2 == parity:
            score += 20  # Corner egg potential

        # Exploration bonus - reward being far from your own eggs
        # This encourages spreading eggs across the board
        if board.eggs_player:
            min_egg_distance = min(
                abs(location[0] - ex) + abs(location[1] - ey)
                for ex, ey in board.eggs_player
            )
            # Reward being far from existing eggs (up to 10 points)
            score += min(min_egg_distance, 10)

        # Mobility: having more valid moves is good
        moves = board.get_valid_moves()
        score += len(moves)

        return score