from collections.abc import Callable
from typing import List, Set, Tuple
import numpy as np
from game import *

"""
Melvin is the dumbest agent of all. He randomly selects a move from the list of valid moves.
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
        self.turn_count += 1
        
        print(f"Turn {self.turn_count} at {location}, time: {time_left():.1f}s")
        
        moves = board.get_valid_moves()
        
        # If only one move available, return it immediately (no need to search)
        if len(moves) == 1:
            return moves[0]
        
        # Adaptive time management: allocate time based on game phase and remaining time
        remaining = time_left()
        if remaining > 300:  # Early game (>5 min left)
            time_budget = min(20.0, remaining * 0.08)  # Use up to 8% per move, max 20s
        elif remaining > 150:  # Mid game
            time_budget = min(15.0, remaining * 0.10)  # Use up to 10%, max 15s
        else:  # Late game
            time_budget = min(10.0, remaining * 0.12)  # Use up to 12%, max 10s
        
        stop_time = max(0.3, remaining - time_budget)
        
        # Use iterative deepening to maximize depth within time budget
        best_move = moves[0]
        max_depth = 30  # Much higher ceiling than Gojo's 20
        
        try:
            for depth in range(1, max_depth + 1):
                if time_left() <= stop_time:
                    break
                
                # Search at current depth with negamax
                move, score = self.negamax_root(board, depth, time_left, stop_time)
                if move is not None:
                    best_move = move
                    print(f"Depth {depth}: score={score:.1f}")
                
        except Exception as e:
            print(f"Search interrupted: {e}")
        
        print(f"Playing {best_move}")
        return best_move

    def negamax_root(self, board: board.Board, depth: int, time_left: Callable, stop_time: float):
        """
        Root negamax search with advanced move ordering for better pruning.
        
        Args:
            board: Current game state
            depth: Maximum depth to search
            time_left: Function to check remaining time
            stop_time: Time threshold to stop search
            
        Returns:
            Tuple of (best_move, best_score)
        """
        moves = board.get_valid_moves()
        
        # Advanced move ordering: prioritize moves more intelligently
        moves = self.order_moves_advanced(board, moves)
        
        best_move = None
        best_score = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        
        for move in moves:
            if time_left() <= stop_time:
                raise TimeoutError()
            
            direction, move_type = move
            next_board = board.forecast_move(direction, move_type)
            
            if next_board is None:
                continue
            
            # Reverse perspective for opponent's turn
            next_board.reverse_perspective()
            
            # Negamax: negate score since we reversed perspective
            score = -self.negamax(next_board, depth - 1, -beta, -alpha, time_left, stop_time)
            
            if score > best_score:
                best_score = score
                best_move = move
            
            alpha = max(alpha, best_score)
        
        return best_move, best_score

    def negamax(self, board: board.Board, depth: int, alpha: float, beta: float,
                time_left: Callable, stop_time: float):
        """
        Negamax algorithm with alpha-beta pruning (cleaner than separate min/max).
        
        Args:
            board: Current game state
            depth: Remaining depth to search
            alpha: Lower bound
            beta: Upper bound
            time_left: Function to check remaining time
            stop_time: Time threshold to stop
            
        Returns:
            Evaluation score for this position
        """
        if time_left() <= stop_time:
            raise TimeoutError()
        
        # Terminal conditions
        if depth == 0:
            return self.evaluate_board(board)
        
        if board.is_game_over():
            return self.evaluate_terminal(board)
        
        moves = board.get_valid_moves()
        
        if len(moves) == 0:
            return self.evaluate_terminal(board)
        
        # Order moves for better pruning
        moves = self.order_moves_advanced(board, moves)
        
        max_eval = float('-inf')
        
        for move in moves:
            direction, move_type = move
            next_board = board.forecast_move(direction, move_type)
            
            if next_board is None:
                continue
            
            next_board.reverse_perspective()
            
            # Negamax: negate and swap alpha/beta
            eval_score = -self.negamax(next_board, depth - 1, -beta, -alpha, time_left, stop_time)
            max_eval = max(max_eval, eval_score)
            alpha = max(alpha, eval_score)
            
            if beta <= alpha:
                break  # Beta cutoff
        
        return max_eval

    def order_moves_advanced(self, board: board.Board, moves):
        """
        Advanced move ordering for superior alpha-beta pruning.
        Orders by: EGG moves > TURD moves > moves toward corners > other moves.
        
        Args:
            board: Current board state
            moves: List of valid moves
            
        Returns:
            Ordered list of moves
        """
        from game.enums import MoveType
        
        def move_priority(move):
            direction, move_type = move
            score = 0
            
            # Highest priority: egg-laying moves (primary win condition)
            if move_type == MoveType.EGG:
                score += 1000
                # Extra bonus for corner eggs
                next_board = board.forecast_move(direction, move_type)
                if next_board:
                    loc = next_board.chicken_enemy.get_location()  # After reverse, we're enemy
                    if loc in [(0,0), (0,7), (7,0), (7,7)]:
                        score += 500
            
            # Second priority: turd moves (blocking)
            elif move_type == MoveType.TURD:
                score += 200
            
            # Tertiary: plain moves toward corners with parity
            else:
                next_board = board.forecast_move(direction, move_type)
                if next_board:
                    next_board.reverse_perspective()
                    loc = next_board.chicken_player.get_location()
                    # Reward moving toward corners where we can lay eggs
                    if next_board.can_lay_egg_at_loc((0,0)):
                        score += max(0, 14 - (abs(loc[0]) + abs(loc[1])))
                    if next_board.can_lay_egg_at_loc((7,7)):
                        score += max(0, 14 - (abs(loc[0]-7) + abs(loc[1]-7)))
            
            return score
        
        return sorted(moves, key=move_priority, reverse=True)
    
    def evaluate_terminal(self, board: board.Board):
        """
        Evaluate a terminal game state (game over).
        
        Args:
            board: Terminal game state (from current player's perspective)
            
        Returns:
            Large positive score if current player won, negative if lost
        """
        winner = board.get_winner()
        
        # Board is from current player's perspective
        if winner == enums.Result.PLAYER:
            return 10000.0
        elif winner == enums.Result.ENEMY:
            return -10000.0
        else:
            return 0.0

    def evaluate_board(self, board: board.Board):
        """
        Superior evaluation function - beats Gojo through better feature weights.
        
        Args:
            board: Game state to evaluate (from current player's perspective)
            
        Returns:
            Positive score if favorable, negative if unfavorable
        """
        score = 0.0
        
        player_chicken = board.chicken_player
        enemy_chicken = board.chicken_enemy
        player_loc = player_chicken.get_location()
        enemy_loc = enemy_chicken.get_location()
        
        # 1. EGG DIFFERENTIAL - Most critical (weight: 150, higher than Gojo's 100)
        player_eggs = player_chicken.get_eggs_laid()
        enemy_eggs = enemy_chicken.get_eggs_laid()
        score += 150.0 * (player_eggs - enemy_eggs)
        
        # 2. PARITY BONUS - Reward being on egg-laying square (weight: 8, better than Gojo's 5)
        if board.can_lay_egg():
            score += 8.0
        
        # 3. CORNER STRATEGY - Multi-tier corner evaluation
        corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
        
        # 3a. Corner proximity with parity awareness
        for corner in corners:
            if board.can_lay_egg_at_loc(corner):
                player_dist = abs(player_loc[0] - corner[0]) + abs(player_loc[1] - corner[1])
                enemy_dist = abs(enemy_loc[0] - corner[0]) + abs(enemy_loc[1] - corner[1])
                # Being closer to egg-laying corners is worth 2 points per step
                score += 2.0 * (enemy_dist - player_dist)
                
                # Big bonus for actually being on a corner
                if player_loc == corner:
                    score += 25.0
        
        # 4. MOBILITY - More moves = better position (weight: 3)
        player_moves = len(board.get_valid_moves(enemy=False))
        enemy_moves = len(board.get_valid_moves(enemy=True))
        score += 3.0 * (player_moves - enemy_moves)
        
        # Penalty for having very low mobility (danger of being blocked)
        if player_moves <= 1:
            score -= 20.0
        if enemy_moves <= 1:
            score += 20.0
        
        # 5. TURD RESOURCES - Having turds for blocking (weight: 7, higher than Gojo's 3)
        player_turds = player_chicken.get_turds_left()
        enemy_turds = enemy_chicken.get_turds_left()
        score += 7.0 * (player_turds - enemy_turds)
        
        # 6. BOARD CONTROL - Egg spread (better than Gojo's exploration)
        # Reward having eggs in multiple quadrants
        if hasattr(board, 'eggs_player') and len(board.eggs_player) > 0:
            quadrants = set()
            for egg_pos in board.eggs_player:
                quad = (0 if egg_pos[0] < 4 else 1, 0 if egg_pos[1] < 4 else 1)
                quadrants.add(quad)
            score += 5.0 * len(quadrants)  # Reward board control
        
        # 7. CENTER CONTROL - Being in center can be advantageous early game
        center_dist = abs(player_loc[0] - 3.5) + abs(player_loc[1] - 3.5)
        enemy_center_dist = abs(enemy_loc[0] - 3.5) + abs(enemy_loc[1] - 3.5)
        # Small bonus for center control (helps with flexibility)
        score += 0.5 * (enemy_center_dist - center_dist)
        
        return score