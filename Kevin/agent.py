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
        
        print(f"At {location}, time: {time_left():.1f}s")
        
        moves = board.get_valid_moves()
        
        # If only one move available, return it immediately
        if len(moves) == 1:
            return moves[0]
        
        # TERRITORIAL STRATEGY: Try to find a strategic territorial move first
        territorial_move = self.get_territorial_move(board, moves)
        if territorial_move:
            print(f"Playing territorial move: {territorial_move}")
            return territorial_move
        
        # FALLBACK TO MINIMAX: Use tactical search if no clear territorial move
        print("Using minimax search...")
        
        # Adaptive time management
        remaining = time_left()
        if remaining > 300:
            time_budget = min(15.0, remaining * 0.06)  # Reduced from 8% for faster moves
        elif remaining > 150:
            time_budget = min(10.0, remaining * 0.08)
        else:
            time_budget = min(8.0, remaining * 0.10)
        
        stop_time = max(0.3, remaining - time_budget)
        
        # Iterative deepening minimax
        best_move = moves[0]
        max_depth = 25  # Reduced from 30 since territorial handles strategy
        
        try:
            for depth in range(1, max_depth + 1):
                if time_left() <= stop_time:
                    break
                
                move, score = self.negamax_root(board, depth, time_left, stop_time)
                if move is not None:
                    best_move = move
                    print(f"Depth {depth}: score={score:.1f}")
                
        except Exception as e:
            print(f"Search interrupted: {e}")
        
        print(f"Playing minimax move: {best_move}")
        return best_move
    
    def get_territorial_move(self, board: board.Board, moves):
        """
        Determine if there's a clear territorial move to make.
        Priority:
        1. Lay eggs on main diagonals (backslash or /)
        2. Place turds to mark territory boundaries
        3. Fill current territory quadrant
        4. Move toward next strategic position
        
        Returns:
            Move tuple if territorial move found, None otherwise
        """
        from game.enums import MoveType, Direction
        
        location = board.chicken_player.get_location()
        
        # Initialize instance variables if not present
        if not hasattr(self, 'diagonal_type'):
            self.diagonal_type = None
        if not hasattr(self, 'diagonal_positions'):
            self.diagonal_positions = set()
        if not hasattr(self, 'center_turds_placed'):
            self.center_turds_placed = False
        if not hasattr(self, 'current_territory'):
            self.current_territory = None
        
        # Determine diagonal type on first move if not set
        if self.diagonal_type is None:
            self.diagonal_type = self.determine_diagonal(location)
            print(f"Using diagonal type: {self.diagonal_type}")
        
        # PHASE 1: Diagonal egg laying
        diagonal_move = self.try_diagonal_egg(board, moves, location)
        if diagonal_move:
            return diagonal_move
        
        # PHASE 2: Territory boundary turds (center blocking)
        if not self.center_turds_placed and board.chicken_player.get_turds_left() > 0:
            boundary_move = self.try_boundary_turd(board, moves, location)
            if boundary_move:
                return boundary_move
        
        # PHASE 3: Fill territory systematically
        territory_fill_move = self.try_territory_fill(board, moves, location)
        if territory_fill_move:
            return territory_fill_move
        
        # No clear territorial move - use minimax
        return None
    
    def determine_diagonal(self, start_loc):
        r"""
        Determine which diagonal (\ or /) based on starting position.
        \ diagonal: top-left to bottom-right (row == col)
        / diagonal: top-right to bottom-left (row + col == 7)
        """
        row, col = start_loc
        
        # Check if we're closer to \ diagonal or / diagonal
        dist_to_backslash = abs(row - col)
        dist_to_slash = abs(row + col - 7)
        
        if dist_to_backslash <= dist_to_slash:
            return "\\"  # Top-left to bottom-right
        else:
            return "/"  # Top-right to bottom-left
    
    def try_diagonal_egg(self, board: board.Board, moves, location):
        r"""
        Try to lay an egg on the main diagonal.
        """
        from game.enums import MoveType
        
        # Check if we can lay egg at current location on diagonal
        if board.can_lay_egg():
            row, col = location
            
            # Check if current position is on our diagonal
            if self.diagonal_type == "\\" and row == col:
                # On \ diagonal
                for move in moves:
                    if move[1] == MoveType.EGG:
                        self.diagonal_positions.add(location)
                        print(f"Laying diagonal egg at {location}")
                        return move
            
            elif self.diagonal_type == "/" and row + col == 7:
                # On / diagonal
                for move in moves:
                    if move[1] == MoveType.EGG:
                        self.diagonal_positions.add(location)
                        print(f"Laying diagonal egg at {location}")
                        return move
        
        # Try to move toward nearest diagonal position where we can lay egg
        target = self.get_nearest_diagonal_target(board, location)
        if target:
            move = self.find_move_toward(board, moves, location, target)
            if move:
                print(f"Moving toward diagonal position {target}")
                return move
        
        return None
    
    def get_nearest_diagonal_target(self, board: board.Board, location):
        r"""
        Find nearest position on diagonal where we can lay an egg.
        """
        row, col = location
        candidates = []
        
        if self.diagonal_type == "\\":
            # \ diagonal positions: (0,0), (1,1), (2,2), ..., (7,7)
            for i in range(8):
                if (i, i) not in self.diagonal_positions:
                    # Check if we can lay egg there (parity check)
                    if self.can_eventually_lay_egg_at(board, (i, i)):
                        dist = abs(row - i) + abs(col - i)
                        candidates.append(((i, i), dist))
        
        else:  # "/" diagonal
            # / diagonal positions: (0,7), (1,6), (2,5), ..., (7,0)
            for i in range(8):
                pos = (i, 7 - i)
                if pos not in self.diagonal_positions:
                    if self.can_eventually_lay_egg_at(board, pos):
                        dist = abs(row - i) + abs(col - (7 - i))
                        candidates.append((pos, dist))
        
        if candidates:
            # Return nearest position
            candidates.sort(key=lambda x: x[1])
            return candidates[0][0]
        
        return None
    
    def can_eventually_lay_egg_at(self, board: board.Board, target):
        """
        Check if we can eventually lay an egg at target position (parity check).
        """
        # Simple parity check: can we reach it on an even move?
        current_loc = board.chicken_player.get_location()
        manhattan_dist = abs(current_loc[0] - target[0]) + abs(current_loc[1] - target[1])
        
        # We can lay egg if we reach on even parity
        # (assuming we currently have even parity if we can lay egg now)
        current_parity = board.can_lay_egg()
        target_parity = (manhattan_dist % 2) == 0
        
        return current_parity == target_parity
    
    def try_boundary_turd(self, board: board.Board, moves, location):
        """
        Place turds on territory boundaries (center area) to block opponent.
        """
        from game.enums import MoveType
        
        # Define boundary positions (center area to mark territory)
        if self.diagonal_type == "\\":
            # For \ diagonal, place turds on the / diagonal to mark boundary
            boundary_positions = [(2, 5), (3, 4), (4, 3), (5, 2)]
        else:
            # For / diagonal, place turds on the \ diagonal to mark boundary
            boundary_positions = [(2, 2), (3, 3), (4, 4), (5, 5)]
        
        # Check if we're on a boundary position and can place turd
        if location in boundary_positions:
            for move in moves:
                if move[1] == MoveType.TURD:
                    print(f"Placing boundary turd at {location}")
                    # Check if we've placed enough turds
                    if len([p for p in boundary_positions if self.is_blocked(board, p)]) >= 2:
                        self.center_turds_placed = True
                    return move
        
        # Move toward a boundary position
        for boundary_pos in boundary_positions:
            if not self.is_blocked(board, boundary_pos):
                move = self.find_move_toward(board, moves, location, boundary_pos)
                if move:
                    print(f"Moving toward boundary position {boundary_pos}")
                    return move
        
        self.center_turds_placed = True  # All boundaries handled
        return None
    
    def is_blocked(self, board: board.Board, position):
        """
        Check if a position is blocked (has egg or turd).
        """
        # Check if position has any obstacle
        if hasattr(board, 'eggs_player') and position in board.eggs_player:
            return True
        if hasattr(board, 'eggs_enemy') and position in board.eggs_enemy:
            return True
        if hasattr(board, 'turds_player') and position in board.turds_player:
            return True
        if hasattr(board, 'turds_enemy') and position in board.turds_enemy:
            return True
        return False
    
    def try_territory_fill(self, board: board.Board, moves, location):
        """
        Systematically fill territory quadrant with eggs.
        """
        from game.enums import MoveType
        
        # Determine which territory to fill
        if self.current_territory is None:
            self.current_territory = self.choose_territory(location)
            print(f"Claiming territory: {self.current_territory}")
        
        # Try to lay egg if we can
        if board.can_lay_egg() and self.is_in_territory(location, self.current_territory):
            for move in moves:
                if move[1] == MoveType.EGG:
                    print(f"Filling territory at {location}")
                    return move
        
        # Move toward next unfilled spot in territory
        next_target = self.get_next_territory_target(board, self.current_territory)
        if next_target:
            move = self.find_move_toward(board, moves, location, next_target)
            if move:
                print(f"Moving to fill territory position {next_target}")
                return move
        else:
            # Territory filled, move to next one
            self.current_territory = self.get_next_territory(self.current_territory)
            if self.current_territory:
                print(f"Moving to next territory: {self.current_territory}")
                return self.try_territory_fill(board, moves, location)
        
        return None
    
    def choose_territory(self, location):
        r"""
        Choose initial territory based on diagonal type.
        \ diagonal -> top-right quadrant first
        / diagonal -> top-left quadrant first
        """
        if self.diagonal_type == "\\":
            return "top-right"  # Quadrant: rows 0-3, cols 4-7
        else:
            return "top-left"   # Quadrant: rows 0-3, cols 0-3
    
    def is_in_territory(self, location, territory):
        """Check if location is in the specified territory."""
        row, col = location
        
        if territory == "top-left":
            return row <= 3 and col <= 3
        elif territory == "top-right":
            return row <= 3 and col >= 4
        elif territory == "bottom-left":
            return row >= 4 and col <= 3
        elif territory == "bottom-right":
            return row >= 4 and col >= 4
        
        return False
    
    def get_next_territory_target(self, board: board.Board, territory):
        """
        Get next position in territory to fill with egg.
        """
        # Define positions in territory (in filling order)
        positions = []
        
        if territory == "top-left":
            positions = [(r, c) for r in range(4) for c in range(4)]
        elif territory == "top-right":
            positions = [(r, c) for r in range(4) for c in range(4, 8)]
        elif territory == "bottom-left":
            positions = [(r, c) for r in range(4, 8) for c in range(4)]
        elif territory == "bottom-right":
            positions = [(r, c) for r in range(4, 8) for c in range(4, 8)]
        
        # Find first unfilled position where we can lay egg
        for pos in positions:
            if not self.is_blocked(board, pos) and self.can_eventually_lay_egg_at(board, pos):
                return pos
        
        return None
    
    def get_next_territory(self, current):
        """Get next territory to claim."""
        order = ["top-left", "top-right", "bottom-left", "bottom-right"]
        
        try:
            idx = order.index(current)
            if idx < len(order) - 1:
                return order[idx + 1]
        except ValueError:
            pass
        
        return None
    
    def find_move_toward(self, board: board.Board, moves, current, target):
        """
        Find a move that gets us closer to target position.
        """
        from game.enums import MoveType, Direction
        
        best_move = None
        best_dist = float('inf')
        
        curr_dist = abs(current[0] - target[0]) + abs(current[1] - target[1])
        
        for move in moves:
            direction, move_type = move
            
            # Only consider plain moves (not egg/turd) for navigation
            if move_type != MoveType.PLAIN:
                continue
            
            # Simulate move to see where it takes us
            next_board = board.forecast_move(direction, move_type)
            if next_board:
                next_loc = next_board.chicken_player.get_location()
                next_dist = abs(next_loc[0] - target[0]) + abs(next_loc[1] - target[1])
                
                # Pick move that gets us closer
                if next_dist < best_dist:
                    best_dist = next_dist
                    best_move = move
        
        # Only return move if it actually gets us closer
        if best_dist < curr_dist:
            return best_move
        
        return None

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
        Territorial-focused evaluation function.
        
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
        
        # 1. EGG DIFFERENTIAL - Most critical (weight: 200)
        player_eggs = player_chicken.get_eggs_laid()
        enemy_eggs = enemy_chicken.get_eggs_laid()
        score += 200.0 * (player_eggs - enemy_eggs)
        
        # 2. DIAGONAL CONTROL - Reward eggs on diagonals
        if hasattr(board, 'eggs_player'):
            diagonal_eggs = 0
            for egg_pos in board.eggs_player:
                row, col = egg_pos
                # Check if on \ diagonal
                if row == col:
                    diagonal_eggs += 1
                # Check if on / diagonal
                if row + col == 7:
                    diagonal_eggs += 1
            score += 15.0 * diagonal_eggs
        
        # 3. TERRITORIAL DENSITY - Reward eggs in claimed territory
        if hasattr(board, 'eggs_player') and len(board.eggs_player) > 0:
            # Count eggs in each quadrant
            quadrant_counts = {}
            for egg_pos in board.eggs_player:
                row, col = egg_pos
                if row <= 3 and col <= 3:
                    quad = "top-left"
                elif row <= 3 and col >= 4:
                    quad = "top-right"
                elif row >= 4 and col <= 3:
                    quad = "bottom-left"
                else:
                    quad = "bottom-right"
                
                quadrant_counts[quad] = quadrant_counts.get(quad, 0) + 1
            
            # Reward having a densely filled quadrant (territorial control)
            if quadrant_counts:
                max_density = max(quadrant_counts.values())
                score += 10.0 * max_density
                # Also reward number of quadrants controlled
                score += 8.0 * len(quadrant_counts)
        
        # 4. PARITY BONUS - Reward being on egg-laying square
        if board.can_lay_egg():
            score += 12.0
        
        # 5. CORNER STRATEGY - Bonus for corner positions
        corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
        for corner in corners:
            if board.can_lay_egg_at_loc(corner):
                player_dist = abs(player_loc[0] - corner[0]) + abs(player_loc[1] - corner[1])
                enemy_dist = abs(enemy_loc[0] - corner[0]) + abs(enemy_loc[1] - corner[1])
                score += 1.5 * (enemy_dist - player_dist)
                
                if player_loc == corner:
                    score += 20.0
        
        # 6. MOBILITY - More moves = better position
        player_moves = len(board.get_valid_moves(enemy=False))
        enemy_moves = len(board.get_valid_moves(enemy=True))
        score += 4.0 * (player_moves - enemy_moves)
        
        # Heavy penalty for being trapped
        if player_moves <= 1:
            score -= 30.0
        if enemy_moves <= 1:
            score += 30.0
        
        # 7. TURD RESOURCES - Strategic blocking capability
        player_turds = player_chicken.get_turds_left()
        enemy_turds = enemy_chicken.get_turds_left()
        score += 6.0 * (player_turds - enemy_turds)
        
        # 8. BOUNDARY CONTROL - Reward turds in center (boundary marking)
        if hasattr(board, 'turds_player'):
            center_turds = 0
            for turd_pos in board.turds_player:
                row, col = turd_pos
                # Check if turd is in center area (2-5, 2-5)
                if 2 <= row <= 5 and 2 <= col <= 5:
                    center_turds += 1
            score += 8.0 * center_turds
        
        return score