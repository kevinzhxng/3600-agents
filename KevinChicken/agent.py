from collections.abc import Callable
from typing import List, Set, Tuple
import numpy as np
import time, random
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
        
        # If only one move available, return it immediately (no need to search)
        if len(moves) == 1:
            return moves[0]

        # Endgame hybrid trigger: switch to MCTS when few turns left or tight score.
        if self.should_use_mcts(board, moves, time_left()):
            # Allocate small slice of time for MCTS (max 1s or 5% of remaining time)
            mcts_time = min(1.0, max(0.25, time_left() * 0.05))
            try:
                mcts_move = self.mcts_endgame(board, time_left, mcts_time)
                if mcts_move is not None:
                    print(f"MCTS chose {mcts_move} in {mcts_time:.2f}s")
                    return mcts_move
            except Exception as e:
                print(f"MCTS fallback due to error: {e}")
        
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

    # ----------------- HYBRID MCTS HELPERS -----------------
    class _Node:
        __slots__ = ("board", "move", "parent", "children", "wins", "visits", "unexpanded")
        def __init__(self, board, move=None, parent=None, children=None):
            self.board = board
            self.move = move
            self.parent = parent
            self.children = [] if children is None else children
            self.wins = 0.0
            self.visits = 0
            # Track moves not yet expanded (ordered for policy guidance)
            self.unexpanded = []

    def should_use_mcts(self, board: board.Board, moves, remaining_time: float) -> bool:
        """Decide if we switch to MCTS for endgame.
        Heuristics:
        - Few moves (<=4) OR many eggs laid (>=30 total) OR time is low (<120s)
        - Egg gap small (<=3)
        """
        player_eggs = board.chicken_player.get_eggs_laid()
        enemy_eggs = board.chicken_enemy.get_eggs_laid()
        egg_gap = abs(player_eggs - enemy_eggs)
        total_eggs = player_eggs + enemy_eggs
        return (len(moves) <= 4 or total_eggs >= 30 or remaining_time < 120) and egg_gap <= 3

    def mcts_endgame(self, board: board.Board, time_left: Callable, max_time: float):
        """Run a lightweight MCTS from the current position for up to max_time seconds.
        Returns best move found or None.
        """
        start = time.time()
        end_time = start + max_time

        # Root node setup
        root = self._Node(board)
        moves = board.get_valid_moves()
        ordered = self.order_moves_advanced(board, moves)
        root.unexpanded = list(ordered)

        if not ordered:
            return None

        # Pre-expand one child to seed tree
        self._expand_node(root)

        # Main loop
        iterations = 0
        while time.time() < end_time:
            iterations += 1
            path = [root]
            node = root
            # Selection: descend using UCT while node fully expanded and has children
            while node.unexpanded == [] and node.children:
                node = self._select_child(node)
                path.append(node)

            # Expansion (if possible)
            if node.unexpanded and time.time() < end_time:
                node = self._expand_node(node)
                path.append(node)

            # Simulation / rollout
            result = self._rollout(node.board, end_time)

            # Backpropagation
            for n in path:
                n.visits += 1
                n.wins += result

        # Choose move with highest visits (robust) from root
        best = None
        best_visits = -1
        for child in root.children:
            if child.visits > best_visits:
                best_visits = child.visits
                best = child.move

        return best

    def _expand_node(self, node: '_Node'):
        move = node.unexpanded.pop(0)
        direction, move_type = move
        new_board = node.board.forecast_move(direction, move_type)
        if new_board is None:
            # Skip invalid forecast and try next if exists
            return node if not node.unexpanded else self._expand_node(node)
        new_board.reverse_perspective()
        child = self._Node(new_board, move=move, parent=node)
        # Order child move list for later expansion
        child_moves = new_board.get_valid_moves()
        child.unexpanded = list(self.order_moves_advanced(new_board, child_moves))
        node.children.append(child)
        return child

    def _select_child(self, node: '_Node'):
        # UCT selection
        parent_visits = max(1, node.visits)
        C = 0.5  # exploration constant tuned low for deterministic game
        best_score = -1e9
        best_child = None
        for c in node.children:
            if c.visits == 0:
                return c
            mean = c.wins / c.visits
            uct = mean + C * np.sqrt(np.log(parent_visits) / c.visits)
            if uct > best_score:
                best_score = uct
                best_child = c
        return best_child

    def _rollout(self, board: board.Board, end_time: float, depth_limit: int = 12):
        depth = 0
        while depth < depth_limit and not board.is_game_over() and time.time() < end_time:
            moves = board.get_valid_moves()
            if not moves:
                break
            ordered = self.order_moves_advanced(board, moves)
            # Bias: choose among top 3 ordered moves randomly
            slice_moves = ordered[:3] if len(ordered) >= 3 else ordered
            direction, move_type = random.choice(slice_moves)
            new_board = board.forecast_move(direction, move_type)
            if new_board is None:
                break
            new_board.reverse_perspective()
            board = new_board
            depth += 1

        # Terminal check
        if board.is_game_over():
            winner = board.get_winner()
            if winner == enums.Result.PLAYER:
                return 1.0
            elif winner == enums.Result.ENEMY:
                return 0.0
            else:
                return 0.5

        # Fallback heuristic scaling to [0,1]
        heuristic = self.evaluate_board(board)
        # Compress large range using tanh
        return 0.5 + 0.5 * np.tanh(heuristic / 2000.0)

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