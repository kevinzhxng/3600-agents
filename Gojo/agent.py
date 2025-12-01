from collections.abc import Callable
from time import sleep
from typing import List, Set, Tuple
import numpy as np
from game import *
from game.enums import Direction, MoveType, loc_after_direction
from game.game_map import prob_hear, prob_feel

"""
Minimax agent with alpha-beta pruning and Bayesian trapdoor avoidance.
"""


class PlayerAgent:
    """
    /you may add functions, however, __init__ and play are the entry points for
    your program and should not be changed.
    """

    def __init__(self, board: board.Board, time_left: Callable):
        pass

    def _lazy_init(self):
        """
        Lazy initialization called on first play().
        """
        # Initialize Bayesian trapdoor probability grids
        # Trapdoors spawn in center area (rows/cols 2-5) with higher prob toward center
        # Trapdoor A is on even-parity squares, Trapdoor B on odd-parity
        self.trapdoor_prob = [np.zeros((8, 8)), np.zeros((8, 8))]
        
        # Set initial prior based on spawn distribution
        # From trapdoor_manager: cells [2:dim-2, 2:dim-2] have weight 1, [3:dim-3, 3:dim-3] have weight 2
        for x in range(8):
            for y in range(8):
                parity = (x + y) % 2
                # Only valid spawn area
                if 2 <= x <= 5 and 2 <= y <= 5:
                    # Inner area (3-4) has higher weight
                    if 3 <= x <= 4 and 3 <= y <= 4:
                        self.trapdoor_prob[parity][x, y] = 2.0
                    else:
                        self.trapdoor_prob[parity][x, y] = 1.0
        
        # Normalize each grid
        for i in range(2):
            total = np.sum(self.trapdoor_prob[i])
            if total > 0:
                self.trapdoor_prob[i] /= total
        
        # Track visit counts for each position to prevent oscillation
        # Key: (x, y), Value: number of times visited
        self.visit_counts = {}

    def update_trapdoor_beliefs(self, board: board.Board, sensor_data: List[Tuple[bool, bool]]):
        """
        Update trapdoor probability grids using Bayesian inference.
        sensor_data[i] = (did_hear, did_feel) for trapdoor i
        """
        location = board.chicken_player.get_location()
        
        for trap_idx in range(2):
            did_hear, did_feel = sensor_data[trap_idx]
            parity = trap_idx  # Trapdoor 0 is even parity, trapdoor 1 is odd parity
            
            # Calculate likelihood for each cell
            likelihood = np.zeros((8, 8))
            for x in range(8):
                for y in range(8):
                    # Only consider cells with matching parity
                    if (x + y) % 2 != parity:
                        continue
                    
                    # Calculate distance
                    delta_x = abs(x - location[0])
                    delta_y = abs(y - location[1])
                    
                    # Get probabilities of hearing/feeling at this distance
                    p_hear = prob_hear(delta_x, delta_y)
                    p_feel = prob_feel(delta_x, delta_y)
                    
                    # Calculate likelihood of observed sensor readings
                    if did_hear:
                        hear_likelihood = p_hear
                    else:
                        hear_likelihood = 1.0 - p_hear
                    
                    if did_feel:
                        feel_likelihood = p_feel
                    else:
                        feel_likelihood = 1.0 - p_feel
                    
                    likelihood[x, y] = hear_likelihood * feel_likelihood
            
            # Bayesian update: posterior = prior * likelihood
            self.trapdoor_prob[trap_idx] *= likelihood
            
            # Normalize
            total = np.sum(self.trapdoor_prob[trap_idx])
            if total > 0:
                self.trapdoor_prob[trap_idx] /= total

    def get_trapdoor_danger(self, x: int, y: int) -> float:
        """
        Get the combined probability that (x, y) contains a trapdoor.
        """
        return self.trapdoor_prob[0][x, y] + self.trapdoor_prob[1][x, y]

    def play(
        self,
        board: board.Board,
        sensor_data: List[Tuple[bool, bool]],
        time_left: Callable,
    ):
        # Lazy initialization on first call
        if not hasattr(self, 'trapdoor_prob'):
            self._lazy_init()
        
        location = board.chicken_player.get_location()
        print(f"I'm at {location}.")
        print(f"Trapdoor A: heard? {sensor_data[0][0]}, felt? {sensor_data[0][1]}")
        print(f"Trapdoor B: heard? {sensor_data[1][0]}, felt? {sensor_data[1][1]}")
        print(f"Starting to think with {time_left()} seconds left.")

        # Update trapdoor beliefs with new sensor data
        self.update_trapdoor_beliefs(board, sensor_data)
        
        # Debug: print high-probability trapdoor locations
        for trap_idx in range(2):
            max_prob = np.max(self.trapdoor_prob[trap_idx])
            if max_prob > 0.1:
                max_loc = np.unravel_index(np.argmax(self.trapdoor_prob[trap_idx]), (8, 8))
                print(f"Trapdoor {trap_idx} most likely at {max_loc} (prob={max_prob:.2f})")

        moves = board.get_valid_moves()

        # Use minimax with alpha-beta pruning to find the best move
        best_move = self.get_best_move(board, time_left)
        result = best_move if best_move is not None else moves[0]

        print(f"I have {time_left()} seconds left. Playing {result}.")
        
        # Update visit count for current position (before we move away)
        if location in self.visit_counts:
            self.visit_counts[location] += 1
        else:
            self.visit_counts[location] = 1
        
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
        
        # Filter out moves to high-danger trapdoor squares
        moves = self.filter_safe_moves(board, moves)
        
        location = board.chicken_player.get_location()

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
            
            # ANTI-OSCILLATION: Penalize moves that return to previously visited squares
            new_loc = loc_after_direction(location, direction)
            visit_count = self.visit_counts.get(new_loc, 0)
            if visit_count == 1:
                # First revisit: small penalty
                score -= 50
            elif visit_count >= 2:
                # Already revisited once: massive penalty to prevent further revisits
                score -= 10000

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
        Also penalize moves toward high-probability trapdoor locations.
        """
        location = board.chicken_player.get_location()
        
        def move_score(move):
            direction, move_type = move
            score = 0
            
            # Strongly prefer egg-laying moves
            if move_type == MoveType.EGG:
                score += 100
            # Prefer turd moves over plain moves
            elif move_type == MoveType.TURD:
                score += 50
            
            # Penalize moves toward dangerous squares
            new_loc = loc_after_direction(location, direction)
            if 0 <= new_loc[0] < 8 and 0 <= new_loc[1] < 8:
                danger = self.get_trapdoor_danger(new_loc[0], new_loc[1])
                # Heavy penalty for high-probability trapdoor squares
                score -= danger * 500
            
            return score

        return sorted(moves, key=move_score, reverse=True)

    def filter_safe_moves(self, board: board.Board, moves, danger_threshold: float = 0.3):
        """
        Filter out moves that would land on high-probability trapdoor squares.
        Only filters if there are safe alternatives available.
        
        Args:
            moves: List of valid moves
            danger_threshold: Filter moves to squares with danger >= this value
        
        Returns:
            Filtered list of moves, or original list if no safe moves exist
        """
        location = board.chicken_player.get_location()
        
        safe_moves = []
        filtered_moves = []  # Track what we filter out
        
        for move in moves:
            direction, move_type = move
            new_loc = loc_after_direction(location, direction)
            
            if 0 <= new_loc[0] < 8 and 0 <= new_loc[1] < 8:
                danger = self.get_trapdoor_danger(new_loc[0], new_loc[1])
                if danger < danger_threshold:
                    safe_moves.append(move)
                else:
                    filtered_moves.append((move, new_loc, danger))
            else:
                # Off-board moves won't be valid anyway, but include for safety
                safe_moves.append(move)
        
        # Print filtered moves
        for move, new_loc, danger in filtered_moves:
            print(f"Filtered move {move} -> {new_loc} (danger={danger:.2f})")
        
        # Only use filtered list if we have safe options
        # Otherwise, we have to pick the least dangerous option
        if safe_moves:
            return safe_moves
        else:
            # All moves are dangerous - return all and let evaluation pick least bad
            print("WARNING: All moves are dangerous, no safe alternatives!")
            return moves

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
        Includes trapdoor avoidance penalties.
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
        x, y = location
        
        # EGG-LAYING OPPORTUNITIES
        # Bonus for being on a square where we can lay an egg
        if (x + y) % 2 == parity:
            # Check if square is empty (can actually lay here)
            if location not in board.eggs_player and location not in board.turds_player:
                score += 25  # Can lay egg here right now
            else:
                score += 5  # Right parity but occupied
        
        # Count nearby empty egg-laying squares (squares we could reach and lay on)
        nearby_egg_opportunities = 0
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                
                # Skip out of bounds
                if not (0 <= nx < 8 and 0 <= ny < 8):
                    continue
                
                # Check if this is an egg-laying square for us
                if (nx + ny) % 2 == parity:
                    # Check if it's empty (no egg or turd already there)
                    if ((nx, ny) not in board.eggs_player and 
                        (nx, ny) not in board.eggs_enemy and
                        (nx, ny) not in board.turds_player and
                        (nx, ny) not in board.turds_enemy):
                        
                        distance = abs(dx) + abs(dy)
                        if distance == 1:
                            nearby_egg_opportunities += 8  # Adjacent: very valuable
                        elif distance == 2:
                            nearby_egg_opportunities += 4  # Close: valuable
                        else:
                            nearby_egg_opportunities += 2  # Nearby: somewhat valuable
        
        score += nearby_egg_opportunities

        # Corner proximity bonus - reward being closer to any corner
        corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
        min_corner_distance = min(
            abs(x - cx) + abs(y - cy) 
            for cx, cy in corners
        )
        # Max distance to a corner is 14 (from center to opposite corner)
        # Give up to 14 points for being close to a corner
        score += (14 - min_corner_distance)
        
        # Extra bonus for actually being on a corner with egg-laying potential
        if location in corners and (x + y) % 2 == parity:
            score += 30  # Corner egg potential (increased from 20)

        # Mobility: having more valid moves is good
        moves = board.get_valid_moves()
        score += len(moves)

        # FRONTIER BONUS: Reward being near empty squares where we can expand
        # This encourages the agent to take more space on the board
        # (Note: egg-laying opportunities already counted above, so only count non-egg squares here)
        frontier_score = 0
        
        # Check all adjacent and nearby squares (within 2 steps)
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                
                # Skip out of bounds
                if not (0 <= nx < 8 and 0 <= ny < 8):
                    continue
                
                # Skip egg-laying squares (already counted above)
                if (nx + ny) % 2 == parity:
                    continue
                
                # Check if this square is "empty" (no eggs or turds from either player)
                is_empty = (
                    (nx, ny) not in board.eggs_player and
                    (nx, ny) not in board.eggs_enemy and
                    (nx, ny) not in board.turds_player and
                    (nx, ny) not in board.turds_enemy
                )
                
                if is_empty:
                    # Weight by distance: adjacent squares worth more than 2-away
                    distance = abs(dx) + abs(dy)
                    if distance == 1:
                        frontier_score += 2  # Orthogonally adjacent
                    elif distance == 2:
                        frontier_score += 1  # Diagonal or 2 steps away
        
        score += frontier_score

        # TRAPDOOR AVOIDANCE
        # Penalize being on or near high-probability trapdoor squares
        x, y = location
        danger = self.get_trapdoor_danger(x, y)
        
        # Massive penalty for standing on likely trapdoor
        # At 89% confidence, this should be -8900 points - completely unacceptable
        # (4 eggs = 400 points, but we want to NEVER step on a high-confidence trapdoor)
        score -= danger * 10000
        
        # Smaller penalty for being adjacent to likely trapdoors
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < 8 and 0 <= ny < 8:
                    adjacent_danger = self.get_trapdoor_danger(nx, ny)
                    score -= adjacent_danger * 100  # Smaller penalty for nearby danger

        # ENEMY SIDE AVOIDANCE
        # Penalize being on the enemy's side of the board - that area is likely full of turds
        # which restrict movement and can get us blocked/trapped.
        # Penalty is minimal near the middle but ramps up sharply near the enemy's edge.
        my_spawn = board.chicken_player.get_spawn()
        
        # Determine which side we started on (x=0 means left side, x=7 means right side)
        if my_spawn[0] == 0:
            # We started on the left, enemy is on the right
            # x=0-3 is our side (safe), x=4-7 is enemy side (increasingly dangerous)
            enemy_side_depth = max(0, x - 3)  # 0 at x<=3, 1 at x=4, 2 at x=5, 3 at x=6, 4 at x=7
        else:
            # We started on the right, enemy is on the left
            # x=4-7 is our side (safe), x=0-3 is enemy side (increasingly dangerous)
            enemy_side_depth = max(0, 4 - x)  # 0 at x>=4, 1 at x=3, 2 at x=2, 3 at x=1, 4 at x=0
        
        # Exponential penalty: minimal at depth 0-1, ramps up sharply at depth 3-4
        # depth 0: 0, depth 1: 10, depth 2: 40, depth 3: 90, depth 4: 160
        enemy_side_penalty = (enemy_side_depth ** 2) * 10
        score -= enemy_side_penalty

        return score