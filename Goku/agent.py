from collections.abc import Callable
from time import sleep
from typing import List, Set, Tuple
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

    def _init_trapdoor_beliefs(self):
        """
        Lazy initialization of Bayesian trapdoor probability grids.
        Called on first play() if not already initialized.
        """
        # trapdoor_prob[trap_idx][x][y] as plain lists
        self.trapdoor_prob = [
            [[0.0 for _ in range(8)] for _ in range(8)],
            [[0.0 for _ in range(8)] for _ in range(8)],
        ]

        # Set initial prior based on spawn distribution
        for x in range(8):
            for y in range(8):
                parity = (x + y) % 2
                if 2 <= x <= 5 and 2 <= y <= 5:
                    if 3 <= x <= 4 and 3 <= y <= 4:
                        self.trapdoor_prob[parity][x][y] = 2.0
                    else:
                        self.trapdoor_prob[parity][x][y] = 1.0

        # Normalize each grid
        for t in range(2):
            total = 0.0
            for x in range(8):
                for y in range(8):
                    total += self.trapdoor_prob[t][x][y]
            if total > 0.0:
                for x in range(8):
                    for y in range(8):
                        self.trapdoor_prob[t][x][y] /= total

        # Track whether we've entered the enemy spawn danger zone
        self.near_enemy_spawn = False

        # Territorial strategy state
        self.diagonal_type = None
        self.territory_phase = "diagonal"
        self.diagonal_eggs_laid = set()
        self.diagonal_turds_placed = set()
        self.center_marks_placed = False
        self.corners_filled = set()
        self.my_parity = None
        self.last_diagonal_action = None

        # Stuck detection
        self.position_history = []
        self.stuck_counter = 0
        self.stuck_threshold = 10

        # Opening diagonal-ladder state
        self.diagonal_step_index = 0
        self.diagonal_base_dirs = None


    def update_trapdoor_beliefs(self, board: board.Board, sensor_data: List[Tuple[bool, bool]]):
        """
        Update trapdoor probability grids using Bayesian inference.
        sensor_data[i] = (did_hear, did_feel) for trapdoor i
        """
        location = board.chicken_player.get_location()

        for trap_idx in range(2):
            did_hear, did_feel = sensor_data[trap_idx]
            parity = trap_idx  # Trapdoor 0 is even parity, trapdoor 1 is odd parity

            # Build a fresh posterior grid
            new_grid = [[0.0 for _ in range(8)] for _ in range(8)]

            for x in range(8):
                for y in range(8):
                    # Only consider cells with matching parity
                    if (x + y) % 2 != parity:
                        continue

                    prior = self.trapdoor_prob[trap_idx][x][y]
                    if prior == 0.0:
                        continue

                    delta_x = abs(x - location[0])
                    delta_y = abs(y - location[1])

                    p_hear = prob_hear(delta_x, delta_y)
                    p_feel = prob_feel(delta_x, delta_y)

                    hear_likelihood = p_hear if did_hear else (1.0 - p_hear)
                    feel_likelihood = p_feel if did_feel else (1.0 - p_feel)

                    likelihood = hear_likelihood * feel_likelihood
                    new_grid[x][y] = prior * likelihood

            # Normalize new_grid
            total = 0.0
            for x in range(8):
                for y in range(8):
                    total += new_grid[x][y]

            if total > 0.0:
                for x in range(8):
                    for y in range(8):
                        new_grid[x][y] /= total
                self.trapdoor_prob[trap_idx] = new_grid
            else:
                # If everything went to zero (weird sensor combination),
                # keep the old belief to avoid NaNs/zeros everywhere.
                pass

    def get_trapdoor_danger(self, x: int, y: int) -> float:
        """
        Get the combined probability that (x, y) contains a trapdoor.
        """
        return self.trapdoor_prob[0][x][y] + self.trapdoor_prob[1][x][y]

    def play(
        self,
        board: board.Board,
        sensor_data: List[Tuple[bool, bool]],
        time_left: Callable,
    ):
        # Lazy initialization of trapdoor beliefs on first call
        if not hasattr(self, 'trapdoor_prob'):
            self._init_trapdoor_beliefs()
        
        location = board.chicken_player.get_location()
        print(f"I'm at {location}.")
        print(f"Trapdoor A: heard? {sensor_data[0][0]}, felt? {sensor_data[0][1]}")
        print(f"Trapdoor B: heard? {sensor_data[1][0]}, felt? {sensor_data[1][1]}")
        print(f"Starting to think with {time_left()} seconds left.")

        # Update trapdoor beliefs with new sensor data
        self.update_trapdoor_beliefs(board, sensor_data)
        
        # Initialize territorial strategy on first move
        if self.diagonal_type is None:
            self.initialize_territorial_strategy(board)
        
        # Check if we've moved within 5 squares of enemy spawn (activates avoidance)
        enemy_spawn = board.chicken_enemy.get_spawn()
        dist_to_enemy_spawn = abs(location[0] - enemy_spawn[0]) + abs(location[1] - enemy_spawn[1])
        if dist_to_enemy_spawn <= 5:
            if not self.near_enemy_spawn:
                print(f"Entered enemy spawn danger zone! Distance: {dist_to_enemy_spawn}")
            self.near_enemy_spawn = True
        
        for trap_idx in range(2):
            grid = self.trapdoor_prob[trap_idx]
            max_prob = 0.0
            max_loc = None
            for x in range(8):
                for y in range(8):
                    if grid[x][y] > max_prob:
                        max_prob = grid[x][y]
                        max_loc = (x, y)
            if max_prob > 0.1 and max_loc is not None:
                print(f"Trapdoor {trap_idx} most likely at {max_loc} (prob={max_prob:.2f})")

        moves = board.get_valid_moves()
        
        # Try territorial move FIRST (prioritize egg-laying)
        print(f"Territory phase: {self.territory_phase}")
        territorial_move = self.get_territorial_move(board, moves)
        if territorial_move:
            print(f"Playing territorial move: {territorial_move}")
            return territorial_move
        
        # Only try strategic turd placement if we can't lay an egg
        if board.chicken_player.get_turds_left() > 0:
            turd_move = self.try_strategic_turd(board, moves)
            if turd_move:
                print(f"Playing strategic turd: {turd_move}")
                return turd_move
        
        # Fallback to minimax if no clear territorial move
        print("No clear territorial move, using minimax...")
        best_move = self.get_best_move(board, time_left)
        result = best_move if best_move is not None else moves[0]

        print(f"I have {time_left()} seconds left. Playing {result}.")
        return result
    def initialize_territorial_strategy(self, board: board.Board):
        """
        Initialize territorial strategy based on starting position.

        We define an opening 'ladder' that walks diagonally away from our spawn
        while alternating eggs and turds.

        Example for spawn (3, 0) (top left-ish):
            (3,0)  --EGG+DOWN-->    (3,1)
            (3,1)  --TURD+RIGHT-->  (4,1)
            (4,1)  --EGG+DOWN-->    (4,2)
            (4,2)  --TURD+RIGHT-->  (5,2)
            ...
        """
        spawn = board.chicken_player.get_spawn()
        x, y = spawn  # x = column, y = row

        # Store spawn and parity
        self.my_spawn = spawn
        self.my_parity = (x + y) % 2

        # Pick which way to grow the diagonal ladder:
        # we look at which quadrant of the board our spawn is in
        # and choose an (egg_dir, turd_dir) pair.
        if x <= 3 and y <= 3:
            # Top left: grow down and right
            self.diagonal_type = "\\"
            egg_dir = Direction.DOWN
            turd_dir = Direction.RIGHT
        elif x >= 4 and y <= 3:
            # Top right: grow down and left
            self.diagonal_type = "/"
            egg_dir = Direction.DOWN
            turd_dir = Direction.LEFT
        elif x <= 3 and y >= 4:
            # Bottom left: grow up and right
            self.diagonal_type = "/"
            egg_dir = Direction.UP
            turd_dir = Direction.RIGHT
        else:
            # Bottom right: grow up and left
            self.diagonal_type = "\\"
            egg_dir = Direction.UP
            turd_dir = Direction.LEFT

        self.diagonal_base_dirs = (egg_dir, turd_dir)
        self.diagonal_step_index = 0

        print(
            f"Initialized territorial strategy: spawn={spawn}, "
            f"parity={self.my_parity}, diagonal={self.diagonal_type}, "
            f"egg_dir={egg_dir}, turd_dir={turd_dir}"
        )

    def _dynamic_danger_threshold(self, board: board.Board) -> float:
        """
        Calculate danger threshold based on how far behind we are.
        More behind = more willing to take risks.
        """
        player_eggs = len(board.eggs_player)
        enemy_eggs = len(board.eggs_enemy)
        egg_diff = player_eggs - enemy_eggs
        behind = max(0, -egg_diff)

        if behind <= 3:
            return 0.5
        elif behind <= 5:
            return 0.7
        else:
            return 0.85

    def in_my_territory(self, pos: Tuple[int, int]) -> bool:
        """
        Return True if pos is on our side of the board based on spawn and diagonal_type.
        This replaces hard-coded special squares for "our territory".
        """
        if self.diagonal_type is None or not hasattr(self, "my_spawn"):
            return True  # fallback: treat whole board as ours

        x, y = pos
        sx, sy = self.my_spawn

        if self.diagonal_type == "\\":  # use main diagonal
            if sx > sy:
                # spawn is "below" the main diagonal -> claim x >= y side
                return x >= y
            else:
                # spawn is "above" the main diagonal -> claim x <= y side
                return x <= y
        else:  # "/"
            spawn_sum = sx + sy
            if spawn_sum > 7:
                # spawn is on "bottom/right" side of / -> claim x + y >= 7
                return x + y >= 7
            elif spawn_sum < 7:
                # spawn is on "top/left" side of / -> claim x + y <= 7
                return x + y <= 7
            else:
                # spawn exactly on diagonal: pick one side consistently
                return x + y >= 7

    def is_boundary_square(self, pos: Tuple[int, int]) -> bool:
        """
        A boundary square is in our territory, but at least one 4-neighbor is in opponent territory.
        We like to put turds here to form a fence.
        """
        if not self.in_my_territory(pos):
            return False

        x, y = pos
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 8 and 0 <= ny < 8:
                if not self.in_my_territory((nx, ny)):
                    return True
        return False

    def try_strategic_turd(self, board: board.Board, moves):
        """
        Place turds strategically to:
        1. Block opponent's diagonal/corners (early game)
        2. Protect our egg clusters (mid game)
        
        Returns turd move if we should place one, None otherwise.
        """
        location = board.chicken_player.get_location()
        turds_remaining = board.chicken_player.get_turds_left()
        player_eggs = len(board.eggs_player) if hasattr(board, 'eggs_player') else 0
        enemy_eggs = len(board.eggs_enemy) if hasattr(board, 'eggs_enemy') else 0
        
        # Only use turds in first 30 turns or if we have 5+ eggs
        turn_estimate = (5 - turds_remaining) + player_eggs  # Rough turn estimate
        if turn_estimate > 30 and player_eggs < 5:
            return None
        
        # STRATEGY 1: Early game - block opponent's key positions
        if turds_remaining >= 3:
            # Determine opponent's diagonal type based on their spawn
            enemy_spawn = board.chicken_enemy.get_spawn()
            opp_row, opp_col = enemy_spawn
            if opp_row < 4:
                if opp_col < 4:
                    opp_diagonal = "\\"
                else:
                    opp_diagonal = "/"
            else:
                if opp_col < 4:
                    opp_diagonal = "\\"
                else:
                    opp_diagonal = "/"
            
            # Block opponent's diagonal positions (opposite parity to enemy)
            enemy_parity = (enemy_spawn[0] + enemy_spawn[1]) % 2
            our_parity = self.my_parity
            
            # If we're on opponent's key diagonal position, place turd
            if opp_diagonal == "\\":
                # Check if we're on their diagonal
                if location[0] == location[1] and (location[0] + location[1]) % 2 != enemy_parity:
                    # This is opponent's diagonal but wrong parity for them - block it!
                    for move in moves:
                        if move[1] == MoveType.TURD:
                            print(f"Blocking opponent diagonal at {location}")
                            return move
            else:  # "/"
                if location[0] + location[1] == 7 and (location[0] + location[1]) % 2 != enemy_parity:
                    for move in moves:
                        if move[1] == MoveType.TURD:
                            print(f"Blocking opponent diagonal at {location}")
                            return move
        
        # STRATEGY 2: Mid game - protect our egg clusters
        if turds_remaining >= 1 and player_eggs >= 3:
            # If we're adjacent to our own eggs and on wrong parity, place defensive turd
            if player_eggs + 1 < enemy_eggs:
                # we’re behind; better to move/egg later, skip defensive turd
                return None
            if not board.can_lay_egg():  # Wrong parity for us
                for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                    adj_pos = (location[0] + dx, location[1] + dy)
                    if hasattr(board, 'eggs_player') and adj_pos in board.eggs_player:
                        # Adjacent to our egg - place defensive turd
                        for move in moves:
                            if move[1] == MoveType.TURD:
                                print(f"Placing defensive turd near our eggs at {location}")
                                return move
        
        return None
    
    def get_territorial_move(self, board: board.Board, moves):
        """
        Execute territorial strategy with trapdoor avoidance.
        Priority:
        1. Lay eggs diagonally to establish territory
        2. Mark territory center (avoiding trapdoors)
        3. Fill corners based on parity
        4. Fill remaining territory
        
        Includes time-based phase advancement to ensure territory filling.
        """
        location = board.chicken_player.get_location()
        # NEW: if we’ve turned off scripted territory, never try it again
        if self.territory_phase == "off":
            return None
        
        # Filter out dangerous moves first
        safe_moves = self.filter_safe_moves(board, moves, danger_threshold=0.25)
        if not safe_moves:
            safe_moves = moves  # If all moves dangerous, use all
        
        # Check if we should advance phases based on progress
        self.maybe_advance_phase(board)
        
        # PHASE 1: Build diagonal fence (alternate eggs and turds)
        if self.territory_phase == "diagonal":
            move = self.try_diagonal_fence(board, safe_moves, location)
            if move:
                return move
            # Diagonal fence complete, move to fill phase
            self.territory_phase = "fill"
            print("Diagonal fence complete, moving to territory fill phase")
        
        # PHASE 2: Fill remaining territory
        if self.territory_phase == "fill":
            move = self.try_territory_filling(board, safe_moves, location)
            if move:
                return move
        
        # No clear territorial move
        return None
    
    def maybe_advance_phase(self, board: board.Board):
        """
        Advance to next phase if we're spending too much time or have enough eggs.
        Ensures we don't get stuck in early phases.
        """
        player_eggs = len(board.eggs_player) if hasattr(board, 'eggs_player') else 0
        enemy_eggs = len(board.eggs_enemy) if hasattr(board, 'eggs_enemy') else 0
        turds_remaining = board.chicken_player.get_turds_left()
        total_eggs = player_eggs + enemy_eggs

        # Only turn off scripted territory VERY late in the game
        # (board really is crowded by then)
        if total_eggs >= 30:  # was 18 before
            if self.territory_phase != "off":
                print("Board very crowded; turning off scripted territory and relying on minimax.")
            self.territory_phase = "off"
            return

        # Advance from diagonal to fill phase when:
        # 1. We have 4+ diagonal eggs AND 2+ turds placed, OR
        # 2. We've run out of turds (can't place more fence), OR
        # 3. We have 6+ total eggs (enough boundary established)
        if self.territory_phase == "diagonal":
            if (len(self.diagonal_eggs_laid) >= 4 and len(self.diagonal_turds_placed) >= 2) or \
               turds_remaining == 0 or \
               player_eggs >= 6:
                self.territory_phase = "fill"
                print("Diagonal fence complete, advancing to territory fill phase")

    def try_diagonal_fence(self, board: board.Board, moves, location):
        """
        Build a spawn-based diagonal 'ladder' starting from our spawn.

        Pattern (for top-left spawn):
            egg step:   move along egg_dir (e.g. DOWN) with MoveType.EGG
            turd step:  move along turd_dir (e.g. RIGHT) with MoveType.TURD

        This gives a fence like:
            (3,0) egg
            (3,1) turd
            (4,1) egg
            (4,2) turd
            ...
        """
        row, col = location

        # Track recent positions to detect oscillation
        self.position_history.append(location)
        if len(self.position_history) > 6:
            self.position_history.pop(0)

        if len(self.position_history) >= 4:
            unique_positions = set(self.position_history[-4:])
            if len(unique_positions) <= 2:
                self.stuck_counter += 1
            else:
                self.stuck_counter = 0

        # If we've been bouncing in place, abandon the diagonal phase
        if self.stuck_counter >= self.stuck_threshold:
            print(f"Stuck oscillating near {location} for {self.stuck_counter} turns, advancing to fill phase")
            self.territory_phase = "fill"
            self.stuck_counter = 0
            return None

        # Hard stop on fence length and when turds are gone
        turds_remaining = board.chicken_player.get_turds_left()
        board_size = 8
        max_fence_steps = min(board_size, turds_remaining * 2)
        if self.diagonal_step_index >= max_fence_steps or turds_remaining == 0:
            print("Diagonal fence complete (step limit or no turds), advancing to fill phase")
            self.territory_phase = "fill"
            return None

        # Make sure we know our base directions
        if not self.diagonal_base_dirs:
            return None
        egg_dir, turd_dir = self.diagonal_base_dirs

        can_egg_here = board.can_lay_egg()

        # Decide what we WANT to do here
        if can_egg_here:
            desired_dir = egg_dir
            desired_type = MoveType.EGG
        else:
            # Wrong parity for egg: try to lay a turd on the fence, if we can
            if turds_remaining > 0:
                desired_dir = turd_dir
                desired_type = MoveType.TURD
            else:
                # No turds left, just walk along egg direction
                desired_dir = egg_dir
                desired_type = MoveType.PLAIN

        # Try to find a move that matches (desired_dir, desired_type)
        move = self.pick_directional_move(board, moves, location, desired_dir, desired_type)
        if move is None:
            # Relax type constraint: any move in desired_dir
            move = self.pick_directional_move(board, moves, location, desired_dir, None)

        if move is None:
            # We cannot follow the ladder at all, give up on the diagonal
            print(f"Diagonal fence blocked from {location}, advancing to fill phase")
            self.territory_phase = "fill"
            return None

        direction, move_type = move
        new_loc = loc_after_direction(location, direction)

        # Check trap probability - trust Bayesian belief instead of hard-coded central band
        danger = self.get_trapdoor_danger(new_loc[0], new_loc[1])
        if danger > 0.5:
            print(f"Planned diagonal step into {new_loc} too dangerous (prob={danger:.2f}), skipping fence move")
            return None

        # Record what we just left behind at 'location'
        if move_type == MoveType.EGG:
            self.diagonal_eggs_laid.add(location)
            self.last_diagonal_action = "egg"
            print(f"Laying diagonal fence egg at {location} -> moving {direction.name}")
        elif move_type == MoveType.TURD:
            self.diagonal_turds_placed.add(location)
            self.last_diagonal_action = "turd"
            print(f"Placing diagonal fence turd at {location} -> moving {direction.name}")

        self.stuck_counter = 0
        self.diagonal_step_index += 1
        return move


    def try_center_marking(self, board: board.Board, moves, location):
        """
        Place eggs/turds near the center to mark territory boundary (avoiding trapdoors).
        Also lays eggs opportunistically whenever possible.
        """
        # Define boundary positions based on diagonal
        if self.diagonal_type == "\\":
            # For \ diagonal, mark positions on the / side
            boundary_positions = [(2, 5), (3, 4), (4, 3), (5, 2)]
        else:  # "/"
            # For / diagonal, mark positions on the \ side
            boundary_positions = [(2, 2), (3, 3), (4, 4), (5, 5)]
        
        # Filter out positions with high trapdoor probability
        safe_boundary_positions = [
            pos for pos in boundary_positions 
            if self.get_trapdoor_danger(pos[0], pos[1]) < 0.3
        ]
        
        # PRIORITY: Always lay egg if possible
        if board.can_lay_egg():
            for move in moves:
                if move[1] == MoveType.EGG:
                    if location in safe_boundary_positions:
                        print(f"Marking territory center at {location}")
                        self.center_marks_placed = True
                    else:
                        print(f"Laying opportunistic egg at {location} (center marking phase)")
                    return move
        
        # Move toward a safe boundary position
        for pos in safe_boundary_positions:
            if not self.is_position_blocked(board, pos):
                move = self.find_safe_move_toward(board, moves, location, pos)
                if move:
                    print(f"Moving toward boundary position {pos}")
                    return move
        
        # All boundaries handled or too dangerous
        self.center_marks_placed = True
        return None
    
    def try_corner_filling(self, board: board.Board, moves, location):
        """
        Fill the corners based on parity.
        Even parity: (0,0) and (7,7)
        Odd parity: (0,7) and (7,0)
        Also lays eggs opportunistically whenever possible.
        """
        if self.my_parity == 0:  # Even parity
            target_corners = [(0, 0), (7, 7)]
        else:  # Odd parity
            target_corners = [(0, 7), (7, 0)]
        
        # PRIORITY: Always lay egg if possible
        if board.can_lay_egg():
            for move in moves:
                if move[1] == MoveType.EGG:
                    if location in target_corners:
                        self.corners_filled.add(location)
                        print(f"Laying corner egg at {location}")
                    else:
                        print(f"Laying opportunistic egg at {location} (corner phase)")
                    return move
        
        # Move toward nearest unfilled corner that we can actually reach with correct parity
        unfilled_corners = [c for c in target_corners if c not in self.corners_filled]
        
        # Filter out corners that are blocked or unreachable due to parity
        reachable_corners = []
        for corner in unfilled_corners:
            # Skip if blocked
            if self.is_position_blocked(board, corner):
                continue
            
            # Check if we can reach this corner with correct parity
            current_parity = (location[0] + location[1]) % 2
            target_parity = (corner[0] + corner[1]) % 2
            manhattan_dist = abs(location[0] - corner[0]) + abs(location[1] - corner[1])
            
            # We can reach with correct parity if: current parity + dist parity == target parity
            # Since we need target_parity == my_parity, and dist changes parity if odd
            # We need: (current_parity + dist) % 2 == target_parity
            if (current_parity + manhattan_dist) % 2 == target_parity:
                reachable_corners.append(corner)
        
        if reachable_corners:
            # Sort by distance
            reachable_corners.sort(key=lambda c: abs(location[0] - c[0]) + abs(location[1] - c[1]))
            target = reachable_corners[0]
            
            move = self.find_safe_move_toward(board, moves, location, target)
            if move:
                print(f"Moving toward corner {target}")
                return move
        
        # No reachable corners - advance to fill phase
        print("No reachable corners, advancing to fill phase")
        return None
    
    def try_territory_filling(self, board: board.Board, moves, location):
        """
        Fill remaining territory systematically.
        
        NEW BEHAVIOR:
          - If we can lay an egg and we're in our territory: do it.
          - If we cannot lay an egg and we're on a boundary square and have turds: fence with a turd.
          - Otherwise, move toward an unfilled square in our territory (edges preferred),
            using trap-aware pathing.
        """
        # Stuck detection: if we are bouncing between the same 2 squares,
        # give up on scripted filling and fall back to minimax.
        self.position_history.append(location)
        if len(self.position_history) > 6:
            self.position_history.pop(0)

        if len(self.position_history) >= 4:
            recent = self.position_history[-4:]
            unique_positions = set(recent)
            if len(unique_positions) <= 2:
                self.stuck_counter += 1
            else:
                self.stuck_counter = 0

        if self.stuck_counter >= self.stuck_threshold:
            print(f"Stuck filling near {location}, falling back to minimax")
            self.stuck_counter = 0
            self.territory_phase = "off"
            return None

        # 1) Lay egg at current location if possible and in our territory
        if board.can_lay_egg() and self.in_my_territory(location):
            for move in moves:
                if move[1] == MoveType.EGG:
                    print(f"Filling territory at {location} with egg")
                    return move

        # 2) If we can't lay egg, but we are on a boundary and have turds, fence it
        if (not board.can_lay_egg()
            and board.chicken_player.get_turds_left() > 0
            and self.is_boundary_square(location)
            and self.get_trapdoor_danger(location[0], location[1]) < 0.8):
            for move in moves:
                if move[1] == MoveType.TURD:
                    print(f"Fencing boundary at {location} with turd")
                    return move

        # 3) Move toward nearest unfilled position in our territory (edge squares prioritized)
        territory_positions = self.get_territory_positions()

        unfilled = []
        for pos in territory_positions:
            if self.is_position_blocked(board, pos):
                continue

            # still only care about our parity squares
            if (pos[0] + pos[1]) % 2 != self.my_parity:
                continue

            is_edge = (pos[0] == 0 or pos[0] == 7 or pos[1] == 0 or pos[1] == 7)
            priority = 0 if is_edge else 1
            distance = abs(location[0] - pos[0]) + abs(location[1] - pos[1])
            unfilled.append((priority, distance, pos))

        if unfilled:
            unfilled.sort(key=lambda t: (t[0], t[1]))
            target = unfilled[0][2]
            move = self.find_safe_move_toward(board, moves, location, target)
            if move:
                print(f"Moving to fill territory position {target}")
                return move

        # No clear territorial move
        return None
    
    def get_territory_positions(self):
        """
        Get all positions in our claimed territory based on diagonal.
        ONLY returns positions matching our parity (where we can actually lay eggs).
        """
        positions = []
        spawn = self.my_spawn if hasattr(self, 'my_spawn') else (0, 0)
        
        if self.diagonal_type == "\\":
            # \ diagonal: row = col line
            # If spawn row > col: claim bottom-right (row >= col)
            # If spawn row < col: claim top-left (row <= col)
            if spawn[0] > spawn[1]:  # Bottom-right quadrant
                for row in range(8):
                    for col in range(8):
                        if row >= col and (row + col) % 2 == self.my_parity:
                            positions.append((row, col))
            else:  # Top-left quadrant (spawn[0] <= spawn[1])
                for row in range(8):
                    for col in range(8):
                        if row <= col and (row + col) % 2 == self.my_parity:
                            positions.append((row, col))
        else:  # "/"
            # / diagonal: row + col = 7 line
            # If spawn row + col > 7: claim bottom-left/right side (row + col >= 7)
            # If spawn row + col < 7: claim top-left/right side (row + col <= 7)
            # Special case: if spawn row + col = 7, we're ON the diagonal - claim the side with more space
            spawn_sum = spawn[0] + spawn[1]
            
            if spawn_sum > 7:  # Bottom side (away from (0,0))
                for row in range(8):
                    for col in range(8):
                        if row + col >= 7 and (row + col) % 2 == self.my_parity:
                            positions.append((row, col))
            elif spawn_sum < 7:  # Top side (near (0,0))
                for row in range(8):
                    for col in range(8):
                        if row + col <= 7 and (row + col) % 2 == self.my_parity:
                            positions.append((row, col))
            else:  # spawn_sum == 7, on the diagonal
                # Claim the bottom-left side (larger row values)
                for row in range(8):
                    for col in range(8):
                        if row + col >= 7 and (row + col) % 2 == self.my_parity:
                            positions.append((row, col))
        
        return positions
    
    def can_eventually_lay_egg_at(self, board: board.Board, target):
        """
        Check if we can eventually lay an egg at target (parity check).
        """
        current_loc = board.chicken_player.get_location()
        manhattan_dist = abs(current_loc[0] - target[0]) + abs(current_loc[1] - target[1])
        
        # Check parity compatibility
        current_parity = board.can_lay_egg()
        target_parity = (manhattan_dist % 2) == 0
        
        return current_parity == target_parity
    
    def is_position_blocked(self, board: board.Board, position):
        """
        Check if a position is blocked by an egg or turd.
        """
        if hasattr(board, 'eggs_player') and position in board.eggs_player:
            return True
        if hasattr(board, 'eggs_enemy') and position in board.eggs_enemy:
            return True
        if hasattr(board, 'turds_player') and position in board.turds_player:
            return True
        if hasattr(board, 'turds_enemy') and position in board.turds_enemy:
            return True
        return False
    
    def find_safe_move_toward(self, board: board.Board, moves, current, target):
        """
        Find a move that gets us closer to target while avoiding trapdoors.
        """
        best_move = None
        best_score = float('-inf')
        
        curr_dist = abs(current[0] - target[0]) + abs(current[1] - target[1])
        
        for move in moves:
            direction, move_type = move
            
            # Only consider plain moves for navigation
            if move_type != MoveType.PLAIN:
                continue
            
            new_loc = loc_after_direction(current, direction)
            
            if 0 <= new_loc[0] < 8 and 0 <= new_loc[1] < 8:
                new_dist = abs(new_loc[0] - target[0]) + abs(new_loc[1] - target[1])
                
                # Score: prefer moves that get closer and avoid trapdoors
                score = (curr_dist - new_dist) * 100  # Reward getting closer
                danger = self.get_trapdoor_danger(new_loc[0], new_loc[1])
                score -= danger * 1000  # Heavy penalty for trapdoor danger
                
                if score > best_score:
                    best_score = score
                    best_move = move
        
        return best_move

    def pick_directional_move(self, board: board.Board, moves, current, desired_dir, desired_type=None):
        """
        Among the available moves, pick one that uses desired_dir and,
        if desired_type is not None, the desired MoveType.

        We prefer moves that go to lower trapdoor danger.
        """
        best = None  # (direction, move_type, danger)

        for direction, move_type in moves:
            if direction != desired_dir:
                continue
            if desired_type is not None and move_type != desired_type:
                continue

            new_loc = loc_after_direction(current, direction)
            if not (0 <= new_loc[0] < 8 and 0 <= new_loc[1] < 8):
                continue

            danger = self.get_trapdoor_danger(new_loc[0], new_loc[1])

            if best is None or danger < best[2]:
                best = (direction, move_type, danger)

        if best is None:
            return None
        return (best[0], best[1])


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
        
        # How far behind are we? (positive means we're losing)
        player_eggs = len(board.eggs_player)
        enemy_eggs = len(board.eggs_enemy)
        egg_diff = player_eggs - enemy_eggs
        behind = max(0, -egg_diff)

        # Base behavior: cautious when the game is close.
        # If we fall further behind, we gradually become more willing
        # to step on medium-risk squares, but never on near-certain traps.
        if behind <= 3:
            # Close game → pretty safe
            danger_threshold = 0.5
        elif behind <= 5:
            # Moderately behind → take some risks
            danger_threshold = 0.7
        else:
            # Way behind → quite risky, but still avoid near-certain traps
            danger_threshold = 0.85

        moves = self.filter_safe_moves(board, moves, danger_threshold)

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

        safe_moves = []      # danger < danger_threshold
        fallback_moves = []  # danger in [danger_threshold, 0.9)
        hard_banned = []     # danger >= 0.9

        for move in moves:
            direction, move_type = move
            new_loc = loc_after_direction(location, direction)

            if 0 <= new_loc[0] < 8 and 0 <= new_loc[1] < 8:
                danger = self.get_trapdoor_danger(new_loc[0], new_loc[1])

                # Hard ban: near-certain trapdoor (>= 0.9)
                if danger >= 0.9:
                    hard_banned.append(move)
                    continue

                # Safe according to our current threshold
                if danger < danger_threshold:
                    safe_moves.append(move)
                else:
                    # Not great, but better than a near-certain trap
                    fallback_moves.append(move)
            else:
                # Off-board moves should be invalid anyway, but don't treat as dangerous here
                safe_moves.append(move)

        # 1. Prefer truly safe moves if we have any
        if safe_moves:
            return safe_moves

        # 2. Otherwise, accept medium-risk moves (but still < 0.9 danger)
        if fallback_moves:
            return fallback_moves

        # 3. Everything is >= 0.9 → we're in hell. Let caller pick the least bad.
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
        
        # Bonus for being on a square where we can lay an egg
        if (location[0] + location[1]) % 2 == parity:
            score += 5  # Can lay egg here

        # Find the closest empty square of our parity (where we could lay an egg
        # at some point), and give a bonus for being closer to it.
        nearest_egg_dist = None

        for x in range(8):
            for y in range(8):
                # Only squares where we can ever lay eggs (same parity as our chicken)
                if (x + y) % 2 != parity:
                    continue

                # Skip squares that already have eggs or turds on them
                if self.is_position_blocked(board, (x, y)):
                    continue

                # Manhattan distance from our current location
                dist = abs(location[0] - x) + abs(location[1] - y)

                if nearest_egg_dist is None or dist < nearest_egg_dist:
                    nearest_egg_dist = dist

        if nearest_egg_dist is not None:
            # On an 8x8 board, max Manhattan distance is 14 (corner to corner).
            # Closer to a potential egg square = bigger bonus.
            max_dist = 14
            score += (max_dist - nearest_egg_dist) * 5

        # Corner proximity bonus - reward being closer to any corner
        corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
        min_corner_distance = min(
            abs(location[0] - cx) + abs(location[1] - cy) 
            for cx, cy in corners
        )
        # Max distance to a corner is 14 (from center to opposite corner)
        # Give up to 14 points for being close to a corner
        score += (14 - min_corner_distance)
        
        # Extra bonus for actually being on a corner with egg-laying potential
        if location in corners and (location[0] + location[1]) % 2 == parity:
            score += 20  # Corner egg potential

        # Mobility: having more valid moves is good
        moves = board.get_valid_moves()
        score += len(moves)

        # TRAPDOOR AVOIDANCE
        # Penalize being on or near high-probability trapdoor squares
        x, y = location
        danger = self.get_trapdoor_danger(x, y)
        
        # Massive penalty for standing on likely trapdoor
        # At 89% confidence, this should be -8900 points - completely unacceptable
        # (4 eggs = 400 points, but we want to NEVER step on a high-confidence trapdoor)
        egg_diff = player_eggs - enemy_eggs

        if egg_diff < -3:
            # We're losing badly → be more willing to risk trapdoors
            danger_weight_on = 800
            danger_weight_adj = 10
        else:
            # Even or ahead → stay cautious
            danger_weight_on = 1500
            danger_weight_adj = 20

        score -= danger * danger_weight_on
        
        # Smaller penalty for being adjacent to likely trapdoors
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < 8 and 0 <= ny < 8:
                    adjacent_danger = self.get_trapdoor_danger(nx, ny)
                    score -= adjacent_danger * danger_weight_adj # Smaller penalty for nearby danger

        # ENEMY SPAWN AVOIDANCE
        # Once we've been within 5 squares of enemy spawn, penalize getting closer
        if self.near_enemy_spawn:
            enemy_spawn = board.chicken_enemy.get_spawn()
            dist_to_enemy_spawn = abs(x - enemy_spawn[0]) + abs(y - enemy_spawn[1])
            
            # Penalize being close to enemy spawn (they might block our return if we hit trapdoor)
            # Closer = higher penalty
            if dist_to_enemy_spawn <= 5:
                # Scale penalty: at distance 0 (on spawn) = 500, at distance 5 = 100
                spawn_penalty = (6 - dist_to_enemy_spawn) * 100
                score -= spawn_penalty

        return score