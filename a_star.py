import heapq
import random
import math

# ----------------------------
# Heuristic Functions (adapted to dynamic goal)
# ----------------------------

def manhattan(state, goal):
    """Manhattan distance: Sum of absolute row and column differences for each tile."""
    distance = 0
    for i, tile in enumerate(state):
        if tile == 0:
            continue  # Skip the blank
        goal_index = goal.index(tile)
        distance += abs(goal_index % 3 - i % 3) + abs(goal_index // 3 - i // 3)
    return distance

def hamming(state, goal):
    """Hamming distance: Count of misplaced tiles (excluding the blank)."""
    return sum(1 for i, tile in enumerate(state) if tile != 0 and tile != goal[i])

def euclidean(state, goal):
    """Euclidean distance: Sum of Euclidean distances for each tile."""
    distance = 0.0
    for i, tile in enumerate(state):
        if tile == 0:
            continue
        goal_index = goal.index(tile)
        dx = abs(goal_index % 3 - i % 3)
        dy = abs(goal_index // 3 - i // 3)
        distance += math.sqrt(dx*dx + dy*dy)
    return distance

def linear_conflict(state, goal):
    """
    Linear Conflict: Manhattan distance plus extra cost for each pair of conflicting tiles.
    Two tiles conflict if they are in the same row or column and their goal positions are
    also in that row or column but reversed relative to each other.
    """
    # Base: Manhattan distance
    md = manhattan(state, goal)
    lc = 0

    # Row conflicts
    for row in range(3):
        row_indices = [row * 3 + col for col in range(3)]
        tiles = [state[i] for i in row_indices]
        for i in range(3):
            for j in range(i+1, 3):
                if tiles[i] != 0 and tiles[j] != 0:
                    # Check if both tiles should be in the same row in the goal
                    if goal.index(tiles[i]) // 3 == row and goal.index(tiles[j]) // 3 == row:
                        if tiles[i] > tiles[j]:
                            lc += 2

    # Column conflicts
    for col in range(3):
        col_indices = [col + 3 * row for row in range(3)]
        tiles = [state[i] for i in col_indices]
        for i in range(3):
            for j in range(i+1, 3):
                if tiles[i] != 0 and tiles[j] != 0:
                    if goal.index(tiles[i]) % 3 == col and goal.index(tiles[j]) % 3 == col:
                        if tiles[i] > tiles[j]:
                            lc += 2

    return md + lc

# Dictionary mapping heuristic names to functions.
HEURISTICS = {
    "manhattan": manhattan,
    "hamming": hamming,
    "euclidean": euclidean,
    "linear_conflict": linear_conflict
}

# ----------------------------
# A* Search Implementation
# ----------------------------

def get_neighbors(state):
    """Generate neighbors for the 8-puzzle by moving the blank (0) in all valid directions."""
    neighbors = []
    i = state.index(0)
    x, y = i % 3, i // 3  # column, row
    moves = []
    if x > 0: moves.append((-1, 0))  # left
    if x < 2: moves.append((1, 0))   # right
    if y > 0: moves.append((0, -1))  # up
    if y < 2: moves.append((0, 1))   # down
    for dx, dy in moves:
        new_x, new_y = x + dx, y + dy
        j = new_y * 3 + new_x
        new_state = list(state)
        new_state[i], new_state[j] = new_state[j], new_state[i]
        neighbors.append(tuple(new_state))
    return neighbors

def a_star(start, goal, heuristic_fn):
    """
    Perform A* search for the 8-puzzle using a given heuristic function.
    
    Returns:
        solution: List of states from start to goal.
        search_data: List of dictionaries for each expanded node (state, g, h, f).
    """
    open_set = []
    initial_h = heuristic_fn(start, goal)
    heapq.heappush(open_set, (initial_h, 0, start, []))
    closed_set = set()
    search_data = []  # to record each expanded node

    while open_set:
        f, g, current, path = heapq.heappop(open_set)
        current_h = f - g
        search_data.append({
            "state": current,
            "g": g,
            "h": current_h,
            "f": f
        })

        if current == goal:
            return path + [current], search_data

        if current in closed_set:
            continue
        closed_set.add(current)

        for neighbor in get_neighbors(current):
            if neighbor in closed_set:
                continue
            new_g = g + 1
            new_h = heuristic_fn(neighbor, goal)
            new_f = new_g + new_h
            heapq.heappush(open_set, (new_f, new_g, neighbor, path + [current]))
    
    return None, search_data

# ----------------------------
# Puzzle Generation Helpers
# ----------------------------

def generate_random_state(goal, moves=30):
    """
    Generate a random 8-puzzle state by starting from the goal state and
    making a series of random moves.
    """
    state = goal
    for _ in range(moves):
        state = random.choice(get_neighbors(state))
    return state

def parse_state_input(state_str):
    """
    Convert a space-separated string of 9 numbers into a tuple representing the state.
    """
    parts = state_str.strip().split()
    if len(parts) != 9:
        raise ValueError("State must have exactly 9 numbers.")
    return tuple(int(x) for x in parts)

# ----------------------------
# Main Routine
# ----------------------------

def main():
    # Input the goal state.
    goal_input = input("Enter the goal state (9 numbers separated by spaces, use 0 for blank): ")
    try:
        goal_state = parse_state_input(goal_input)
    except ValueError as ve:
        print("Error:", ve)
        return

    # Choose the heuristic.
    print("Available heuristics:")
    for key in HEURISTICS.keys():
        print(f" - {key}")
    heuristic_choice = input("Enter the heuristic to use: ").strip().lower()
    if heuristic_choice not in HEURISTICS:
        print("Invalid heuristic choice. Exiting.")
        return
    heuristic_fn = HEURISTICS[heuristic_choice]

    # Input starting state or generate one.
    start_choice = input("Do you want to enter a starting state? (y/n): ").strip().lower()
    if start_choice == 'y':
        start_input = input("Enter the starting state (9 numbers separated by spaces, use 0 for blank): ")
        try:
            start_state = parse_state_input(start_input)
        except ValueError as ve:
            print("Error:", ve)
            return
    else:
        # Generate a random state by scrambling the goal state.
        start_state = generate_random_state(goal_state)
        print("Generated starting state:", start_state)

    print("\nStarting A* search...")
    solution, search_data = a_star(start_state, goal_state, heuristic_fn)
    if solution:
        print(f"Solution found in {len(solution) - 1} moves!")
        print("Solution path:")
        for state in solution:
            print(state)
    else:
        print("No solution found.")

    # (Optional) Print out the search data for debugging or analysis.
    print("\nSearch Data (first 5 expanded nodes):")
    for data in search_data[:5]:
        print(data)

if __name__ == '__main__':
    main()