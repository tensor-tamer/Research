import heapq
import random
import math
import json

# ----------------------------
# Heuristic Functions
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
    md = manhattan(state, goal)
    lc = 0
    # Row conflicts
    for row in range(3):
        row_indices = [row * 3 + col for col in range(3)]
        tiles = [state[i] for i in row_indices]
        for i in range(3):
            for j in range(i+1, 3):
                if tiles[i] != 0 and tiles[j] != 0:
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
    Perform A* search for the 8-puzzle using the provided heuristic function.
    
    Returns:
        solution: List of states from start to goal.
        search_data: List of dictionaries recording (state, g, h, f) for each expanded node.
    """
    open_set = []
    initial_h = heuristic_fn(start, goal)
    heapq.heappush(open_set, (initial_h, 0, start, []))
    closed_set = set()
    search_data = []

    while open_set:
        f, g, current, path = heapq.heappop(open_set)
        current_h = f - g
        # Log search data for this node.
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
# State Augmentation Functions
# ----------------------------
def rotate_state(state, times=1):
    """Rotate the 3x3 puzzle state 90 degrees clockwise 'times' times."""
    matrix = [list(state[i*3:(i+1)*3]) for i in range(3)]
    for _ in range(times):
        # Rotate the matrix 90 degrees clockwise
        matrix = [list(row) for row in zip(*matrix[::-1])]
    return tuple(sum(matrix, []))

def mirror_state(state):
    """Mirror the 3x3 puzzle state horizontally."""
    matrix = [list(state[i*3:(i+1)*3]) for i in range(3)]
    mirrored = [row[::-1] for row in matrix]
    return tuple(sum(mirrored, []))

def augment_state(state):
    """Randomly apply a rotation or mirror transformation to the state with 50% probability."""
    if random.random() < 0.5:
        # Choose a random number of 90-degree rotations (0-3)
        times = random.randint(0, 3)
        state = rotate_state(state, times)
    if random.random() < 0.5:
        state = mirror_state(state)
    return state

def inject_noise(state, noise_prob=0.05):
    """
    With a small probability, swap two non-blank tiles.
    Note: Use with caution; too much noise might break solvability.
    """
    if random.random() < noise_prob:
        state = list(state)
        indices = [i for i, tile in enumerate(state) if tile != 0]
        if len(indices) >= 2:
            i1, i2 = random.sample(indices, 2)
            state[i1], state[i2] = state[i2], state[i1]
        return tuple(state)
    return state

# ----------------------------
# Puzzle Generation Helpers
# ----------------------------
def scramble_state(state, min_moves=20, max_moves=40):
    """
    Scramble a given state by performing a series of random moves.
    The number of moves is randomly selected between min_moves and max_moves.
    Optionally apply noise injection and augmentation.
    """
    moves = random.randint(min_moves, max_moves)
    for _ in range(moves):
        state = random.choice(get_neighbors(state))
    state = inject_noise(state, noise_prob=0.05)
    state = augment_state(state)
    return state

def generate_fixed_balanced_dataset(num_per_heuristic=125, output_file="balanced_eval_data.json"):
    all_data = []
    solved_state = (1, 2, 3, 4, 5, 6, 7, 8, 0)
    for heuristic_name, heuristic_fn in HEURISTICS.items():
        print(f"Generating {num_per_heuristic} episodes for: {heuristic_name}")
        count = 0
        while count < num_per_heuristic:
            goal = scramble_state(solved_state, 15, 25)
            start = scramble_state(goal, 20, 35)
            solution, search_data = a_star(start, goal, heuristic_fn)
            if not solution:
                continue
            episode = {
                "goal_state": goal,
                "start_state": start,
                "solution": solution,
                "search_data": search_data,
                "heuristic_used": heuristic_name
            }
            all_data.append(episode)
            count += 1
    with open(output_file, "w") as f:
        json.dump(all_data, f, indent=2)
    print(f"Saved {len(all_data)} episodes to {output_file}")

def generate_episode(goal_min_moves=15, goal_max_moves=25, start_min_moves=25, start_max_moves=35):
    """
    Generate one synthetic episode:
      - Create a random goal state by scrambling the fixed solved state.
      - Generate a start state by further scrambling the goal state.
      - Randomly choose one heuristic (uniformly for now).
      - Solve the puzzle using A* with the chosen heuristic.
    
    Returns a dictionary containing:
      - goal_state, start_state, heuristic_used, solution path, and search data.
    """
    solved_state = (1, 2, 3, 4, 5, 6, 7, 8, 0)
    # Generate random goal state with variable scramble depth.
    goal_state = scramble_state(solved_state, min_moves=goal_min_moves, max_moves=goal_max_moves)
    # Generate start state by further scrambling the goal state.
    start_state = scramble_state(goal_state, min_moves=start_min_moves, max_moves=start_max_moves)
    # Randomly choose a heuristic (can later be weighted for balance).
    heuristic_name, heuristic_fn = random.choice(list(HEURISTICS.items()))
    # Run A* search
    solution, search_data = a_star(start_state, goal_state, heuristic_fn)
    return {
        "goal_state": goal_state,
        "start_state": start_state,
        "heuristic_used": heuristic_name,
        "solution": solution,
        "search_data": search_data
    }

# ----------------------------
# Main Synthetic Data Generation Routine
# ----------------------------
def main():
    num_episodes = 500  # Adjust number of episodes as needed
    episodes = []
    for i in range(num_episodes):
        print(f"Generating episode {i+1}...")
        episode = generate_episode()
        episodes.append(episode)
    
    # Save the synthetic data to a JSON file for later ML experiments.
    with open("NEW4_synthetic_a_star_data.json", "w") as f:
        json.dump(episodes, f, indent=2)
    print(f"Data for {num_episodes} episodes saved to synthetic_a_star_data.json")

if __name__ == '__main__':
    main()
