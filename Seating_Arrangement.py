import math
import random

conflict_matrix = [
    [0, 3, 8, 2],  
    [3, 0, 6, 4],  
    [8, 6, 0, 5],  
    [2, 4, 5, 0]
    ]
index_to_person = ['A', 'B', 'C', 'D']

# initial state: C, A, B, D
def initial_state():
    return [2, 0, 1, 3]  

def calculate_cost(state):
    score = 0
    n = len(state)
    for i in range(n):
        a = state[i]
        b = state[(i + 1) % n]
        score += conflict_matrix[a][b]
    return score

def get_neighbor(state):
    new_state = state[:]
    i, j = random.sample(range(len(state)), 2)
    new_state[i], new_state[j] = new_state[j], new_state[i]
    return new_state

def simulated_annealing(temperature=100, alpha=0.95, max_iterations=1000):
    current = initial_state()
    best = current
    current_cost = calculate_cost(current)
    best_cost = current_cost

    for iteration in range(max_iterations):
        neighbor = get_neighbor(current)
        neighbor_cost = calculate_cost(neighbor)

        delta = neighbor_cost - current_cost

        if delta < 0:
            status = "Good - Accepted"
            current = neighbor
            current_cost = neighbor_cost
        else:
            probability = math.exp(-delta/temperature)
            if random.random() < probability:
                status = "Bad - Accepted"
                current = neighbor
                current_cost = neighbor_cost
            else:
                status = "Bad - Rejected"
        
        if current_cost < best_cost:
                best = current
                best_cost = current_cost

        print(f"Iteration {iteration}: Current Score = {current_cost}, {status}")
        temperature *= alpha

    return best, best_cost

best_state, best_score = simulated_annealing()
best_seating = [index_to_person[i] for i in best_state]

print("\nOptimal Circular Seating Arrangement:", " -> ".join(best_seating))
print("Minimum Conflict Score:", best_score)