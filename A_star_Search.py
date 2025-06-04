import math
import heapq

def heuristic(n, goal, coordinates):
    x1, y1 = coordinates[n]
    x2, y2 = coordinates[goal]
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def a_star_search(start, goal, graphs, coordinates):
    minQ = []
    heapq.heappush(minQ, (heuristic(start, goal, coordinates), 0, start, [start]))
    visited = set()

    while minQ:
        f, path_cost, current, path = heapq.heappop(minQ)
        if current == goal:
            return path, path_cost
        if current in visited:
            continue
        visited.add(current)
        for neighbor, cost in graphs.get(current, []):
            if neighbor not in visited:
                new_g = path_cost + cost
                heu = heuristic(neighbor, goal, coordinates)
                new_f = new_g + heu
                new_path = path + [neighbor]
                heapq.heappush(minQ, (new_f, new_g, neighbor, new_path))
    return None

def main():
    V = int(input())
    coordinates = {}
    for _ in range(V):
        parts = input().split()
        n = parts[0]
        x = int(parts[1])
        y = int(parts[2])
        coordinates[n] = (x, y)

    E = int(input())
    graphs = {}
    for _ in range(E):
        u, v, cost = input().split()
        cost = int(cost)
        graphs.setdefault(u, []).append((v, cost))

    start = input().strip()
    goal = input().strip()

    path, path_cost = a_star_search(start, goal, graphs, coordinates)

    if path:
        print("Solution path:", " -> ".join(path))
        print("Solution cost:", path_cost)
    else:
        print("No path found.")

if __name__ == "__main__":
    main()
