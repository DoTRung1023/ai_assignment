import heapq
import math
import sys
from collections import deque

# get input
argv = sys.argv
MODE = argv[1].lower()
MAP_PATH = argv[2]
ALGO = argv[3].lower()

HEUR = (argv[4].lower() if len(argv) >= 5 else "manhattan")

lines = []
with open(MAP_PATH, "r", encoding="utf-8") as f:
    for ln in f:
        ln = ln.strip()
        if ln != "":
            lines.append(ln)

ROWS, COLS = map(int, lines[0].split())
sr, sc = map(int, lines[1].split())
gr, gc = map(int, lines[2].split())
sr -= 1
sc -= 1
gr -= 1
gc -= 1
START = (sr, sc)
GOAL = (gr, gc)

GRID = [["X" for _ in range(COLS)] for _ in range(ROWS)]
grid_lines = lines[3:]
for i in range(ROWS):
    parts = grid_lines[i].split()
    for j in range(COLS):
        t = parts[j]
        if t != "X":
            GRID[i][j] = int(t)

# calculate step cost
def step_cost(grid, cur_pos, next_pos):
    r, c = cur_pos
    nr, nc = next_pos
    cur_height = grid[r][c]
    next_height = grid[nr][nc]
    return 1 + max(0, next_height - cur_height)

# get neighbors
def get_neighbors(rows, cols, grid, cur_pos):
    r, c = cur_pos
    out = []
    candidates = ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
    for nr, nc in candidates:
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != "X":
            out.append((nr, nc))
    return out

# stats for debug
def make_stats(rows, cols):
    visits = [[0 for _ in range(cols)] for _ in range(rows)]
    first = [[None for _ in range(cols)] for _ in range(rows)]
    last = [[None for _ in range(cols)] for _ in range(rows)]
    return [visits, first, last, 0]

# update stats when visit
def record_visit(stats, cur_pos):
    r, c = cur_pos

    # update visit times
    visits = stats[0]
    visits[r][c] += 1

    # update count
    count = stats[3]
    count += 1
    stats[3] = count

    # update first and last visit
    first = stats[1]
    last = stats[2]
    if first[r][c] is None:
        first[r][c] = count
    last[r][c] = count

# construct path
def construct_path(parent, start, goal):
    path = []

    cur = goal
    path.append(cur)
    while cur != start:
        cur = parent[cur]
        path.append(cur)
    path.reverse()

    return path

# bfs algorithm
def bfs(rows, cols, start, goal, grid, stats):
    q = deque()
    q.append(start)
    parent = {}
    visited = set()

    while q:
        u = q.popleft()

        # update stats
        record_visit(stats, u)

        # return path if goal is reached
        if u == goal:
            return construct_path(parent, start, goal)

        # skip if already visited
        if u in visited:
            continue

        # add neighbors to queue
        visited.add(u)
        for v in get_neighbors(rows, cols, grid, u):
            if v not in visited:
                q.append(v)
                if v != start and v not in parent:
                    parent[v] = u
    return None

# ucs algorithm
def ucs(rows, cols, start, goal, grid, stats):
    pq = []
    tieBreaker = 0
    heapq.heappush(pq, (0, tieBreaker, start))
    parent = {}
    best_g = {start: 0}
    visited = set()

    while pq:
        g, _, u = heapq.heappop(pq)

        # update stats
        record_visit(stats, u)

        # return path if goal is reached
        if u == goal:
            return construct_path(parent, start, goal)

        # skip if already visited
        if u in visited:
            continue

        # add neighbors to priority queue
        visited.add(u)
        for v in get_neighbors(rows, cols, grid, u):
            ng = g + step_cost(grid, u, v)
            if v not in best_g or ng < best_g[v]:
                best_g[v] = ng
                parent[v] = u
                tieBreaker += 1
                heapq.heappush(pq, (ng, tieBreaker, v))
    return None

# calculate heuristic
def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def euclidean(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


# astar algorithm
def astar(rows, cols, start, goal, grid, stats, hfun):
    pq = []
    tieBreaker = 0
    heapq.heappush(pq, (hfun(start, goal), tieBreaker, 0, start))
    parent = {}
    best_g = {start: 0}
    visited = set()

    while pq:
        _, _, g, u = heapq.heappop(pq)

        # update stats
        record_visit(stats, u)

        # return path if goal is reached
        if u == goal:
            return construct_path(parent, start, goal)

        # skip if already visited
        if u in visited:
            continue

        # add neighbors to priority queue
        visited.add(u)
        for v in get_neighbors(rows, cols, grid, u):
            ng = g + step_cost(grid, u, v)
            if v not in best_g or ng < best_g[v]:
                best_g[v] = ng
                parent[v] = u
                tieBreaker += 1

                # calculate f = cost + heuristic
                f = ng + hfun(v, goal)
                heapq.heappush(pq, (f, tieBreaker, ng, v))
    return None

# print path
def print_path(rows, cols, grid, path):
    if path is None:
        return None
    s = set(path)
    out = [["" for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == "X":
                out[i][j] = "X"
            else:
                out[i][j] = "*" if (i, j) in s else str(grid[i][j])
    return out

# print release
def print_release(path_grid):
    if path_grid is None:
        print("null")
        return
    for row in path_grid:
        print(" ".join(row))

# print debug
def print_debug(rows, cols, grid, path_grid, stats):
    print("path:")
    if path_grid is None:
        print("null")
    else:
        for row in path_grid:
            print(" ".join(row))

    print("#visits:")
    for i in range(rows):
        parts = []
        for j in range(cols):
            if grid[i][j] == "X":
                parts.append("X")
            else:
                v = stats[0][i][j]
                parts.append(str(v) if v > 0 else ".")
        print(" ".join(parts))

    def print_order(mat):
        for i in range(rows):
            parts = []
            for j in range(cols):
                if grid[i][j] == "X":
                    parts.append("X".rjust(3))
                else:
                    v = mat[i][j]
                    parts.append(("." if v is None else str(v)).rjust(3))
            print(" ".join(parts))

    print("first visit:")
    print_order(stats[1])
    print("last visit:")
    print_order(stats[2])


def main():
    stats = make_stats(ROWS, COLS)
    path = None
    if ALGO == "bfs":
        path = bfs(ROWS, COLS, START, GOAL, GRID, stats)
    elif ALGO == "ucs":
        path = ucs(ROWS, COLS, START, GOAL, GRID, stats)
    else:
        if HEUR == "euclidean":
            hfun = euclidean
        else:
            hfun = manhattan
        path = astar(ROWS, COLS, START, GOAL, GRID, stats, hfun)
    
    # print output
    path_grid = print_path(ROWS, COLS, GRID, path)

    if MODE == "release":
        print_release(path_grid)
    else:
        print_debug(ROWS, COLS, GRID, path_grid, stats)

main()