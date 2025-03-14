import tkinter as tk  
import random

# -------------------- 全局参数 --------------------
CELL_SIZE = 10         # 每个格子的像素尺寸
MAZE_ROWS = 50         # 迷宫行数
MAZE_COLS = 50         # 迷宫列数
DELAY = 5              # 每帧延时（毫秒）
STEPS_PER_UPDATE = 100 # 每次动画帧处理的步数

# -------------------- 算法说明（用于下方显示） --------------------
gen_descriptions = {
    "DFS": "深度优先搜索 (DFS) 生成迷宫：利用回溯法沿随机路径扩展，生成一个无环连通的完美迷宫。",
    "Kruskal": "随机化 Kruskal 算法生成迷宫：通过随机排列边并利用并查集连接区域，构造完美迷宫。",
    "Prim": "随机化 Prim 算法生成迷宫：从起点出发扩展邻边，随机选择进行连接，生成连通迷宫。",
    "RecursiveDivision": "递归分割算法生成迷宫：根据区域比例选择分割方向，在分割墙上留缺口，递归生成结构独特的迷宫。"
}

path_descriptions = {
    "BFS": "广度优先搜索 (BFS) 寻路：不考虑目标位置，层层展开，直至找到终点。",
    "DFS": "深度优先搜索 (DFS) 寻路：沿每个分支深入搜索，直到无法深入为止，再回溯寻找其他分支。",
    "A*": "A* 算法寻路：结合实际距离 (G) 与启发式距离 (H)，始终扩展 F=G+H 最小的节点，快速找到最优路径。"
}

# -------------------- 数据结构 --------------------
class Cell:
    def __init__(self, row, col):
        self.row = row
        self.col = col
        # 初始四面墙均存在
        self.walls = {'top': True, 'right': True, 'bottom': True, 'left': True}
        self.visited = False  # 用于生成算法标记

    def draw(self, canvas):
        x = self.col * CELL_SIZE
        y = self.row * CELL_SIZE
        if self.walls['top']:
            canvas.create_line(x, y, x + CELL_SIZE, y)
        if self.walls['right']:
            canvas.create_line(x + CELL_SIZE, y, x + CELL_SIZE, y + CELL_SIZE)
        if self.walls['bottom']:
            canvas.create_line(x + CELL_SIZE, y + CELL_SIZE, x, y + CELL_SIZE)
        if self.walls['left']:
            canvas.create_line(x, y + CELL_SIZE, x, y)

class Maze:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.grid = [[Cell(r, c) for c in range(cols)] for r in range(rows)]

    def draw(self, canvas):
        canvas.delete("all")
        for row in self.grid:
            for cell in row:
                cell.draw(canvas)
        canvas.update()

    def get_neighbors(self, cell):
        neighbors = []
        r, c = cell.row, cell.col
        if r > 0:
            neighbors.append(self.grid[r - 1][c])
        if c < self.cols - 1:
            neighbors.append(self.grid[r][c + 1])
        if r < self.rows - 1:
            neighbors.append(self.grid[r + 1][c])
        if c > 0:
            neighbors.append(self.grid[r][c - 1])
        return neighbors

    def get_accessible_neighbors(self, cell):
        accessible = []
        r, c = cell.row, cell.col
        if r > 0 and not cell.walls['top']:
            accessible.append(self.grid[r - 1][c])
        if c < self.cols - 1 and not cell.walls['right']:
            accessible.append(self.grid[r][c + 1])
        if r < self.rows - 1 and not cell.walls['bottom']:
            accessible.append(self.grid[r + 1][c])
        if c > 0 and not cell.walls['left']:
            accessible.append(self.grid[r][c - 1])
        return accessible

    def remove_walls(self, current, next_cell):
        dr = next_cell.row - current.row
        dc = next_cell.col - current.col
        if dr == 1:
            current.walls['bottom'] = False
            next_cell.walls['top'] = False
        elif dr == -1:
            current.walls['top'] = False
            next_cell.walls['bottom'] = False
        elif dc == 1:
            current.walls['right'] = False
            next_cell.walls['left'] = False
        elif dc == -1:
            current.walls['left'] = False
            next_cell.walls['right'] = False

    def add_wall(self, cell1, cell2):
        dr = cell2.row - cell1.row
        dc = cell2.col - cell1.col
        if dr == 1:
            cell1.walls['bottom'] = True
            cell2.walls['top'] = True
        elif dr == -1:
            cell1.walls['top'] = True
            cell2.walls['bottom'] = True
        elif dc == 1:
            cell1.walls['right'] = True
            cell2.walls['left'] = True
        elif dc == -1:
            cell1.walls['left'] = True
            cell2.walls['right'] = True

    def clear_interior_walls(self):
        for r in range(self.rows):
            for c in range(self.cols):
                cell = self.grid[r][c]
                if r > 0:
                    cell.walls['top'] = False
                if r < self.rows - 1:
                    cell.walls['bottom'] = False
                if c > 0:
                    cell.walls['left'] = False
                if c < self.cols - 1:
                    cell.walls['right'] = False

# -------------------- 生成算法 --------------------
def generate_maze_dfs(maze, canvas, callback=None):
    stack = []
    start = maze.grid[0][0]
    start.visited = True
    stack.append(start)
    def step():
        nonlocal stack
        if not stack:
            if callback:
                callback()
            return
        count = 0
        while stack and count < STEPS_PER_UPDATE:
            current = stack[-1]
            neighbors = [n for n in maze.get_neighbors(current) if not n.visited]
            if neighbors:
                next_cell = random.choice(neighbors)
                maze.remove_walls(current, next_cell)
                next_cell.visited = True
                stack.append(next_cell)
            else:
                stack.pop()
            count += 1
        maze.draw(canvas)
        canvas.after(DELAY, step)
    step()

class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [0] * size
    def find(self, a):
        if self.parent[a] != a:
            self.parent[a] = self.find(self.parent[a])
        return self.parent[a]
    def union(self, a, b):
        rootA = self.find(a)
        rootB = self.find(b)
        if rootA == rootB:
            return False
        if self.rank[rootA] < self.rank[rootB]:
            self.parent[rootA] = rootB
        elif self.rank[rootA] > self.rank[rootB]:
            self.parent[rootB] = rootA
        else:
            self.parent[rootB] = rootA
            self.rank[rootA] += 1
        return True

def generate_maze_kruskal(maze, canvas, callback=None):
    uf = UnionFind(MAZE_ROWS * MAZE_COLS)
    edges = []
    for r in range(MAZE_ROWS):
        for c in range(MAZE_COLS):
            cell = maze.grid[r][c]
            if c < MAZE_COLS - 1:
                edges.append((cell, maze.grid[r][c+1]))
            if r < MAZE_ROWS - 1:
                edges.append((cell, maze.grid[r+1][c]))
    random.shuffle(edges)
    edge_index = 0
    total_edges = len(edges)
    def step():
        nonlocal edge_index
        if edge_index >= total_edges:
            if callback:
                callback()
            return
        count = 0
        while edge_index < total_edges and count < STEPS_PER_UPDATE:
            cell1, cell2 = edges[edge_index]
            index1 = cell1.row * MAZE_COLS + cell1.col
            index2 = cell2.row * MAZE_COLS + cell2.col
            if uf.union(index1, index2):
                maze.remove_walls(cell1, cell2)
            edge_index += 1
            count += 1
        maze.draw(canvas)
        canvas.after(DELAY, step)
    step()

def generate_maze_prim(maze, canvas, callback=None):
    linked = [[False] * MAZE_COLS for _ in range(MAZE_ROWS)]
    linked[0][0] = True
    candidates = []
    start = maze.grid[0][0]
    for neighbor in maze.get_neighbors(start):
        candidates.append((start, neighbor))
    def step():
        nonlocal candidates
        if not candidates:
            if callback:
                callback()
            return
        count = 0
        while candidates and count < STEPS_PER_UPDATE:
            edge = random.choice(candidates)
            candidates.remove(edge)
            cell, neighbor = edge
            if linked[neighbor.row][neighbor.col]:
                count += 1
                continue
            maze.remove_walls(cell, neighbor)
            linked[neighbor.row][neighbor.col] = True
            for nb in maze.get_neighbors(neighbor):
                if not linked[nb.row][nb.col]:
                    candidates.append((neighbor, nb))
            count += 1
        maze.draw(canvas)
        canvas.after(DELAY, step)
    step()

def recursive_division(maze, left, top, right, bottom, actions):
    if right - left < 1 or bottom - top < 1:
        return
    width = right - left + 1
    height = bottom - top + 1
    if width < height:
        orientation = 'H'
    elif width > height:
        orientation = 'V'
    else:
        orientation = random.choice(['H', 'V'])
    if orientation == 'H':
        wall_row = random.randint(top, bottom - 1)
        gap_col = random.randint(left, right)
        for col in range(left, right + 1):
            if col == gap_col:
                continue
            actions.append(('H', wall_row, col))
        recursive_division(maze, left, top, right, wall_row, actions)
        recursive_division(maze, left, wall_row + 1, right, bottom, actions)
    else:
        wall_col = random.randint(left, right - 1)
        gap_row = random.randint(top, bottom)
        for row in range(top, bottom + 1):
            if row == gap_row:
                continue
            actions.append(('V', row, wall_col))
        recursive_division(maze, left, top, wall_col, bottom, actions)
        recursive_division(maze, wall_col + 1, top, right, bottom, actions)

def generate_maze_recursive_division(maze, canvas, callback=None):
    maze.clear_interior_walls()
    actions = []
    recursive_division(maze, 0, 0, MAZE_COLS - 1, MAZE_ROWS - 1, actions)
    action_index = 0
    total_actions = len(actions)
    def step():
        nonlocal action_index
        if action_index >= total_actions:
            if callback:
                callback()
            return
        count = 0
        while action_index < total_actions and count < STEPS_PER_UPDATE:
            act = actions[action_index]
            action_index += 1
            if act[0] == 'H':
                _, row, col = act
                cell1 = maze.grid[row][col]
                cell2 = maze.grid[row+1][col]
                maze.add_wall(cell1, cell2)
            else:
                _, row, col = act
                cell1 = maze.grid[row][col]
                cell2 = maze.grid[row][col+1]
                maze.add_wall(cell1, cell2)
            count += 1
        maze.draw(canvas)
        canvas.after(DELAY, step)
    step()

# -------------------- 辅助函数：绘制探索状态和直达路径 --------------------
def draw_cell_explored(canvas, row, col, color):
    x = col * CELL_SIZE
    y = row * CELL_SIZE
    canvas.create_rectangle(x+1, y+1, x+CELL_SIZE-1, y+CELL_SIZE-1, fill=color, outline="")

def draw_direct_blue_path(canvas, maze, path):
    points = []
    for (r, c) in path:
        x = c * CELL_SIZE + CELL_SIZE / 2
        y = r * CELL_SIZE + CELL_SIZE / 2
        points.append((x, y))
    flattened = []
    for pt in points:
        flattened.extend(pt)
    canvas.create_line(*flattened, fill="blue", width=2)

# -------------------- 寻路算法（动态展现） --------------------
def dynamic_solve_bfs(maze, canvas, cancel_control, callback=None):
    start = (0, 0)
    goal = (maze.rows - 1, maze.cols - 1)
    queue = [start]
    came_from = {start: None}
    visited = set([start])
    drawn_explored = set([start])
    draw_cell_explored(canvas, 0, 0, "light blue")
    def step():
        if cancel_control["cancelled"]:
            return
        nonlocal queue
        count = 0
        while queue and count < STEPS_PER_UPDATE:
            current = queue.pop(0)
            if current == goal:
                animate_path(came_from, goal)
                return
            r, c = current
            cell = maze.grid[r][c]
            for nb in maze.get_accessible_neighbors(cell):
                nb_coord = (nb.row, nb.col)
                if nb_coord not in visited:
                    visited.add(nb_coord)
                    came_from[nb_coord] = current
                    queue.append(nb_coord)
                    if nb_coord not in drawn_explored:
                        drawn_explored.add(nb_coord)
                        draw_cell_explored(canvas, nb.row, nb.col, "light blue")
            count += 1
        canvas.after(400, step)
    def animate_path(came_from, goal):
        if cancel_control["cancelled"]:
            return
        path = []
        cur = goal
        while cur is not None:
            path.append(cur)
            cur = came_from.get(cur)
        path.reverse()
        def draw_path_step(index):
            if cancel_control["cancelled"]:
                return
            if index < len(path):
                r, c = path[index]
                draw_cell_explored(canvas, r, c, "red")
                canvas.after(1, lambda: draw_path_step(index+1))
            else:
                draw_direct_blue_path(canvas, maze, path)
                if callback:
                    callback()
        draw_path_step(0)
    step()

def dynamic_solve_dfs(maze, canvas, cancel_control, callback=None):
    start = (0, 0)
    goal = (maze.rows - 1, maze.cols - 1)
    stack = [start]
    came_from = {start: None}
    visited = set([start])
    drawn_explored = set([start])
    draw_cell_explored(canvas, 0, 0, "light blue")
    def step():
        if cancel_control["cancelled"]:
            return
        nonlocal stack
        count = 0
        while stack and count < STEPS_PER_UPDATE:
            current = stack.pop()
            r, c = current
            draw_cell_explored(canvas, r, c, "yellow")
            if current == goal:
                animate_path(came_from, goal)
                return
            cell = maze.grid[r][c]
            for nb in maze.get_accessible_neighbors(cell):
                nb_coord = (nb.row, nb.col)
                if nb_coord not in visited:
                    visited.add(nb_coord)
                    came_from[nb_coord] = current
                    stack.append(nb_coord)
                    if nb_coord not in drawn_explored:
                        drawn_explored.add(nb_coord)
                        draw_cell_explored(canvas, nb.row, nb.col, "light blue")
            count += 1
        canvas.after(400, step)
    def animate_path(came_from, goal):
        if cancel_control["cancelled"]:
            return
        path = []
        cur = goal
        while cur is not None:
            path.append(cur)
            cur = came_from.get(cur)
        path.reverse()
        def draw_path_step(index):
            if cancel_control["cancelled"]:
                return
            if index < len(path):
                r, c = path[index]
                draw_cell_explored(canvas, r, c, "red")
                canvas.after(1, lambda: draw_path_step(index+1))
            else:
                draw_direct_blue_path(canvas, maze, path)
                if callback:
                    callback()
        draw_path_step(0)
    step()

def dynamic_solve_astar(maze, canvas, cancel_control, callback=None):
    start = (0, 0)
    goal = (maze.rows - 1, maze.cols - 1)
    open_set = {start}
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    visited = set([start])
    drawn_explored = set([start])
    draw_cell_explored(canvas, 0, 0, "light blue")
    def step():
        if cancel_control["cancelled"]:
            return
        nonlocal open_set
        count = 0
        if not open_set:
            return
        while open_set and count < STEPS_PER_UPDATE:
            current = min(open_set, key=lambda x: f_score.get(x, float('inf')))
            r, c = current
            draw_cell_explored(canvas, r, c, "orange")
            if current == goal:
                animate_path(came_from, goal)
                return
            open_set.remove(current)
            cell = maze.grid[r][c]
            for nb in maze.get_accessible_neighbors(cell):
                nb_coord = (nb.row, nb.col)
                tentative_g = g_score[current] + 1
                if tentative_g < g_score.get(nb_coord, float('inf')):
                    came_from[nb_coord] = current
                    g_score[nb_coord] = tentative_g
                    f_score[nb_coord] = tentative_g + heuristic(nb_coord, goal)
                    open_set.add(nb_coord)
                    if nb_coord not in drawn_explored:
                        drawn_explored.add(nb_coord)
                        draw_cell_explored(canvas, nb.row, nb.col, "light blue")
            count += 1
        canvas.after(400, step)
    def animate_path(came_from, goal):
        if cancel_control["cancelled"]:
            return
        path = []
        cur = goal
        while cur is not None:
            path.append(cur)
            cur = came_from.get(cur)
        path.reverse()
        def draw_path_step(index):
            if cancel_control["cancelled"]:
                return
            if index < len(path):
                r, c = path[index]
                draw_cell_explored(canvas, r, c, "red")
                canvas.after(1, lambda: draw_path_step(index+1))
            else:
                draw_direct_blue_path(canvas, maze, path)
                if callback:
                    callback()
        draw_path_step(0)
    step()

def heuristic(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

# -------------------- 主界面 --------------------
class MazeApp:
    def __init__(self, master):
        self.master = master
        master.title("迷宫生成与寻路")
        
        # 使用两个框架分隔上半部分（迷宫和控制区）与下半部分（算法说明区）
        self.top_frame = tk.Frame(master)
        self.top_frame.grid(row=0, column=0, sticky="nsew")
        self.bottom_frame = tk.Frame(master)
        self.bottom_frame.grid(row=1, column=0, sticky="ew")
        
        # 左侧：迷宫画布
        self.canvas = tk.Canvas(self.top_frame, width=MAZE_COLS * CELL_SIZE, height=MAZE_ROWS * CELL_SIZE, bg="white")
        self.canvas.grid(row=0, column=0)
        
        # 右侧：控制区
        self.control_frame = tk.Frame(self.top_frame)
        self.control_frame.grid(row=0, column=1, sticky="ns", padx=10, pady=10)
        
        # 生成算法相关控件
        self.algorithm = tk.StringVar(value="DFS")
        label_gen = tk.Label(self.control_frame, text="选择生成算法：")
        label_gen.grid(row=0, column=0, sticky="w", pady=5)
        gen_algorithms = [
            ("深度优先搜索 (DFS)", "DFS"),
            ("随机化 Kruskal 算法", "Kruskal"),
            ("随机化 Prim 算法", "Prim"),
            ("递归分割算法", "RecursiveDivision")
        ]
        for i, (text, mode) in enumerate(gen_algorithms):
            tk.Radiobutton(self.control_frame, text=text, variable=self.algorithm, value=mode,
                           command=self.update_algorithm_explanations).grid(row=i+1, column=0, sticky="w", pady=2)
        self.generate_button = tk.Button(self.control_frame, text="生成迷宫", command=self.generate_maze)
        self.generate_button.grid(row=5, column=0, sticky="we", pady=10)
        
        # 分隔线
        separator = tk.Label(self.control_frame, text="-------------------")
        separator.grid(row=6, column=0, pady=5)
        
        # 寻路算法相关控件
        self.path_algo = tk.StringVar(value="BFS")
        label_path = tk.Label(self.control_frame, text="选择寻路算法：")
        label_path.grid(row=7, column=0, sticky="w", pady=5)
        path_algorithms = [
            ("广度优先搜索 (BFS)", "BFS"),
            ("深度优先搜索 (DFS)", "DFS"),
            ("A* 算法", "A*")
        ]
        for i, (text, mode) in enumerate(path_algorithms):
            tk.Radiobutton(self.control_frame, text=text, variable=self.path_algo, value=mode,
                           command=self.update_algorithm_explanations).grid(row=8+i, column=0, sticky="w", pady=2)
        self.find_path_button = tk.Button(self.control_frame, text="寻路", command=self.find_path)
        self.find_path_button.grid(row=11, column=0, sticky="we", pady=10)
        
        # 下方：固定大小的文本框用于显示算法说明，设置自动换行
        self.alg_text = tk.Text(self.bottom_frame, width=80, height=5, wrap="word")
        self.alg_text.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        # 设置为只读状态
        self.alg_text.config(state=tk.DISABLED)
        
        # 初始化迷宫和绘制
        self.maze = Maze(MAZE_ROWS, MAZE_COLS)
        self.maze.draw(self.canvas)
        
        # 初始化说明文本内容
        self.update_algorithm_explanations()
        
        self.search_in_progress = False
        self.search_control = None  # 用于控制动态寻路取消

    def update_algorithm_explanations(self):
        # 根据当前选择更新生成和寻路算法的说明
        gen_mode = self.algorithm.get()
        path_mode = self.path_algo.get()
        gen_desc = gen_descriptions.get(gen_mode, "")
        path_desc = path_descriptions.get(path_mode, "")
        text = "生成算法说明: " + gen_desc + "\n\n" + "寻路算法说明: " + path_desc
        self.alg_text.config(state=tk.NORMAL)
        self.alg_text.delete("1.0", tk.END)
        self.alg_text.insert(tk.END, text)
        self.alg_text.config(state=tk.DISABLED)

    def generate_maze(self):
        # 如果有正在进行的寻路动画，则取消它
        if self.search_control is not None:
            self.search_control["cancelled"] = True
            self.search_in_progress = False
            self.find_path_button.config(state="normal")
        # 禁用生成和寻路按钮，防止在生成过程中点击
        self.generate_button.config(state="disabled")
        self.find_path_button.config(state="disabled")
        self.maze = Maze(MAZE_ROWS, MAZE_COLS)
        self.maze.draw(self.canvas)
        def finish_callback():
            self.maze.grid[0][0].walls['top'] = False
            self.maze.grid[MAZE_ROWS-1][MAZE_COLS-1].walls['bottom'] = False
            self.maze.draw(self.canvas)
            # 迷宫生成完成后重新启用按钮
            self.generate_button.config(state="normal")
            self.find_path_button.config(state="normal")
        algo = self.algorithm.get()
        if algo == "DFS":
            generate_maze_dfs(self.maze, self.canvas, finish_callback)
        elif algo == "Kruskal":
            generate_maze_kruskal(self.maze, self.canvas, finish_callback)
        elif algo == "Prim":
            generate_maze_prim(self.maze, self.canvas, finish_callback)
        elif algo == "RecursiveDivision":
            generate_maze_recursive_division(self.maze, self.canvas, finish_callback)

    def find_path(self):
        if self.search_in_progress:
            return
        self.search_in_progress = True
        self.find_path_button.config(state="disabled")
        self.search_control = {"cancelled": False}
        algo = self.path_algo.get()
        def search_finished():
            self.search_in_progress = False
            self.find_path_button.config(state="normal")
        if algo == "BFS":
            dynamic_solve_bfs(self.maze, self.canvas, self.search_control, callback=search_finished)
        elif algo == "DFS":
            dynamic_solve_dfs(self.maze, self.canvas, self.search_control, callback=search_finished)
        elif algo == "A*":
            dynamic_solve_astar(self.maze, self.canvas, self.search_control, callback=search_finished)

if __name__ == "__main__":
    root = tk.Tk()
    app = MazeApp(root)
    root.mainloop()
