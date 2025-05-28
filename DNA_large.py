import numpy as np
from bisect import bisect_right
from typing import List, Tuple, Dict
from collections import defaultdict
from sequence import REFERENCE, QUERY
import heapq
import tracemalloc


def show_top(snapshot, label):
    print(f"--- {label} ---")
    top10 = snapshot.statistics('lineno')[:10]
    for stat in top10:
        print(stat)
    current, peak = tracemalloc.get_traced_memory()
    print(f"[{label}] Current={current/1024**2:.1f}MB; Peak={peak/1024**2:.1f}MB\n")

seg_dtype = np.dtype([
    ('x0',       np.int32),  # 起点 x
    ('y0',       np.int32),  # 起点 y
    ('length',   np.int32),  # 段长度
    ('direction',np.int8 ),  # 方向：1 主对角 / 0 副对角
    ('distance', np.int32)   # 错误数
])

# 判断两碱基是否互补
def is_complement(a: str, b: str) -> bool:
    """A-T, T-A, C-G, G-C 是互补配对"""
    return (a == 'A' and b == 'T') or \
           (a == 'T' and b == 'A') or \
           (a == 'C' and b == 'G') or \
           (a == 'G' and b == 'C')

# 1. 向量化 dot-plot 构建

def build_dotplot(query: str, reference: str) -> np.ndarray:
    """
    第一步：构建 dot-plot 网格
    - 坐标系：x 轴是 query（长度 M），y 轴是 reference（长度 N）
    - 相同：标 'B'（蓝点），互补：标 'R'（红点），否则空字符串
    """
    M, N = len(query), len(reference)
    grid = np.zeros((M, N), dtype=np.uint8)
    for x, qb in enumerate(query):
        for y, rb in enumerate(reference):
            if qb == rb:
                grid[x][y] = 1
            elif is_complement(qb, rb):
                grid[x][y] = 2
            else:
                grid[x][y] = 0
    return grid

def init_diag_tables(grid):
    """
    根据 grid 初始化四个对角线方向的累积计数表：
      main_ld  - 主对角线向左下（i-1, j-1）方向累积
      main_ru  - 主对角线向右上（i+1, j+1）方向累积
      anti_lu  - 副对角线向左上（i-1, j+1）方向累积
      anti_rd  - 副对角线向右下（i+1, j-1）方向累积

    grid: M x N 的字符矩阵，非空字符串表示有色点
    返回 (main_ld, main_ru, anti_lu, anti_rd)，它们都是 M x N 的整数矩阵
    """
    M = len(grid)
    N = len(grid[0]) if M > 0 else 0

    # 初始化四个表
    main_ld = np.zeros((M, N), dtype=np.int16)
    anti_lu = np.zeros((M, N), dtype=np.int16)

    # main_ld: 从上到下、从左到右
    for i in range(M):
        for j in range(N):
            cnt = 1 if grid[i][j]== 1 else 0
            if i-1 >= 0 and j-1 >= 0:
                cnt += main_ld[i-1][j-1]
            main_ld[i][j] = cnt


    # anti_lu: 从上到下、从右到左
    for i in range(M):
        for j in range(N-1, -1, -1):
            cnt = 1 if grid[i][j]== 2 else 0
            if i-1 >= 0 and j+1 < N:
                cnt += anti_lu[i-1][j+1]
            anti_lu[i][j] = cnt

    return main_ld, anti_lu
class Segment:
    """表示一个对角线或副对角线上的连续匹配段"""
    def __init__(self, x0: int, y0: int, length: int, direction: str, distance: int):
        """
        x0, y0: 起点坐标
        length: 段长度（点的数量）
        direction: 'main' 或 'anti'
        """
        self.x0 = x0
        self.y0 = y0
        self.length = length
        self.direction = direction
        self.distance = distance
        # 计算终点坐标
        self.x1 = x0 + length - 1
        if direction == 'main':
            self.y1 = y0 + length - 1
        else:
            self.y1 = y0 - (length - 1)

    def __repr__(self):
        if self.direction == "main":
            # return f"({self.x0},{self.x1 + 1},{self.y0},{self.y1 + 1}, {1})"
            return f"({self.x0},{self.x1 + 1},{self.y0},{self.y1 + 1})"
        else:
            # return f"({self.x0},{self.x1 + 1},{self.y1},{self.y0 + 1}, {0})"
            return f"({self.x0},{self.x1 + 1},{self.y1},{self.y0 + 1})"
        
    def update_endpoints(self):
        self.x1 = self.x0 + self.length - 1
        if self.direction == 'main':
            self.y1 = self.y0 + self.length - 1
        else:
            self.y1 = self.y0 - (self.length - 1)

    def update_startpoints(self):
        self.x0 = self.x1 - self.length - 1
        if self.direction == 'main':
            self.y0 = self.y1 - self.length + 1
        else:
            self.y0 = self.y1 + (self.length - 1)

    def recalc_distance(self):
        if self.direction == 'main':
            prev = main_ld[self.x0-1, self.y0-1] if self.x0 > 0 and self.y0 > 0 else 0
            correct = main_ld[self.x1, self.y1] - prev
        else:
            prev = anti_lu[self.x0-1, self.y0+1] if self.x0 > 0 and self.y0+1 < anti_lu.shape[1] else 0
            correct = anti_lu[self.x1, self.y1] - prev
        self.distance = self.length - correct

def find_diagonal_segments_np(grid: np.ndarray) -> np.ndarray:
    M, N = grid.shape
    records = []
    # 定义 run-length 提取函数
    def extract_runs(line):
        mask = (line != 0).astype(np.int8)  # 转成 0/1 数组
        padded = np.concatenate(([0], mask, [0]))  # 在两端填充 0
        diff = np.diff(padded)
        starts = np.where(diff==1)[0]
        ends = np.where(diff==-1)[0]
        return starts, ends

    # 主对角线
    for d in range(-M+1, N):
        diag = np.diagonal(grid, offset=d)  # 获取对角线
        starts, ends = extract_runs(diag)
        base_x = max(0, -d)
        for s,e in zip(starts, ends):
            length = e-s
            if length>0:
                x0 = base_x + s
                y0 = x0 + d
                records.append((x0, y0, length, 1, 0))  # 暂不计算 distance
    # 副对角线
    flipped = np.fliplr(grid)
    for d in range(-M+1, N):
        diag = np.diagonal(flipped, offset=d)
        starts, ends = extract_runs(diag)
        base_x = max(0, -d)
        for s,e in zip(starts, ends):
            length = e-s
            if length>0:
                x0 = base_x + s
                y0 = N-1 - (base_x + s + d)
                records.append((x0, y0, length, 0, 0))
    # 一次性构造结构化数组，节省多次 append 的开销
    if not records:
        return np.empty(0, dtype=seg_dtype)
    segs = np.array(records, dtype=seg_dtype)
    
    return segs
def find_diagonal_segments(grid: np.ndarray) -> np.ndarray:
    """
    第二步准备：在主对角线（右下方向）和副对角线（右上方向）上
    寻找所有连续的匹配段（不做阈值过滤）
    """
    M, N = len(grid), len(grid[0])
    segs: List[Segment] = []

    # 主对角线方向（dx=+1, dy=+1）
    for x in range(M):
        for y in range(N):
            if grid[x][y] and (x == 0 or y == 0 or grid[x-1][y-1] == 0):
                # 遇到一个新的段起点
                length = 0
                xx, yy = x, y
                while xx < M and yy < N and grid[xx][yy]:
                    length += 1
                    xx += 1; yy += 1
                segs.append(Segment(x, y, length, 'main', 0))

    # 副对角线方向（dx=+1, dy=-1）
    for x in range(M):
        for y in range(N):
            if grid[x][y] and (x == 0 or y == N-1 or grid[x-1][y+1] == 0):
                length = 0
                xx, yy = x, y
                while xx < M and yy >= 0 and grid[xx][yy]:
                    length += 1
                    xx += 1; yy -= 1
                segs.append(Segment(x, y, length, 'anti', 0))

    return segs


def find_segments_in_range_np(x_start: int,
                              x_end:   int) -> np.ndarray:
    """
    NumPy 版：在 query x ∈ [x_start, x_end] 范围内，
    提取所有主/副对角线连续匹配段，返回结构化数组。
    """
    M, N = grid.shape
    # 临时收集所有 segment 信息的普通 Python 列表
    recs: List[Tuple[int,int,int,int,int]] = []

    # --- 主对角线 (direction=1) ---
    for x in range(max(0, x_start), min(M, x_end+1)):
        for y in range(N):
            # 找到一个新段的起点
            if grid[x, y] == 1 and (x == 0 or y == 0 or grid[x-1, y-1] != 1):
                length = 0
                xx, yy = x, y
                # 顺着 (dx=+1, dy=+1) 扩展
                while xx < M and yy < N and grid[xx, yy] == 1:
                    length += 1
                    xx += 1; yy += 1

                # 截断到 x_end
                end_x, end_y = xx-1, yy-1
                if end_x > x_end:
                    end_x = x_end
                    end_y = y + (x_end - x)
                    length = end_x - x + 1

                # 计算 distance（错误数）
                prev = main_ld[x-1, y-1] if x>0 and y>0 else 0
                correct = main_ld[end_x, end_y] - prev
                distance = length - correct

                # 收集一条记录 (x0, y0, length, direction, distance)
                recs.append((x, y, length, 1, distance))

    # --- 副对角线 (direction=0) ---
    for x in range(max(0, x_start), min(M, x_end+1)):
        for y in range(N):
            if grid[x, y] == 2 and (x == 0 or y == N-1 or grid[x-1, y+1] != 2):
                length = 0
                xx, yy = x, y
                # 顺着 (dx=+1, dy=-1) 扩展
                while xx < M and yy >= 0 and grid[xx, yy] == 2:
                    length += 1
                    xx += 1; yy -= 1

                # 截断到 x_end
                end_x, end_y = xx-1, yy+1
                if end_x > x_end:
                    end_x = x_end
                    end_y = y - (x_end - x)
                    length = end_x - x + 1

                # 计算 distance
                prev = anti_lu[x-1, y+1] if x>0 and y+1<N else 0
                correct = anti_lu[end_x, end_y] - prev
                distance = length - correct

                recs.append((x, y, length, 0, distance))

    # 如果一个都没找到，直接返回空结构化数组
    if not recs:
        return np.empty(0, dtype=seg_dtype)

    # 一次性打包成 NumPy 结构化数组
    segs = np.array(recs, dtype=seg_dtype)

    # 调用你之前改好的 merge_with_tolerance（接受 np.ndarray）
    segs = merge_with_tolerance_np(segs,
                                max_gap=5,
                                max_error_rate=0.095)
    return segs
def find_segments_in_range(x_start: int, x_end: int) -> List['Segment']:
    """
    在 query x 范围 [x_start, x_end] 内，提取所有主/副对角线连续匹配段。
    """
    M, N = len(grid), len(grid[0])
    segs = []
    # 主对角线
    for x in range(max(0, x_start), min(M, x_end+1)):
        for y in range(N):
            if grid[x][y] == 1 and (x == 0 or y == 0 or grid[x-1][y-1] != 1):
                length = 0
                xx, yy = x, y
                while xx < M and yy < N and grid[xx][yy] == 1:
                    length += 1
                    xx += 1; yy += 1

                end_x, end_y = xx-1, yy-1
                if xx-1 > x_end:
                    end_x = x_end
                    end_y = y + (x_end - x)
                    length = end_x - x + 1

                prev = main_ld[x][y] if x>0 and y>0 else 0
                affective_distance = main_ld[end_x][end_y] - prev
                distance = length - affective_distance
                segs.append(Segment(x, y, length, 'main', distance))
                
    # 副对角线
    for x in range(max(0, x_start), min(M, x_end+1)):
        for y in range(N):
            if grid[x][y] == 2 and (x == 0 or y == N-1 or grid[x-1][y+1] != 2):
                length = 0
                xx, yy = x, y
                while xx < M and yy >= 0 and grid[xx][yy] == 2:
                    length += 1
                    xx += 1; yy -= 1

                end_x, end_y = xx-1, yy+1
                if xx-1 > x_end:
                    end_x = x_end
                    end_y = y - (x_end - x)
                    length = end_x - x + 1

                prev = anti_lu[x][y] if x>0 and y+1<len(grid[0]) else 0
                affective_distance = anti_lu[end_x][end_y] - prev
                distance = length - affective_distance
                segs.append(Segment(x, y, length, 'anti', distance))

    segs = merge_with_tolerance(segs,
                                    max_gap=5,
                                    max_error_rate=0.095)
    # segs = minimal_interval_cover2(segs)
    # print(segs)
    return segs

############################################  WRONG  #########################################################

def weighted_interval_scheduling_np(segs: np.ndarray) -> np.ndarray:
    # 提取字段
    x0 = segs['x0']
    length = segs['length']
    x1 = x0 + length - 1
    # 排序索引
    order = np.argsort(x1, kind = 'mergesort')
    x0s = x0[order]
    x1s = x1[order]
    lens = length[order]
    n = len(lens)
    # 计算 p
    p = np.searchsorted(x1s, x0s) - 1
    dp = np.zeros(n+1, dtype=int)
    choose = np.zeros(n, dtype=bool)
    for j in range(1, n+1):
        w = lens[j-1]
        if w + dp[p[j-1]+1] > dp[j-1]:
            dp[j] = w + dp[p[j-1]+1]
            choose[j-1] = True
        else:
            dp[j] = dp[j-1]
    # 回溯
    res_idx = []
    j = n
    while j>0:
        if choose[j-1]:
            res_idx.append(order[j-1])  # 原始索引
            j = p[j-1] + 1
        else:
            j -= 1
    res_idx.reverse()
    return segs[res_idx]  # 返回原数组切片
# 在一组可选区间 segs中，挑选一组互不重叠的区间，使得它们的总“权重”最大
def weighted_interval_scheduling(segs: List[Segment]) -> List[Segment]:
    segs = sorted(segs, key=lambda s: s.x1)
    ends = [s.x1 for s in segs]
    n = len(segs)
    p = [bisect_right(ends, segs[j].x0 - 1) - 1 for j in range(n)]
    dp = [0]*(n+1)
    choose = [False]*n
    for j in range(1, n+1):
        w = segs[j-1].length
        if w + dp[p[j-1]+1] > dp[j-1]:
            dp[j] = w + dp[p[j-1]+1]
            choose[j-1] = True
        else:
            dp[j] = dp[j-1]
    res = []
    j = n
    while j>0:
        if choose[j-1]:
            res.append(segs[j-1])
            j = p[j-1]+1
        else:
            j -= 1
    return list(reversed(res))


def merge_in_blanks_np(segs: np.ndarray, rate: float) -> np.ndarray:
    """
    NumPy 版 merge_in_blanks：
      - segs: np.ndarray，dtype=seg_dtype
      - rate: 合并容错阈值
    保持原来线性扫描合并的逻辑不变，只是用结构化数组替代 Segment 对象。
    """
    # 0. 如果没有任何 segment，直接返回空
    if segs.size == 0:
        return segs

    # 1. 先按 x0 升序（与你原来遍历顺序保持一致）
    order = np.argsort(segs['x0'], kind = 'mergesort')
    segs = segs[order]

    # 2. 取字段到本地变量，方便后面多次访问
    x0_arr   = segs['x0']
    y0_arr   = segs['y0']
    len_arr  = segs['length']
    dir_arr  = segs['direction']
    dist_arr = segs['distance']

    # 3. 预计算每条记录的 x1, y1（终点坐标）
    x1_arr = x0_arr + len_arr - 1
    y1_arr = np.where(
        dir_arr == 1,
        y0_arr + len_arr - 1,
        y0_arr - (len_arr - 1)
    )

    merged = []  # 用 Python list 临时收集结果

    # 4. 初始化 prev 为第一个 segment
    prev_x0 = int(x0_arr[0])
    prev_y0 = int(y0_arr[0])
    prev_len = int(len_arr[0])
    prev_dir = int(dir_arr[0])
    prev_dist = int(dist_arr[0])
    prev_x1 = int(x1_arr[0])
    prev_y1 = int(y1_arr[0])

    # 5. 从第二个开始，逐条尝试与 prev 合并
    for i in range(1, segs.size):
        sx0   = int(x0_arr[i])
        sy0   = int(y0_arr[i])
        slen  = int(len_arr[i])
        sdir  = int(dir_arr[i])
        sdist = int(dist_arr[i])
        sx1   = int(x1_arr[i])
        sy1   = int(y1_arr[i])

        if sdir == prev_dir:
            # 6. 只在同一个方向上才考虑合并
            same_diag = (
                (sdir == 1 and (sy0 - sx0) == (prev_y0 - prev_x0)) or
                (sdir == 0 and (sx0 + sy0) == (prev_x0 + prev_y0))
            )
            if same_diag:
                # 7. 计算合并后长度 new_len
                new_len = sx1 - prev_x0 + 1
                # 8. 计算合并后的错误数 affective_distance
                if prev_dir == 1:
                    prev_count = main_ld[prev_x0 - 1, prev_y0 - 1] if prev_x0 > 0 and prev_y0 > 0 else 0
                    affective = main_ld[sx1, sy1] - prev_count
                else:
                    prev_count = anti_lu[prev_x0 - 1, prev_y0 + 1] if prev_x0 > 0 and prev_y0 + 1 < grid.shape[1] else 0
                    affective = anti_lu[sx1, sy1] - prev_count
                distance = new_len - affective

                # 9. 判断是否满足 error_rate < rate
                if distance / new_len < rate:
                    # —— 合并：只更新 prev_* 相关值，不立即输出 —— #
                    prev_len = new_len
                    prev_dist = distance
                    prev_x1 = sx1
                    prev_y1 = sy1
                    continue  # 下一个 seg 继续尝试跟扩展后的 prev 合并

        # 10. 如果不能合并，先把 prev 输出，再把当前 s 设成新 prev
        merged.append((prev_x0, prev_y0, prev_len, prev_dir, prev_dist))
        prev_x0, prev_y0 = sx0, sy0
        prev_len, prev_dir, prev_dist = slen, sdir, sdist
        prev_x1, prev_y1 = sx1, sy1

    # 11. 循环结束后，别忘了输出最后一个 prev
    merged.append((prev_x0, prev_y0, prev_len, prev_dir, prev_dist))

    # 12. 打包成结构化数组返回
    return np.array(merged, dtype=segs.dtype)
def merge_in_blanks(segs: List['Segment'], rate) -> List['Segment']:
    prev_seg = Segment(0, 0, 0, "main", 0)
    result = []
    for seg in segs:
        if seg.direction == prev_seg.direction:
            new_length = seg.x1 - prev_seg.x0 + 1
            if seg.direction == 'main' and seg.x0-seg.y0 == prev_seg.x0-prev_seg.y0:
                prev = main_ld[prev_seg.x0-1][prev_seg.y0-1] if prev_seg.x0>0 and prev_seg.y0>0 else 0
                affective_distance = main_ld[seg.x1][seg.y1] - prev
            elif seg.direction == 'anti' and seg.x0+seg.y0 == prev_seg.x0+prev_seg.y0:
                prev = anti_lu[prev_seg.x0-1][prev_seg.y0+1] if prev_seg.x0>0 and prev_seg.y0+1<len(grid[0]) else 0
                affective_distance = anti_lu[seg.x1][seg.y1] - prev
            else:
                result.append(seg)
                prev_seg = seg
                continue
            distance = new_length - affective_distance
            if distance/new_length < rate:
                result.append(Segment(prev_seg.x0, prev_seg.y0, new_length, seg.direction, distance))
                if prev_seg in result:
                    result.remove(prev_seg)
            else:
                result.append(seg)
        else:
            result.append(seg)
        prev_seg = seg
    
    return result


############################################  CHANGED  #########################################################
# # 对每条对角线在 [x_start,x_end] 范围内做 DP & 区间调度
def find_large_valid_segments_in_range_np(x_start: int,
                                       x_end:   int,
                                       rate:    float,
                                       min_len: int = 30) -> np.ndarray:
    """
    在 query x ∈ [x_start, x_end] 范围内，用双指针＋DP 找到各条主/副对角线上的误差率 ≤ rate
    且长度 ≥ min_len 的最优匹配片段，返回 dtype=seg_dtype 的结构化数组。
    """
    M, N = grid.shape

    recs = []  # 用 Python 列表先收集所有候选的 (x0,y0,length,direction,errs)

    groups: Dict[Tuple[int,int], np.ndarray] = {}
    # 1) 构造每条主对角线的点列表
    for d in range(-(M-1), N):
        lo = max(0, -d, x_start)
        hi = min(M-1, N-1-d, x_end)
        L  = hi - lo + 1
        # if L >= min_len:
        if 1:
            xs = np.arange(lo, hi+1)
            ys = xs + d
            groups[(1, d)] = np.stack((xs, ys), axis=1)

    # 2) 构造每条副对角线的点列表
    for d in range(0, M+N-1):
        lo = max(0, d-(N-1), x_start)
        hi = min(M-1, d, x_end)
        L  = hi - lo + 1
        # if L >= min_len:
        if 1:
            xs = np.arange(lo, hi+1)
            ys = d - xs
            groups[(0, d)] = np.stack((xs, ys), axis=1)

    # 3) 对每条线做双指针 + DP
    for (direction, _), pts in groups.items():
        L = pts.shape[0]

        # 3.1) 计算前缀错误计数 mis[k] = pts[:k] 中不匹配的点数
        mis = np.zeros(L+1, dtype=int)
        # 这里用 vectorized 一次性比较 grid[pts[:,0], pts[:,1]]
        if direction == 1:
            mask = (grid[pts[:,0], pts[:,1]] != 1)
        else:
            mask = (grid[pts[:,0], pts[:,1]] != 2)
        mis[1:] = np.cumsum(mask)

        # 3.2) 标准 DP 数组和回溯指针
        dp       = np.zeros(L+1, dtype=int)
        prev_idx = np.zeros(L+1, dtype=int)
        L_ptr    = 0

        for R in range(1, L+1):
            # 保证从 L_ptr 到 R-1 的错误率 ≤ rate
            while L_ptr < R and (mis[R] - mis[L_ptr]) > rate * (R - L_ptr):
                L_ptr += 1
            # 不选第 R 段
            dp[R] = dp[R-1]
            prev_idx[R]  = R-1
            # 选第 R 段 (长度 = R - L_ptr)
            if R - L_ptr >= min_len:
                length = R - L_ptr
                if dp[L_ptr] + length > dp[R]:
                    dp[R] = dp[L_ptr] + length
                    prev_idx[R] = L_ptr

        # 3.3) 回溯选出的最佳子区间
        idx = L
        while idx > 0:
            pi = prev_idx[idx]
            if pi < idx - 1:
                start = pi
                end   = idx - 1
                x0, y0 = pts[start]
                length = end - start + 1
                errs   = mis[end+1] - mis[start]
                recs.append((int(x0), int(y0), int(length), direction, int(errs)))
                idx = pi
            else:
                idx -= 1

    # 4) 打包成结构化 NumPy 数组返回
    if recs:
        return np.array(recs, dtype=seg_dtype)
    else:
        return np.empty(0, dtype=seg_dtype)
    
def find_large_valid_segments_in_range(x_start: int, x_end: int,
                                       rate:float,
                                       min_len: int=30) -> List[Segment]:
    M, N = len(grid), len(grid[0])
    results: List[Segment] = []
    groups: Dict[Tuple[str,int], List[Tuple[int,int]]] = {}
    # 构造每条对角线的点列表 pts，仅保留 x∈[x_start,x_end]
    for d in range(-(M-1), N):
        pts = []
        lo = max(0, -d, x_start)
        hi = min(M-1, N-1-d, x_end)
        for x in range(lo, hi+1):
            y = x + d
            pts.append((x,y))
        if len(pts) >= min_len:
            groups[("main",d)] = pts
    for d in range(0, M+N-1):
        pts = []
        lo = max(0, d-(N-1), x_start)
        hi = min(M-1, d, x_end)
        for x in range(lo, hi+1):
            y = d - x
            pts.append((x,y))
        if len(pts) >= min_len:
            groups[("anti",d)] = pts

    # 对每条线做双指针+DP
    for (direction, d), pts in groups.items():
        L = len(pts)
        mis = [0]*(L+1)
        for i,(x,y) in enumerate(pts):
            if direction == "main":
                mis[i+1] = mis[i] + (1 if grid[x][y] != 1 else 0)
            else:
                mis[i+1] = mis[i] + (1 if grid[x][y] != 2 else 0)
        dp = [0]*(L+1)
        prev_idx = [0]*(L+1)
        L_ptr = 0
        for R in range(1, L+1):
            while L_ptr<R and (mis[R]-mis[L_ptr]) > rate*(R-L_ptr):
                L_ptr += 1
            dp[R] = dp[R-1]
            prev_idx[R] = R-1
            if R-L_ptr >= min_len:
                length = R - L_ptr
                if dp[L_ptr] + length > dp[R]:
                    dp[R] = dp[L_ptr] + length
                    prev_idx[R] = L_ptr
        # 回溯
        idx = L
        while idx>0:
            pi = prev_idx[idx]
            if pi < idx-1:
                start,end = pi, idx-1
                x0,y0 = pts[start]
                length = end-start+1
                errs = mis[end+1] - mis[start]
                results.append(Segment(x0,y0,length,direction,errs))
                idx = pi
            else:
                idx -= 1
    # print("start, end, results = ", x_start, " ", x_end, " ", results)
    return results

# 全局填补空白并合并相邻段

def fill_in_blanks_global_np(segs: np.ndarray,
                          rate: float,
                          min_gap: int = 30) -> np.ndarray:
    """
    NumPy 版：全局填补空白并合并相邻段
    segs: np.ndarray, dtype=seg_dtype
    返回同dtype的 np.ndarray
    """
    # 如果输入为空，直接返回空结构化数组
    if segs.size == 0:
        return np.empty(0, dtype=seg_dtype)
    
    # 按 x0 升序排序 (字段访问)  # 新增：对结构化数组按 'x0' 字段排序
    order = np.argsort(segs['x0'], kind = 'mergesort')  # 新增：获取排序索引
    sorted_segs = segs[order]       # 新增：生成排序后的数组

    # 用 Python 列表暂存中间结果 (tuple 格式)
    recs = []  # 新增：临时存储所有段的 tuple

    # 遍历每对相邻段，计算空隙并填充
    for prev, curr in zip(sorted_segs, sorted_segs[1:]):  # 新增：按相邻对迭代
        recs.append((int(prev['x0']), int(prev['y0']), int(prev['length']), int(prev['direction']), int(prev['distance'])))  # 新增：加入前一段的 tuple
        # prev.x1 = prev.x0 + prev.length - 1, 所以 g0 = prev.x1+1 = prev.x0 + prev.length
        g0 = int(prev['x0'] + prev['length'])  # 新增：空隙起点
        g1 = int(curr['x0'] - 1)               # 新增：空隙终点
        gap_len = g1 - g0 + 1                  # 新增：计算空隙长度

        if gap_len >= min_gap:  # 新增：只有当 gap_len ≥ min_gap 时才进行补段
            
            # 子区间 DP 调度，返回 numpy 结构化数组  # 已修改：调用 numpy 版 find_large_valid_segments_in_range
            extras = find_large_valid_segments_in_range_np(g0, g1, rate, min_gap)
            # 如果没有任何补段，再退回到基本提取函数
            # if extras.size == 0:
            #     extras = find_segments_in_range_np(g0, g1)  # 新增：fallback
            # 将 extras 逐条加入 recs
            for e in extras:
                recs.append((int(e['x0']), int(e['y0']), int(e['length']), int(e['direction']), int(e['distance'])))  # 新增：拆解结构化元素为 tuple

            # 补完后再加入 curr
            # recs.append((int(curr['x0']), int(curr['y0']), int(curr['length']), int(curr['direction']), int(curr['distance'])))  # 新增：加入当前段

    # 别忘了把最后一个段也加上  # 新增：添加尾段
    last = sorted_segs[-1]
    recs.append((int(last['x0']), int(last['y0']), int(last['length']), int(last['direction']), int(last['distance'])))  # 新增：加入最后一段

    # # 一次性打包为结构化数组  # 新增：tuple 列表转换为 np.ndarray
    final = np.array(recs, dtype=seg_dtype)

    # 合并相邻同向段  # 已修改：调用 numpy 版 merge_with_tolerance
    
    final = merge_in_blanks_np(final, rate) 
    # get_detail_np(final)
    # 区间调度，选出最优覆盖  # 已修改：调用 numpy 版 weighted_interval_scheduling
    final = weighted_interval_scheduling_np(final) # !!!!WRONG

    return final  # 返回 dtype=seg_dtype 的 np.ndarray
def fill_in_blanks_global(segs: List[Segment],
                          rate: float,
                          min_gap: int=30) -> List[Segment]:
    segs = sorted(segs, key=lambda s: s.x0)
    filled: List[Segment] = []
    for prev, curr in zip(segs, segs[1:]):
        filled.append(prev)
        g0, g1 = prev.x1+1, curr.x0-1
        gap_len = g1 - g0 + 1
        if gap_len >= min_gap:
            # 子区间 DP 调度（不分方向）
            extras = find_large_valid_segments_in_range(g0, g1, rate, min_gap)
            # extras = find_segments_in_range(g0, g1)
            if extras == []:
                extras = find_segments_in_range(g0, g1)
            filled.extend(extras)
            filled.append(curr)
    if segs:
        filled.append(segs[-1])
    # 最后合并相邻同向段
    filled = merge_in_blanks(filled, rate)
    # filled = dedupe_by_start_max_length(filled)
    final = weighted_interval_scheduling(filled)
    # final = minimal_interval_cover2(filled)
    return final

def extend_end_backward_np(segs: np.ndarray, rate: float) -> np.ndarray:
    """
    将原来基于 Segment 对象的 extend_end_backward 改写为
    接受并返回结构化数组版本，等价功能、同样复杂度。
    """
    if segs.size == 0:
        return segs  # 空数组直接返回

    # 1. 先按 x0 倒序，相当于原来传进来时 segments 已经是倒序
    order_desc = np.argsort(segs['x0'], kind = 'mergesort')[::-1]
    arr = segs[order_desc]

    M, N = grid.shape
    out_records = []
    # 2. 记录上一个 segment 在 query 上的结束 +1
    #    原版是 segments[-1].x1+1，倒序后就是 arr[0]
    x0 = arr[0]['x0']; length = arr[0]['length']
    prev_end = x0 + length

    # 3. 逐个处理
    for seg in arr:
        ox0, oy0, olen, odir = seg['x0'], seg['y0'], seg['length'], seg['direction']
        ox1 = ox0 + olen - 1
        oy1 = oy0 + (olen - 1) * (1 if odir == 1 else -1)

        target_end = prev_end - 1
        space = target_end - ox1

        if ox1 < target_end:
            # 4. 尝试把终点推到 target_end
            cand_x1 = target_end
            step = 1 if odir == 1 else -1
            cand_y1 = oy1 + step * space
            cand_len = olen + space

            # 5. 循环内边界检查 & 错配率检查
            while cand_x1 > ox1:
                # 5a. 越界就往回退一步
                if cand_x1 >= M or cand_y1 < 0 or cand_y1 >= N:
                    cand_x1 -= 1
                    cand_y1 -= step
                    cand_len -= 1
                    continue

                # 5b. 计算累积 correct
                if odir == 1:
                    prev_corr = main_ld[ox0-1, oy0-1] if ox0>0 and oy0>0 else 0
                    corr = main_ld[cand_x1, cand_y1] - prev_corr
                else:
                    prev_corr = anti_lu[ox0-1, oy0+1] if ox0>0 and oy0+1<N else 0
                    corr = anti_lu[cand_x1, cand_y1] - prev_corr

                dist = cand_len - corr
                # 5c. 错配率 OK 就停
                if dist / cand_len < rate:
                    break
                # 否则再退一步
                cand_x1 -= 1
                cand_y1 -= step
                cand_len -= 1

            # 6. 计算新的 distance
            if odir == 1:
                prev_corr = main_ld[ox0-1, oy0-1] if ox0>0 and oy0>0 else 0
                corr = main_ld[cand_x1, cand_y1] - prev_corr
            else:
                prev_corr = anti_lu[ox0-1, oy0+1] if ox0>0 and oy0+1<N else 0
                corr = anti_lu[cand_x1, cand_y1] - prev_corr

            new_dist = cand_len - corr
            # 7. 收集进输出
            out_records.append((ox0, oy0, cand_len, odir, int(new_dist)))
        else:
            # 8. 不需要扩展，保留原段
            out_records.append((ox0, oy0, olen, odir, int(seg['distance'])))

        prev_end = ox0  # 更新 prev_end 为当前段的 x0

    # 9. 打包成结构化数组，最后再按 x0 升序返回
    out_arr = np.array(out_records, dtype=seg_dtype)
    return out_arr[np.argsort(out_arr['x0'], kind = 'mergesort')]
def extend_end_backward(segments: List[Segment],
                          rate: float) -> List[Segment]:
    """
    对每一个 segments 中的 segment：
    - 记住上一个 segment 在 query 上结束的 x_end(prev_end)
    - 如果当前 segment 的 x0 与 prev_end+1 之间有 gap > 1，
      则尝试将当前 segment 的 x0 往前推到 prev_end+1，
      并调整 y0 与 length，确保 error_rate <= rate。
    """
    prev_end = segments[-1].x1 + 1
    new_segs = []
    M, N = grid.shape

    for seg in segments:

        target_end = prev_end - 1
        space = target_end - seg.x1
        if seg.x1 < target_end:
            # 尝试延申结尾位置
            cand_x1 = target_end
            direction = 1 if seg.direction == "main" else -1
            cand_y1 = (seg.y1 + direction * space)
            cand_length = seg.length + space
            # Move start backward step by step
            while cand_x1 > seg.x1:
                # ensure within bounds
                if cand_x1 >= M or cand_y1 < 0 or cand_y1 >= N:
                    cand_x1 -= 1
                    cand_y1 -= direction
                    cand_length -= 1
                    continue
                if seg.direction == 'main':
                    prev = main_ld[seg.x0-1, seg.y0-1] if seg.x0>0 and seg.y0>0 else 0
                    correct = main_ld[cand_x1, cand_y1] - prev
                else:
                    prev = anti_lu[seg.x0-1, seg.y0+1] if seg.x0>0 and seg.y0+1<N else 0
                    correct = anti_lu[cand_x1, cand_y1] - prev
                distance = cand_length - correct
                if distance / cand_length < rate:
                    break
                else:
                    cand_x1 -= 1
                    cand_y1 -= direction
                    cand_length -= 1
            # update seg
            seg.x1 = cand_x1
            seg.y1 = cand_y1
            seg.length = cand_length
            seg.recalc_distance()

        new_segs.append(seg)
        prev_end = seg.x0

    return new_segs

def extend_start_backward_np(segs: np.ndarray, rate: float) -> np.ndarray:
    """
    将原来基于 Segment 对象的 extend_start_backward 改写为
    接受并返回结构化数组版本，等价功能、同样复杂度。
    """
    if segs.size == 0:
        return segs

    # 1. 按 x0 升序（原版就是升序）
    order_asc = np.argsort(segs['x0'], kind = 'mergesort')
    arr = segs[order_asc]

    M, N = grid.shape
    out_records = []
    prev_end = -1  # 上一个段结束的位置

    # 2. 逐个处理
    for seg in arr:
        ox0, oy0, olen, odir = seg['x0'], seg['y0'], seg['length'], seg['direction']
        ox1 = ox0 + olen - 1
        oy1 = oy0 + (olen - 1) * (1 if odir == 1 else -1)

        target_start = prev_end + 1
        space = ox0 - target_start

        if ox0 > target_start:
            # 3. 尝试把起点推到 target_start
            cand_x0 = target_start
            step = 1 if odir == 1 else -1
            cand_y0 = oy0 - step * space
            cand_len = olen + space

            # 4. 循环内检查 & 错配率
            while cand_x0 < ox0:
                # 4a. 越界就停
                if cand_x0 < 0 or cand_y0 < 0 or cand_y0 >= N:
                    break

                # 4b. 计算累积 correct
                if odir == 1:
                    prev_corr = main_ld[cand_x0-1, cand_y0-1] if cand_x0>0 and cand_y0>0 else 0
                    corr = main_ld[cand_x0 + cand_len - 1, cand_y0 + cand_len - 1] - prev_corr
                else:
                    prev_corr = anti_lu[cand_x0-1, cand_y0+1] if cand_x0>0 and cand_y0+1<N else 0
                    corr = anti_lu[cand_x0 + cand_len - 1, cand_y0 - (cand_len - 1)] - prev_corr

                dist = cand_len - corr
                # 4c. 错配率 OK 就停
                if dist / cand_len < rate:
                    break
                # 否则再推进一步
                cand_x0 += 1
                cand_y0 += step
                cand_len -= 1

            # 5. 重新计算 distance
            if odir == 1:
                prev_corr = main_ld[cand_x0-1, cand_y0-1] if cand_x0>0 and cand_y0>0 else 0
                corr = main_ld[cand_x0 + cand_len - 1, cand_y0 + cand_len - 1] - prev_corr
            else:
                prev_corr = anti_lu[cand_x0-1, cand_y0+1] if cand_x0>0 and cand_y0+1<N else 0
                corr = anti_lu[cand_x0 + cand_len - 1, cand_y0 - (cand_len - 1)] - prev_corr

            new_dist = cand_len - corr
            # 6. 收集
            out_records.append((cand_x0, cand_y0, cand_len, odir, int(new_dist)))
        else:
            # 7. 不需要扩展，保留原段
            out_records.append((ox0, oy0, olen, odir, int(seg['distance'])))

        prev_end = out_records[-1][0] + out_records[-1][2] - 1  # 更新 prev_end 为新段的结束

    # 8. 最终打包并返回
    out_arr = np.array(out_records, dtype=seg_dtype)
    return out_arr
def extend_start_backward(segments: List[Segment],
                          rate: float) -> List[Segment]:
    """
    对每一个 segments 中的 segment：
    - 记住上一个 segment 在 query 上结束的 x_end(prev_end)
    - 如果当前 segment 的 x0 与 prev_end+1 之间有 gap > 1，
      则尝试将当前 segment 的 x0 往前推到 prev_end+1，
      并调整 y0 与 length，确保 error_rate <= rate。
    """
    prev_end = -1
    new_segs = []
    M, N = grid.shape

    for seg in segments:
        # print(seg.x0)
        # 计算 gap
        target_start = prev_end + 1
        space = seg.x0 - target_start
        if seg.x0 > target_start:
            # 尝试延申起始位置
            
            cand_x0 = target_start
            direction = 1 if seg.direction == "main" else -1
            cand_y0 = (seg.y0 - direction * space)
            cand_length = seg.length + space
            # Move start backward step by step
            while cand_x0 < seg.x0:
                # ensure within bounds
                if cand_x0 < 0 or cand_y0 < 0 or cand_y0 >= N:
                    break
                if seg.direction == 'main':
                    prev = main_ld[cand_x0-1, cand_y0-1] if cand_x0>0 and cand_y0>0 else 0
                    correct = main_ld[cand_x0 + cand_length - 1, cand_y0 + cand_length - 1] - prev
                else:
                    prev = anti_lu[cand_x0-1, cand_y0+1] if cand_x0>0 and cand_y0+1<N else 0
                    correct = anti_lu[cand_x0 + cand_length - 1, cand_y0 - (cand_length - 1)] - prev
                distance = cand_length - correct
                if distance / cand_length < rate:
                    break
                else:
                    cand_x0 += 1
                    cand_y0 += direction
                    cand_length -= 1
            # update seg
            seg.x0 = cand_x0
            seg.y0 = cand_y0
            seg.length = cand_length
            seg.update_endpoints()
            seg.recalc_distance()

        new_segs.append(seg)
        prev_end = seg.x1

    return new_segs

def chose_segs_np(segs: np.ndarray, min_len: int) -> np.ndarray:
    return segs[segs['length'] >= min_len]  # 向量化过滤
def chose_segs(segments: List[Segment], length) -> List[Segment]:
    answer = []
    for seg in segments:
        if seg.length < length:
            continue
        answer.append(seg)
    return answer


def minimal_interval_cover2_np(segs: np.ndarray,
                               rate: float,
                               length_thresh: int) -> np.ndarray:
    """
    NumPy版 minimal_interval_cover2：
    - segs: np.ndarray，dtype=seg_dtype
    - rate: 错误率阈值
    - length_thresh: 最短保留长度
    返回同样 dtype=seg_dtype 的覆盖最少片段集合
    """
    # 1. 无数据时直接返回空
    if segs.size == 0:
        return np.empty(0, dtype=segs.dtype)

    # 2. 向量化提取字段
    x0_arr = segs['x0']                               # 起点 x
    len_arr = segs['length']                          # 段长
    dir_arr = segs['direction']                       # 方向(1=main,0=anti)

    # 3. 计算每条段的终点 x1
    x1_arr = x0_arr + len_arr - 1                     # 终点 x1

    # 4. 全局覆盖范围
    min_x = int(x0_arr.min())                         # 最小起点
    max_x = int(x1_arr.max())                         # 最大终点

    # 5. 桶：对于每个可能的起点 x0，记录能到达的最远 x1 及对应段的索引
    best_end = np.full(max_x+1, -1, dtype=int)        # 存储最远 x1
    best_idx = np.full(max_x+1, -1, dtype=int)        # 存储对应的段索引
    for idx in range(segs.size):
        bx0 = int(x0_arr[idx])
        bx1 = int(x1_arr[idx])
        # 如果当前段能到达更远，就更新桶
        if bx1 > best_end[bx0]:
            best_end[bx0] = bx1
            best_idx[bx0] = idx

    covered_end = min_x - 1                           # 当前已覆盖的最远位置
    out_recs = []                                     # 结果列表，先存 tuple

    # 6. 线性扫描：每次从 covered_end+1 向前选最优片段
    while covered_end < max_x:
        start_pos = covered_end + 1

        # 6.1 在 [min_x..start_pos] 区间内找能到达最远的那条
        window = best_end[min_x:start_pos+1]         # 取出子数组
        if window.size == 0:
            break
        rel = np.argmax(window)                      # 子数组中最大值位置
        candidate_end = int(window[rel])
        candidate_i   = int(best_idx[min_x + rel])  # 全局索引

        # 无法推进时退出
        if candidate_i < 0 or candidate_end < start_pos:
            break

        # 6.2 截断选中的段
        orig = segs[candidate_i]
        ox0 = int(orig['x0'])
        oy0 = int(orig['y0'])
        olen= int(orig['length'])
        odir= int(orig['direction'])
        oend= ox0 + olen - 1

        new_len = min(oend, candidate_end) - start_pos + 1

        # 6.3 重新计算截断后 y0 和 distance
        if odir == 1:
            new_y0 = oy0 + (start_pos - ox0)
            prev   = main_ld[start_pos-1, new_y0-1] if start_pos>0 and new_y0>0 else 0
            correct= main_ld[start_pos+new_len-1, new_y0+new_len-1] - prev
        else:
            new_y0 = oy0 - (start_pos - ox0)
            prev   = anti_lu[start_pos-1, new_y0+1] if start_pos>0 and new_y0+1<grid.shape[1] else 0
            correct= anti_lu[start_pos+new_len-1, new_y0-new_len+1] - prev

        distance = new_len - correct

        # 6.4 满足阈值则加入结果
        if new_len >= length_thresh and (distance / new_len) < rate:
            out_recs.append((start_pos,
                             new_y0,
                             new_len,
                             odir,
                             int(distance)))

        # 6.5 更新覆盖端
        covered_end = candidate_end

    # 7. 打包成结构化数组返回
    return np.array(out_recs, dtype=segs.dtype)
def minimal_interval_cover2(segs: List[Segment], rate, length) -> List[Segment]:
    """
    使用桶 + 线性贪心算法，最少段数覆盖 [min_x0, max_x1] 上所有匹配位置：
      - 原始段可重叠
      - 每次选取在当前覆盖端点前的候选集中，x1 最大的段
      - 截断该段并加入结果
      - 跳过已选段，直到覆盖至全局终点
    """
    if not segs:
        return []

    # 全局覆盖范围
    min_x = min(s.x0 for s in segs)
    max_x = max(s.x1 for s in segs)

    # 构建一个 size = max_x+1 的桶，记录每个 x0 上能到达的最远 x1 及对应段索引
    best_end = [-1] * (max_x + 1)
    best_idx = [-1] * (max_x + 1)
    for idx, s in enumerate(segs):
        if s.x1 > best_end[s.x0]:
            best_end[s.x0] = s.x1
            best_idx[s.x0] = idx

    covered_end = min_x - 1
    result: List[Segment] = []

    # 线性扫描：每次从 last_cover+1 回顾到 min_x，选出能让 x1 最大的那段
    while covered_end < max_x:
        start_pos = covered_end + 1

        # 在 [min_x .. start_pos] 范围内找 best_end 最大的 x1
        candidate_end = -1
        candidate_i = -1
        # 注意：min_x 是固定的最小 x0
        for x0 in range(min_x, start_pos + 1):
            if best_end[x0] > candidate_end:
                candidate_end = best_end[x0]
                candidate_i = best_idx[x0]

        if candidate_i < 0 or candidate_end < start_pos:
            # 无法继续覆盖
            break

        # 取出那段，做截断
        best = segs[candidate_i]
        new_length = min(best.x1, candidate_end) - start_pos + 1

        # 计算截断后的 y0 与 distance
        if best.direction == 'main':
            new_y0 = best.y0 + (start_pos - best.x0)
            prev = main_ld[start_pos-1][new_y0-1] if start_pos>0 and new_y0>0 else 0
            correct = main_ld[start_pos+new_length-1][new_y0+new_length-1] - prev
        else:
            new_y0 = best.y0 - (start_pos - best.x0)
            prev = anti_lu[start_pos-1][new_y0+1] if start_pos>0 and new_y0+1 < len(grid[0]) else 0
            correct = anti_lu[start_pos+new_length-1][new_y0-new_length+1] - prev

        distance = new_length - correct

        # 仅保留符合阈值的段
        if distance/new_length < rate and new_length >= length:
            new_seg = Segment(start_pos, new_y0, new_length, best.direction, distance)
            result.append(new_seg)

        # 更新覆盖端
        covered_end = candidate_end

    return result

def merge_with_tolerance_np(segs: np.ndarray,
                            max_gap: int = 1,
                            max_error_rate: float = 0.1) -> np.ndarray:
    """
    NumPy版 merge_with_tolerance：
    - segs: np.ndarray，dtype=seg_dtype
    - max_gap: x 轴允许的最大空隙
    - max_error_rate: (gap + dist1 + dist2)/(merged_length) ≤ 此值 才合并
    返回同样 dtype=seg_dtype 的 np.ndarray。
    """
    # 如果没有任何 segment，直接返回空
    if segs.size == 0:
        return np.empty(0, dtype=segs.dtype)

    # 提取各字段为向量
    x0       = segs['x0']                              # 起点 x
    y0       = segs['y0']                              # 起点 y
    length   = segs['length']                          # 段长
    direction= segs['direction']                       # 方向：1=main,0=anti
    distance = segs['distance']                        # 错误数

    # 计算终点坐标 x1, y1
    x1 = x0 + length - 1                                # 终点 x1
    y1 = np.where(direction==1,
                  y0 + length - 1,                     # 主对角线 dy=+1
                  y0 - (length - 1))                    # 副对角线 dy=-1

    # 计算对角线 id：主对角线用 y0-x0，副对角线用 y0+x0
    diag_id = np.where(direction==1,
                       y0 - x0,
                       y0 + x0)

    merged_records = []

    # 分组：先按 direction 再按 diag_id
    for dir_val in np.unique(direction):
        mask_dir = (direction == dir_val)              # 选出同一方向的记录
        for d in np.unique(diag_id[mask_dir]):
            mask = mask_dir & (diag_id == d)           # 选出同一条对角线的记录
            group = segs[mask]                         # 结构化数组切片
            # 按 x0 升序
            order = np.argsort(group['x0'], kind = 'mergesort')
            grp = group[order]

            # 初始化 curr 为该组第一个片段
            c = grp[0]
            cx0, cy0, clen, cdir, cdist = int(c['x0']), int(c['y0']), int(c['length']), int(c['direction']), int(c['distance'])
            cx1 = cx0 + clen - 1
            cy1 = cy0 + (clen - 1 if cdir==1 else -(clen - 1))

            # 遍历该组的剩余片段
            for s in grp[1:]:
                sx0, sy0, slen, sdir, sdist = int(s['x0']), int(s['y0']), int(s['length']), int(s['direction']), int(s['distance'])
                sx1 = sx0 + slen - 1
                sy1 = sy0 + (slen - 1 if sdir==1 else -(slen - 1))

                gap        = sx0 - cx1 - 1                  # x 轴间隙
                merged_len = sx1 - cx0 + 1                  # 合并后长度

                # 合并条件
                cond_gap   = gap <= max_gap
                cond_err   = (gap + cdist + sdist) / merged_len <= max_error_rate
                cond_align = ((dir_val==1 and sy0 - cy1 == gap+1) or
                              (dir_val==0 and cy1 - sy0 == gap+1))

                if cond_gap and cond_err and cond_align:
                    # 能合并：更新 curr 的长度和 distance
                    cdist = gap + cdist + sdist
                    clen  = merged_len
                    cx1   = cx0 + clen - 1
                    cy1   = cy0 + (clen - 1 if cdir==1 else -(clen - 1))
                else:
                    # 不能合并：把 curr 推入结果，重置 curr
                    merged_records.append((cx0, cy0, clen, cdir, cdist))
                    cx0, cy0, clen, cdir, cdist = sx0, sy0, slen, sdir, sdist
                    cx1 = cx0 + clen - 1
                    cy1 = cy0 + (clen - 1 if cdir==1 else -(clen - 1))

            # 组尾：记得把最后一个 curr 加入
            merged_records.append((cx0, cy0, clen, cdir, cdist))

    # 打包回结构化数组并返回
    return np.array(merged_records, dtype=segs.dtype)
def merge_with_tolerance(segs: List[Segment],
                         max_gap: int = 1,
                         max_error_rate: float = 0.1) -> List[Segment]:
    """
    对每条平行于主/副对角线的线段组：
      - 如果两段在 x 轴的间隙 gap ≤ max_gap，且 gap/(merged_length) ≤ max_error_rate，
        就把它们合并成一段。
    """
    from collections import defaultdict

    groups = defaultdict(list)
    # 按方向 & 对角线 id 分组（不变）
    for s in segs:
        if s.direction == 'main':
            diag = s.y0 - s.x0
        else:
            diag = s.y0 + s.x0
        groups[(s.direction, diag)].append(s)

    merged_all = []
    # 对每个组，原本要 sort，这里改成「桶扫描」
    for (direction, diag), group in groups.items():
        # 找出该组内的最大 possible x0
        max_x0 = max(s.x0 for s in group)
        # 建桶：每个 x0 对应一个列表
        bucket = [[] for _ in range(max_x0 + 1)]
        for s in group:
            bucket[s.x0].append(s)
        # 线性扫描 x0，从小到大，相当于已经排好序
        curr = None
        for x0 in range(len(bucket)):
            for s in bucket[x0]:
                if curr is None:
                    curr = s
                    continue
                gap = s.x0 - curr.x1 - 1
                merged_len = s.x1 - curr.x0 + 1
                cond_gap = gap <= max_gap
                cond_err = (gap + curr.distance + s.distance) / merged_len <= max_error_rate
                cond_align = (direction == 'main' and s.y0 - curr.y1 == gap+1) or \
                             (direction == 'anti' and curr.y1 - s.y0 == gap+1)
                if cond_gap and cond_err and cond_align:
                    # 合并
                    curr = Segment(curr.x0,
                                   curr.y0,
                                   merged_len,
                                   direction,
                                   gap + curr.distance + s.distance)
                else:
                    merged_all.append(curr)
                    curr = s
        if curr is not None:
            merged_all.append(curr)

    return merged_all

def find_best_matches_np(query: str, reference: str) -> List[Segment]:
    """
    主函数：给定 query 和 reference，返回保留下来的所有最长匹配段
    """
    global main_ld, grid, anti_lu
    """
    主函数：给定 query 和 reference，返回保留下来的所有最长匹配段
    """
    # 0. 构建 dot-plot
    # tracemalloc.start()
    grid = build_dotplot(query, reference)
    # snapshot1 = tracemalloc.take_snapshot()
    # show_top(snapshot1, "build_dotplot 完成")
    main_ld, anti_lu = init_diag_tables(grid)
    # snapshot2 = tracemalloc.take_snapshot()
    # show_top(snapshot2, "init_diag_tables 完成")
    final_segs = find_diagonal_segments_np(grid)
    # snapshot3 = tracemalloc.take_snapshot()
    # show_top(snapshot3, "find_diagonal_segments_np 完成")


    # 2091!!!!!
    final_segs = merge_with_tolerance_np(final_segs, 1, 0.057)   
    # snapshot4 = tracemalloc.take_snapshot()
    # show_top(snapshot4, "merge_with_tolerance_np 完成")
    final_segs = minimal_interval_cover2_np(final_segs, 0.08, 20)
    # snapshot5 = tracemalloc.take_snapshot()
    # show_top(snapshot5, "minimal_interval_cover2_np 完成")
    final_segs = fill_in_blanks_global_np(final_segs, 0.065, 25)
    # snapshot6 = tracemalloc.take_snapshot()
    # show_top(snapshot6, "fill_in_blanks_global_np 完成")
    
    
    final_segs = chose_segs_np(final_segs, 25)


    final_segs = extend_start_backward_np(final_segs, 0.1)
    # snapshot8 = tracemalloc.take_snapshot()
    # show_top(snapshot8, "extend_start_backward_np 完成")

    # 假设 final_segs 是 dtype=seg_dtype 的结构化数组
    # 1. 先拿到按照 x0 升序的索引
    order = np.argsort(final_segs['x0'], kind = 'mergesort')
    # 2. 倒序（reverse=True）
    order_desc = order[::-1]
    # 3. 用索引重排
    final_segs = final_segs[order_desc]


    final_segs = extend_end_backward_np(final_segs, 0.1)
    # snapshot9 = tracemalloc.take_snapshot()
    # show_top(snapshot9, "extend_end_backward_np 完成")

    # final_segs 是结构化数组，按照 x0 字段升序排列
    final_segs = final_segs[np.argsort(final_segs['x0'], kind = 'mergesort')]

    return final_segs
def find_best_matches(query: str, reference: str) -> List[Segment]:
    """
    主函数：给定 query 和 reference，返回保留下来的所有最长匹配段
    """
    global main_ld, grid, anti_lu, all_segs, x0_groups
    """
    主函数：给定 query 和 reference，返回保留下来的所有最长匹配段
    """
    # 0. 构建 dot-plot
    grid = build_dotplot(query, reference)
    main_ld, anti_lu = init_diag_tables(grid)
    final_segs = find_diagonal_segments(grid)


    # 2091!!!!!
    final_segs = merge_with_tolerance(final_segs, 1, 0.057)   
    final_segs = minimal_interval_cover2(final_segs, 0.08, 20)
    final_segs = fill_in_blanks_global(final_segs, 0.065, 25)
    
    
    # final_segs = weighted_interval_scheduling(final_segs)
    final_segs = chose_segs(final_segs, 25)


    final_segs = extend_start_backward(final_segs, 0.1)

    final_segs = sorted(final_segs, key=lambda s: s.x0, reverse = True)

    final_segs = extend_end_backward(final_segs, 0.1)

    final_segs = sorted(final_segs, key=lambda s: s.x0)
    
    
    return final_segs


def get_answer(matches):
    print("Remain segments:")
    for seg in matches:
        if seg.length >= 30:
            print(seg, end=",")
    print()

def get_detail(matches):
    point = 0
    for seg in matches:
        length = seg.x1 - seg.x0 + 1
        if length >= 30:
            print(seg, end="\t")
            print(";length = ", length, end="\t")
            if seg.direction == "main":
                prev = main_ld[seg.x0-1][seg.y0-1] if seg.x0>0 and seg.y0>0 else 0
                affective_length = main_ld[seg.x1][seg.y1] - prev
            else:
                prev = anti_lu[seg.x0-1][seg.y0+1] if seg.x0>0 and seg.y0+1<len(grid[0]) else 0
                affective_length = anti_lu[seg.x1][seg.y1] - prev
            print("affective length = ",affective_length, end = '\t')
            if affective_length/length > 0.9:
                print("True")
                point += affective_length
            else:
                print("False")
            # print(seg)
    print("final score = ", point)

def get_answer_np(matches: np.ndarray):
    """打印所有长度 ≥30 的 segment 的简要列表"""
    print("Remain segments:")
    for seg in matches:
        length = int(seg['length'])
        if length >= 30:
        # if length:
            x0 = int(seg['x0'])
            y0 = int(seg['y0'])
            direction = int(seg['direction'])
            # 计算 x1,y1
            x1 = x0 + length - 1
            if direction == 1:  # main
                y1 = y0 + length - 1
                # __repr__ 输出 (x0, x1+1, y0, y1+1)
                print(f"({x0},{x1+1},{y0},{y1+1})", end=",")
            else:              # anti
                y1 = y0 - (length - 1)
                # __repr__ 输出 (x0, x1+1, y1, y0+1)
                print(f"({x0},{x1+1},{y1},{y0+1})", end=",")
    print()

def get_detail_np(matches: np.ndarray):
    """对每个长度 ≥30 的 segment 打印详细信息并累计分数"""
    point = 0
    for seg in matches:
        length = int(seg['length'])
        # if length < 30:
        if length <= 0:
            continue
        x0 = int(seg['x0'])
        y0 = int(seg['y0'])
        direction = int(seg['direction'])
        x1 = x0 + length - 1
        if direction == 1:
            y1 = y0 + length - 1
            prev = main_ld[x0-1, y0-1] if x0>0 and y0>0 else 0
            affective = main_ld[x1, y1] - prev
        else:
            y1 = y0 - (length - 1)
            prev = anti_lu[x0-1, y0+1] if x0>0 and y0+1<anti_lu.shape[1] else 0
            affective = anti_lu[x1, y1] - prev

        # 复刻原来的输出格式
        if direction == 1:
            print(f"({x0},{x1+1},{y0},{y1+1})\t;length = {length}\t", end="")
        else:
            print(f"({x0},{x1+1},{y1},{y0+1})\t;length = {length}\t", end="")

        print(f"affective length = {affective}\t", end="")
        if affective / length > 0.9:
            print("True")
            point += affective
        else:
            print("False")
    print("final score = ", point)


if __name__ == "__main__":

    reference = "TGATTTAGAACGGACTTAGCAGACATTGAAACTCGAGGGGTATAGCAATAGATGCCCAAAAAGGTAAGCGCCATAAGCGTGGTTCTACGAGCCAGGTGCTCATGCCTAAGTTCTGCGCCTTCGCTGTCACTTGGAAATACTGTAATGGATCATGCCTAGGTTATGCGCCTTCGGGGTCACTTCAACATACTGTAATGGATCATGCCTAGGTTTTGCGTGTTCGCTGTCATTTCGAAATACTCCAATGGATGATGCCTAGGTTCTGTGCCTTCGCTGACGCATGGAAATACTTTAACGGATCATGCCCAGGCTCTGCGCCTTCGCTGAAACTTCGAAATACTCTAATGGATCATGCCTCGGTGCTCCACCTTCGCTTTCATTCCGAAATACTCTAATGGATCGCGTCCGTGTAACAACTTCGTACTGTTATATCGACCATCAGAATACCCATCCCTCGGGGAGGTAACCTATATTCACGTCGCAAGTTTCGATCTACAGTATGCTGACTGTTTGCCGCGATTTTAAGTCAAGAAGCGCGTCCACATGGTCGCGGTCGTCAACTTCAGTACGCTCATATGACACCAAAAGATCTACCTACAGCCCGTGCAGCTCGACTTTTGTGCTCTAGGGCACGACGGGTGGCGTTTGCTCCCGCGCATCTCGACTTTTAAGCTCTATGGCACAACGTGTGGCGTTTGCCCCCGCGCAGCTCGACTTTTGTGCTCTAGGGCACGGCGGGTGGCGTTTGCCCTCGCCCAGCTTGACTTTTGTGCTCTAGGGCACGACGGGTGGCGTTTGCCCCCGTGCAGCCCGACTTTTGTACTCTAGTGCACGACGGGTGGCGTTTGCCCCCGCACCGCTCGACTTTTGTGATCTAGGGCACTACGAGTAGCGTTGGCCCAGACAGATCAACGCACATGGATTCTCTAACAGGCCCCGCGCTTCTCATTGGCCCGTGAGACGGGTCTGAGAGGAAGACATTAGGTAGATACGGAAAGCTTTGGCGTAGTTCGTATCTTTCAGCGGTGAAGCGTCTTCGGTCCGGGCTGCGTTATGCCTGCGGGAGGAAGGCTCCACTAGATGGTTTACGAGACATAATGTCAGCTTCCTAAAGGTACGGCAGCGCCTGCGTATATCACAGGACGAATTGTCAGTTTGCTAGGGGTACGGGAGCGCTTGCGTATTACATAGGACGAATCGTCAGCTTCCTAAAGGGACGGTAGCGCTTGCGTGTTACATAGGACGAATTGTCAGCTTCGTAAAGGTACGGTAGTTCTTGCGTATTACATAGGATGCATTGTCCGCTTCCTAAAGGTACGCTGGCGCTTGCGTATCACATAGGACGGATAGCGCGATTGCTAAAGGTACGGGAGCGCTTGCGTCTTAGAGCGCACGAATCGGATATAAGCTGCGCCCGCGTCTGGCGAGCAAAAATCGTGGGAGCCAGCGAGGGAAAAACTGCTCGGGCGACTTAAACGGAATTACAAGACTCATTGCCATCGAGGACGTTAGACTAAAGAGCCCCTGCGTGCCTCCTTTGTATAGCTCGATGTAGTGGCCCGTGTATGTGGAACAGGAATGCTCGATCTAAGGTAGTAGTGGCTACAGCTCCGAGAGTTTGCGTACTGCGGTGCCAGGGATTTTGCCTGCGGGTACAGCCTCTGCGCACGCCGGTCTGTGATCAAGAACTAAACTAGAGA"
    query = "TGATTTAGAACGGACTTAGCAGACATTGAAACTCGAGGGGTATAGCAATAGATGCCCAAAAAGGTAAGCGCCATAAGCGTGTTTCTACGAGCCAGGTGCTCATGCCTAAGTTCTGCGCCTTCGCTGTCACTGGGAAATACTGTAATGGATCATCCGTAGGTTATGCGCCTTCGGGGTCACTTCAACATACTGTAACGGATCGTGCCTAGGTTTTGCGTATTCGCTGTCATTTCGAATTACACCAATGGATGATGCCTAGGTTCTGTGCCTCCGCTGACGCATCGAAATACTTTAACGGATCGCGTCCGAGTAACAACTTCGTACTGTTATATAGGCAATCAGAATACCCATGCCTCGGGGAGGTAACCTATATTCACGTCGCAAGTTTCGATCTACAGTACTGTAGGTATATCTTTTGGTGTCATATGAGGGTACTGAACTTGACGACCGCGACCATGTGGATGCGCTTCTTGACTTAAAATCGCGGCAAACAGTAAGCATCCGTGAAGCTCGACTTTTGTGCTCTAGGGCACGACGGGTGGCGTTTGCTCCCGCGCATCTCGAGTTGTAAGCTCTATGGCACAACGGGTGGCGTTTGCCGCCGAGCAGCTCGACTTTTGTGCTCTAGGGCACGGCGGGTGGCGTTTGCCCTCGCCCAGCTTGACTTTTGTGCTCTAGGGCACGACGGGTGGCCTTTGCCCCCGCGCAGCTGGACTTTTGTGCTCTAGGGCACGGCGGGTGGCGTTTGCCCTCCCCCAGCTTGACTACTGTGCTCTAGGGCACGACGGGTGGCGTTTGCCCCCGCGCAGCTCGACTTTTGTGCTCTATGGCACGGGGGGTGGCGTTTGCCCTCGCCCAGCTTGACTTTTGCGCTCTAGGGCACGACGGGTGGCGTTTGCCGGCAAACGCCACACGTCGTGCCCTAGAGCACAAACGTCAAGCTGGGCGAGGGCAACCGCCACCCGCCCTGCCCTAGAGCACAAAAGTCGAGCTGCGCGGGCCCGCGCAGCTCGACTTTTGTGCTCTAGGACACGGCGGGTGGCGTTTGCCCTCGCCCAGCTTGACTCTTGTGCTCTAGGGCACGACGGGTGGCGTTTGCCCCAGCGCAGCCCGACTTTTGTACTCTAGAGCACGACGGTTGGCATTTGCCCCCGCACCGCTCGACTTTTGTGATCTAGGGCCCTAGGAGTAGCGTTGGCCAGCTTTCCGTATCTACCTAATGTCTTCCTCTCAGACCCGTCTCACGGGCCAATGAGAAGCGCGGGGCCTATTAGAGAATCCATGTGCGTTGATCTGTCTGCAGACAGCTCAACGCACATGGATTCGCTAGCAGGCCCCGCGCTTCTCATTGGCCCGTGAGACGGGTCTGAGAGGAAGACATAAGGTAGATACGGCAAGCTCACGTCCGTGTAACAACGTCGTACTGTTATATCGACCATCAGAATCCCCATCCCGCGAGGAGGTAACCTATATTCAGGTCGCAAGTTTCGATCTACAGTATTGGCGTAGTTCGTATCTTTCAGCGGTGAAGCTTCTTCGGTCCGGGCTGCGTTATGCCTGCTGGAGGACGGCTCCACTAGATGGTTTACGAGACATAATGTCCGCTTCCTAAAGGTACACTGGCGCTTGAGTATCACATAGGACGGATAGCTCGATTCCTAAAGGGACGGGAGCGCTTGCGTCTTAGAGCGCATGAATCGTCAGCTTCCCAAAGGGACCGTAGCGCTTGCGTGTTATATAGGAAGAATGGTCAGCTTTGTAAAGGTACGGTAGTTCTTGCGTATTACAGAGGATGCATTGTCTACTACCTAAAGGTACGGCAGCGCCTGCGTATATCACAGGACGAATTGTCAGTTTGCTAGGGGTACGGGAGCGCTTGCATATTACATAGGACGAATCGGATATAAGCTGCGCCCGCGTCTGGCGATAAAAAATCGTGGTAGCCAGCGAGGGAAAAACTGCTCGGGCGACTTAAACGGAATTAAAAGACTCATTGCCGTGACAGACTTCCGTATAGCAACCTCTGGGATGTCGATGCGGTGTCCCCAGTCTGCGCTGAGCGGGGGCAGACAGACTTAGTTATAGTATGCATCTGTTTTAGCTAGACATCACGACCTAGTGGGGTTCATGTTGAGATTCTAGGGCGGTACGCAGCCGGTGGATTATTACTTCCCCAGAAATTCTGACTTCGTCACTGGATGGATTGTACTATCCGGTCAACCTTACAAGGTTTCAACAGGGACGAAGGGTAAACGTATGAAGCTTGGATGCCGTTACCGTAAAGGGCCCTATTGAAGTGTCGAGGACGTTAGACTAAAGAGCCCCTGCGTGCCTCCTTTGTATAGCTCGAGGTAGTGGCCCGGATATGTGGAACAGGAATGCTCGATCTAAGGTAGTAGTGGGTACCGCTCCGAGAGTTTGCGTACTGCGGTGCCCGGGATTTTGCCTGCGGGTACAGCCTCTGCGCACGCCGGTCTGTAATCAAGAACTAAACTAGAGA"
    # reference = REFERENCE
    # query = QUERY

    # 原来
    # matches = find_best_matches(query, reference)
    # get_answer(matches)
    # get_detail(matches)

    # 现在
    matches = find_best_matches_np(query, reference)
    get_answer_np(matches)
    # get_detail_np(matches)