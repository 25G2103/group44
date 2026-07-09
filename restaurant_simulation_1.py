"""
飲食店来客シミュレーション（新フロアレイアウト + 通行人グループ + 視野モデル）
--------------------------------------------------------------------------
店内レイアウト（手描きフロア図を反映。入口は左下）：
  券売所（入口すぐ上）
  カウンター(1人席)10席：上部・横一列
  2人席：中央に3ブロック（各ブロック 2列×3行 = 18卓）
  4人席×5：右側・縦一列
  4人席×12：下部・横一列

視野モデル：
  入口（券売所前）から半径 VISIBLE_DISTANCE 以内にある卓だけが、
  店外の通行人グループから「見える席」として扱われる。
  通行人グループは、入店を判断するタイミングでこの「見える席」の
  埋まり率を社会的証明として利用する。

エージェントの入店判断：
  ①価格帯の適合度（budget） ②混雑許容度（crowd_tolerance）
  ③社会的証明（見える席の埋まり率）の重み付き合成で入店確率を決める。
"""

import random
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation

# ============================================================
# 日本語フォント設定
# ============================================================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'IPAexGothic', 'Meiryo', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# フロアレイアウトの基準値（手描き図を再現）
# ============================================================
FLOOR_W = 24.0
FLOOR_H = 12.0
ENTRANCE_POS = (0.9, 0.0)      # 入口の座標（左下）
VISIBLE_DISTANCE = 3.0         # 入口からこの距離以内の卓は「見える席」

ENTRANCE_X = ENTRANCE_POS[0]
ENTRANCE_Y = ENTRANCE_POS[1]

SEAT_COLOR_OCC_VISIBLE = "#c0622a"
SEAT_COLOR_OCC_HIDDEN = "#8b9e6b"
SEAT_COLOR_EMPTY_VISIBLE = "#fde8d8"
SEAT_COLOR_EMPTY_HIDDEN = "#eee"

ROAD_Y = -2.5

# ============================================================
# 通行人グループ・エージェントのパラメータ
# ============================================================
GROUP_SIZES = [1, 2, 3, 4]
GROUP_SIZE_WEIGHTS = [0.50, 0.30, 0.15, 0.05]   # 1人客が最も多い分布
GROUP_MEMBER_GAP = 0.35

WALK_SPEED_NORMAL = 0.35
WALK_SPEED_SLOW = 0.10
SLOWDOWN_RANGE = 4.0

VIEW_DECISION_X = ENTRANCE_X   # このx座標まで来たら入店を判断する

SPAWN_INTERVAL = 5
N_STEPS = 320
RANDOM_SEED = 11
INITIAL_FILL_RATE = 0.3

LEAVE_PROB = 0.012

STORE_PRICE_LEVEL = 3
WEIGHT_SOCIAL_PROOF = 0.40
WEIGHT_PRICE_FIT = 0.35
WEIGHT_CROWD_FIT = 0.25


# ============================================================
# 座席クラス
# ============================================================
class Seat:
    def __init__(self, index, x, y, w, h, capacity, kind, is_window):
        self.index = index
        self.x = x                   # 描画用：左下のx
        self.y = y                   # 描画用：左下のy
        self.w = w
        self.h = h
        self.cx = x + w / 2          # 中心座標（移動・視野判定用）
        self.cy = y + h / 2
        self.capacity = capacity
        self.kind = kind             # "table4" / "table2" / "counter"
        self.is_window = is_window   # 入口から見える席か
        self.occupied = False
        self.party_size = 0


def create_layout_seats():
    """手描きフロア図のレイアウトを再現して座席リストを生成する。"""
    seats = []
    index = 0

    def add(kind, x, y, w, h, capacity, is_window=False):
        nonlocal index
        seats.append(Seat(index, x, y, w, h, capacity, kind, is_window))
        index += 1

    # --- カウンター(1人席)：上部、横一列 10席 ---
    counter_n = 10
    cx0, cx1 = 4.2, 22.5
    cy, ch = 9.6, 1.0
    seat_w = (cx1 - cx0) / counter_n
    for i in range(counter_n):
        add("counter", cx0 + i * seat_w, cy, seat_w * 0.9, ch, 1)

    # --- 2人席：2ブロック、各ブロック仕切りで2列×2行 ---
    block_starts = [7.6, 11.0]
    cell_w, cell_h = 1.0, 1.0
    gap_row = 0.15
    gap_col_partition = 0.35
    for bx in block_starts:
        for col in range(2):
            for row in range(2):
                x = bx + col * (cell_w + gap_col_partition)
                y = 5.3 + row * (cell_h + gap_row)
                add("table2", x, y, cell_w, cell_h, 2)

    # --- 3人席：2人席ブロックの右隣に独立配置（2列×2行）---
    bx = 4.2
    for col in range(1):
        for row in range(3):
            x = bx + col * (cell_w + gap_col_partition)
            y = 4.3 + row * (cell_h + gap_row)
            add("table3", x, y, cell_w, cell_h, 3)

    # --- 4人席×5：右側、縦一列 ---
    col4_x, col4_w = 20.3, 2.6
    col4_y0, col4_y1 = 3.3, 8.6
    n5 = 5
    gap5 = 0.2
    h5 = (col4_y1 - col4_y0 - gap5 * (n5 - 1)) / n5
    for i in range(n5):
        y = col4_y0 + i * (h5 + gap5)
        add("table4", col4_x, y, col4_w, h5, 4)

    # --- 4人席×12：下部、横一列（仕切りあり） → ここが「見える席（窓側）」 ---
    row12_x0, row12_x1 = 2.3, 23.3
    row12_y, row12_h = 0.6, 2.3
    n12 = 12
    gap12 = 0.08
    w12 = (row12_x1 - row12_x0 - gap12 * (n12 - 1)) / n12
    for i in range(n12):
        x = row12_x0 + i * (w12 + gap12)
        add("table4", x, row12_y, w12, row12_h, 4, is_window=True)   # ← is_window=True を明示

    return seats


# ============================================================
# 通行人グループ・エージェントクラス
# ============================================================
class Group:
    _id_counter = 0

    def __init__(self, x, size):
        self.id = Group._id_counter
        Group._id_counter += 1
        self.x = x
        self.y = ROAD_Y
        self.size = size
        self.state = "walking"     # walking / entering / seated / left / exited
        self.decided = False
        self.target_seat = None

        self.budget = random.randint(1, 5)
        self.crowd_tolerance = random.random()
        self.is_smoker = random.random() < 0.3   # 30%の確率で喫煙者
        #滞在時間
        self.time_seated = 0
        self.max_stay = random.randint(30, 120)  # 30〜120フレーム滞在


    def current_speed(self):
        d = abs(self.x - ENTRANCE_X)
        if d < SLOWDOWN_RANGE:
            ratio = d / SLOWDOWN_RANGE
            return WALK_SPEED_SLOW + (WALK_SPEED_NORMAL - WALK_SPEED_SLOW) * ratio
        return WALK_SPEED_NORMAL

    def visible_seats(self, seats):
        """入口から見える席（is_window=True）の一覧。フロア全体に対して固定。"""
        return [s for s in seats if s.is_window]

    def member_offsets(self):
        n = self.size
        return [(-(n - 1) / 2 + i) * GROUP_MEMBER_GAP for i in range(n)]


def pick_group_size():
    return random.choices(GROUP_SIZES, weights=GROUP_SIZE_WEIGHTS, k=1)[0]


def find_seat_for_group(seats, group_size, strategy="window"):
    """
    エージェントの移動ルール：
      ① 人数以上の席であること（capacity >= group_size）
      ② strategy="window" : 窓側（is_window=True）を最優先で埋める
         strategy="depth"  : 奥側（x が大きい）を最優先で埋める
      ③ 同条件内では、席を無駄にしないよう capacity が小さい席を優先
    """
    candidates = [s for s in seats if (not s.occupied) and (s.capacity >= group_size)]
    if not candidates:
        return None

    if strategy == "window":
        # is_window=True を優先（Falseより優先）、次にcapacityが小さい順
        candidates.sort(key=lambda s: (not s.is_window, s.capacity))
    elif strategy == "depth":
        # x が大きい（奥）ほど優先、同じ奥行きならcapacityが小さい順
        candidates.sort(key=lambda s: (-s.x, s.capacity))
    else:
        raise ValueError(f"unknown strategy: {strategy}")

    return candidates[0]

def calc_entry_probability(group, visible_rate, current_occupancy_rate):
    price_fit = max(0, 1 - abs(STORE_PRICE_LEVEL - group.budget) * 0.3)
    crowd_fit = 1 - abs(current_occupancy_rate - group.crowd_tolerance)
    social_proof = visible_rate

    score = (
        WEIGHT_PRICE_FIT * price_fit +
        WEIGHT_CROWD_FIT * crowd_fit +
        WEIGHT_SOCIAL_PROOF * social_proof
    )

    # 喫煙者は入店確率がやや上がる（喫煙可能店として設定）
    if group.is_smoker:
        score += 0.15

    return max(0, min(1, score))


# ============================================================
# シミュレーション本体
# ============================================================
def run_simulation(n_steps=N_STEPS, spawn_interval=SPAWN_INTERVAL, seed=RANDOM_SEED,
                    initial_fill_rate=INITIAL_FILL_RATE, window_fill_rate=None,
                    nonwindow_fill_rate=None, seat_strategy="window"):

    seats = create_layout_seats()

    if seed is not None:
        random.seed(seed)
    Group._id_counter = 0

    # 初期埋まり
    groups = []

    for s in seats:
        s.occupied = False
        s.party_size = 0

        fill_rate = window_fill_rate if s.is_window else nonwindow_fill_rate
        if fill_rate is not None and random.random() < fill_rate:
            party_size = s.capacity if s.kind != "counter" else 1

            s.occupied = True
            s.party_size = party_size

            g = Group(x=s.cx, size=party_size)
            g.y = s.cy
            g.state = "seated"
            g.decided = True
            g.target_seat = s
            g.time_seated = random.randint(0, 20)
            g.max_stay = random.randint(60, 150)
            groups.append(g)

    # 初期満席の内訳を記録
    initial_window_tables = sum(1 for s in seats if s.is_window and s.occupied)
    initial_window_people = sum(s.party_size for s in seats if s.is_window and s.occupied)
    initial_nonwindow_tables = sum(1 for s in seats if (not s.is_window) and s.occupied)
    initial_nonwindow_people = sum(s.party_size for s in seats if (not s.is_window) and s.occupied)

    total_window_tables = sum(1 for s in seats if s.is_window)
    total_window_capacity = sum(s.capacity for s in seats if s.is_window)
    total_nonwindow_tables = sum(1 for s in seats if not s.is_window)
    total_nonwindow_capacity = sum(s.capacity for s in seats if not s.is_window)

    road_x_start = FLOOR_W + 6.0
    road_x_end = ENTRANCE_X - 6.0

    history = []
    occupied_seat_count_history = []
    leave_history = []
    passed_cumulative_history = []

    entered_total = 0
    passed_total = 0
    groups_entered = 0
    groups_passed = 0
    entered_budgets = []
    entered_crowd_tol = []
    entered_smokers = 0
    total_smokers = 0

    passed_smokers_total = 0
    passed_nonsmokers_total = 0
    passed_smokers_cumulative_history = []
    passed_nonsmokers_cumulative_history = []
    #素通り判定

    total_seats_capacity = sum(s.capacity for s in seats)

    window_seats = [s for s in seats if s.is_window]
    nonwindow_seats = [s for s in seats if not s.is_window]
    window_occ_history = []
    nonwindow_occ_history = []

    # ============================================================
    # フレームループ
    # ============================================================
    for frame in range(n_steps):

        # 新規スポーン
        if frame % spawn_interval == 0:
            size = pick_group_size()
            groups.append(Group(x=road_x_start, size=size))

        occupied_capacity = sum(s.party_size for s in seats)
        current_occupancy_rate = occupied_capacity / total_seats_capacity

        # グループ更新
        for g in groups:
            if g.state == "walking":
                if not g.decided and g.x <= VIEW_DECISION_X:
                    visible = g.visible_seats(seats)
                    if visible:
                        occ_rate = sum(1 for s in visible if s.occupied) / len(visible)
                    else:
                        occ_rate = 0.0

                    entry_prob = calc_entry_probability(g, occ_rate, current_occupancy_rate)

                    g.decided = True
                    if g.is_smoker:
                        total_smokers += 1

                    target = find_seat_for_group(seats, g.size, strategy=seat_strategy)
                    if random.random() < entry_prob and target is not None:
                        target.occupied = True
                        target.party_size = g.size
                        g.target_seat = target
                        g.state = "entering"

                        entered_total += g.size
                        groups_entered += 1
                        entered_budgets.append(g.budget)
                        entered_crowd_tol.append(g.crowd_tolerance)
                        if g.is_smoker:
                            entered_smokers += 1
                    else:
                        passed_total += g.size
                        groups_passed += 1
                        # 喫煙者/非喫煙者を分けて累積
                        if g.is_smoker:
                            passed_smokers_total += g.size
                        else:
                            passed_nonsmokers_total += g.size

                if g.state == "walking":
                    g.x -= g.current_speed()
                    if g.x < road_x_end:
                        g.state = "exited"

            elif g.state == "entering":
                tx, ty = g.target_seat.cx, g.target_seat.cy
                dx = tx - g.x
                dy = ty - g.y
                dist = math.hypot(dx, dy)
                if dist < 0.15:
                    g.x, g.y = tx, ty
                    g.state = "seated"
                else:
                    step = min(dist, WALK_SPEED_SLOW)
                    g.x += dx / dist * step
                    g.y += dy / dist * step

            elif g.state == "seated":
                if not hasattr(g, "time_seated"):
                    g.time_seated = 0
                    g.max_stay = random.randint(20, 60)
                g.time_seated += 1

                leave_reason = None

                if g.time_seated >= g.max_stay:
                    leave_reason = "finished_meal"
                elif current_occupancy_rate > g.crowd_tolerance + 0.2:
                    if random.random() < 0.05:
                        leave_reason = "too_crowded"
                elif abs(STORE_PRICE_LEVEL - g.budget) >= 3:
                    leave_reason = "too_expensive"
                elif g.is_smoker and random.random() < 0.02:
                    leave_reason = "go_smoking"
                elif random.random() < LEAVE_PROB:
                    leave_reason = "random"

                if leave_reason is not None:
                    g.target_seat.occupied = False
                    g.target_seat.party_size = 0
                    g.state = "left"
                    leave_history.append((frame, g.size, g.is_smoker, leave_reason))

        # 消去
        groups = [g for g in groups if g.state not in ("exited", "left")]

        # スナップショット保存
        snapshot = {
            "groups": [
                {
                    "x": g.x, "y": g.y,
                    "state": g.state, "size": g.size,
                    "is_smoker": g.is_smoker,
                    "offsets": g.member_offsets()
                }
                for g in groups
            ],
            "seats_state": [(s.occupied, s.party_size) for s in seats],
        }
        history.append(snapshot)
        occupied_seat_count_history.append(sum(1 for s in seats if s.occupied))
        passed_cumulative_history.append(passed_total)
        passed_smokers_cumulative_history.append(passed_smokers_total)
        passed_nonsmokers_cumulative_history.append(passed_nonsmokers_total)

        w_rate = (sum(1 for s in window_seats if s.occupied) / len(window_seats)) if window_seats else 0.0
        nw_rate = (sum(1 for s in nonwindow_seats if s.occupied) / len(nonwindow_seats)) if nonwindow_seats else 0.0
        window_occ_history.append(w_rate)
        nonwindow_occ_history.append(nw_rate)

   # ============================================================
# ループ終了後の集計
# ============================================================
    walking_history = []
    entered_history = []
    passed_history = []
    
    smoker_walking_history = []
    smoker_entered_history = []

# --- 退店人数（フレーム軸）の累積推移 ---
    left_per_frame = [0] * n_steps
    for (frame, size, is_smoker, reason) in leave_history:
        if frame < n_steps:
            left_per_frame[frame] += size

    left_cumulative_frame = []
    total_left = 0
    for x in left_per_frame:
        total_left += x
        left_cumulative_frame.append(total_left)

# --- 退店人数（イベント軸）の累積（あなたの元コード） ---
    left_cumulative_history = []
    cumulative_left = 0
    for (frame, size, is_smoker, reason) in leave_history:
        cumulative_left += size
        left_cumulative_history.append(cumulative_left)

# --- 歩行中・入店中の人数をフレームごとに集計 ---
    for snap in history:
        walking = sum(grp["size"] for grp in snap["groups"] if grp["state"] == "walking")
        entered = sum(grp["size"] for grp in snap["groups"] if grp["state"] == "entering")

    walking_history.append(walking)
    entered_history.append(entered)

    smoker_walking = sum(
        grp["size"] for grp in snap["groups"]
        if grp["state"] == "walking" and grp["is_smoker"]
    )
    smoker_entered = sum(
        grp["size"] for grp in snap["groups"]
        if grp["state"] == "entering" and grp["is_smoker"]
    )

        # --- 入店人数（フレーム軸）の累積推移 ---
    entered_per_frame = [0] * n_steps
    for snap in history:
        frame_entered = sum(
            grp["size"] for grp in snap["groups"]
            if grp["state"] == "entering"
        )
        entered_per_frame.append(frame_entered)

    entered_per_frame = []
    for snap in history:
        frame_entered = sum(
            grp["size"] for grp in snap["groups"]
            if grp["state"] == "entering"
            )
        entered_per_frame.append(frame_entered)



    smoker_walking_history.append(smoker_walking)
    smoker_entered_history.append(smoker_entered)

    for snap in history:
        walking = sum(grp["size"] for grp in snap["groups"] if grp["state"] == "walking")
        entered = sum(grp["size"] for grp in snap["groups"] if grp["state"] == "entering")

        walking_history.append(walking)
        entered_history.append(entered)

        smoker_walking = sum(
            grp["size"] for grp in snap["groups"]
            if grp["state"] == "walking" and grp["is_smoker"]
        )
        smoker_entered = sum(
            grp["size"] for grp in snap["groups"]
            if grp["state"] == "entering" and grp["is_smoker"]
        )

        smoker_walking_history.append(smoker_walking)
        smoker_entered_history.append(smoker_entered)

    return {
        "history": history,
        "occupied_seat_count_history": occupied_seat_count_history,
        "seats": seats,
        "entered_total": entered_total,
        "passed_total": passed_total,
        "groups_entered": groups_entered,
        "groups_passed": groups_passed,
        "entered_budgets": entered_budgets,
        "entered_crowd_tol": entered_crowd_tol,
        "entered_smokers": entered_smokers,
        "total_smokers": total_smokers,
        "road_x_start": road_x_start,
        "road_x_end": road_x_end,
        "total_seats_capacity": total_seats_capacity,
        "walking_history": walking_history,
        "entered_history": entered_history,
        "passed_history": passed_cumulative_history,
        "smoker_walking_history": smoker_walking_history,
        "smoker_entered_history": smoker_entered_history,
        "smoker_passed_history": passed_smokers_cumulative_history,        #変更
        "nonsmoker_passed_history": passed_nonsmokers_cumulative_history,  #追加
        "leave_history": leave_history,
        "window_occ_history": window_occ_history,
        "nonwindow_occ_history": nonwindow_occ_history,
        "initial_window_tables": initial_window_tables,
        "initial_window_people": initial_window_people,
        "initial_nonwindow_tables": initial_nonwindow_tables,
        "initial_nonwindow_people": initial_nonwindow_people,
        "total_window_tables": total_window_tables,
        "total_window_capacity": total_window_capacity,
        "total_nonwindow_tables": total_nonwindow_tables,
        "total_nonwindow_capacity": total_nonwindow_capacity,
        "passed_cumulative_history": passed_cumulative_history,
        "left_cumulative_history": left_cumulative_history,
        "left_cumulative_frame": left_cumulative_frame,
    }



# ============================================================
# 結果の表示
# ============================================================
def print_results(result):
    total = result["entered_total"] + result["passed_total"]
    rate = (result["entered_total"] / total * 100) if total > 0 else 0
    total_groups = result["groups_entered"] + result["groups_passed"]
    group_rate = (result["groups_entered"] / total_groups * 100) if total_groups > 0 else 0
    n_tables = len(result["seats"])
    print("=" * 55)
    print("新フロアレイアウト・通行人グループシミュレーション結果")
    print("=" * 55)
    print(f"卓数（カウンター含む） : {n_tables}卓　総収容人数: {result['total_seats_capacity']}人")
    print(f"店の価格帯              : {STORE_PRICE_LEVEL}")

    # ↓ 追加：初期満席の内訳
    print(f"窓側　　: 全{result['total_window_tables']}卓（上限{result['total_window_capacity']}人）"
          f" のうち初期満席 {result['initial_window_tables']}卓 / {result['initial_window_people']}人")
    print(f"非窓側　: 全{result['total_nonwindow_tables']}卓（上限{result['total_nonwindow_capacity']}人）"
          f" のうち初期満席 {result['initial_nonwindow_tables']}卓 / {result['initial_nonwindow_people']}人")


    print(f"判断した人数の総数      : {total}人（{total_groups}グループ）")
    print(f"入店した人数            : {result['entered_total']}人（{result['groups_entered']}グループ）")
    print(f"素通りした人数          : {result['passed_total']}人（{result['groups_passed']}グループ）")
    print(f"入店率（人数ベース）     : {rate:.1f}%")
    print(f"入店率（グループベース） : {group_rate:.1f}%")
    print(f"最終的な使用卓数        : {result['occupied_seat_count_history'][-1]}/{n_tables}")
    if result["entered_budgets"]:
        avg_budget = sum(result["entered_budgets"]) / len(result["entered_budgets"])
        avg_crowd = sum(result["entered_crowd_tol"]) / len(result["entered_crowd_tol"])
        print(f"入店グループの平均budget         : {avg_budget:.2f}")
        print(f"入店グループの平均crowd_tolerance: {avg_crowd:.2f}")
    smoker_rate = (result["entered_smokers"] / result["total_smokers"] * 100) if result["total_smokers"] > 0 else 0
    print(f"喫煙者グループの入店率           : {smoker_rate:.1f}% ({result['entered_smokers']}/{result['total_smokers']}グループ)")
    print("=" * 55)

#通常客と喫煙者の来客数の推移グラフ
def plot_customer_graph(result, save_path="customer_graph.png"):
    # --- 全体（通常客＋喫煙者含む） ---
    walking = result["walking_history"]
    entered = result["entered_history"]
    passed = result["passed_history"]

    # --- 喫煙者 ---
    smoker_walking = result["smoker_walking_history"]
    smoker_entered = result["smoker_entered_history"]
    smoker_passed = result["smoker_passed_history"]

    fig, ax = plt.subplots(figsize=(10, 6))

    # --- 全体（実線） ---
    ax.plot(walking, label="歩行中（全体）", color="#5a9bd4", linewidth=2)
    ax.plot(entered, label="入店（全体・累積）", color="#e8956a", linewidth=2)
    ax.plot(passed, label="素通り（全体・累積）", color="#999", linewidth=2)

    # --- 喫煙者（破線） ---
    ax.plot(smoker_walking, label="歩行中（喫煙者）", color="#2a6bb4", linestyle="--")
    ax.plot(smoker_entered, label="入店（喫煙者・累積）", color="#d47a4a", linestyle="--")
    ax.plot(smoker_passed, label="素通り（喫煙者・累積）", color="#666", linestyle="--")

    ax.set_xlabel("フレーム")
    ax.set_ylabel("人数")
    ax.set_title("来客者の推移（通常客＋喫煙者）")
    ax.grid(alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    plt.close()
    print(f"来客者グラフを保存しました: {save_path}")

def plot_entered_only_graph(result, save_path="entered_only_graph.png"):
    entered = result["entered_history"]  # フレームごとの入店人数

    entered_cumsum = []
    total = 0
    for x in entered:
        total += x
        entered_cumsum.append(total)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(entered_cumsum, color="#e8956a", linewidth=2, label="入店（累積）")

    ax.set_xlabel("フレーム")
    ax.set_ylabel("人数")
    ax.set_title("入店人数の推移（累積）")
    ax.grid(alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    plt.close()
    print(f"入店人数グラフを保存しました: {save_path}")


def plot_left_only_graph(result, save_path="left_only_graph.png"):
    left = result.get("left_cumulative_history", [])

    if not left:
        print("退店履歴がありません。")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(left, color="#5a9bd4", linewidth=2, label="退店（累積）")

    ax.set_xlabel("フレーム")
    ax.set_ylabel("人数")
    ax.set_title("退店人数の推移（累積）")
    ax.grid(alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    plt.close()
    print(f"退店人数グラフを保存しました: {save_path}")

def plot_left_frame_graph(result, save_path="left_frame_graph.png"):
    left = result.get("left_cumulative_frame", [])

    if not left:
        print("退店履歴がありません。")
        return

    # ★★★ ここで累積化し直す ★★★
    left_cumsum = []
    total = 0
    for x in left:
        total += x
        left_cumsum.append(total)
    # ★★★ ここまで ★★★

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(left_cumsum, color="#5a9bd4", linewidth=2, label="退店（累積・フレーム軸）")

    ax.set_xlabel("フレーム")
    ax.set_ylabel("人数")
    ax.set_title("退店人数の推移（累積・フレーム軸）")
    ax.grid(alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    plt.close()
    print(f"退店人数（フレーム軸）グラフを保存しました: {save_path}")

#退店理由のグラフ
def plot_leave_reason_graph(result, save_path="leave_reason_graph.png"):
    leave_history = result.get("leave_history", [])

    if not leave_history:
        print("退店履歴がありません。")
        return

    # 理由ごとにカウント
    reasons = ["finished_meal", "too_crowded", "too_expensive", "go_smoking", "random"]
    reason_labels = {
        "finished_meal": "食事終了",
        "too_crowded": "混雑で退店",
        "too_expensive": "価格帯不一致",
        "go_smoking": "喫煙所へ",
        "random": "ランダム退店"
    }

    reason_counts = {r: 0 for r in reasons}

    for (_, size, is_smoker, reason) in leave_history:
        if reason in reason_counts:
            reason_counts[reason] += size

    # グラフ描画
    fig, ax = plt.subplots(figsize=(10, 6))

    labels = [reason_labels[r] for r in reasons]
    counts = [reason_counts[r] for r in reasons]

    ax.bar(labels, counts, color=["#c0622a", "#e8956a", "#999", "#5a9bd4", "#8b9e6b"])

    ax.set_title("退店理由の人数分布")
    ax.set_ylabel("人数")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    plt.close()

    print(f"退店理由グラフを保存しました: {save_path}")

def plot_window_occupancy_comparison(result_window, result_nonwindow, save_path="window_occupancy_comparison.png"):
    """窓側満席スタート vs 非窓側満席スタートで、窓側の埋まり率がどう推移するかを比較"""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(result_window["window_occ_history"],
            label="窓側埋まり率（窓側満席スタート）", color="#c0622a", linewidth=2)
    ax.plot(result_nonwindow["window_occ_history"],
            label="窓側埋まり率（非窓側満席スタート）", color="#5a9bd4", linewidth=2)

    ax.set_xlabel("フレーム")
    ax.set_ylabel("窓側席の埋まり率")
    ax.set_ylim(0, 1.05)
    ax.set_title("窓側席の埋まり率の推移比較")
    ax.grid(alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    plt.close()
    print(f"窓側埋まり率比較グラフを保存しました: {save_path}")

# ============================================================
# 座席の色を決める共通ヘルパー
# ============================================================
def seat_color(seat, occupied):
    if occupied:
        return SEAT_COLOR_OCC_VISIBLE if seat.is_window else SEAT_COLOR_OCC_HIDDEN
    else:
        return SEAT_COLOR_EMPTY_VISIBLE if seat.is_window else SEAT_COLOR_EMPTY_HIDDEN


def _draw_static_layout(ax, seats):
    """フロアの壁・入口・券売所・全座席を描画し、座席パッチの辞書を返す。"""
    ax.set_xlim(-1.5, FLOOR_W + 1.5)
    ax.set_ylim(ROAD_Y - 1.3, FLOOR_H + 1.0)
    ax.set_aspect("equal")
    ax.axis("off")

    # 外壁
    ax.add_patch(mpatches.Rectangle((0, 0), FLOOR_W, FLOOR_H, facecolor="#fff7ee",
                                     edgecolor="black", linewidth=2, zorder=1))
    # 喫煙所
    ax.add_patch(mpatches.Rectangle((0, 9.2), 3.2, 2.8, facecolor="#dddddd",
                                     edgecolor="black", zorder=2))
    ax.text(1.6, 10.6, "喫煙所", ha="center", va="center", fontsize=8, zorder=3)
    # 入口
    ax.add_patch(mpatches.Rectangle((0, -0.4), 1.8, 0.4, facecolor="white",
                                     edgecolor="black", zorder=2))
    ax.text(0.9, -0.9, "入口", ha="center", va="center", fontsize=9, zorder=3)

   # 見える範囲の目安（入口手前の4人席×12列を枠で囲む）
    row12_x0, row12_x1 = 2.3, 23.3
    row12_y, row12_h = 0.6, 2.3
    view_rect = mpatches.Rectangle(
        (row12_x0 - 0.15, row12_y - 0.15),
        (row12_x1 - row12_x0) + 0.3, row12_h + 0.3,
        fill=False, edgecolor="#5a9bd4", linewidth=1.5,
        linestyle="--", alpha=0.6, zorder=2
    )

    ax.add_patch(view_rect)
    seat_patches = {}
    for s in seats:
        color = seat_color(s, s.occupied)
        rect = mpatches.FancyBboxPatch(
            (s.x, s.y), s.w, s.h,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            facecolor=color, edgecolor="#555", linewidth=0.7, zorder=3
        )
        ax.add_patch(rect)
        seat_patches[s.index] = rect

    ax.text(13.3, 10.8, "カウンター(1人席)", ha="center", fontsize=9, zorder=3)
    ax.text(7.6, 8.9, "2人席", ha="center", fontsize=9, color="#c0622a", zorder=3)
    ax.text(21.6, 8.9, "4人席×5", ha="center", fontsize=9, zorder=3)
    ax.text(12.8, 3.1, "4人席×12", ha="center", fontsize=9, zorder=3)

    # 道路
    road_x_min, road_x_max = ENTRANCE_X - 8.0, FLOOR_W + 8.0
    road_rect = mpatches.Rectangle(
        (road_x_min, ROAD_Y - 0.8), road_x_max - road_x_min, 1.6,
        facecolor="#e8e8e8", edgecolor="none", zorder=4
    )
    ax.add_patch(road_rect)
    ax.text(road_x_min + 0.3, ROAD_Y - 1.15, "道路", fontsize=9, color="#999", zorder=5)

    return seat_patches


# ============================================================
# アニメーション(GIF)作成
# ============================================================
STATE_COLOR = {
    "walking": "#5a9bd4",
    "entering": "#e8956a",
    "seated": "#c0622a",
}


def create_animation(result, save_path="restaurant_layout_simulation.gif"):
    history = result["history"]
    occupied_seat_count_history = result["occupied_seat_count_history"]
    seats = result["seats"]
    road_x_start = result["road_x_start"]

    fig, (ax_map, ax_chart) = plt.subplots(
        2, 1, figsize=(13, 10.5), gridspec_kw={"height_ratios": [2.6, 1]}
    )

    seat_patches = _draw_static_layout(ax_map, seats)
    ax_map.set_xlim(ENTRANCE_X - 9.0, max(road_x_start + 1.0, FLOOR_W + 2.0))

    title_text = ax_map.set_title("")
    pedestrian_dots = ax_map.scatter([], [], s=130, zorder=6)

    # --- 下段: 使用卓数の推移グラフ ---
    n_tables = len(seats)
    ax_chart.set_xlim(0, len(occupied_seat_count_history) - 1)
    ax_chart.set_ylim(0, n_tables + 1)
    ax_chart.axhline(n_tables, color="gray", linestyle="--", linewidth=1, label="満卓")
    ax_chart.set_xlabel("フレーム")
    ax_chart.set_ylabel("使用卓数")
    ax_chart.grid(alpha=0.3)
    line, = ax_chart.plot([], [], color="#c0622a", linewidth=2)
    point, = ax_chart.plot([], [], "o", color="#c0622a", markersize=5)
    ax_chart.legend(loc="lower right", fontsize=9)

    def update(frame):
        snap = history[frame]

        for s in seats:
            occ, _ = snap["seats_state"][s.index]
            seat_patches[s.index].set_facecolor(seat_color(s, occ))

        xs, ys, colors = [], [], []
        walking_people = 0
        for grp in snap["groups"]:
            if grp["state"] == "walking":
                walking_people += grp["size"]
            for off in grp["offsets"]:
                xs.append(grp["x"] + off)
                ys.append(grp["y"])
                colors.append(STATE_COLOR.get(grp["state"], "#999"))

        pedestrian_dots.set_offsets(list(zip(xs, ys)) if xs else [[None, None]])
        pedestrian_dots.set_color(colors)

        occ_now = occupied_seat_count_history[frame]
        title_text.set_text(
            f"フレーム {frame}　歩行中の人数 {walking_people}　使用卓数 {occ_now}/{n_tables}"
        )

        line.set_data(range(frame + 1), occupied_seat_count_history[:frame + 1])
        point.set_data([frame], [occ_now])

        return list(seat_patches.values()) + [pedestrian_dots, line, point, title_text]

    ani = animation.FuncAnimation(
        fig, update, frames=len(history), interval=110, blit=False
    )

    plt.tight_layout()
    print("GIFを生成中...")
    ani.save(save_path, writer="pillow", fps=10)
    plt.close()
    print(f"GIFを保存しました: {save_path}")


# ============================================================
# 静止画：店舗レイアウト確認用
# ============================================================
def plot_layout_check(seats, save_path="layout_check.png"):
    fig, ax = plt.subplots(figsize=(9, 6))
    _draw_static_layout(ax, seats)
    for s in seats:
        ax.text(s.cx, s.cy, str(s.capacity), ha="center", va="center", fontsize=6, zorder=4)
    ax.set_xlim(-1.5, FLOOR_W + 1.5)
    ax.set_ylim(-2.0, FLOOR_H + 0.8)
    ax.set_title("座席配置の確認図（数字=収容人数、オレンジ=見える席／緑=見えない席）")
    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    plt.close()
    print(f"レイアウト確認図を保存しました: {save_path}")


# ============================================================
# メイン処理
# ============================================================
if __name__ == "__main__":
    result = run_simulation()
    print_results(result)
    plot_layout_check(result["seats"])
    create_animation(result)
    plot_customer_graph(result)
    plot_leave_reason_graph(result)
    result_window = run_simulation(window_fill_rate=1.0, nonwindow_fill_rate=0.0)
    result_nonwindow = run_simulation(window_fill_rate=0.0, nonwindow_fill_rate=1.0)
    plot_entered_only_graph(result_window, "entered_only_window.png")
    plot_left_only_graph(result_window, "left_only_window.png")
    plot_left_frame_graph(result_window, "left_frame_window.png")



    # ============================================================
    # 実験：窓側だけ初期満席 × 座席選択戦略（前 or 奥から詰める）
    # ============================================================

    # --- 窓側満席 × 前から詰める（window戦略） ---
    result_window_front = run_simulation(
        window_fill_rate=1.0,
        nonwindow_fill_rate=0.0,
        spawn_interval=SPAWN_INTERVAL,
        seed=RANDOM_SEED,
        seat_strategy="window"
    )
    print("=== 実験：窓側だけ初期満席 × 前から詰める ===")
    print_results(result_window_front)
    plot_customer_graph(result_window_front, "customer_window_front.png")
    plot_leave_reason_graph(result_window_front, save_path="leave_reason_window_front.png")

    # --- 窓側満席 × 奥から詰める（depth戦略） ---
    result_window_depth = run_simulation(
        window_fill_rate=1.0,
        nonwindow_fill_rate=0.0,
        spawn_interval=SPAWN_INTERVAL,
        seed=RANDOM_SEED,
        seat_strategy="depth"
    )
    print("=== 実験：窓側だけ初期満席 × 奥から詰める ===")
    print_results(result_window_depth)
    plot_customer_graph(result_window_depth, "customer_window_depth.png")
    plot_leave_reason_graph(result_window_depth, save_path="leave_reason_window_depth.png")

    # ============================================================
    # 実験：非窓側だけ初期満席 × 座席選択戦略（前 or 奥から詰める）
    # ============================================================

    # --- 非窓側満席 × 前から詰める（window戦略） ---
    result_nonwindow_front = run_simulation(
        window_fill_rate=0.0,
        nonwindow_fill_rate=1.0,
        spawn_interval=SPAWN_INTERVAL,
        seed=RANDOM_SEED,
        seat_strategy="window"
    )
    print("=== 実験：非窓側だけ初期満席 × 前から詰める ===")
    print_results(result_nonwindow_front)
    plot_customer_graph(result_nonwindow_front, save_path="customer_nonwindow_front.png")
    plot_leave_reason_graph(result_nonwindow_front, save_path="leave_reason_nonwindow_front.png")

    # --- 非窓側満席 × 奥から詰める（depth戦略） ---
    result_nonwindow_depth = run_simulation(
        window_fill_rate=0.0,
        nonwindow_fill_rate=1.0,
        spawn_interval=SPAWN_INTERVAL,
        seed=RANDOM_SEED,
        seat_strategy="depth"
    )
    print("=== 実験：非窓側だけ初期満席 × 奥から詰める ===")
    print_results(result_nonwindow_depth)
    plot_customer_graph(result_nonwindow_depth, save_path="customer_nonwindow_depth.png")
    plot_leave_reason_graph(result_nonwindow_depth, save_path="leave_reason_nonwindow_depth.png")

    plot_window_occupancy_comparison(result_window_front, result_nonwindow_front)

    # ============================================================
    # 比較リザルト
    # ============================================================
    print("\n=== 窓側 vs 非窓側 × 前 vs 奥 詰め比較 ===")
    print(f"窓側満席・前詰め   → 最終使用卓数: {result_window_front['occupied_seat_count_history'][-1]} / 入店人数: {result_window_front['entered_total']}")
    print(f"窓側満席・奥詰め   → 最終使用卓数: {result_window_depth['occupied_seat_count_history'][-1]} / 入店人数: {result_window_depth['entered_total']}")
    print(f"非窓側満席・前詰め → 最終使用卓数: {result_nonwindow_front['occupied_seat_count_history'][-1]} / 入店人数: {result_nonwindow_front['entered_total']}")
    print(f"非窓側満席・奥詰め → 最終使用卓数: {result_nonwindow_depth['occupied_seat_count_history'][-1]} / 入店人数: {result_nonwindow_depth['entered_total']}")