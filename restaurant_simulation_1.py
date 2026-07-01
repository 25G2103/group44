"""
飲食店来客シミュレーション（実店舗レイアウト + 通行人グループ + 視野モデル）
--------------------------------------------------------------------------
店内レイアウト（入口は右側、x=0を基準に左へ奥行きが伸びる）：
  [4人席12卓(窓)] - [2人席11卓(窓側5卓+中央1卓+厨房側5卓)] - [4人席5卓(壁内)]
  カウンター10席は厨房前の別レーン（壁内・道路からは見えない）
 
視野モデル：
  窓に面した座席（4人席12卓＋2人席の窓側5卓）だけが道路から見える。
  通行人グループは、自分の位置(x)からVIEW_RANGE以内にある「窓側座席」だけを
  見て、混雑具合（社会的証明）を判断材料にする。
 
エージェントの入店判断：
  ①価格帯の適合度（budget） ②混雑許容度（crowd_tolerance）
  ③社会的証明（視野内の埋まり率）の重み付き合成で入店確率を決める。
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
# 座席サイズ・隙間の基準値
# ============================================================
SEAT_W4 = 1.0
SEAT_W2 = 0.75
SEAT_WC = 0.55
GAP4 = 0.5
GAP2 = 0.4
GAPC = 0.2
BLOCK_GAP = 1.3
 
SEAT_DRAW_SIZE = {"table4": 0.85, "table2": 0.62, "counter": 0.42}
SEAT_COLOR_OCC = {"table4": "#c0622a", "table2": "#8b9e6b", "counter": "#6a8fb5"}
SEAT_COLOR_EMPTY = {"table4": "#eee", "table2": "#eee", "counter": "#e4e4e4"}
SEAT_COLOR_WINDOW_EMPTY = "#fde8d8"
 
# 奥行き方向(y)の基準
Y_WINDOW = 0.0
Y_KITCHEN = 1.4
Y_COUNTER = 2.6
 
ENTRANCE_X = -35.0   # 入口を左端に変更（座席は右方向へ伸びる）
 
ROAD_Y = -2.5
 
# ============================================================
# 通行人グループ・エージェントのパラメータ
# ============================================================
GROUP_SIZES = [1, 2, 4]
GROUP_MEMBER_GAP = 0.35
 
WALK_SPEED_NORMAL = 0.35
WALK_SPEED_SLOW = 0.10
SLOWDOWN_RANGE = 4.0
 
VIEW_RANGE = 5.0
VIEW_DECISION_X = ENTRANCE_X
 
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
    def __init__(self, index, x, y, capacity, kind, is_window):
        self.index = index
        self.x = x
        self.y = y
        self.capacity = capacity
        self.kind = kind             # "table4" / "table2" / "counter"
        self.is_window = is_window   # 道路から見える座席か
        self.occupied = False
        self.party_size = 0
 
 
def create_layout_seats():
    """店内レイアウトを生成する。入口は左側(ENTRANCE_X)で、
    座席は右方向（x増加方向）へ積み上げる。"""
    seats = []
    index = 0

    def add_row(n, width, gap, x_start, y, is_window, capacity, kind):
        """x_startを起点に右方向(+x)へn卓を並べる"""
        nonlocal index
        x = x_start
        for _ in range(n):
            seats.append(Seat(index, x, y, capacity, kind, is_window))
            index += 1
            x += (width + gap)
        return x

    # ① 入口右の 4人席 12卓（窓側・見える）
    x_cursor = ENTRANCE_X + 1.0
    x_cursor = add_row(12, SEAT_W4, GAP4, x_cursor, Y_WINDOW,
                        is_window=True, capacity=4, kind="table4")
    x_cursor += BLOCK_GAP

    # ② 2人席：窓側5卓（見える）
    window2_start = x_cursor
    x_cursor = add_row(5, SEAT_W2, GAP2, x_cursor, Y_WINDOW,
                        is_window=True, capacity=2, kind="table2")
    window2_end = x_cursor - (SEAT_W2 + GAP2)

    # 中央の1卓（窓と厨房の中間の高さ）
    mid_x = (window2_start + window2_end) / 2
    mid_y = (Y_WINDOW + Y_KITCHEN) / 2
    seats.append(Seat(index, mid_x, mid_y, 2, "table2", is_window=False))
    index += 1

    # 2人席：厨房側5卓（見えない）
    add_row(5, SEAT_W2, GAP2, window2_start, Y_KITCHEN,
            is_window=False, capacity=2, kind="table2")

    x_cursor += BLOCK_GAP

    # ③ 奥の4人席 5卓（見えない）
    x_cursor = add_row(5, SEAT_W4, GAP4, x_cursor, Y_WINDOW,
                        is_window=False, capacity=4, kind="table4")

    # ④ カウンター席10席（厨房前・右奥、見えない）
    counter_start = ENTRANCE_X + 1.0
    add_row(10, SEAT_WC, GAPC, counter_start, Y_COUNTER,
            is_window=False, capacity=1, kind="counter")

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
        self.size = size
        self.state = "walking"     # walking / entering / seated / left / exited
        self.decided = False
        self.target_seat = None
 
        self.budget = random.randint(1, 5)
        self.crowd_tolerance = random.random()
        self.is_smoker = random.random() < 0.3   # 30%の確率で喫煙者
 
    def current_speed(self):
        d = abs(self.x - ENTRANCE_X)
        if d < SLOWDOWN_RANGE:
            ratio = d / SLOWDOWN_RANGE
            return WALK_SPEED_SLOW + (WALK_SPEED_NORMAL - WALK_SPEED_SLOW) * ratio
        return WALK_SPEED_NORMAL
 
    def visible_seats(self, seats):
        """窓に面した座席のうち、自分の位置からVIEW_RANGE以内にあるものだけが見える"""
        return [s for s in seats if s.is_window and abs(s.x - self.x) <= VIEW_RANGE]
 
    def member_offsets(self):
        n = self.size
        return [(-(n - 1) / 2 + i) * GROUP_MEMBER_GAP for i in range(n)]
 
 
def pick_group_size():
    return random.choice(GROUP_SIZES)
 
 
def find_seat_for_group(seats, group_size):
    """
    エージェントの移動ルール：
      ① 人数以上の席であること（capacity >= group_size）
      ② 奥側から座る（入口から離れた席 = x が大きい席を優先）
    """
    # ① 人数以上の席だけを候補にする
    candidates = [s for s in seats if (not s.occupied) and (s.capacity >= group_size)]
    if not candidates:
        return None

    # ② 奥側（入口から遠い = x が大きい）を優先
    #    ただし、同じ奥行きなら「席の小ささ（capacity）」が小さい方を優先
    candidates.sort(key=lambda s: (s.x, -s.capacity), reverse=True)

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
                    initial_fill_rate=INITIAL_FILL_RATE):
    seats = create_layout_seats()
 
    if seed is not None:
        random.seed(seed)
    Group._id_counter = 0
 
    for s in seats:
        if random.random() < initial_fill_rate:
            s.occupied = True
            s.party_size = s.capacity if s.kind != "counter" else 1
 
    road_x_start = max(s.x for s in seats) + 6.0   # 通行人は右端から出現
    road_x_end = ENTRANCE_X - 6.0                   # 左端（入口の外）まで歩いたら退場
 
    groups = []
    history = []
    occupied_seat_count_history = []
    entered_total = 0
    passed_total = 0
    groups_entered = 0
    groups_passed = 0
    entered_budgets = []
    entered_crowd_tol = []
    entered_smokers = 0    # 入店した喫煙者グループ数
    total_smokers = 0      # 判断した喫煙者グループ数
 
    total_seats_capacity = sum(s.capacity for s in seats)
 
    for frame in range(n_steps):
        if frame % spawn_interval == 0:
            size = pick_group_size()
            groups.append(Group(x=road_x_start, size=size))
 
        for g in groups:
            if g.state == "walking":
                if not g.decided and g.x <= VIEW_DECISION_X:
                    visible = g.visible_seats(seats)
                    if visible:
                        occ_rate = sum(1 for s in visible if s.occupied) / len(visible)
                    else:
                        occ_rate = 0.0
                    occupied_capacity = sum(s.party_size for s in seats)
                    current_occupancy_rate = occupied_capacity / total_seats_capacity
                    entry_prob = calc_entry_probability(g, occ_rate, current_occupancy_rate)
                    g.decided = True
                    if g.is_smoker:
                        total_smokers += 1

                    target = find_seat_for_group(seats, g.size)
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
 
                if g.state == "walking":
                    g.x -= g.current_speed()    # 右から左へ歩く
                    if g.x < road_x_end:
                        g.state = "exited"
 
            elif g.state == "entering":
                dx = g.target_seat.x - g.x
                if abs(dx) < 0.15:
                    g.x = g.target_seat.x
                    g.state = "seated"
                else:
                    g.x += math.copysign(min(abs(dx), WALK_SPEED_SLOW), dx)
 
            elif g.state == "seated":
                if random.random() < LEAVE_PROB:
                    g.target_seat.occupied = False
                    g.target_seat.party_size = 0
                    g.state = "left"
 
        groups = [g for g in groups if g.state not in ("exited", "left")]
 
        snapshot = {
            "groups": [
                {"x": g.x,
                 "y": (g.target_seat.y if g.target_seat and g.state in ("entering", "seated") else ROAD_Y),
                 "state": g.state, "size": g.size,
                 "offsets": g.member_offsets()} for g in groups
            ],
            "seats_state": [(s.occupied, s.party_size) for s in seats],
        }
        history.append(snapshot)
        occupied_seat_count_history.append(sum(1 for s in seats if s.occupied))
 
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
    print("実店舗レイアウト・通行人グループシミュレーション結果")
    print("=" * 55)
    print(f"卓数（カウンター含む） : {n_tables}卓　総収容人数: {result['total_seats_capacity']}人")
    print(f"店の価格帯              : {STORE_PRICE_LEVEL}")
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
    road_x_end = result["road_x_end"]
 
    fig, (ax_map, ax_chart) = plt.subplots(
        2, 1, figsize=(13, 8.5), gridspec_kw={"height_ratios": [2.4, 1]}
    )
 
    seat_x_min = min(s.x for s in seats) - 1.0
    seat_x_max = max(s.x for s in seats) + 1.0
    seat_y_min = min(s.y for s in seats) - 0.7
    seat_y_max = max(s.y for s in seats) + 0.7
 
    road_x_min = min(road_x_start, seat_x_min) - 1.0
    road_x_max = max(road_x_end, seat_x_max) + 1.0
 
    ax_map.set_xlim(road_x_min, road_x_max)
    ax_map.set_ylim(ROAD_Y - 1.3, seat_y_max + 1.0)
    ax_map.set_aspect("equal")
    ax_map.axis("off")
 
    # 店舗の背景（道路より下のzorder）
    store_rect = mpatches.Rectangle(
        (seat_x_min - 0.3, seat_y_min - 0.3),
        (seat_x_max - seat_x_min) + 0.6, (seat_y_max - seat_y_min) + 0.6,
        facecolor="#fff7ee", edgecolor="#8a6a4a", linewidth=1.8, zorder=0
    )
    ax_map.add_patch(store_rect)
    ax_map.text(0, seat_y_max + 0.45, "店内座席エリア", fontsize=10,
                color="#7a4a20", fontweight="bold", ha="center", zorder=2)
 
    # 窓のライン（入口から右方向の窓側座席の手前）
    window_line_y = Y_WINDOW - 0.5
    ax_map.plot([ENTRANCE_X - 0.2, seat_x_max + 0.2], [window_line_y, window_line_y],
                color="#5a9bd4", linewidth=2.2, linestyle="--", zorder=2)
    ax_map.text((ENTRANCE_X + seat_x_max) / 2, window_line_y - 0.4,
                "窓（道路から見える範囲）", ha="center", fontsize=8.5,
                color="#3a78b5", fontweight="bold", zorder=2)

    # 入口・レジ（左端）
    ax_map.add_patch(mpatches.Rectangle(
        (ENTRANCE_X - 0.2, Y_WINDOW - 0.45), 0.45, 1.1,
        facecolor="#d9c08f", edgecolor="#8a6a4a", linewidth=1.1, zorder=2))
    ax_map.text(ENTRANCE_X + 0.05, Y_WINDOW + 0.8, "入口\nレジ", ha="center", fontsize=7.5,
                color="#5a4326", fontweight="bold", zorder=2)

    # 厨房ラベル（右奥）
    ax_map.text(seat_x_max - 0.3, seat_y_max + 0.1, "厨房", fontsize=9,
                color="#888", fontweight="bold", zorder=2)
 
    # 道路の背景（店舗より手前=zorderを高くして必ず見えるようにする）
    road_rect = mpatches.Rectangle(
        (road_x_min, ROAD_Y - 0.8), road_x_max - road_x_min, 1.6,
        facecolor="#e8e8e8", edgecolor="none", zorder=3
    )
    ax_map.add_patch(road_rect)
    ax_map.text(road_x_min + 0.3, ROAD_Y - 1.15, "道路", fontsize=9, color="#999", zorder=4)
 
    # 座席の描画
    seat_patches = []
    for s in seats:
        size = SEAT_DRAW_SIZE.get(s.kind, 0.7)
        base_color = SEAT_COLOR_WINDOW_EMPTY if s.is_window else SEAT_COLOR_EMPTY[s.kind]
        rect = mpatches.FancyBboxPatch(
            (s.x - size / 2, s.y - size / 2), size, size,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=base_color, edgecolor="#999", linewidth=0.8, zorder=2
        )
        ax_map.add_patch(rect)
        seat_patches.append(rect)
 
    title_text = ax_map.set_title("")
    pedestrian_dots = ax_map.scatter([], [], s=130, zorder=5)
 
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
 
        for patch, (occ, party_size), s in zip(seat_patches, snap["seats_state"], seats):
            if occ:
                patch.set_facecolor(SEAT_COLOR_OCC[s.kind])
            else:
                patch.set_facecolor(SEAT_COLOR_WINDOW_EMPTY if s.is_window else SEAT_COLOR_EMPTY[s.kind])
 
        xs, ys, colors = [], [], []
        walking_people = 0
        for grp in snap["groups"]:
            base_y = grp["y"] if grp["state"] != "walking" else ROAD_Y
            if grp["state"] == "walking":
                walking_people += grp["size"]
            for off in grp["offsets"]:
                xs.append(grp["x"] + off)
                ys.append(base_y)
                colors.append(STATE_COLOR.get(grp["state"], "#999"))
 
        pedestrian_dots.set_offsets(list(zip(xs, ys)) if xs else [[None, None]])
        pedestrian_dots.set_color(colors)
 
        occ_now = occupied_seat_count_history[frame]
        title_text.set_text(
            f"フレーム {frame}　歩行中の人数 {walking_people}　使用卓数 {occ_now}/{n_tables}"
        )
 
        line.set_data(range(frame + 1), occupied_seat_count_history[:frame + 1])
        point.set_data([frame], [occ_now])
 
        return seat_patches + [pedestrian_dots, line, point, title_text]
 
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
    fig, ax = plt.subplots(figsize=(13, 5))
    for s in seats:
        size = SEAT_DRAW_SIZE.get(s.kind, 0.7)
        color = SEAT_COLOR_WINDOW_EMPTY if s.is_window else "#eee"
        rect = mpatches.FancyBboxPatch(
            (s.x - size / 2, s.y - size / 2), size, size,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=color, edgecolor="#999"
        )
        ax.add_patch(rect)
        ax.text(s.x, s.y, str(s.capacity), ha="center", va="center", fontsize=6)
    ax.set_xlim(min(s.x for s in seats) - 1, ENTRANCE_X + 1)
    ax.set_ylim(min(s.y for s in seats) - 1, max(s.y for s in seats) + 1)
    ax.set_aspect("equal")
    ax.set_title("座席配置の確認図（数字=収容人数、オレンジ=窓側）")
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