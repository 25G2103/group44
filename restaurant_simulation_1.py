"""
飲食店来客シミュレーション（社会的証明モデル）
------------------------------------------------
入口付近の席の埋まり具合（視認できる混雑度）が
通行人エージェントの入店判断に与える影響をシミュレーションする。

社会的証明（Social Proof）：
  他者が利用している様子を見ると、自分もその場を
  良いものだと判断しやすくなる心理的傾向。
"""

import random
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
# 定数・パラメータ
# ============================================================
COLS = 6
ROWS = 4
TOTAL_SEATS = COLS * ROWS
ENTRANCE_COLS = 2          # 入口側の列数（通行人から見える列）

LEAVE_PROB = 0.10           # 1ステップあたりの退席確率
N_STEPS = 60                # シミュレーションするステップ数
RANDOM_SEED = 42            # 再現性のための乱数シード（Noneでランダム）

PRESETS = [
    {"label": "空席多い (0%)",   "rate": 0.0},
    {"label": "少し埋まり (30%)", "rate": 0.3},
    {"label": "半分埋まり (50%)", "rate": 0.5},
    {"label": "かなり埋まり (70%)", "rate": 0.7},
]


# ============================================================
# 関数定義
# ============================================================
def get_initial_seats(fill_rate):
    """入口側(左ENTRANCE_COLS列)を fill_rate の割合で埋めた初期状態を作る"""
    seats = []
    for i in range(TOTAL_SEATS):
        col = i % COLS
        is_entrance = col < ENTRANCE_COLS
        if is_entrance and random.random() < fill_rate:
            seats.append("occupied")
        else:
            seats.append("empty")
    return seats


def get_visible_occupancy_rate(seats):
    """入口側の席の埋まり率を計算する"""
    entrance_seats = [s for i, s in enumerate(seats) if (i % COLS) < ENTRANCE_COLS]
    occupied = entrance_seats.count("occupied")
    return occupied / len(entrance_seats)


def calc_entry_probability(visible_rate):
    """社会的証明：見える埋まり率が高いほど入店確率が上がる
    0% -> 5%, 50% -> 45%, 100% -> 85%
    """
    return 0.05 + visible_rate * 0.80


def step(seats):
    """1ステップ進める：入店判断 → 着席 → ランダム退席"""
    visible_rate = get_visible_occupancy_rate(seats)
    entry_prob = calc_entry_probability(visible_rate)
    next_seats = seats.copy()

    # 通行人の入店判断
    if random.random() < entry_prob:
        empty_idx = [i for i, s in enumerate(next_seats) if s == "empty"]
        if empty_idx:
            # 入口に近い列を優先して着席
            empty_idx.sort(key=lambda i: i % COLS)
            next_seats[empty_idx[0]] = "occupied"

    # ランダム退席
    occupied_idx = [i for i, s in enumerate(next_seats) if s == "occupied"]
    if occupied_idx and random.random() < LEAVE_PROB:
        leaving = random.choice(occupied_idx)
        next_seats[leaving] = "empty"

    return next_seats, visible_rate, entry_prob


# ============================================================
# シミュレーション実行
# ============================================================
def run_simulation(initial_rate, n_steps=N_STEPS, seed=RANDOM_SEED):
    """1つの初期条件についてシミュレーションを実行し、
    各ステップの座席状態(seats)も含めて履歴を返す（アニメーション用）"""
    if seed is not None:
        random.seed(seed)

    seats = get_initial_seats(initial_rate)

    seats_history = [seats.copy()]
    occupied_history = [seats.count("occupied")]
    visible_rate_history = [get_visible_occupancy_rate(seats)]
    entry_prob_history = [calc_entry_probability(visible_rate_history[0])]

    for _ in range(n_steps):
        seats, vis_rate, entry_prob = step(seats)
        seats_history.append(seats.copy())
        occupied_history.append(seats.count("occupied"))
        visible_rate_history.append(vis_rate)
        entry_prob_history.append(entry_prob)

    return {
        "final_seats": seats,
        "seats_history": seats_history,
        "occupied_history": occupied_history,
        "visible_rate_history": visible_rate_history,
        "entry_prob_history": entry_prob_history,
    }


def run_all_presets(n_steps=N_STEPS, seed=RANDOM_SEED):
    """4つの初期条件すべてでシミュレーションを実行する"""
    results = {}
    for preset in PRESETS:
        # 各プリセットで同じシードから始める（条件間の比較を公平にする）
        results[preset["label"]] = run_simulation(preset["rate"], n_steps, seed)
    return results


# ============================================================
# 結果の表示
# ============================================================
def print_results(results):
    print("=" * 60)
    print(f"飲食店来客シミュレーション結果（{N_STEPS}ステップ経過後）")
    print("=" * 60)
    for label, res in results.items():
        final_count = res["occupied_history"][-1]
        max_count = max(res["occupied_history"])
        avg_count = sum(res["occupied_history"]) / len(res["occupied_history"])
        print(f"{label:14s} | 最終在席数: {final_count:2d}/{TOTAL_SEATS} | "
              f"最大: {max_count:2d} | 平均: {avg_count:5.1f}")
    print("=" * 60)


# ============================================================
# 静止画グラフ作成
# ============================================================
def plot_results(results, save_path="restaurant_simulation_result.png"):
    colors = ["#9e9e9e", "#e8956a", "#c0622a", "#8b3a10"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # --- (1) 在席数の推移（条件別比較） ---
    ax = axes[0, 0]
    for (label, res), color in zip(results.items(), colors):
        ax.plot(res["occupied_history"], label=label, color=color, linewidth=2)
    ax.axhline(TOTAL_SEATS, color="gray", linestyle="--", linewidth=1, label="満席")
    ax.set_title("在席数の推移（初期条件別）")
    ax.set_xlabel("ステップ")
    ax.set_ylabel("在席数（人）")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # --- (2) 入店確率の推移 ---
    ax = axes[0, 1]
    for (label, res), color in zip(results.items(), colors):
        prob_pct = [p * 100 for p in res["entry_prob_history"]]
        ax.plot(prob_pct, label=label, color=color, linewidth=2)
    ax.set_title("入店確率の推移")
    ax.set_xlabel("ステップ")
    ax.set_ylabel("入店確率（%）")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # --- (3) 最終的な店内レイアウト（代表例：50%プリセット） ---
    ax = axes[1, 0]
    sample_label = "半分埋まり (50%)"
    final_seats = results[sample_label]["final_seats"]
    for i, s in enumerate(final_seats):
        col = i % COLS
        row = i // COLS
        is_entrance = col < ENTRANCE_COLS
        if s == "occupied":
            color = "#c0622a" if is_entrance else "#8b9e6b"
        else:
            color = "#fde8d8" if is_entrance else "#eee"
        rect = mpatches.FancyBboxPatch(
            (col, ROWS - row - 1), 0.85, 0.85,
            boxstyle="round,pad=0.02,rounding_size=0.1",
            facecolor=color, edgecolor="#bbb"
        )
        ax.add_patch(rect)
    ax.axvline(ENTRANCE_COLS, color="#c0622a", linestyle="--", linewidth=1.5)
    ax.text(ENTRANCE_COLS / 2, ROWS + 0.3, "入口から見える席",
            ha="center", fontsize=9, color="#c0622a", fontweight="bold")
    ax.text(ENTRANCE_COLS + (COLS - ENTRANCE_COLS) / 2, ROWS + 0.3, "奥の席",
            ha="center", fontsize=9, color="#888")
    ax.set_xlim(0, COLS)
    ax.set_ylim(0, ROWS + 0.7)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"最終レイアウト例（{sample_label}）", pad=10)

    # --- (4) 最終在席数の比較（バー） ---
    ax = axes[1, 1]
    labels = list(results.keys())
    finals = [res["occupied_history"][-1] for res in results.values()]
    bars = ax.bar(labels, finals, color=colors)
    ax.set_title("最終在席数の比較")
    ax.set_ylabel("在席数（人）")
    ax.set_ylim(0, TOTAL_SEATS + 3)
    for bar, val in zip(bars, finals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val}人", ha="center", va="bottom", fontsize=9)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"グラフを保存しました: {save_path}")
    plt.close()


# ============================================================
# アニメーション(GIF)作成
# ============================================================
def create_animation(result, label, save_path="restaurant_simulation.gif"):
    """1つの初期条件のシミュレーション結果から、
    座席の埋まり方が変化していく様子をGIFアニメーションとして保存する"""
    seats_history = result["seats_history"]
    occupied_history = result["occupied_history"]
    entry_prob_history = result["entry_prob_history"]

    fig, (ax_seats, ax_chart) = plt.subplots(
        1, 2, figsize=(11, 5), gridspec_kw={"width_ratios": [1, 1.1]}
    )

    # --- 左側: 座席レイアウト ---
    ax_seats.set_xlim(0, COLS)
    ax_seats.set_ylim(0, ROWS + 0.7)
    ax_seats.set_aspect("equal")
    ax_seats.axis("off")
    ax_seats.axvline(ENTRANCE_COLS, color="#c0622a", linestyle="--", linewidth=1.5)
    ax_seats.text(ENTRANCE_COLS / 2, ROWS + 0.3, "入口から見える席",
                  ha="center", fontsize=9, color="#c0622a", fontweight="bold")
    ax_seats.text(ENTRANCE_COLS + (COLS - ENTRANCE_COLS) / 2, ROWS + 0.3, "奥の席",
                  ha="center", fontsize=9, color="#888")

    # 席の四角を最初に作成し、毎フレームで色だけ更新する
    seat_patches = []
    for i in range(TOTAL_SEATS):
        col = i % COLS
        row = i // COLS
        rect = mpatches.FancyBboxPatch(
            (col, ROWS - row - 1), 0.85, 0.85,
            boxstyle="round,pad=0.02,rounding_size=0.1",
            facecolor="#eee", edgecolor="#bbb"
        )
        ax_seats.add_patch(rect)
        seat_patches.append(rect)

    title_text = ax_seats.set_title("")

    # --- 右側: 在席数の推移グラフ ---
    ax_chart.set_xlim(0, len(occupied_history) - 1)
    ax_chart.set_ylim(0, TOTAL_SEATS + 2)
    ax_chart.axhline(TOTAL_SEATS, color="gray", linestyle="--", linewidth=1, label="満席")
    ax_chart.set_xlabel("ステップ")
    ax_chart.set_ylabel("在席数（人）")
    ax_chart.grid(alpha=0.3)
    line, = ax_chart.plot([], [], color="#c0622a", linewidth=2)
    point, = ax_chart.plot([], [], "o", color="#c0622a", markersize=6)
    ax_chart.legend(loc="lower right", fontsize=9)

    def update(frame):
        seats = seats_history[frame]
        for i, s in enumerate(seats):
            col = i % COLS
            is_entrance = col < ENTRANCE_COLS
            if s == "occupied":
                color = "#c0622a" if is_entrance else "#8b9e6b"
            else:
                color = "#fde8d8" if is_entrance else "#eee"
            seat_patches[i].set_facecolor(color)

        occ = occupied_history[frame]
        prob = entry_prob_history[frame]
        title_text.set_text(
            f"{label}　ステップ {frame}　在席 {occ}/{TOTAL_SEATS}　入店確率 {prob*100:.0f}%"
        )

        line.set_data(range(frame + 1), occupied_history[:frame + 1])
        point.set_data([frame], [occ])

        return seat_patches + [line, point, title_text]

    ani = animation.FuncAnimation(
        fig, update, frames=len(seats_history), interval=200, blit=False
    )

    plt.tight_layout()
    print(f"GIFを生成中... ({label})")
    ani.save(save_path, writer="pillow", fps=5)
    plt.close()
    print(f"GIFを保存しました: {save_path}")


# ============================================================
# メイン処理
# ============================================================
if __name__ == "__main__":
    results = run_all_presets()
    print_results(results)
    plot_results(results)

    # 代表として「空席多い(0%)」と「かなり埋まり(70%)」をGIFアニメーション化
    # （対比がわかりやすいため。全プリセットをGIF化したい場合はループで回す）
    create_animation(results["空席多い (0%)"], "空席多い (0%)",
                      save_path="restaurant_empty_start.gif")
    create_animation(results["かなり埋まり (70%)"], "かなり埋まり (70%)",
                      save_path="restaurant_crowded_start.gif")
