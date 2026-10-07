"""6단계: 단계별 결과를 song_id로 합쳐 분위기 형용사를 붙이고 시각화한다.

입력: outputs/01_lyrics.csv, 03_va_predictions.csv, 05_song_emotion_color.csv
출력: outputs/final_result.csv, outputs/playlist_analysis.png
"""
from __future__ import annotations

import math

import pandas as pd

import config
from src.common import get_logger, load_playlist, read_csv, write_csv

log = get_logger(__name__)

BG = "#0D0D1A"


# ─────────────────────────────────────────────
# 형용사 부여
# ─────────────────────────────────────────────
def _is_nan(x: float) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def _clamp(x: float, low: float = -1.0, high: float = 1.0) -> float:
    # NaN은 그대로 NaN (원본은 min(high, nan) == high 가 되어 예측값 없는 곡이 +1 '활기찬 & 밝은'이 됐음)
    if _is_nan(x):
        return float("nan")
    return max(low, min(high, x))


def to_signed(x: float) -> float:
    """DEAM 척도 값 → -1~1 ((x-5)/5). 값이 없으면 NaN."""
    return _clamp((x - config.VA_CENTER) / config.VA_HALF_RANGE)


def _three_level(score: float, labels: tuple[str, str, str], low: float, high: float) -> str:
    if _is_nan(score):
        return config.FAILED_LABEL
    if score > high:
        return labels[0]
    if score > low:
        return labels[1]
    return labels[2]


def get_thresholds(cfg: config.Config) -> dict[str, tuple[float, float]]:
    """형용사 임계값 (DEAM 척도, (낮은 경계, 높은 경계)).

    calibrated: models/train_meta.json의 OOF 3등분 값. 없으면 경고 후 fixed.
    fixed: 원본 ±0.2 → (x-5)/5 기준이라 DEAM 척도로 4.0 / 6.0.
    """
    t = config.LABEL_THRESHOLD
    fixed = {k: (config.VA_CENTER - t * config.VA_HALF_RANGE, config.VA_CENTER + t * config.VA_HALF_RANGE)
             for k in ("valence", "arousal")}
    if config.LABEL_THRESHOLD_MODE != "calibrated":
        return fixed
    from src.mood_model import load_label_thresholds

    saved = load_label_thresholds(cfg)
    if not saved:
        log.warning("보정 임계값이 없습니다(%s). 원본 ±0.2 기준을 씁니다. --steps 3 --retrain으로 다시 학습하세요.",
                    cfg.model_meta_path)
        return fixed
    return {k: (saved[k][0], saved[k][1]) for k in ("valence", "arousal")}


def valence_label(v: float, low: float, high: float) -> str:
    """Valence 예측값(DEAM 척도) → 밝은/담담한/우울한."""
    return _three_level(v, config.VALENCE_LABELS, low, high)


def arousal_label(a: float, low: float, high: float) -> str:
    """Arousal 예측값(DEAM 척도) → 활기찬/포근한/차분한."""
    return _three_level(a, config.AROUSAL_LABELS, low, high)


def build_final_table(cfg: config.Config) -> pd.DataFrame:
    """가사·예측·색을 song_id 기준 left join하고 형용사 컬럼을 추가한다."""
    if cfg.lyrics_csv.exists():
        base = read_csv(cfg.lyrics_csv)
    else:
        log.warning("%s 없음 → 플레이리스트만으로 진행", cfg.lyrics_csv)
        base = load_playlist(cfg.input_csv)

    preds = read_csv(cfg.va_predictions_csv)[["song_id", *config.PRED_COLS]]
    df = base.merge(preds, on="song_id", how="left")

    if cfg.song_emotion_color_csv.exists():
        colors = read_csv(cfg.song_emotion_color_csv)[["song_id", "main_color_text"]]
        df = df.merge(colors, on="song_id", how="left")
    else:
        log.warning("%s 없음 → main_color_text 비움", cfg.song_emotion_color_csv)
        df["main_color_text"] = pd.NA

    th = get_thresholds(cfg)
    log.info("형용사 임계값 (%s): Valence %s, Arousal %s", config.LABEL_THRESHOLD_MODE, th["valence"], th["arousal"])
    df["valence_score"] = df["valence_mean_pred"].apply(to_signed)   # 그래프 좌표 (-1~1)
    df["arousal_score"] = df["arousal_mean_pred"].apply(to_signed)
    df["valence_label"] = [valence_label(v, *th["valence"]) for v in df["valence_mean_pred"]]
    df["arousal_label"] = [arousal_label(a, *th["arousal"]) for a in df["arousal_mean_pred"]]
    df["va_mood"] = [
        config.FAILED_LABEL if config.FAILED_LABEL in (a, v) else f"{a} & {v}"
        for a, v in zip(df["arousal_label"], df["valence_label"])
    ]  # 색깔과 분리된 VA 분위기
    df["main_color_text"] = df["main_color_text"].fillna(config.FAILED_LABEL)

    failed = df[df["va_mood"] == config.FAILED_LABEL]["title"].tolist()
    if failed:
        log.warning("음원 분석 실패로 형용사를 붙이지 못한 곡: %s", ", ".join(map(str, failed)))
    return df


# ─────────────────────────────────────────────
# 시각화
# ─────────────────────────────────────────────
def _donut(ax, counts: pd.Series, palette: dict[str, str], title: str) -> None:
    palette = {**palette, config.FAILED_LABEL: config.FAILED_COLOR}
    colors = [palette.get(m, "#888888") for m in counts.index]
    _, _, autotexts = ax.pie(
        counts.values, labels=counts.index, autopct="%1.1f%%", colors=colors, startangle=90,
        wedgeprops=dict(width=0.55, edgecolor=BG, linewidth=2),
        textprops=dict(color="white", fontsize=10))
    for at in autotexts:
        at.set_fontsize(9)
        at.set_color("#EEEEEE")
    ax.set_title(title, color="white", fontsize=13, pad=12)


def _draw_va_scatter(ax, df: pd.DataFrame, th: dict[str, tuple[float, float]]) -> None:
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches

    ax.set_facecolor("#12122A")
    # 축 범위: 절대 척도 예측은 0 근처에 모이므로 데이터·경계선이 다 보이는 만큼만 (최소 ±0.3, 최대 ±1.15)
    pts = df[["valence_score", "arousal_score"]].abs().to_numpy().ravel()
    bounds = [abs(to_signed(x)) for k in ("valence", "arousal") for x in th[k]]
    vals = [v for v in [*pts, *bounds] if not _is_nan(v)]
    lim = min(1.15, max(0.3, 1.25 * max(vals, default=1.0)))

    ax.axhline(0, color="#444466", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="#444466", linewidth=0.8, linestyle="--")
    # 형용사 경계선 (점선)
    for v in th["valence"]:
        ax.axvline(to_signed(v), color="#55557A", linewidth=0.7, linestyle=":")
    for a in th["arousal"]:
        ax.axhline(to_signed(a), color="#55557A", linewidth=0.7, linestyle=":")
    for (x0, y0, x1, y1, c) in [
            (0, 0, lim, lim, "#1A2A1A"), (-lim, 0, 0, lim, "#2A1A1A"),
            (0, -lim, lim, 0, "#1A1A2A"), (-lim, -lim, 0, 0, "#1E1A22")]:
        ax.fill([x0, x1, x1, x0], [y0, y0, y1, y1], color=c, alpha=0.5, zorder=0)

    for _, row in df.dropna(subset=["valence_score", "arousal_score"]).iterrows():  # 분석 실패 곡은 점 없음
        ax.scatter(row["valence_score"], row["arousal_score"],
                   color=row["color"], s=160, zorder=3, marker=row["marker"],
                   edgecolors=row["lyrics_color"], linewidths=2.5)
        ax.annotate(row["title"], (row["valence_score"], row["arousal_score"]),
                    textcoords="offset points", xytext=(8, 5), fontsize=8, color="#CCCCEE")

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Valence  (우울한 ←————→ 밝은)", color="#AAAACC", fontsize=10)
    ax.set_ylabel("Arousal  (차분한 ←————→ 활기찬)", color="#AAAACC", fontsize=10)
    ax.set_title("Valence – Arousal 분포\n(채움: Arousal / 테두리: 가사 색감 / 마커: Valence / 점선: 형용사 경계)",
                 color="white", fontsize=11, pad=10)
    ax.tick_params(colors="#666688")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")
    for (x, y, t, ha, va) in [(0.98, 0.98, "활기찬 & 밝은", "right", "top"), (0.02, 0.98, "활기찬 & 우울한", "left", "top"),
                              (0.98, 0.02, "차분한 & 밝은", "right", "bottom"), (0.02, 0.02, "차분한 & 우울한", "left", "bottom")]:
        ax.text(x, y, t, fontsize=8, color="#666688", ha=ha, va=va, transform=ax.transAxes)

    legend_style = dict(fontsize=7, facecolor="#1A1A2E", edgecolor="#333355",
                        labelcolor="white", framealpha=0.85, title_fontsize=8)
    a_patches = [mpatches.Patch(color=c, label=m) for m, c in config.AROUSAL_COLORS.items()]
    # 범례는 그래프 아래 바깥쪽 (아래 모서리의 '차분한 & ...' 라벨을 가리지 않게)
    leg1 = ax.legend(handles=a_patches, title="Arousal", loc="upper right", bbox_to_anchor=(1.0, -0.13),
                     ncol=3, **legend_style)
    leg1.get_title().set_color("white")

    v_handles = [mlines.Line2D([0], [0], marker=config.VALENCE_MARKERS[m], color="w",
                               markerfacecolor=config.VALENCE_COLORS[m], markersize=8,
                               label=f"{config.VALENCE_SYMBOLS[m]} {m}", linestyle="None")
                 for m in config.VALENCE_MARKERS]
    leg2 = ax.legend(handles=v_handles, title="Valence", loc="upper left", bbox_to_anchor=(0.0, -0.13),
                     ncol=3, **legend_style)
    leg2.get_title().set_color("white")
    ax.add_artist(leg1)


LINE_IN = 0.19          # 그룹 목록 텍스트 한 줄 높이(인치)
MOOD_BLOCK_LINES = 3.4  # 분위기 그룹: 곡당 제목·VA·색 3줄 + 여백
COLOR_BLOCK_LINES = 2.4  # 색 그룹: 곡당 제목·분위기 2줄 + 여백
HEADER_LINES = 2.6
COLOR_MARK = "■"        # 🎨 이모지는 한글 폰트에 없어 □로 깨졌음


def _group_row_height(max_items: int, block_lines: float) -> float:
    """그룹 목록 행 높이(인치): 가장 큰 그룹이 겹치지 않고 들어갈 만큼."""
    return (HEADER_LINES + max_items * block_lines + 0.5) * LINE_IN


def _line_height(ax) -> float:
    """axes 좌표계에서 텍스트 한 줄 높이."""
    return LINE_IN / (ax.get_position().height * ax.figure.get_figheight())


def _draw_mood_groups(ax, df: pd.DataFrame, total: int) -> None:
    ax.axis("off")
    ax.set_facecolor(BG)
    ax.set_title("비슷한 분위기끼리 모아봤어요!", color="white", fontsize=13, pad=10)

    lh = _line_height(ax)
    grouped = df.groupby("va_mood")
    col_width = 1.0 / len(grouped)
    for i, (mood, group) in enumerate(grouped):
        cx = col_width * i + 0.01
        color = config.AROUSAL_COLORS.get(mood.split(" & ")[0], "#888888")
        ax.text(cx, 1.0, mood, fontsize=9.5, fontweight="bold",
                color=color, transform=ax.transAxes, va="top")
        ax.text(cx, 1.0 - lh * 1.2, f"{len(group)}곡 ({len(group) / total * 100:.0f}%)",
                fontsize=8, color="#AAAACC", transform=ax.transAxes, va="top")
        for j, (_, row) in enumerate(group.iterrows()):
            y_pos = 1.0 - lh * (HEADER_LINES + j * MOOD_BLOCK_LINES)
            lc = config.LYRICS_COLOR_MAP.get(row["main_color_text"], "#CCCCCC")
            ax.text(cx, y_pos, f"♪ {row['title']}", fontsize=8.5, color="#DDDDEE",
                    transform=ax.transAxes, va="top")
            va_text = ("  음원 분석 실패" if _is_nan(row["valence_score"])
                       else f"  V:{row['valence_score']:+.2f} A:{row['arousal_score']:+.2f}")
            ax.text(cx, y_pos - lh, va_text,
                    fontsize=7.5, color="#7777AA", transform=ax.transAxes, va="top")
            ax.text(cx, y_pos - 2 * lh, f"  {COLOR_MARK} {row['main_color_text']}",
                    fontsize=7.5, color=lc, transform=ax.transAxes, va="top")


def _draw_color_groups(ax, df: pd.DataFrame) -> None:
    ax.axis("off")
    ax.set_facecolor(BG)
    ax.set_title("가사 색감으로 모아봤어요!", color="white", fontsize=13, pad=10)

    lh = _line_height(ax)
    grouped = df.groupby("main_color_text")
    col_width = 1.0 / len(grouped)
    for i, (color_label, group) in enumerate(grouped):
        cx = col_width * i + 0.01
        lc = config.LYRICS_COLOR_MAP.get(color_label, "#CCCCCC")
        ax.text(cx, 1.0, f"{COLOR_MARK} {color_label}", fontsize=9.5, fontweight="bold",
                color=lc, transform=ax.transAxes, va="top")
        ax.text(cx, 1.0 - lh * 1.2, f"{len(group)}곡", fontsize=8, color="#AAAACC",
                transform=ax.transAxes, va="top")
        for j, (_, row) in enumerate(group.iterrows()):
            y_pos = 1.0 - lh * (HEADER_LINES + j * COLOR_BLOCK_LINES)
            ax.text(cx, y_pos, f"♪ {row['title']}", fontsize=8.5, color="#DDDDEE",
                    transform=ax.transAxes, va="top")
            ax.text(cx, y_pos - lh, f"  {row['va_mood']}", fontsize=7.5, color="#7777AA",
                    transform=ax.transAxes, va="top")


def plot(df: pd.DataFrame, out_path, th: dict[str, tuple[float, float]]) -> None:
    """분석 결과 대시보드 PNG 저장 (도넛 3개 + VA 산점도 + 그룹 목록 2개)."""
    import matplotlib
    matplotlib.use("Agg")
    import koreanize_matplotlib  # noqa: F401 - 한글 폰트 등록
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt

    df = df.copy()
    df["color"] = df["arousal_label"].map(config.AROUSAL_COLORS).fillna("#CCCCCC")
    df["marker"] = df["valence_label"].map(config.VALENCE_MARKERS).fillna("o")
    df["lyrics_color"] = df["main_color_text"].map(config.LYRICS_COLOR_MAP).fillna("#CCCCCC")

    total = len(df)
    va_mood_counts = df["va_mood"].value_counts()
    dominant_mood = va_mood_counts.index[0]
    dominant_pct = va_mood_counts.iloc[0] / total * 100

    # 위 3행은 원본 크기(인치) 유지, 아래 그룹 목록 2행은 가장 큰 그룹에 맞춰 늘림
    max_mood = int(va_mood_counts.max())
    max_color = int(df["main_color_text"].value_counts(dropna=False).max())
    heights = [1.08, 4.75, 4.75,
               max(2.5, _group_row_height(max_mood, MOOD_BLOCK_LINES)),
               max(2.0, _group_row_height(max_color, COLOR_BLOCK_LINES))]
    hspace = 0.55
    fig_h = sum(heights) * (1 + hspace * (len(heights) - 1) / len(heights)) / (0.93 - 0.04)
    fig = plt.figure(figsize=(20, fig_h), facecolor=BG)
    gs = gridspec.GridSpec(5, 2, figure=fig, height_ratios=heights,
                           hspace=hspace, wspace=0.35, left=0.06, right=0.96, top=0.93, bottom=0.04)
    ax_title = fig.add_subplot(gs[0, :])
    ax_donut_a = fig.add_subplot(gs[1, 0])
    ax_donut_v = fig.add_subplot(gs[1, 1])
    ax_donut_c = fig.add_subplot(gs[2, 0])
    ax_va = fig.add_subplot(gs[2, 1])
    ax_songs = fig.add_subplot(gs[3, :])
    ax_color_group = fig.add_subplot(gs[4, :])
    for ax in (ax_title, ax_donut_a, ax_donut_v, ax_donut_c, ax_va, ax_songs, ax_color_group):
        ax.set_facecolor(BG)

    ax_title.axis("off")
    ax_title.text(0.5, 0.80, "당신의 플레이리스트 분석 결과", ha="center", va="center",
                  fontsize=22, fontweight="bold", color="white", transform=ax_title.transAxes)
    ax_title.text(0.5, 0.10, f"가장 많은 분위기: '{dominant_mood}' ({dominant_pct:.0f}%)  |  총 {total}곡",
                  ha="center", va="center", fontsize=12, color="#AAAADD",
                  transform=ax_title.transAxes)

    _donut(ax_donut_a, df["arousal_label"].value_counts(), config.AROUSAL_COLORS, "Arousal 분위기 구성")
    _donut(ax_donut_v, df["valence_label"].value_counts(), config.VALENCE_COLORS, "Valence 감성 구성")
    _donut(ax_donut_c, df["main_color_text"].value_counts(), config.LYRICS_COLOR_MAP, "가사 감성 색감 구성")
    _draw_va_scatter(ax_va, df, th)
    _draw_mood_groups(ax_songs, df, total)
    _draw_color_groups(ax_color_group, df)

    plt.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    log.info("시각화 저장: %s", out_path)


def run(cfg: config.Config) -> pd.DataFrame:
    """final_result.csv와 playlist_analysis.png를 만든다."""
    df = build_final_table(cfg)
    write_csv(df, cfg.final_csv, log)
    plot(df, cfg.figure_png, get_thresholds(cfg))
    return df
