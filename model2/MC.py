import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import koreanize_matplotlib

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
OUTPUT_DIR  = os.path.join(BASE_DIR, "output")

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR,  exist_ok=True)

# ── 1. 데이터 로드 & 전처리
input_file  = os.path.join(DATASET_DIR, "Playlists_lyrics_mood.csv")
output_file = os.path.join(DATASET_DIR, "audio_va_result.csv")

def clamp(x, low=-1.0, high=1.0):
    return max(low, min(high, x))

def valence_to_signed(x):
    return clamp((x - 5) / 5)

def arousal_to_signed(x):
    return clamp((x - 5) / 5)

# ── Valence / Arousal 3단계 레이블
def valence_label(v):
    if v > 0.2:    return "밝은"
    elif v > -0.2: return "담담한"
    else:          return "우울한"

def arousal_label(a):
    if a > 0.2:    return "활기찬"
    elif a > -0.2: return "포근한"
    else:          return "차분한"

df = pd.read_csv(input_file)
df["valence_score"] = df["valence_mean_pred"].apply(valence_to_signed)
df["arousal_score"] = df["arousal_mean_pred"].apply(arousal_to_signed)
df["valence_label"] = df["valence_score"].apply(valence_label)
df["arousal_label"] = df["arousal_score"].apply(arousal_label)

# ── mood를 VA 기준으로만 구성 (색깔은 분리)
df["va_mood"] = df["arousal_label"] + " & " + df["valence_label"]

df.to_csv(output_file, index=False, encoding="utf-8-sig")
print("CSV 저장 완료:", output_file)

# ── 2. 색상 & 마커 팔레트
arousal_colors = {
    "활기찬": "#FF4C4C",
    "포근한": "#4CA3FF",
    "차분한": "#A0A0C8",
}
valence_colors_palette = {
    "밝은":   "#FFD700",
    "담담한": "#87CEEB",
    "우울한": "#708090",
}
valence_markers = {
    "밝은":   "o",
    "담담한": "D",
    "우울한": "s",
}
valence_symbols = {
    "밝은":   "●",
    "담담한": "◆",
    "우울한": "■",
}

# ── 가사 색감 9가지 원색 그대로
lyrics_color_map = {
    "노란색": "#FFD700",
    "파란색": "#4CA3FF",
    "빨간색": "#FF4C4C",
    "보라색": "#9B59B6",
    "초록색": "#7EC8A4",
    "갈색":   "#8B6347",
    "주황색": "#FF9933",
    "분홍색": "#FF69B4",
    "회색":   "#A0A0A0",
}

df["color"]        = df["arousal_label"].map(arousal_colors).fillna("#CCCCCC")
df["marker"]       = df["valence_label"].map(valence_markers).fillna("o")
df["lyrics_color"] = df["main_color_text"].map(lyrics_color_map).fillna("#CCCCCC")

# ── 3. 통계
total          = len(df)
arousal_counts = df["arousal_label"].value_counts()
valence_counts = df["valence_label"].value_counts()
va_mood_counts = df["va_mood"].value_counts()
color_counts   = df["main_color_text"].value_counts()
dominant_mood  = va_mood_counts.index[0]
dominant_pct   = va_mood_counts.iloc[0] / total * 100

# ── 4. Figure 구성 (5행 2열)
fig = plt.figure(figsize=(20, 22), facecolor="#0D0D1A")
gs = gridspec.GridSpec(5, 2, figure=fig,
                       height_ratios=[0.5, 2.2, 2.2, 2.8, 2.5],
                       hspace=0.55, wspace=0.35,
                       left=0.06, right=0.96, top=0.93, bottom=0.04)

ax_title        = fig.add_subplot(gs[0, :])
ax_donut_a      = fig.add_subplot(gs[1, 0])
ax_donut_v      = fig.add_subplot(gs[1, 1])
ax_donut_c      = fig.add_subplot(gs[2, 0])
ax_va           = fig.add_subplot(gs[2, 1])
ax_songs        = fig.add_subplot(gs[3, :])
ax_color_group  = fig.add_subplot(gs[4, :])

for ax in [ax_title, ax_donut_a, ax_donut_v, ax_donut_c,
           ax_va, ax_songs, ax_color_group]:
    ax.set_facecolor("#0D0D1A")

# ══ TITLE ══
ax_title.axis('off')
ax_title.text(0.5, 0.80, "당신의 플레이리스트 분석 결과",
              ha='center', va='center', fontsize=22, fontweight='bold',
              color='white', transform=ax_title.transAxes)
ax_title.text(0.5, 0.10,
              f"가장 많은 분위기: '{dominant_mood}' ({dominant_pct:.0f}%)  |  총 {total}곡",
              ha='center', va='center', fontsize=12, color='#AAAADD',
              transform=ax_title.transAxes)

# ══ DONUT — Arousal ══
colors_a = [arousal_colors.get(m, "#888888") for m in arousal_counts.index]
wedges, texts, autotexts = ax_donut_a.pie(
    arousal_counts.values, labels=arousal_counts.index,
    autopct='%1.1f%%', colors=colors_a, startangle=90,
    wedgeprops=dict(width=0.55, edgecolor='#0D0D1A', linewidth=2),
    textprops=dict(color='white', fontsize=10))
for at in autotexts:
    at.set_fontsize(9); at.set_color('#EEEEEE')
ax_donut_a.set_title("Arousal 분위기 구성", color='white', fontsize=13, pad=12)

# ══ DONUT — Valence ══
colors_v = [valence_colors_palette.get(m, "#888888") for m in valence_counts.index]
wedges2, texts2, autotexts2 = ax_donut_v.pie(
    valence_counts.values, labels=valence_counts.index,
    autopct='%1.1f%%', colors=colors_v, startangle=90,
    wedgeprops=dict(width=0.55, edgecolor='#0D0D1A', linewidth=2),
    textprops=dict(color='white', fontsize=10))
for at in autotexts2:
    at.set_fontsize(9); at.set_color('#EEEEEE')
ax_donut_v.set_title("Valence 감성 구성", color='white', fontsize=13, pad=12)

# ══ DONUT — 가사 색감 (9가지 원색) ══
colors_c = [lyrics_color_map.get(m, "#888888") for m in color_counts.index]
wedges3, texts3, autotexts3 = ax_donut_c.pie(
    color_counts.values, labels=color_counts.index,
    autopct='%1.1f%%', colors=colors_c, startangle=90,
    wedgeprops=dict(width=0.55, edgecolor='#0D0D1A', linewidth=2),
    textprops=dict(color='white', fontsize=10))
for at in autotexts3:
    at.set_fontsize(9); at.set_color('#EEEEEE')
ax_donut_c.set_title("가사 감성 색감 구성", color='white', fontsize=13, pad=12)

# ══ VA SCATTER ══
ax_va.set_facecolor("#12122A")
ax_va.axhline(0, color='#444466', linewidth=0.8, linestyle='--')
ax_va.axvline(0, color='#444466', linewidth=0.8, linestyle='--')

for (x0, y0, x1, y1, c) in [
    (0, 0, 1.1, 1.1, "#1A2A1A"), (-1.1, 0, 0, 1.1, "#2A1A1A"),
    (0, -1.1, 1.1, 0, "#1A1A2A"), (-1.1, -1.1, 0, 0, "#1E1A22")]:
    ax_va.fill([x0,x1,x1,x0], [y0,y0,y1,y1], color=c, alpha=0.5, zorder=0)

for _, row in df.iterrows():
    ax_va.scatter(row["valence_score"], row["arousal_score"],
                  color=row["color"], s=160, zorder=3,
                  marker=row["marker"],
                  edgecolors=row["lyrics_color"],
                  linewidths=2.5)
    ax_va.annotate(row["title"],
                   (row["valence_score"], row["arousal_score"]),
                   textcoords="offset points", xytext=(8, 5),
                   fontsize=8, color='#CCCCEE')

ax_va.set_xlim(-1.15, 1.15); ax_va.set_ylim(-1.15, 1.15)
ax_va.set_xlabel("Valence  (우울한 ←————→ 밝은)", color='#AAAACC', fontsize=10)
ax_va.set_ylabel("Arousal  (차분한 ←————→ 활기찬)", color='#AAAACC', fontsize=10)
ax_va.set_title("Valence – Arousal 분포\n(채움: Arousal / 테두리: 가사 색감 / 마커: Valence)",
                color='white', fontsize=11, pad=10)
ax_va.tick_params(colors='#666688')
for spine in ax_va.spines.values():
    spine.set_edgecolor('#333355')

for (x, y, t) in [(0.55, 1.08, "활기찬 & 밝은"), (-1.1, 1.08, "활기찬 & 우울한"),
                   (0.55, -1.12, "차분한 & 밝은"), (-1.1, -1.12, "차분한 & 우울한")]:
    ax_va.text(x, y, t, fontsize=8, color='#666688')

a_patches = [mpatches.Patch(color=c, label=m) for m, c in arousal_colors.items()]
leg1 = ax_va.legend(handles=a_patches, title="Arousal", loc='lower right',
                    fontsize=7, facecolor='#1A1A2E', edgecolor='#333355',
                    labelcolor='white', framealpha=0.85, title_fontsize=8)
leg1.get_title().set_color('white')

v_handles = [mlines.Line2D([0],[0], marker=valence_markers[m], color='w',
                    markerfacecolor=valence_colors_palette[m],
                    markersize=8, label=f"{valence_symbols[m]} {m}", linestyle='None')
             for m in valence_markers]
leg2 = ax_va.legend(handles=v_handles, title="Valence", loc='lower left',
                    fontsize=7, facecolor='#1A1A2E', edgecolor='#333355',
                    labelcolor='white', framealpha=0.85, title_fontsize=8)
leg2.get_title().set_color('white')
ax_va.add_artist(leg1)

# ══ 곡 목록 (VA mood 기준 그룹핑) ══
ax_songs.axis('off')
ax_songs.set_facecolor("#0D0D1A")
ax_songs.set_title("비슷한 분위기끼리 모아봤어요!",
                   color='white', fontsize=13, pad=10)

grouped = df.groupby("va_mood")
n_groups = len(grouped)
col_width = 1.0 / n_groups

for i, (mood, group) in enumerate(grouped):
    a_lbl = mood.split(" & ")[0]
    cx = col_width * i + 0.01
    color = arousal_colors.get(a_lbl, "#888888")

    ax_songs.text(cx, 0.95, mood, fontsize=9.5, fontweight='bold',
                  color=color, transform=ax_songs.transAxes, va='top')
    ax_songs.text(cx, 0.80, f"{len(group)}곡 ({len(group)/total*100:.0f}%)",
                  fontsize=8, color='#AAAACC', transform=ax_songs.transAxes, va='top')

    for j, (_, row) in enumerate(group.iterrows()):
        y_pos = 0.64 - j * 0.20
        lc = lyrics_color_map.get(row["main_color_text"], "#CCCCCC")

        ax_songs.text(cx, y_pos, f"♪ {row['title']}",
                      fontsize=8.5, color='#DDDDEE',
                      transform=ax_songs.transAxes, va='top')
        ax_songs.text(cx, y_pos - 0.11,
                      f"  V:{row['valence_score']:+.2f} A:{row['arousal_score']:+.2f}",
                      fontsize=7.5, color='#7777AA',
                      transform=ax_songs.transAxes, va='top')
        ax_songs.text(cx, y_pos - 0.20,
                      f"  🎨 {row['main_color_text']}",
                      fontsize=7.5, color=lc,
                      transform=ax_songs.transAxes, va='top')

# ══ 가사 색감 기준 그룹 섹션 (9가지 원색) ══
ax_color_group.axis('off')
ax_color_group.set_facecolor("#0D0D1A")
ax_color_group.set_title("가사 색감으로 모아봤어요!",
                          color='white', fontsize=13, pad=10)

color_grouped = df.groupby("main_color_text")
n_cgroups = len(color_grouped)
col_width_c = 1.0 / n_cgroups

for i, (color_label, group) in enumerate(color_grouped):
    cx = col_width_c * i + 0.01
    lc = lyrics_color_map.get(color_label, "#CCCCCC")

    ax_color_group.text(cx, 0.95, f"🎨 {color_label}",
                        fontsize=9.5, fontweight='bold',
                        color=lc, transform=ax_color_group.transAxes, va='top')
    ax_color_group.text(cx, 0.80, f"{len(group)}곡",
                        fontsize=8, color='#AAAACC',
                        transform=ax_color_group.transAxes, va='top')

    for j, (_, row) in enumerate(group.iterrows()):
        y_pos = 0.64 - j * 0.18
        ax_color_group.text(cx, y_pos, f"♪ {row['title']}",
                            fontsize=8.5, color='#DDDDEE',
                            transform=ax_color_group.transAxes, va='top')
        ax_color_group.text(cx, y_pos - 0.10,
                            f"  {row['va_mood']}",
                            fontsize=7.5, color='#7777AA',
                            transform=ax_color_group.transAxes, va='top')

plt.savefig(os.path.join(OUTPUT_DIR, "playlist_analysis.png"),
            dpi=160, bbox_inches='tight', facecolor='#0D0D1A')
plt.close()
print("시각화 저장 완료:", os.path.join(OUTPUT_DIR, "playlist_analysis.png"))
