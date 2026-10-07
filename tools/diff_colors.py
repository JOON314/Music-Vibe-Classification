"""_baseline 대비 곡별 가사 색·주감정·형용사 변화를 표로 보여준다 (compare_baseline.py 실행 후 사용).

    python tools/diff_colors.py [--out outputs/_compare]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "_baseline" / "dataset"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "outputs" / "_compare"))
    out = Path(ap.parse_args().out)

    old_s = pd.read_csv(BASE / "song_emotion_scores_color.csv")[["title", "primary_emotion", "main_color_text"]]
    new_s = pd.read_csv(out / "05_song_emotion_color.csv")[["title", "primary_emotion", "main_color_text"]]
    old_f = pd.read_csv(BASE / "audio_va_result.csv")[["title", "va_mood"]]
    new_f = pd.read_csv(out / "final_result.csv")[["title", "va_mood"]]

    m = (old_s.merge(new_s, on="title", how="outer", suffixes=("_원본", "_새"))
         .merge(old_f.merge(new_f, on="title", suffixes=("_원본", "_새")), on="title", how="outer"))
    m["색 변경"] = (m["main_color_text_원본"] != m["main_color_text_새"]).map({True: "◀", False: ""})
    m["분위기 변경"] = (m["va_mood_원본"] != m["va_mood_새"]).map({True: "◀", False: ""})
    pd.set_option("display.width", 250)
    print(m.to_string(index=False))
    print(f"\n색 변경 {(m['색 변경'] != '').sum()}곡, 분위기 변경 {(m['분위기 변경'] != '').sum()}곡 / {len(m)}곡")


if __name__ == "__main__":
    main()
