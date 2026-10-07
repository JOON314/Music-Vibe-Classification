"""_baseline 입력으로 4~6단계를 다시 돌려 원본 산출물과 비교한다.

원본 공유 CSV(_baseline/dataset/Playlists_lyrics_mood.csv)의 가사·예측값을
01_lyrics.csv / 03_va_predictions.csv 형식으로 옮긴 뒤 4~6단계를 실행하고,
원본 CSV들과 song_id 컬럼을 뺀 나머지가 완전히 같은지 확인한다.

    python tools/compare_baseline.py [--out outputs/_compare]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

import config  # noqa: E402
from src import lyrics_color, lyrics_keywords, report  # noqa: E402
from src.common import load_playlist, setup_logging, write_csv  # noqa: E402

BASE = ROOT / "_baseline" / "dataset"
log = logging.getLogger("compare")

# (새 산출물 속성명, 원본 파일명)
PAIRS = [
    ("tokens_csv", "lyrics_tokens_long.csv"),
    ("core_keywords_csv", "core_keywords_by_song.csv"),
    ("keywords_summary_csv", "core_keywords_summary.csv"),
    ("matched_emotions_csv", "matched_keywords_emotions.csv"),
    ("song_emotion_color_csv", "song_emotion_scores_color.csv"),
    ("final_csv", "audio_va_result.csv"),
]


def prepare_inputs(cfg: config.Config) -> None:
    """원본 공유 CSV → 01_lyrics.csv, 03_va_predictions.csv (song_id는 Playlist 행 번호)."""
    orig = pd.read_csv(BASE / "Playlists_lyrics_mood.csv", encoding=config.CSV_ENCODING)
    for c in ("title", "artist"):
        orig[c] = orig[c].astype(str).str.strip()
    ids = load_playlist(cfg.input_csv)[["song_id", "title", "artist"]]
    df = ids.merge(orig, on=["title", "artist"], how="left", validate="one_to_one")

    write_csv(df[["song_id", "title", "artist", "lyrics"]], cfg.lyrics_csv)
    preds = df.dropna(subset=config.PRED_COLS)[["song_id", "title", "artist", *config.PRED_COLS]]
    write_csv(preds, cfg.va_predictions_csv)


def compare(new_path: Path, old_path: Path) -> list[str]:
    """두 CSV를 song_id 제외하고 비교. 차이 설명 목록(없으면 빈 목록)을 돌려준다."""
    new = pd.read_csv(new_path, encoding=config.CSV_ENCODING).drop(columns=["song_id"], errors="ignore")
    old = pd.read_csv(old_path, encoding=config.CSV_ENCODING)
    problems: list[str] = []
    if list(new.columns) != list(old.columns):
        problems.append(f"컬럼 다름\n    new={list(new.columns)}\n    old={list(old.columns)}")
        common = [c for c in old.columns if c in new.columns]
        new, old = new[common], old[common]
    if len(new) != len(old):
        problems.append(f"행 수 다름 new={len(new)} old={len(old)}")
        return problems
    try:
        pd.testing.assert_frame_equal(new.reset_index(drop=True), old.reset_index(drop=True),
                                      check_dtype=False, check_exact=False, rtol=1e-9, atol=1e-12)
    except AssertionError as e:
        problems.append(str(e).strip().splitlines()[0] + " ...\n    " + "\n    ".join(str(e).splitlines()[1:8]))
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "outputs" / "_compare"))
    ap.add_argument("--ref", default=str(BASE),
                    help="비교 기준 폴더 (기본: _baseline/dataset). 원본 코드를 현재 환경에서 돌린 결과 폴더도 가능")
    args = ap.parse_args()
    ref = Path(args.ref).resolve()
    setup_logging()

    cfg = config.Config(output_dir=Path(args.out).resolve())
    cfg.ensure_dirs()
    prepare_inputs(cfg)
    lyrics_keywords.run(cfg)
    lyrics_color.run(cfg)
    report.run(cfg)

    ok = True
    print(f"\n===== 원본 대비 비교 (4~6단계) | 기준: {ref} =====")
    for attr, old_name in PAIRS:
        problems = compare(getattr(cfg, attr), ref / old_name)
        status = "✅ 동일" if not problems else "❌ 차이"
        print(f"{status}  {old_name:34s} ↔ {getattr(cfg, attr).name}")
        for p in problems:
            print("    " + p)
        ok &= not problems
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
