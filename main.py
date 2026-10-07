"""플레이리스트 분위기 분석기 실행 진입점.

예)
    python main.py                  # 1~6단계 전체
    python main.py --steps 4-6      # 4~6단계만 (이미 있는 01·03 산출물 사용)
    python main.py --steps 1,3,5
    python main.py --skip-crawl     # 1단계(멜론 크롤링) 제외
    python main.py --offline        # 네트워크 단계(1·2) 제외
"""
from __future__ import annotations

import argparse
import importlib
import logging
import sys
import time
from pathlib import Path

import config
from src.common import setup_logging

STEPS: dict[int, tuple[str, str]] = {
    1: ("가사 크롤링", "src.crawl_lyrics"),
    2: ("오디오 피처 추출", "src.audio_features"),
    3: ("딥러닝 학습 + 예측", "src.mood_model"),
    4: ("가사 키워드 추출", "src.lyrics_keywords"),
    5: ("가사 감정 → 색", "src.lyrics_color"),
    6: ("형용사 부여 + 시각화", "src.report"),
}

log = logging.getLogger("main")


def parse_steps(spec: str) -> list[int]:
    """'4-6', '1,3,5', '2' 형식을 단계 번호 목록으로 바꾼다."""
    steps: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = (int(x) for x in part.split("-", 1))
            steps.update(range(lo, hi + 1))
        else:
            steps.add(int(part))
    unknown = steps - STEPS.keys()
    if unknown:
        raise argparse.ArgumentTypeError(f"없는 단계: {sorted(unknown)} (1~6만 가능)")
    return sorted(steps)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="플레이리스트 분위기 분석 파이프라인")
    p.add_argument("--steps", type=parse_steps, default=list(STEPS),
                   help="실행할 단계 (예: 4-6, 1,3,5). 기본: 전체")
    p.add_argument("--skip-crawl", action="store_true", help="1단계(가사 크롤링) 제외")
    p.add_argument("--offline", action="store_true", help="네트워크가 필요한 1·2단계 제외")
    p.add_argument("--input", type=str, default=None, help="플레이리스트 CSV 경로 (기본: input/Playlist.csv)")
    p.add_argument("--output-dir", type=str, default=None, help="산출물 폴더 (기본: outputs/)")
    p.add_argument("-v", "--verbose", action="store_true", help="DEBUG 로그 출력")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)

    cfg = config.Config()
    if args.input:
        cfg.input_csv = Path(args.input).resolve()
    if args.output_dir:
        cfg.output_dir = Path(args.output_dir).resolve()
    cfg.ensure_dirs()

    steps = list(args.steps)
    if args.skip_crawl:
        steps = [s for s in steps if s != 1]
    if args.offline:
        steps = [s for s in steps if s not in (1, 2)]

    t0 = time.time()
    log.info("파이프라인 시작: 단계 %s", steps)
    for n in steps:
        label, module_name = STEPS[n]
        log.info("=" * 60)
        log.info("▶ %d단계: %s", n, label)
        t = time.time()
        try:
            importlib.import_module(module_name).run(cfg)
        except Exception:
            log.exception("%d단계 실패. 파이프라인 중단.", n)
            return 1
        log.info("✅ %d단계 완료 (%.1fs)", n, time.time() - t)

    log.info("전체 완료 (총 %.1fs)", time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
