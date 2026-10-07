"""단계들이 함께 쓰는 헬퍼: 로깅, 플레이리스트 로드(song_id 부여), CSV 입출력."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

import config

ID_COLS = ["song_id", "title", "artist"]


def get_logger(name: str) -> logging.Logger:
    """모듈별 로거를 돌려준다. 포맷은 main.setup_logging에서 한 번만 정한다."""
    return logging.getLogger(name)


def setup_logging(level: int = logging.INFO) -> None:
    """루트 로거 포맷 설정."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def load_playlist(path: Path) -> pd.DataFrame:
    """Playlist.csv(title, artist)를 읽고 행 번호(1부터) 기반 song_id를 붙인다.

    title/artist 앞뒤 공백은 제거한다(원본 3·5단계의 strip과 동일).
    """
    df = pd.read_csv(path, encoding=config.CSV_ENCODING, dtype=str)  # 숫자 제목도 문자열로
    df.columns = df.columns.str.strip()
    missing = {"title", "artist"} - set(df.columns)
    if missing:
        raise ValueError(f"{path}에 필요한 컬럼이 없습니다: {missing}")

    df = df[["title", "artist"]].copy()
    for c in ("title", "artist"):
        df[c] = df[c].map(lambda v: v.strip() if isinstance(v, str) else v)
    df.insert(0, "song_id", range(1, len(df) + 1))
    return df


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    """프로젝트 표준 인코딩으로 CSV를 읽는다. 파일이 없으면 어느 단계가 필요한지 알려준다."""
    if not Path(path).exists():
        raise FileNotFoundError(f"입력 파일이 없습니다: {path} (이전 단계를 먼저 실행하세요)")
    return pd.read_csv(path, encoding=config.CSV_ENCODING, **kwargs)


def write_csv(df: pd.DataFrame, path: Path, logger: logging.Logger | None = None) -> None:
    """프로젝트 표준 인코딩으로 CSV를 저장한다."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding=config.CSV_ENCODING)
    if logger:
        logger.info("저장: %s (%d행)", path, len(df))
