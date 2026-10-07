"""5단계: 핵심 키워드 → NRC 8감정 점수 → 곡별 대표 감정 → 색.

입력: outputs/04_core_keywords.csv, data/Korean-NRC-EmoLex.txt
출력: outputs/05_matched_emotions.csv, outputs/05_song_emotion_color.csv

사전: NRC EmoLex의 영어 단어(lang=eng)와 한국어 번역(lang=kor)을 한 표로 쌓아 (word, lang)당 1행으로 만든다.
"""
from __future__ import annotations

import colorsys

import pandas as pd

import config
from src.common import get_logger, read_csv, write_csv

log = get_logger(__name__)

EMO = config.EMOTION_COLS
SCORES = config.SCORE_COLS
GROUP_KEYS = ["title", "song_id"]


def load_core_keywords(cfg: config.Config) -> pd.DataFrame:
    """4단계 핵심 키워드를 읽고 word/lang 소문자·공백 정리, count 숫자화."""
    df = read_csv(cfg.core_keywords_csv)
    missing = {"song_id", "title", "word", "lang", "count"} - set(df.columns)
    if missing:
        raise ValueError(f"core keywords csv missing columns: {missing}")
    df["word"] = df["word"].astype(str).str.strip().str.lower()
    df["lang"] = df["lang"].astype(str).str.strip().str.lower()
    df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(0)
    return df


def load_emotion_lexicon(cfg: config.Config) -> pd.DataFrame:
    """NRC 한·영 사전을 (word, lang)당 1행으로 만든다.

    원본은 이 파일을 쌓아 둔 emotion_lexicon.csv(+ 테스트용 4행)를 썼고, 같은 (word, lang)이
    여러 행(예: 한국어 '당황' 9행)이라 merge 시 키워드가 중복 매칭되어 점수가 여러 번 더해졌다.
    여러 영어 단어가 같은 한국어로 번역된 경우 감정 표시가 서로 달라서, 감정별 평균(해당 감정을
    가진 원어 비율, 0~1)으로 합친다.
    """
    raw = pd.read_csv(cfg.lexicon_txt, sep="\t")
    missing = {"English Word", "Korean Word", *SCORES} - set(raw.columns)
    if missing:
        raise ValueError(f"emotion lexicon missing columns: {missing}")

    parts = [
        raw.rename(columns={"English Word": "word"}).assign(lang="eng"),
        raw.rename(columns={"Korean Word": "word"}).assign(lang="kor"),
    ]
    df = pd.concat(parts, ignore_index=True)[["word", "lang", *SCORES]]
    df = df.dropna(subset=["word"])
    df["word"] = df["word"].astype(str).str.strip().str.lower()
    for col in SCORES:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(float)

    lex = df.groupby(["word", "lang"], as_index=False)[SCORES].mean()
    log.info("감정 사전: %d행 → (word, lang) %d개", len(df), len(lex))
    return lex


def match_keywords(core_df: pd.DataFrame, lexicon_df: pd.DataFrame) -> pd.DataFrame:
    """(word, lang)로 사전과 left join. 사전에 없는 단어도 남기고 is_matched로 표시."""
    matched = core_df.merge(lexicon_df, on=["word", "lang"], how="left")
    matched["is_matched"] = ~matched[SCORES].isna().all(axis=1)
    matched[SCORES] = matched[SCORES].fillna(0)
    return matched


def _secondary_emotion(row: pd.Series) -> str:
    return row[EMO].sort_values(ascending=False).index[1]


def _emotion_combo(row: pd.Series, top_n: int = config.EMOTION_COMBO_TOP_N) -> str:
    if row["emotion_total"] == 0:
        return "unmatched"
    top = row[EMO].sort_values(ascending=False).index[:top_n]
    return "+".join(e for e in top if row[e] > 0)


def score_songs(matched_df: pd.DataFrame) -> pd.DataFrame:
    """단어 점수 × 빈도를 곡별로 합산하고 주/부감정·감정 비율·긍부정 비율을 계산한다."""
    scored = matched_df.copy()
    for col in SCORES:
        scored[col] = scored[col] * scored["count"]

    songs = scored.groupby(GROUP_KEYS)[SCORES].sum().reset_index()
    songs = songs[["song_id", "title", *SCORES]]

    songs["emotion_total"] = songs[EMO].sum(axis=1)
    songs["sentiment_total"] = songs["positive"] + songs["negative"]
    songs["sentiment_balance"] = songs["positive"] - songs["negative"]
    # +1 완전 긍정, -1 완전 부정, 0 중립
    songs["sentiment_ratio"] = (
        (songs["sentiment_balance"] / songs["sentiment_total"])
        .replace([float("inf"), float("-inf")], 0)
        .fillna(0)
    )

    songs["primary_emotion"] = songs[EMO].idxmax(axis=1)
    songs["secondary_emotion"] = songs.apply(_secondary_emotion, axis=1)

    zero = songs["emotion_total"] == 0
    songs.loc[zero, ["primary_emotion", "secondary_emotion"]] = "unmatched"

    for emotion in EMO:
        songs[f"{emotion}_ratio"] = songs[emotion] / songs["emotion_total"]
    songs = songs.fillna(0)

    songs["emotion_combo"] = songs.apply(_emotion_combo, axis=1)
    return songs


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(value, hi))


def hsv_to_hex(h: float, s: float, v: float) -> str:
    """h: 0~360, s/v: 0~1 → '#RRGGBB'."""
    r, g, b = colorsys.hsv_to_rgb(h / 360, s, v)
    return "#{:02X}{:02X}{:02X}".format(int(r * 255), int(g * 255), int(b * 255))


def adjust_hsv_by_sentiment(base_hsv: tuple[float, float, float],
                            sentiment_ratio: float) -> tuple[float, float, float]:
    """긍부정 비율로 V(명도)를 조정한다.

    원본 동작 유지: 마지막 줄에서 V가 sentiment_ratio 자체로 덮어써진다(Phase 2 문제 E).
    """
    h, s, v = base_hsv
    ratio = _clamp(sentiment_ratio, -1, 1)
    v = v + (config.SENTIMENT_VALUE_ADJUST * ratio)
    v = _clamp(sentiment_ratio, -1, 1)
    return h, s, v


def emotion_to_hex(emotion: str, sentiment_ratio: float) -> str:
    """대표 감정 + 긍부정 비율 → HEX 색."""
    base = config.EMOTION_HSV_MAP.get(emotion, config.EMOTION_HSV_MAP["unmatched"])
    return hsv_to_hex(*adjust_hsv_by_sentiment(base, sentiment_ratio))


def emotion_to_color_text(emotion: str) -> str:
    """대표 감정 → 9가지 색 이름. (원본에서 '밝은/어두운' 라벨은 계산 후 쓰이지 않아 생략)"""
    return config.EMOTION_COLOR_NAME_MAP.get(emotion, config.EMOTION_COLOR_NAME_MAP["unmatched"])


def attach_color_columns(songs: pd.DataFrame) -> pd.DataFrame:
    """main_color_hex, main_color_text 컬럼 추가."""
    songs["main_color_hex"] = [
        emotion_to_hex(e, r) for e, r in zip(songs["primary_emotion"], songs["sentiment_ratio"])
    ]
    songs["main_color_text"] = songs["primary_emotion"].map(emotion_to_color_text)
    return songs


def run(cfg: config.Config) -> pd.DataFrame:
    """키워드-감정 매칭과 곡별 색을 계산해 저장하고 곡별 결과를 반환한다."""
    core_df = load_core_keywords(cfg)
    lexicon_df = load_emotion_lexicon(cfg)

    matched = match_keywords(core_df, lexicon_df)
    write_csv(matched, cfg.matched_emotions_csv, log)

    songs = attach_color_columns(score_songs(matched))
    write_csv(songs, cfg.song_emotion_color_csv, log)
    return songs
