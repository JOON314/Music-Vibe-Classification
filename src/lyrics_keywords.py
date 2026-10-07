"""4단계: 가사에서 곡별 핵심 키워드(명사·동사·형용사)를 뽑는다.

입력: outputs/01_lyrics.csv
출력: outputs/04_tokens_long.csv, 04_core_keywords.csv, 04_keywords_summary.csv
한국어는 kiwipiepy, 영어는 spaCy(en_core_web_sm)로 형태소 분석한다.
"""
from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache

import pandas as pd

import config
from src.common import get_logger, read_csv, write_csv

log = get_logger(__name__)

Token = tuple[str, str, str]  # (단어, 품사, 언어)
GROUP_KEYS = ["title", "song_id"]  # 정렬·그룹 기준 (원본 title 정렬 순서 유지 + 동명이곡 구분)


@lru_cache(maxsize=1)
def _nlp():
    """kiwi·불용어·spaCy 모델을 한 번만 로드한다."""
    import spacy
    from kiwipiepy import Kiwi
    from kiwipiepy.utils import Stopwords

    return Kiwi(), Stopwords(), spacy.load(config.SPACY_MODEL)


def split_lyrics(lyrics: object) -> list[str]:
    """가사를 줄 → 문장부호 → 15단어 묶음 순으로 나눈다.

    원본은 15단어 분할이 return 뒤에 들여쓰기되어 실행되지 않았고, 한 줄 가사는 None → 토큰 0개였다.
    """
    text = str(lyrics)

    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if len(lines) > 1:
        return lines

    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    if len(sentences) > 1:
        return sentences

    words = text.split()
    n = config.SPLIT_CHUNK_WORDS
    return [" ".join(words[i:i + n]) for i in range(0, len(words), n)]


def guess_language(text: str) -> str:
    """한글/영문 글자 유무로 kor·eng·mixed·other 판별."""
    has_kor = bool(re.search(r"[가-힣]", text))
    has_eng = bool(re.search(r"[A-Za-z]", text))
    if has_kor and has_eng:
        return "mixed"
    if has_kor:
        return "kor"
    if has_eng:
        return "eng"
    return "other"


def process_korean(text: str) -> list[Token]:
    """kiwi 형태소 분석 → 일반/고유명사, 동사, 형용사만 남긴다."""
    kiwi, ko_stopwords, _ = _nlp()
    results: list[Token] = []
    for token in kiwi.tokenize(text, stopwords=ko_stopwords, normalize_coda=True):
        tag = token.tag
        if tag in {"NNG", "NNP"}:
            pos = "noun"
        elif tag == "VV":
            pos = "verb"
        elif tag == "VA":
            pos = "adjective"
        else:  # 조사(J)·어미(E)·기호(S) 등 나머지 제외
            continue
        results.append((token.form, pos, "kor"))
    return results


def process_english(text: str) -> list[Token]:
    """spaCy → 불용어·비알파벳 제외, 원형 소문자로 명사·동사·형용사만 남긴다."""
    _, _, nlp_en = _nlp()
    results: list[Token] = []
    for token in nlp_en(text):
        if token.is_stop or not token.is_alpha:
            continue
        if token.pos_ in {"NOUN", "PROPN"}:
            pos = "noun"
        elif token.pos_ == "VERB":
            pos = "verb"
        elif token.pos_ == "ADJ":
            pos = "adjective"
        else:
            continue
        results.append((token.lemma_.lower(), pos, "eng"))
    return results


def process_other(text: str) -> list[Token]:
    """기타 언어는 공백 단위로 보존만 한다."""
    return [(word, "unknown", "other") for word in text.split()]


def process_mixed(text: str) -> list[Token]:
    """한·영 혼합 줄은 한글/영문 부분을 각각 따로 분석한다."""
    results: list[Token] = []
    korean_part = " ".join(re.findall(r"[가-힣]+", text))
    english_part = " ".join(re.findall(r"[A-Za-z]+", text))
    if korean_part:
        results.extend(process_korean(korean_part))
    if english_part:
        results.extend(process_english(english_part))
    return results


_PROCESSORS = {"kor": process_korean, "eng": process_english, "mixed": process_mixed}


def analyze_lyrics(lyrics: object) -> list[Token]:
    """가사 한 곡 → 토큰 목록."""
    if pd.isna(lyrics) or str(lyrics).strip() == "":
        return []
    segments = split_lyrics(lyrics)
    if not segments:
        return []

    tokens: list[Token] = []
    for segment in segments:
        tokens.extend(_PROCESSORS.get(guess_language(segment), process_other)(segment))
    return tokens


def build_token_table(df: pd.DataFrame) -> pd.DataFrame:
    """곡별 (단어, 품사, 언어) 빈도표. 짧은 단어·불용어는 제외."""
    rows: list[dict] = []
    for song in df.itertuples(index=False):
        cleaned = []
        for word, pos, lang in analyze_lyrics(song.lyrics):
            word = word.strip().lower()
            if len(word) < config.MIN_WORD_LENGTH or word in config.LYRICS_STOPWORDS:
                continue
            cleaned.append((word, pos, lang))

        for (word, pos, lang), count in Counter(cleaned).items():
            rows.append({"song_id": song.song_id, "title": song.title,
                         "word": word, "pos": pos, "lang": lang, "count": count})
    return pd.DataFrame(rows, columns=["song_id", "title", "word", "pos", "lang", "count"])


def filter_core_keywords(token_df: pd.DataFrame) -> pd.DataFrame:
    """허용 품사 + 곡 내 최소 빈도 이상만 남기고 (곡, 빈도 내림차순) 정렬."""
    if token_df.empty:
        return token_df
    filtered = token_df[token_df["pos"].isin(config.ALLOWED_POS)]
    filtered = filtered[filtered["count"] >= config.MIN_COUNT_IN_SONG]
    # (원본에서 비활성화된 '전체 곡 대비 흔한 단어' 필터는 적용하지 않음: MAX_GLOBAL_RATIO)
    return filtered.sort_values([*GROUP_KEYS, "count"], ascending=[True, True, False])


def build_summary(core_df: pd.DataFrame, top_n: int = config.SUMMARY_TOP_N) -> pd.DataFrame:
    """곡별 빈도 상위 top_n 키워드를 'word(pos/lang), ...' 문자열로 요약."""
    if core_df.empty:
        return pd.DataFrame(columns=["song_id", "title", "core_keywords"])

    df = core_df.sort_values([*GROUP_KEYS, "count"], ascending=[True, True, False]).copy()
    df["keyword_with_pos"] = df["word"] + "(" + df["pos"] + "/" + df["lang"] + ")"
    summary = (
        df.groupby(GROUP_KEYS, sort=True)["keyword_with_pos"]
        .agg(lambda s: ", ".join(s.head(top_n).tolist()))
        .reset_index(name="core_keywords")
    )
    return summary[["song_id", "title", "core_keywords"]]


def run(cfg: config.Config) -> pd.DataFrame:
    """가사 → 토큰표·핵심 키워드·요약 CSV 저장. 핵심 키워드 DataFrame을 반환한다."""
    df = read_csv(cfg.lyrics_csv)
    missing = {"song_id", "title", "lyrics"} - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.dropna(subset=["title", "lyrics"])
    df = df[df["lyrics"].astype(str).str.strip() != ""]
    log.info("Processing %d songs...", len(df))

    token_df = build_token_table(df)
    write_csv(token_df, cfg.tokens_csv, log)

    core_df = filter_core_keywords(token_df)
    write_csv(core_df, cfg.core_keywords_csv, log)

    write_csv(build_summary(core_df), cfg.keywords_summary_csv, log)
    return core_df
