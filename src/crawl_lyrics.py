"""1단계: 멜론에서 곡별 가사를 크롤링한다.

입력: input/Playlist.csv (title, artist)
출력: outputs/01_lyrics.csv (song_id, title, artist, lyrics, lyrics_status, lyrics_error)
캐시: outputs/cache/lyrics_cache.csv — 이미 가사를 받은 (title, artist)는 다시 크롤링하지 않음

실패한 곡은 lyrics를 비워 두고 lyrics_status에 사유를 남긴다 → 4단계에서 자동 제외.
"""
from __future__ import annotations

import pandas as pd

import config
from src.common import get_logger, load_playlist, read_csv, write_csv

log = get_logger(__name__)

CACHE_COLS = ["title", "artist", "lyrics"]


def _is_missing(value: object) -> bool:
    """빈 칸·NaN 여부."""
    s = "" if value is None else str(value).strip()
    return s == "" or s == "nan" or s == "<NA>"


def get_lyrics(driver, title: str, artist: str) -> tuple[str | None, str, str]:
    """멜론에서 한 곡의 가사를 가져온다. (가사 또는 None, 상태, 오류 메시지)를 돌려준다.

    고정 sleep 대신 요소가 나타날 때까지 기다린다(WebDriverWait).
    """
    from selenium.common.exceptions import TimeoutException
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.support.ui import WebDriverWait

    wait = WebDriverWait(driver, config.CRAWL_TIMEOUT)
    stage = "멜론 접속"
    try:
        driver.get(config.MELON_URL)

        # 원본 검색어 순서 유지: "아티스트 제목"
        keyword = f"{artist} {title}"
        log.info("검색 중: %s", keyword)

        search_box = wait.until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, ".ui-autocomplete-input"))
        )
        search_box.clear()
        search_box.send_keys(keyword)
        search_box.send_keys(Keys.RETURN)
        stage = "검색 결과"
        wait.until(EC.url_contains("search"))

        # 검색 결과의 '곡 정보' 버튼: 일정 시간 안에 안 나오면 결과 없음으로 판단
        try:
            detail_btn = WebDriverWait(driver, config.CRAWL_RESULT_TIMEOUT).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, ".btn.btn_icon_detail"))
            )
        except TimeoutException:
            return None, config.LYRICS_NO_RESULT, ""

        stage = "곡 상세 페이지"
        detail_btn.click()
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ".lyric")))

        # 가사 '펼치기' 버튼이 클릭 가능해지면 눌러서 전체 가사 표시 (.lyric → .lyric.on)
        stage = "가사 영역"
        short_wait = WebDriverWait(driver, config.CRAWL_RESULT_TIMEOUT)
        try:
            short_wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, ".button_more.arrow_d"))).click()
            text = short_wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".lyric.on"))).text.strip()
        except TimeoutException:
            # 펼치기 버튼이 없거나 눌러도 펼쳐지지 않으면, 접힌 상태의 가사 전체 텍스트(innerText)를 읽음
            log.debug("  펼치기 실패 → .lyric innerText 사용: %s - %s", artist, title)
            text = (driver.find_element(By.CSS_SELECTOR, ".lyric").get_attribute("innerText") or "").strip()
        if not text:
            return None, config.LYRICS_EMPTY, ""
        return text, config.LYRICS_OK, ""

    except Exception as e:  # noqa: BLE001 - 곡 단위 실패는 기록하고 다음 곡으로
        detail = str(e).strip().splitlines()[0] if str(e).strip() else ""
        msg = f"{stage}: {type(e).__name__} {detail}".strip()
        log.warning("  가사 실패: %s - %s (%s)", artist, title, msg)
        return None, config.LYRICS_ERROR, msg


def load_cache(cfg: config.Config) -> dict[tuple[str, str], str]:
    """(title, artist) → 가사 캐시. 캐시를 쓰지 않거나 파일이 없으면 빈 dict."""
    if not cfg.use_cache or not cfg.lyrics_cache_csv.exists():
        return {}
    df = read_csv(cfg.lyrics_cache_csv, dtype=str).dropna(subset=CACHE_COLS)
    return {(r.title, r.artist): r.lyrics for r in df.itertuples(index=False)}


def save_cache(cfg: config.Config, cache: dict[tuple[str, str], str]) -> None:
    """성공한 가사만 캐시에 저장."""
    df = pd.DataFrame([(t, a, l) for (t, a), l in cache.items()], columns=CACHE_COLS)
    write_csv(df, cfg.lyrics_cache_csv)


def run(cfg: config.Config) -> pd.DataFrame:
    """플레이리스트 전체 가사를 수집(캐시 우선)해 01_lyrics.csv로 저장한다."""
    df = load_playlist(cfg.input_csv)
    cache = load_cache(cfg)

    results: dict[int, tuple[str | None, str, str]] = {}
    to_crawl = []
    for row in df.itertuples(index=False):
        if _is_missing(row.title) or _is_missing(row.artist):
            results[row.song_id] = (None, config.LYRICS_MISSING_INPUT, "")
        elif (row.title, row.artist) in cache:
            results[row.song_id] = (cache[(row.title, row.artist)], config.LYRICS_OK, "")
        else:
            to_crawl.append(row)
    if cache:
        log.info("캐시 사용: %d곡, 새로 크롤링: %d곡", len(df) - len(to_crawl), len(to_crawl))

    if to_crawl:
        from selenium import webdriver
        from tqdm import tqdm

        driver = webdriver.Chrome()
        try:
            for row in tqdm(to_crawl):
                results[row.song_id] = get_lyrics(driver, str(row.title), str(row.artist))
                lyrics, status, _ = results[row.song_id]
                if status == config.LYRICS_OK:
                    cache[(row.title, row.artist)] = lyrics
        finally:
            driver.quit()
        if cfg.use_cache:
            save_cache(cfg, cache)

    df["lyrics"] = [results[i][0] for i in df["song_id"]]
    df["lyrics_status"] = [results[i][1] for i in df["song_id"]]
    df["lyrics_error"] = [results[i][2] for i in df["song_id"]]
    log.info("가사 수집: %d/%d곡 성공", (df["lyrics_status"] == config.LYRICS_OK).sum(), len(df))
    write_csv(df, cfg.lyrics_csv, log)
    return df
