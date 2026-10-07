"""1단계: 멜론에서 곡별 가사를 크롤링한다.

입력: input/Playlist.csv (title, artist)
출력: outputs/01_lyrics.csv (song_id, title, artist, lyrics, lyrics_status, lyrics_error)
실패한 곡은 lyrics를 비워 두고 lyrics_status에 사유를 남긴다 → 4단계에서 자동 제외.
"""
from __future__ import annotations

import time

import pandas as pd

import config
from src.common import get_logger, load_playlist, write_csv

log = get_logger(__name__)


def _is_missing(value: object) -> bool:
    """빈 칸·NaN 여부 (원본: str 변환 후 '' 또는 'nan')."""
    s = "" if value is None else str(value).strip()
    return s == "" or s == "nan" or s == "<NA>"


def get_lyrics(driver, wait, title: str, artist: str) -> tuple[str | None, str, str]:
    """멜론에서 한 곡의 가사를 가져온다. (가사 또는 None, 상태, 오류 메시지)를 돌려준다."""
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.support import expected_conditions as EC

    try:
        driver.get(config.MELON_URL)
        time.sleep(2)

        # 원본 검색어 순서 유지: "아티스트 제목"
        keyword = f"{artist} {title}"
        log.info("검색 중: %s", keyword)

        search_box = wait.until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, ".ui-autocomplete-input"))
        )
        search_box.clear()
        search_box.send_keys(keyword)
        search_box.send_keys(Keys.RETURN)
        time.sleep(3)

        more_info_list = driver.find_elements(By.CSS_SELECTOR, ".btn.btn_icon_detail")
        if len(more_info_list) == 0:
            return None, config.LYRICS_NO_RESULT, ""

        more_info_list[0].click()
        time.sleep(2)

        try:
            driver.find_element(By.CSS_SELECTOR, ".button_more.arrow_d").click()
            time.sleep(1)
        except Exception:
            pass

        text = driver.find_element(By.CSS_SELECTOR, ".lyric.on").text.strip()
        if not text:
            return None, config.LYRICS_EMPTY, ""
        return text, config.LYRICS_OK, ""

    except Exception as e:  # noqa: BLE001 - 곡 단위 실패는 기록하고 다음 곡으로
        msg = str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__
        log.warning("  가사 실패: %s - %s (%s)", artist, title, msg)
        return None, config.LYRICS_ERROR, msg


def run(cfg: config.Config) -> pd.DataFrame:
    """플레이리스트 전체 가사를 크롤링해 01_lyrics.csv로 저장한다."""
    from selenium import webdriver
    from selenium.webdriver.support.ui import WebDriverWait
    from tqdm import tqdm

    df = load_playlist(cfg.input_csv)

    driver = webdriver.Chrome()
    wait = WebDriverWait(driver, 10)
    results: list[tuple[str | None, str, str]] = []
    try:
        for row in tqdm(df.itertuples(index=False), total=len(df)):
            if _is_missing(row.title) or _is_missing(row.artist):
                results.append((None, config.LYRICS_MISSING_INPUT, ""))
                continue
            results.append(get_lyrics(driver, wait, str(row.title), str(row.artist)))
    finally:
        driver.quit()

    df["lyrics"], df["lyrics_status"], df["lyrics_error"] = zip(*results) if results else ([], [], [])
    n_ok = (df["lyrics_status"] == config.LYRICS_OK).sum()
    log.info("가사 수집: %d/%d곡 성공", n_ok, len(df))
    write_csv(df, cfg.lyrics_csv, log)
    return df
