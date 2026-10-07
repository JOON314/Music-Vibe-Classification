"""1단계: 멜론에서 곡별 가사를 크롤링한다.

입력: input/Playlist.csv (title, artist)
출력: outputs/01_lyrics.csv (song_id, title, artist, lyrics)
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


def get_lyrics(driver, wait, title: str, artist: str) -> str:
    """멜론에서 한 곡의 가사를 가져온다. 실패하면 안내 문자열을 돌려준다(원본 동작 유지)."""
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
            return config.LYRICS_NO_RESULT

        more_info_list[0].click()
        time.sleep(2)

        try:
            driver.find_element(By.CSS_SELECTOR, ".button_more.arrow_d").click()
            time.sleep(1)
        except Exception:
            pass

        lyric = driver.find_element(By.CSS_SELECTOR, ".lyric.on")
        return lyric.text.strip()

    except Exception as e:  # noqa: BLE001 - 원본과 같이 곡 단위 실패는 문자열로 기록
        return f"{config.LYRICS_ERROR_PREFIX}{e}"


def run(cfg: config.Config) -> pd.DataFrame:
    """플레이리스트 전체 가사를 크롤링해 01_lyrics.csv로 저장한다."""
    from selenium import webdriver
    from selenium.webdriver.support.ui import WebDriverWait
    from tqdm import tqdm

    df = load_playlist(cfg.input_csv)

    driver = webdriver.Chrome()
    wait = WebDriverWait(driver, 10)
    lyrics_list: list[str] = []
    try:
        for row in tqdm(df.itertuples(index=False), total=len(df)):
            if _is_missing(row.title) or _is_missing(row.artist):
                lyrics_list.append(config.LYRICS_MISSING_INPUT)
                continue
            lyrics_list.append(get_lyrics(driver, wait, str(row.title), str(row.artist)))
    finally:
        driver.quit()

    df["lyrics"] = lyrics_list
    write_csv(df, cfg.lyrics_csv, log)
    return df
