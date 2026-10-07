import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import time
import pandas as pd
from selenium import webdriver
from tqdm import tqdm
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

input_file = "input/Playlist.csv"
output_file = "dataset/Playlists_lyrics_mood.csv"

df = pd.read_csv(input_file, encoding="utf-8-sig")

if len(df.columns) < 2:
    raise ValueError("A열에 가수, B열에 제목이 있어야 합니다.")

artist_col = df.columns[0]
title_col = df.columns[1]

driver = webdriver.Chrome()
wait = WebDriverWait(driver, 10)

def get_lyrics(artist, title):
    try:
        driver.get("https://www.melon.com")
        time.sleep(2)

        keyword = f"{title} {artist}"
        print(f"검색 중: {keyword}")

        search_box = wait.until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, ".ui-autocomplete-input"))
        )
        search_box.clear()
        search_box.send_keys(keyword)
        search_box.send_keys(Keys.RETURN)
        time.sleep(3)

        more_info_list = driver.find_elements(By.CSS_SELECTOR, ".btn.btn_icon_detail")
        if len(more_info_list) == 0:
            return "검색 결과 없음"

        more_info_list[0].click()
        time.sleep(2)

        try:
            driver.find_element(By.CSS_SELECTOR, ".button_more.arrow_d").click()
            time.sleep(1)
        except:
            pass

        lyric = driver.find_element(By.CSS_SELECTOR, ".lyric.on")
        return lyric.text.strip()

    except Exception as e:
        return f"오류: {e}"

lyrics_list = []

for i in tqdm(range(len(df))):
    artist = str(df.iloc[i, 0]).strip()
    title = str(df.iloc[i, 1]).strip()

    if artist == "" or title == "" or artist == "nan" or title == "nan":
        lyrics_list.append("가수/제목 없음")
        continue

    lyrics = get_lyrics(artist, title)
    lyrics_list.append(lyrics)

df["lyrics"] = lyrics_list

df.to_csv(output_file, index=False, encoding="utf-8-sig")
driver.quit()

print("저장 완료")