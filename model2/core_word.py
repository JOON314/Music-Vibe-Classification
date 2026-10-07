# %%
#python -m spacy download en_core_web_sm

import re
import pandas as pd
from collections import Counter
# from langdetect import detect
from kiwipiepy import Kiwi
from kiwipiepy.utils import Stopwords
import spacy
import os
# %%
# 0. Settings
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))  # model2/
DATASET_DIR = os.path.join(BASE_DIR, "dataset")           # model2/dataset/
os.makedirs(DATASET_DIR, exist_ok=True)

input_csv  = os.path.join(DATASET_DIR, "playlists_lyrics_mood.csv")
title_column = "title"
lyrics_column = "lyrics"

output_token_csv   = os.path.join(DATASET_DIR, "lyrics_tokens_long.csv")
output_core_csv    = os.path.join(DATASET_DIR, "core_keywords_by_song.csv")
output_summary_csv = os.path.join(DATASET_DIR, "core_keywords_summary.csv")

min_word_length = 2 # 지나치게 짧은 단어는 제외
min_count_in_song = 2 # 너무 안 나오는 단어도 제외
max_global_ratio = 0.7 # 너무 흔한 단어도 제외

allowed_pos = {"noun", "verb", "adjective"} # 명사 동사 형용사만 후보로 추출

lyrics_stopwords = {
    # Korean
    "나", "너", "우리", "그", "그녀", "저", "이", "것", "거",
    "진짜", "그냥", "너무", "정말", "약간", "음", "아",  # ← 쉼표 추가
    "그게", "와", "하", "사람", "남자", "여자", "아이",  # ← 끝에 쉼표 추가
    # English
    "yeah", "oh", "ooh", "uh", "ah", "la", "hey", "na", "da", "ra",
    "aha", "really", "um", "ha", "I", "me", "you", "us", "he", "she",
    "they", "her", "his", "them", "our", "this", "that", "such", "so",
    "girl", "boy", "man", "woman", "guy", "something", "whoa"
}

# %%
# 1. Load NLP Tools

kiwi = Kiwi()
ko_stopwords = Stopwords()
nlp_en = spacy.load("en_core_web_sm")

# %%
# 2. Split Lyrics

def split_lyrics(lyrics):
  text = str(lyrics)

  # 가능하면 줄 기준 분리 시도
  lines = [l.strip() for l in text.split("\n") if l.strip()]

  if len(lines) > 1:
    return lines

  # 문장부호 기준 분리 시도
  sentences = re.split(r"[.!?]+", text)
  sentences = [s.strip() for s in sentences if s.strip()]

  if len(sentences) > 1:
    return sentences

    words = text.split()
    return [" ".join(words[i:i+15]) for i in range(0, len(words), 15)]

# %%
# 3. Language Detection

def guess_language(text):
  kor_chars = len(re.findall(r"[가-힣]", text)) # 한국어 파트 추출
  eng_chars = len(re.findall(r"[A-Za-z]", text)) # 영어 파트 추출

  if kor_chars > 0 and eng_chars > 0: # 한국어 영어 둘 다 있으면 믹스
    return "mixed"
  if kor_chars > 0: # 한국어만
    return "kor"
  if eng_chars > 0: # 영어만
    return "eng"

  return "other" # 그 외 언어만 있으면 other

# %%
# 4. Korean Processing

def process_korean(text):
  tokens = kiwi.tokenize( # 문장을 형태소 단위로 분리
      text,
      stopwords = ko_stopwords, # kiwi 내부 불용어 제거
      normalize_coda = True # 비정형 텍스트 정규화
  )

  results = []

  for token in tokens:
    word = token.form
    tag = token.tag

    if tag.startswith("J"): # 조사
      continue
    if tag.startswith("E"): # 어미
      continue
    if tag.startswith("S"): # 기호
      continue

    if tag in {"NNG", "NNP"}: # 일반명사, 고유명사
      pos = "noun"
    elif tag == "VV": # 동사
      pos = "verb"
    elif tag == "VA": # 형용사
      pos = "adjective"
    else:
      continue

    results.append((word, pos, "kor")) # (단어, 품사, 언어)

  return results

# %%
# 5. English Processing

def process_english(text):
  doc = nlp_en(text) # 문장 -> 토큰 + 품사 + 원형 + 기타 정보

  results = []

  for token in doc:
    if token.is_stop: # spaCy 기본 불용어 제거
      continue
    if not token.is_alpha: # 알파벳만 남김
      continue

    word = token.lemma_.lower() # 단어를 원형 & 소문자로 변환

    if token.pos_ in {"NOUN", "PROPN"}: # 일반명사, 고유명사
      pos = "noun"
    elif token.pos_ == "VERB": # 동사
      pos = "verb"
    elif token.pos_ == "ADJ":  # 형용사
      pos = "adjective"
    else:
      continue

    results.append((word, pos, "eng")) # 단어, 품사, 언어

  return results

# %%
# 6. Other Language Fallback

def process_other(text): # fallback 은 보존만
  words = text.split()

  return [
      (word, "unknown", "other")
      for word in words
  ]

# %%
# 7. Mixed Language Processing

def process_mixed(text):
  results = []

  korean_part = " ".join(re.findall(r"[가-힣]+", text)) # 한국어 파트 추출
  english_part = " ".join(re.findall(r"[A-Za-z]+", text)) # 영어 파트 추출

  if korean_part: # 한국어 파트 처리
    results.extend(process_korean(korean_part))

  if english_part: # 영어 파트 처리
    results.extend(process_english(english_part))

  return results

# %%
# 8. Analyze One Lyric

def analyze_lyrics(lyrics):
    if pd.isna(lyrics) or str(lyrics).strip() == "":  # ← NaN, 빈값 방어
        return []

    segments = split_lyrics(lyrics)

    if not segments:  # ← 빈 리스트 방어
        return []

    all_tokens = []

    for segment in segments:
        lang = guess_language(segment)

        if lang == "kor":
            all_tokens.extend(process_korean(segment))
        elif lang == "eng":
            all_tokens.extend(process_english(segment))
        elif lang == "mixed":
            all_tokens.extend(process_mixed(segment))
        else:
            all_tokens.extend(process_other(segment))

    return all_tokens

# %%
# 9. Build Token Table

def build_token_table(df):
  rows = []

  for _, row in df.iterrows(): # DataFrame 돌면서 title, lyrics 꺼내기
    title = row[title_column]
    lyrics = row[lyrics_column]

    analyzed_tokens = analyze_lyrics(lyrics) # 가사 적당히 분리 후 처리

    # 불용어 / 길이 기준 1차 필터링
    cleaned = []

    for word, pos, lang in analyzed_tokens:
      word = word.strip().lower() # 공백 처리, 대문자 -> 소문자

      if len(word) < min_word_length: # 최소 길이 미달 단어 필터링
        continue
      if word in lyrics_stopwords: # 불용어 필터링
        continue

      cleaned.append((word, pos, lang)) # 통과한 단어 저장

    counts = Counter(cleaned) # 단어별 빈도 계산       continue

    for (word, pos, lang), count in counts.items(): # 빈도 포함해서 저장
      rows.append({
          "title": title,
          "word": word,
          "pos": pos,
          "lang": lang,
          "count": count
      })

  return pd.DataFrame(rows) # DataFrame 반환

# %%
# 10. Core Keyword Filtering

def filter_core_keywords(token_df):
  if token_df.empty:
    return token_df

  filtered = token_df.copy() # 원본 보존

  # 명사, 동사, 형용사만
  filtered = filtered[filtered["pos"].isin(allowed_pos)]

  # 곡 내부 빈도 기준으로 컷
  filtered = filtered[filtered["count"] >= min_count_in_song]

  # 전체 곡 대부분에 등장하는 흔한 단어도 컷 _ 제외 또는 최적화 필수 코드
  """ total_songs = token_df["title"].nunique()

  doc_freq = ( # 단어가 몇 곡에 나왔는지 계산
      token_df.groupby("word")["title"]
      .nunique()
      .reset_index(name="song_appear_count")
  )

  doc_freq["global_ratio"] = doc_freq["song_appear_count"] / total_songs # 나온 비율 계산

  filtered = filtered.merge(doc_freq, on="word", how="left")
  filtered = filtered[filtered["global_ratio"] <= max_global_ratio] """

  filtered = filtered.sort_values( # title 오름차순, count 내림차순
      ["title", "count"],
      ascending=[True, False]
  )

  return filtered # DataFrame 반환

# %%
# 11. Build Summary Table

def build_summary(core_df, top_n=20): # 빈도 높은 순으로 top_n 개 단어
  if core_df.empty:
    return pd.DataFrame(columns=["title", "core_keywords"])

  core_df["keyword_with_pos"] = (
      core_df["word"]+"("+core_df["pos"]+"/"+core_df["lang"]+")" # 품사와 언어도 같이 기록
  )

  summary = (
      core_df
      .sort_values(["title", "count"], ascending=[True, False]) # title 오름차순, count 내림차순
      .groupby("title") # 같은 곡 내 단어끼리 묶기
      .apply(
          lambda group: ", ".join( # , 로 이어붙인 문자열
              group.head(top_n)["keyword_with_pos"].tolist() # keyword_with_pos 컬럼을 위에서부터 top_n 개
          )
      )
      .reset_index(name="core_keywords")
  )

  return summary # DataFrame 반환

# %%
# 12. Main

def main():
    df = pd.read_csv(input_csv)

    required_columns = {title_column, lyrics_column}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.dropna(subset=[title_column, lyrics_column])
    df = df[df[lyrics_column].str.strip() != ""]

    print(f"Processing {len(df)} songs...")

    token_df = build_token_table(df)
    token_df.to_csv(output_token_csv, index=False, encoding="utf-8-sig")

    core_df = filter_core_keywords(token_df)
    core_df.to_csv(output_core_csv, index=False, encoding="utf-8-sig")

    summary_df = build_summary(core_df)
    summary_df.to_csv(output_summary_csv, index=False, encoding="utf-8-sig")

    print("Done.")
    print(f"Saved: {output_token_csv}")
    print(f"Saved: {output_core_csv}")
    print(f"Saved: {output_summary_csv}")

main()

