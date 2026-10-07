# %%
# 핵심 단어 -> 감정 -> 색
'''
더미 사전 csv 형식은 NRC 기반 csv 와 동일
word, lang, pos, joy, sadness, anger, fear, trust, disgust, surprise, anticipation, positive, negative
매치 컬러는 yellow, blue, red, purple, green, brown, orange, pink, saturation & value
즉, HSV : Hue, Saturation, Value 사용
'''

# %%
# 기본 세팅
import os
import pandas as pd
import colorsys

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))  # model2/
DATASET_DIR = os.path.join(BASE_DIR, "dataset")           # model2/dataset/
os.makedirs(DATASET_DIR, exist_ok=True)
# %%
# 컬럼 list

Core_Keywords_CSV        = os.path.join(DATASET_DIR, "core_keywords_by_song.csv")
Lexicon_CSV              = os.path.join(DATASET_DIR, "dummy_lexicon.csv")

Output_Matched_CSV       = os.path.join(DATASET_DIR, "matched_keywords_emotions.csv")
Output_Song_Scores_Color_CSV  = os.path.join(DATASET_DIR, "song_emotion_scores_color.csv")
Output_Song_Color_Summary_CSV = os.path.join(DATASET_DIR, "song_color_summary.csv")

Emotion_Cols = [
    "joy", "sadness", "anger", "fear", "trust", "disgust", "surprise", "anticipation"
] # NRC 8감정 컬럼 목록
Sentiment_Cols = [
    "positive", "negative"
] # NRC 긍정/부정 컬럼 목록
Score_Cols = Emotion_Cols + Sentiment_Cols # 전체 점수 컬럼 목록

# %%
# core keywords CSV 처리

def load_core_keywords(path):
  df = pd.read_csv(path) # path의 CSV를 읽어서 DataFrame으로 변환

  required = {"title", "word", "lang", "count"} # 이 CSV에 반드시 있어야 하는 컬럼 목록
  missing = required - set(df.columns) # 필수이나 실제 CSV에는 없는 컬럼 찾기
  if missing: # 빠진 컬럼이 있다면 코드 멈추고 에러 메시지 출력
    raise ValueError(f"core keywords csv missing columns: {missing}")

  # str으로 바꾸고 공백 제거, 소문자 통일
  df["word"] = df["word"].astype(str).str.strip().str.lower()
  df["lang"] = df["lang"].astype(str).str.strip().str.lower()
  ##df["pos"] = df["pos"].astype(str).str.strip().str.lower()
  df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(0)
  # 숫자로 바꾸고 이상값은 NaN으로 바꿔서 0으로 채움

  return df # 처리한 core keywords DataFrame 반환

# %%
# emotion lexicon CSV 처리

def load_emotion_lexicon(path):
  df = pd.read_csv(path) # path의 CSV를 읽어서 DataFrame으로 변환

  required = {"word", "lang", *Score_Cols} # 이 CSV에 반드시 있어야 하는 컬럼 목록, Score_Cols 내 원소들 포함
  missing = required - set(df.columns) # 필수이나 실제 CSV에는 없는 컬럼 찾기
  if missing: # 빠진 컬럼이 있다면 코드 멈추고 에러 메시지 출력
    raise ValueError(f"emotion lexicon csv missing columns: {missing}")
  # str으로 바꾸고 공백 제거, 소문자로 통일
  df["word"] = df["word"].astype(str).str.strip().str.lower()
  df["lang"] = df["lang"].astype(str).str.strip().str.lower()
  ##df["pos"] = df["pos"].astype(str).str.strip().str.lower()

  for col in Score_Cols: # Score_Cols 내 컬럼들을 순회하며 처리
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(float)
    # 숫자로 바꾸고 이상값은 NaN으로 바꿔서 0으로 채운 뒤 float으로 맞춤

  return df # 처리한 emotion lexicon DataFrame 반환

# %%
# 핵심 단어 - 감정 매칭

def match_keywords(core_df, lexicon_df):
  matched_df = core_df.merge( # DataFrame의 merge 기능
      lexicon_df,
      on=["word", "lang"], # word, lang, pos가 다 같아야 같은 단어로 판단
      how="left" # core_df 기준으로 lexicon_df를 merge : 사전에 없는 단어도 유지
  )

  matched_df["is_matched"] = ~matched_df[Score_Cols].isna().all(axis=1)
  # Score_Cols 확인해서 NaN이면, 즉, 사전과 매칭이 안 되면 False 되면 True
  matched_df[Score_Cols] = matched_df[Score_Cols].fillna(0) # NaN을 0으로 바꿈

  return matched_df # word - emotion 매칭한 DataFrame 반환

# %%
# 노래별 감정 점수 계산

def score_songs(matched_df):
  scored_df = matched_df.copy() # 원본 보존 위해 copy

  for col in Score_Cols: # Score_Cols 내 컬럼 순회하며 처리
    scored_df[col] = scored_df[col] * scored_df["count"]
    # 감정 및 긍정, 부정 점수에 단어 빈도 곱하기

  song_scores_df = (
      scored_df
      .groupby("title")[Score_Cols]
      .sum()
      .reset_index()
  ) # 곡 제목별로 묶어서 감정 및 긍정, 부정 점수 합산 테이블 생성

  # total 감정 점수
  song_scores_df["emotion_total"] = song_scores_df[Emotion_Cols].sum(axis=1)

  # 곡에서 positive / negative가 얼마나 잡혔는지
  song_scores_df["sentiment_total"] = (
      song_scores_df["positive"] + song_scores_df["negative"]
  )

  # 곡의 positive / negative 정도는 별개로
  song_scores_df["sentiment_balance"] = (
      song_scores_df["positive"] - song_scores_df["negative"]
  )

  # 곡의 positive / negative 비율
  song_scores_df["sentiment_ratio"] = (
      song_scores_df["sentiment_balance"] / song_scores_df["sentiment_total"]
  )

  song_scores_df["sentiment_ratio"] = (
      song_scores_df["sentiment_ratio"]
      .replace([float("inf"), float("-inf")], 0)
      .fillna(0)
  ) # +1 ~ -1 범위, +1이 완전 긍정, -1이 완전 부정 그리고 0이 중립

  # 주감정
  song_scores_df["primary_emotion"] = song_scores_df[Emotion_Cols].idxmax(axis=1)
  # 각 노래에서 8감정 중 가장 점수가 높은 감정을 대표 감정으로 설정하여 반환

  # 부감정
  def get_secondary_emotion(row):
    sorted_emotions = row[Emotion_Cols].sort_values(ascending=False)
    return sorted_emotions.index[1]

  song_scores_df["secondary_emotion"] = song_scores_df.apply(
      get_secondary_emotion,
      axis=1
  )

  # 8개 감정의 점수 합이 0인 곡을 찾아 "unmatched"를 대표 감정으로 설정
  zero_mask = song_scores_df["emotion_total"] == 0
  song_scores_df.loc[zero_mask, "primary_emotion"] = "unmatched"
  song_scores_df.loc[zero_mask, "secondary_emotion"] = "unmatched"

  # 감정 비율 컬럼
  for emotion in Emotion_Cols:
    song_scores_df[f"{emotion}_ratio"] = (
        song_scores_df[emotion] / song_scores_df["emotion_total"]
    )

  song_scores_df = song_scores_df.fillna(0)

  # 상위 감정 조합
  def get_emotion_combo(row, top_n=3):
    if row["emotion_total"] == 0:
      return "unmatched"
    sorted_emotions = row[Emotion_Cols].sort_values(ascending=False)
    top_emotions = [
        emotion for emotion in sorted_emotions.index[:top_n]
        if row[emotion] > 0
    ]

    return "+".join(top_emotions)

  song_scores_df["emotion_combo"] = song_scores_df.apply(
      get_emotion_combo,
      axis=1
  )

  return song_scores_df # 최종 노래별 감정 점수 테이블 반환

# %%
# HSV 내부 계산
'''
NRC 기반 사전이라 형식은 동일
word, lang, pos, joy, sadness, anger, fear, trust, disgust, surprise, anticipation, positive, negative
매치 컬러는 yellow, blue, red, purple, green, brown, orange, pink, value
HSV : Hue, Saturation, Value 사용
'''

Emotion_HSV_Map = {
    "joy": (49, 0.90, 0.90),
    "sadness": (230, 0.94, 0.85),
    "anger": (0, 0.88, 0.85),
    "fear": (279, 0.90, 0.80),
    "trust": (131, 1.00, 0.75),
    "disgust": (43, 0.63, 0.65),
    "surprise": (27, 1.00, 0.90),
    "anticipation": (334, 0.65, 0.90),
    "unmatched": (0, 0.00, 0.60)
} # Hue, Saturation 고정하고 Value 만 조절

Emotion_Color_Name_Map = {
    "joy": "노란색",
    "sadness": "파란색",
    "anger": "빨간색",
    "fear": "보라색",
    "trust": "초록색",
    "disgust": "갈색",
    "surprise": "주황색",
    "anticipation": "분홍색",
    "unmatched": "회색"
}

def clamp(value, min_value=0.0, max_value=1.0):
  return max(min_value,min(value, max_value))

def hsv_to_hex(h, s, v): # h: 0~360, s/v: 0~1
  r, g, b = colorsys.hsv_to_rgb(h / 360, s, v)

  return "#{:02X}{:02X}{:02X}".format(
      int(r * 255),
      int(g * 255),
      int(b * 255)
  )

def adjust_hsv_by_sentiment(base_hsv, sentiment_ratio):
  h, s, v = base_hsv

  ratio = clamp(sentiment_ratio, -1, 1)

  v = v + (0.10 * ratio)
  v = clamp(sentiment_ratio, -1, 1)

  return h, s, v

def emotion_to_adjusted_hex(emotion, sentiment_ratio):
  base_hsv = Emotion_HSV_Map.get(
      emotion,
      Emotion_HSV_Map["unmatched"]
  )

  h, s, v = adjust_hsv_by_sentiment(
      base_hsv,
      sentiment_ratio
  )

  return hsv_to_hex(h, s, v)

def get_value_label(sentiment_ratio):
  if sentiment_ratio >= 0.35:
    return "밝은"
  elif sentiment_ratio <= -0.35:
    return "어두운"
  else:
    return ""

def emotion_to_color_text(emotion, sentiment_ratio):
  color_name = Emotion_Color_Name_Map.get(
      emotion,
      Emotion_Color_Name_Map["unmatched"]
  )
  value = get_value_label(sentiment_ratio)

  return color_name

# %%
# 노래별 감정 - 색 매칭

def attach_color_columns(song_scores_df):
  song_scores_df["main_color_hex"] = song_scores_df.apply(
      lambda row: emotion_to_adjusted_hex(
          row["primary_emotion"],
          row["sentiment_ratio"]
      ),
      axis = 1
  )

  song_scores_df["main_color_text"] = song_scores_df.apply(
      lambda row: emotion_to_color_text(
          row["primary_emotion"],
          row["sentiment_ratio"]
      ),
      axis = 1
  )

  return song_scores_df # 최종 노래별 감정 - 색 매칭 테이블 반환

def build_palette_df(song_scores_df):
  palette_df = song_scores_df[
      [
          "title",
          "primary_emotion",
          "emotion_combo",
          "main_color_text",
          "main_color_hex",
          "sentiment_ratio"
      ]
  ].copy()

  return palette_df

# %%
# CSV 처리 제외 DataFrame으로 바로 다루는 함수 통합

def run_emotion_matching(core_df, lexicon_df):
  matched_df = match_keywords(core_df, lexicon_df) # 단어별 매칭 결과 확인용
  song_scores_df = score_songs(matched_df) # 곡 - 감정 점수 매칭용
  song_scores_df = attach_color_columns(song_scores_df) # 색 컬럼 추가
  palette_df = build_palette_df(song_scores_df) # 요약 테이블 생성

  return matched_df, song_scores_df, palette_df

# %%
# main

def main():
    core_df    = load_core_keywords(Core_Keywords_CSV)
    lexicon_df = load_emotion_lexicon(Lexicon_CSV)

    matched_df, song_scores_df, palette_df = run_emotion_matching(core_df, lexicon_df)

    # 기존 CSV 저장
    matched_df.to_csv(Output_Matched_CSV, index=False, encoding="utf-8-sig")
    song_scores_df.to_csv(Output_Song_Scores_Color_CSV, index=False, encoding="utf-8-sig")

    # ── title + main_color_text 만 추출 ──
    color_text_df = palette_df[["title", "main_color_text"]].copy()
    color_text_df["title"] = color_text_df["title"].astype(str).str.strip()

    # ── Playlists_lyrics_mood.csv 에 main_color_text 컬럼 추가 (항상 덮어쓰기) ──
    playlist_path = os.path.join(DATASET_DIR, "Playlists_lyrics_mood.csv")
    playlist_df   = pd.read_csv(playlist_path, encoding="utf-8-sig")
    playlist_df["title"] = playlist_df["title"].astype(str).str.strip()

    playlist_df.drop(columns=["main_color_text"], errors="ignore", inplace=True)

    playlist_df = playlist_df.merge(
        color_text_df,
        on="title",
        how="left"
    )

    playlist_df.to_csv(playlist_path, index=False, encoding="utf-8-sig")

    print("Done!")
    print(f"Saved: {Output_Matched_CSV}")
    print(f"Saved: {Output_Song_Scores_Color_CSV}")
    print(f"Updated: {playlist_path}  (main_color_text 컬럼 추가)")

if __name__ == "__main__":
    main()

