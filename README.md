# 플레이리스트 분위기 분석기

플레이리스트(곡명·아티스트)를 넣으면 두 갈래로 분석해 한 장의 결과 이미지로 보여줍니다.

- **가사 → 색**: 멜론 가사 크롤링 → 핵심 키워드 → NRC 8감정 → 9가지 색(노랑·파랑·빨강·보라·초록·갈색·주황·분홍·회색)
- **음원 → 분위기 형용사**: 유튜브 음원 → openSMILE 오디오 피처 24개 → DEAM으로 학습한 딥러닝이 Valence/Arousal 예측 → 형용사(밝은/담담한/우울한 × 활기찬/포근한/차분한)

## 1. 설치 (macOS, Apple Silicon 기준)

```bash
brew install python@3.11 ffmpeg deno     # ffmpeg: 음원 → wav 변환, deno: yt-dlp 유튜브 다운로드에 필요
# Google Chrome 설치 필요 (멜론 크롤링, 드라이버는 Selenium이 자동으로 받음)

cd Final
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## 2. 실행

1. `input/Playlist.csv`에 곡을 적습니다 (헤더: `title,artist`).

   ```csv
   title,artist
   어린양,검정치마
   seasons,wave to earth
   ```

2. 실행

   ```bash
   python main.py                 # 전체 1~6단계
   python main.py --steps 4-6     # 가사 분석·시각화만 다시 (01·03 산출물이 있어야 함)
   python main.py --skip-crawl    # 가사 크롤링(1단계)만 건너뜀
   python main.py --offline       # 네트워크 단계(1·2) 건너뜀
   ```

| 단계 | 모듈 | 하는 일 | 산출물 (`outputs/`) |
|---|---|---|---|
| 1 | `src/crawl_lyrics.py` | 멜론 가사 크롤링 (Chrome 창이 뜸) | `01_lyrics.csv` |
| 2 | `src/audio_features.py` | 유튜브 음원 다운로드 + 피처 24개 추출 | `02_audio_features.csv` |
| 3 | `src/mood_model.py` | DEAM 5-Fold 학습 + Valence/Arousal 예측 | `03_va_predictions.csv` |
| 4 | `src/lyrics_keywords.py` | 가사 형태소 분석 → 핵심 키워드 | `04_tokens_long.csv`, `04_core_keywords.csv`, `04_keywords_summary.csv` |
| 5 | `src/lyrics_color.py` | 키워드 → 감정 점수 → 대표 색 | `05_matched_emotions.csv`, `05_song_emotion_color.csv` |
| 6 | `src/report.py` | 결과 합치기 + 형용사 + 시각화 | `final_result.csv`, `playlist_analysis.png` |

각 단계는 자기 산출물만 새로 쓰고, 곡은 `Playlist.csv`의 행 번호인 `song_id`로 연결됩니다.

## 3. 결과 해석 (`outputs/final_result.csv`)

| 컬럼 | 의미 |
|---|---|
| `valence_mean_pred`, `arousal_mean_pred` | 모델 예측값 (DEAM 척도 1~9, 5가 중간). `config.PRED_SCALING="minmax"`이면 원본처럼 플레이리스트 안에서 0~10 재조정 |
| `valence_score`, `arousal_score` | `(x − 5) / 5`, −1~1로 자름 |
| `valence_label` | > 0.2 밝은 / > −0.2 담담한 / 나머지 우울한 |
| `arousal_label` | > 0.2 활기찬 / > −0.2 포근한 / 나머지 차분한 |
| `va_mood` | `arousal_label & valence_label` |
| `main_color_text` | 가사 대표 감정의 색 (joy 노랑, sadness 파랑, anger 빨강, fear 보라, trust 초록, disgust 갈색, surprise 주황, anticipation 분홍, 매칭 없음 회색) |

## 4. 폴더 구조

```
Final/
├── main.py  config.py  requirements.txt  README.md
├── src/          # 단계별 모듈 (각각 run(cfg) 함수)
├── tools/        # compare_baseline.py: 원본 결과와 비교
├── input/        # Playlist.csv
├── data/         # 학습 데이터·감정 사전
│   ├── deam_features_mean.csv     # DEAM 1802곡 openSMILE 피처 곡 평균 (학습 X)
│   ├── deam_labels_avg_mood.csv   # DEAM valence/arousal 평균 1~9 (학습 Y)
│   └── Korean-NRC-EmoLex.txt      # NRC EmoLex 한·영 감정 사전 (5단계에서 (word, lang)당 1행으로 정리해 사용)
├── models/       # 학습된 모델 저장 위치
├── outputs/      # 단계별 산출물
├── _baseline/    # 리팩터링 전 원본 결과 (비교 기준)
└── 참고자료/      # DEAM 전처리 코드, 노트북, 발표 자료
```

`참고자료/Data process.py`는 DEAM 원본 프레임 피처를 곡별 평균으로 만든 코드입니다 (원본 데이터는 포함되어 있지 않음).
