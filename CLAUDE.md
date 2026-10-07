# 프로젝트: 플레이리스트 분위기 분석기 (캡스톤)

## 한 줄 요약
플레이리스트(곡명·아티스트) → 멜론 가사 크롤링 → 가사 핵심 키워드 → 감정 → **색깔** 분류,
유튜브 음원 다운로드 → 오디오 특징 추출 → DEAM 데이터로 학습한 딥러닝으로 Valence/Arousal 예측 → **분위기 형용사** 부여 → 시각화.

## 파이프라인 (`main.py`가 `src/` 단계 모듈의 `run(cfg)`를 순서대로 호출, 설정은 `config.py`)
| 단계 | 파일 | 입력 | 출력 (`outputs/`) |
|---|---|---|---|
| 1 가사 크롤링 | src/crawl_lyrics.py | input/Playlist.csv (title,artist) | 01_lyrics.csv (+lyrics_status) |
| 2 오디오 특징 | src/audio_features.py | input/Playlist.csv | 02_audio_features.csv (24개 피처), 02_audio_failed.csv |
| 3 딥러닝 | src/mood_model.py | data/deam_features_mean.csv(학습 X), data/deam_labels_avg_mood.csv(학습 Y), 02 | 03_va_predictions.csv, models/ |
| 4 키워드 | src/lyrics_keywords.py | 01 | 04_tokens_long.csv, 04_core_keywords.csv, 04_keywords_summary.csv |
| 5 감정→색 | src/lyrics_color.py | 04_core_keywords.csv, data/Korean-NRC-EmoLex.txt | 05_matched_emotions.csv, 05_song_emotion_color.csv |
| 6 형용사·시각화 | src/report.py | 01, 03, 05 (song_id로 병합) | final_result.csv, playlist_analysis.png |

## 핵심 도메인 규칙 (임의로 바꾸지 말 것)
- 학습 데이터: DEAM (1802곡, openSMILE 피처의 곡 평균). 라벨 `deam_labels_avg_mood.csv`의 valence_mean/arousal_mean(1~9 척도), 조인 키 `FileName == song_id`.
- 피처 24개 목록(`config.FEAT_COLS`)과 순서는 학습·추론에서 반드시 동일해야 함.
- 추론 피처: openSMILE ComParE_2016 LLD 곡 평균, 곡별 RMS 에너지를 DEAM 중앙값에 맞춘 뒤 추출 (2026-10 승인, docs/phase2_분석.md).
- 모델: Dense 128-64-32-2, 5-Fold 앙상블, SEED=42 (`keras.utils.set_random_seed` + op determinism으로 재현 가능).
- 예측 척도: DEAM 척도 그대로 (플레이리스트 내 min-max 사용 안 함, 2026-10 승인).
- 형용사: Valence → 밝은/담담한/우울한, Arousal → 활기찬/포근한/차분한.
  임계값은 학습 시 DEAM OOF 예측의 1/3·2/3 분위값(models/train_meta.json, 2026-10 승인). 원래 규칙 ±0.2((x-5)/5)는 `LABEL_THRESHOLD_MODE="fixed"`.
- 색: NRC 8감정 → HSV 색 → 9가지 색 이름(노랑·파랑·빨강·보라·초록·갈색·주황·분홍·회색). 사전은 NRC 원본에서 (word, lang)당 1행(충돌 시 평균).
- `참고자료/Data process.py` = DEAM 원본 피처 → 곡별 평균 CSV 생성 코드(원본 데이터는 로컬에 없음).

## 작업 규칙
- 언어: 설명·주석·커밋 메시지는 한국어.
- 환경: macOS(Apple Silicon), Python 3.11 `.venv`. 외부 의존: Chrome(Selenium), ffmpeg·deno(yt-dlp), spaCy `en_core_web_sm`.
- 가사 원문 파일(01_lyrics.csv, outputs/cache/, _baseline 가사 CSV)은 저작권 때문에 git에 올리지 않음.
- 결과값을 바꾸는 변경(모델 구조·하이퍼파라미터·임계값·스케일링·사전 교체)은 **반드시 먼저 제안하고 승인받은 뒤** 진행.
- 데이터 파일(.csv/.txt)은 삭제 금지. 이동·이름변경은 계획에 적고 승인 후.
- 작업 단위마다 git 커밋. 큰 변경 전에는 계획을 먼저 보여줄 것.
