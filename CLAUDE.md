# 프로젝트: 플레이리스트 분위기 분석기 (캡스톤)

## 한 줄 요약
플레이리스트(곡명·아티스트) → 멜론 가사 크롤링 → 가사 핵심 키워드 → 감정 → **색깔** 분류,
유튜브 음원 다운로드 → 오디오 특징 추출 → DEAM 데이터로 학습한 딥러닝으로 Valence/Arousal 예측 → **분위기 형용사** 부여 → 시각화.

## 파이프라인 (현재 구조, `Final_model.py`가 subprocess로 순서대로 실행)
| 단계 | 파일 | 입력 | 출력 |
|---|---|---|---|
| 1 가사 크롤링 | model2/crawling.py | input/Playlist.csv (title,artist) | dataset/Playlists_lyrics_mood.csv (+lyrics) |
| 2 오디오 특징 | model2/Music_analyze.py | input/Playlist.csv | dataset/playlist_features.csv (24개 피처) |
| 3 딥러닝 | model2/Deep_learning.py | dataset/output.csv(DEAM 학습 X), Avg_mood.csv(학습 Y), playlist_features.csv | Playlists_lyrics_mood.csv에 valence/arousal_mean_pred 병합 |
| 4 키워드 | model2/core_word.py | Playlists_lyrics_mood.csv | lyrics_tokens_long.csv, core_keywords_by_song.csv, core_keywords_summary.csv |
| 5 감정→색 | model2/emotion.py | core_keywords_by_song.csv, dummy_lexicon.csv | matched_keywords_emotions.csv, song_emotion_scores_color.csv, Playlists_lyrics_mood.csv에 main_color_text 병합 |
| 6 형용사·시각화 | model2/MC.py | Playlists_lyrics_mood.csv | audio_va_result.csv, output/playlist_analysis.png |

## 핵심 도메인 규칙 (임의로 바꾸지 말 것)
- 학습 데이터: DEAM (1802곡, openSMILE 피처의 곡 평균). 라벨 `Avg_mood.csv`의 valence_mean/arousal_mean(1~9 척도), 조인 키 `FileName == song_id`.
- 피처 24개 목록(`FEAT_COLS`/`SELECTED`)과 순서는 학습·추론에서 반드시 동일해야 함.
- 모델: Dense 128-64-32-2, 5-Fold 앙상블, SEED=42.
- 형용사: Valence → 밝은/담담한/우울한, Arousal → 활기찬/포근한/차분한 (임계값 ±0.2, (x-5)/5 변환).
- 색: NRC 8감정 → HSV 색 → 9가지 색 이름(노랑·파랑·빨강·보라·초록·갈색·주황·분홍·회색).
- `참고자료/Data process.py` = DEAM 원본 피처 → 곡별 평균 CSV 생성 코드(원본 데이터는 로컬에 없음).

## 작업 규칙
- 언어: 설명·주석·커밋 메시지는 한국어.
- 환경: macOS(Apple Silicon). 외부 의존: Chrome(Selenium), ffmpeg(yt-dlp), spaCy `en_core_web_sm`.
- 결과값을 바꾸는 변경(모델 구조·하이퍼파라미터·임계값·스케일링·사전 교체)은 **반드시 먼저 제안하고 승인받은 뒤** 진행.
- 데이터 파일(.csv/.txt)은 삭제 금지. 이동·이름변경은 계획에 적고 승인 후.
- 작업 단위마다 git 커밋. 큰 변경 전에는 계획을 먼저 보여줄 것.
