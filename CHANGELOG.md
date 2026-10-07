# CHANGELOG

## Phase 1 — 구조 정리 (동작 변경 없음)

### 바뀐 것
- `Final_model.py` + `model2/*.py`(subprocess·모듈 레벨 실행) → `main.py` + `src/` 단계별 `run(cfg)` 함수 (import해서 호출).
- 경로·파일명·피처 목록(`SELECTED`/`FEAT_COLS` 중복 제거)·하이퍼파라미터·임계값·색 매핑 → `config.py` 한 곳.
- 공유 CSV(`Playlists_lyrics_mood.csv`)를 여러 단계가 덮어쓰던 방식 제거 → 단계마다 자기 파일(`outputs/0N_*.csv`)만 쓰고 6단계에서 `final_result.csv`로 합침.
- 곡 연결 키: title(또는 title+artist) → `Playlist.csv` 행 번호 기반 `song_id`.
- print → logging, 타입힌트·docstring, `--steps / --skip-crawl / --offline / --output-dir` 옵션.
- 데이터 파일 이동·이름 변경 (`git mv`, 삭제 없음):

| 원래 | 새 위치 |
|---|---|
| model2/dataset/output.csv | data/deam_features_mean.csv |
| model2/dataset/Avg_mood.csv | data/deam_labels_avg_mood.csv |
| model2/dataset/dummy_lexicon.csv | data/emotion_lexicon.csv |
| model2/dataset/Korean-NRC-EmoLex.txt | data/Korean-NRC-EmoLex.txt |
| model2/input/Playlist.csv | input/Playlist.csv |
| model2/dataset/Lyrics_Processing_ipynb의_사본.ipynb | 참고자료/ |

### 의도적으로 유지한 원본 동작 (Phase 2에서 검토)
- 실패한 곡의 가사 칸에 "검색 결과 없음"/"오류: ..." 문자열 저장
- 한 줄 가사는 `split_lyrics`가 None → 토큰 없음
- 예측값 NaN이 `clamp`에서 +1이 되는 동작, `scale_to_10`, HSV V값 덮어쓰기, 사전 중복 매칭
- 크롤링 고정 sleep, 실행마다 5-Fold 재학습, 음원 임시 파일을 프로젝트 루트에 생성
- 1단계 검색어 순서("아티스트 제목")

### 아주 작은 차이 (현재 데이터 영향 없음)
- 2단계: 제목/아티스트가 빈 칸(NaN)이면 건너뜀 (원본은 문자열 "nan"으로 다운로드 시도)

### 검증
- **같은 환경에서 원본 코드 vs 새 코드 (4~6단계)**: 산출물 6개 모두 완전히 동일.
- **`_baseline` 대비 (4~6단계)**: kiwipiepy 0.23.x 고정 시 핵심 키워드·요약·감정 매칭·곡별 색·최종 결과 모두 동일.
  `04_tokens_long.csv`만 1행 차이(노세노세 '착장', count 1 → 핵심 키워드 필터에서 어차피 제외, 하위 결과 영향 없음).
  kiwipiepy 0.24.0에서는 형태소 결과가 바뀌어 파란달팽이 색이 분홍색 → 보라색으로 달라짐 → requirements에서 0.23.2로 고정.
- **3단계 (딥러닝)**: 원본 코드 자체가 실행할 때마다 결과가 달라짐 (같은 코드 2회 실행 시 Fold 1 MAE 0.730 vs 0.750,
  Arousal 예측은 곡 순위가 거의 뒤집힐 정도로 변동). `tf.random.set_seed`만으로는 Keras 3의 난수가 고정되지 않기 때문.
  검증용으로 `keras.utils.set_random_seed(42)` + op determinism + 단일 스레드를 강제하면
  **원본 코드와 새 코드의 예측값이 완전히 동일(최대 차이 0.0)**, 반복 실행도 동일함을 확인.
  (재현성 고정은 Phase 2 제안 사항)
