"""프로젝트 전체 설정: 경로·파일명·피처 목록·임계값·색 매핑을 한 곳에 모은다.

결과에 영향을 주는 값(피처 순서, 모델 하이퍼파라미터, 임계값, 색 매핑)은
CLAUDE.md의 도메인 규칙이므로 승인 없이 바꾸지 않는다.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent


# ─────────────────────────────────────────────
# 경로
# ─────────────────────────────────────────────
@dataclass
class Config:
    """실행 단위 경로 설정. 테스트/비교 시 output_dir만 바꿔서 쓸 수 있다."""

    input_csv: Path = ROOT_DIR / "input" / "Playlist.csv"
    data_dir: Path = ROOT_DIR / "data"
    output_dir: Path = ROOT_DIR / "outputs"
    models_dir: Path = ROOT_DIR / "models"
    # 유튜브 음원 임시 다운로드 위치. None이면 실행마다 시스템 임시 폴더를 만들고 끝나면 삭제
    audio_tmp_dir: Path | None = None
    # True면 저장된 모델이 있어도 3단계에서 다시 학습
    retrain: bool = False
    # True면 이미 받은 가사·오디오 피처(outputs/cache/)를 재사용
    use_cache: bool = True

    # ── 학습 데이터·사전 (data/)
    @property
    def deam_features_csv(self) -> Path:
        return self.data_dir / "deam_features_mean.csv"

    @property
    def deam_labels_csv(self) -> Path:
        return self.data_dir / "deam_labels_avg_mood.csv"

    @property
    def lexicon_txt(self) -> Path:
        # NRC EmoLex 한·영 버전 (English Word, 감정 10열, Korean Word; 탭 구분)
        return self.data_dir / "Korean-NRC-EmoLex.txt"

    # ── 단계별 산출물 (outputs/)
    @property
    def lyrics_csv(self) -> Path:            # 1단계
        return self.output_dir / "01_lyrics.csv"

    @property
    def audio_features_csv(self) -> Path:    # 2단계
        return self.output_dir / "02_audio_features.csv"

    @property
    def audio_failed_csv(self) -> Path:      # 2단계 실패 목록
        return self.output_dir / "02_audio_failed.csv"

    @property
    def va_predictions_csv(self) -> Path:    # 3단계
        return self.output_dir / "03_va_predictions.csv"

    @property
    def tokens_csv(self) -> Path:            # 4단계
        return self.output_dir / "04_tokens_long.csv"

    @property
    def core_keywords_csv(self) -> Path:     # 4단계
        return self.output_dir / "04_core_keywords.csv"

    @property
    def keywords_summary_csv(self) -> Path:  # 4단계
        return self.output_dir / "04_keywords_summary.csv"

    @property
    def matched_emotions_csv(self) -> Path:  # 5단계
        return self.output_dir / "05_matched_emotions.csv"

    @property
    def song_emotion_color_csv(self) -> Path:  # 5단계
        return self.output_dir / "05_song_emotion_color.csv"

    # ── 캐시 (outputs/cache/)
    @property
    def cache_dir(self) -> Path:
        return self.output_dir / "cache"

    @property
    def lyrics_cache_csv(self) -> Path:
        return self.cache_dir / "lyrics_cache.csv"

    @property
    def audio_cache_csv(self) -> Path:
        # 추출기별로 따로 저장 (librosa 값과 openSMILE 값이 섞이지 않게)
        norm = "_norm" if AUDIO_EXTRACTOR == "opensmile" and LOUDNESS_NORMALIZE else ""
        return self.cache_dir / f"audio_features_cache_{AUDIO_EXTRACTOR}{norm}.csv"

    @property
    def final_csv(self) -> Path:             # 6단계
        return self.output_dir / "final_result.csv"

    @property
    def figure_png(self) -> Path:            # 6단계
        return self.output_dir / "playlist_analysis.png"

    # ── 학습된 모델 (models/)
    def fold_model_path(self, fold: int) -> Path:
        return self.models_dir / f"fold_{fold}.keras"

    @property
    def scalers_path(self) -> Path:
        return self.models_dir / "scalers.joblib"

    @property
    def model_meta_path(self) -> Path:
        return self.models_dir / "train_meta.json"

    def ensure_dirs(self) -> None:
        """출력 폴더가 없으면 만든다."""
        for d in (self.output_dir, self.cache_dir, self.models_dir):
            d.mkdir(parents=True, exist_ok=True)


CSV_ENCODING = "utf-8-sig"


# ─────────────────────────────────────────────
# 1단계: 멜론 가사 크롤링
# ─────────────────────────────────────────────
MELON_URL = "https://www.melon.com"
CRAWL_TIMEOUT = 10         # 페이지 요소를 기다리는 최대 시간(초)
CRAWL_RESULT_TIMEOUT = 5   # 검색 결과가 없다고 판단할 때까지 기다리는 시간(초)
# 가사 칸에는 실제 가사만 넣고, 성공/실패는 lyrics_status 컬럼으로 분리
LYRICS_OK = "ok"
LYRICS_NO_RESULT = "no_result"        # 멜론 검색 결과 없음
LYRICS_ERROR = "error"                # 크롤링 중 예외 (lyrics_error에 메시지)
LYRICS_MISSING_INPUT = "missing_input"  # Playlist.csv에 제목/아티스트 없음
LYRICS_EMPTY = "empty"                # 곡 페이지는 열렸지만 가사가 비어 있음


# ─────────────────────────────────────────────
# 2·3단계: 오디오 피처 (학습·추론 공통, 순서 고정)
# ─────────────────────────────────────────────
FEAT_COLS: list[str] = [
    "F0final_sma_amean",
    "voicingFinalUnclipped_sma_amean",
    "jitterLocal_sma_amean",
    "shimmerLocal_sma_amean",
    "logHNR_sma_amean",
    "audspec_lengthL1norm_sma_amean",
    "pcm_RMSenergy_sma_amean",
    "pcm_zcr_sma_amean",
    "pcm_fftMag_fband250-650_sma_amean",
    "pcm_fftMag_fband1000-4000_sma_amean",
    "pcm_fftMag_spectralRollOff25.0_sma_amean",
    "pcm_fftMag_spectralRollOff50.0_sma_amean",
    "pcm_fftMag_spectralRollOff75.0_sma_amean",
    "pcm_fftMag_spectralRollOff90.0_sma_amean",
    "pcm_fftMag_spectralFlux_sma_amean",
    "pcm_fftMag_spectralCentroid_sma_amean",
    "pcm_fftMag_spectralEntropy_sma_amean",
    "pcm_fftMag_spectralVariance_sma_amean",
    "pcm_fftMag_spectralSkewness_sma_amean",
    "pcm_fftMag_spectralKurtosis_sma_amean",
    "pcm_fftMag_spectralSlope_sma_amean",
    "pcm_fftMag_psySharpness_sma_amean",
    "pcm_fftMag_mfcc_sma[1]_amean",
    "pcm_fftMag_mfcc_sma[4]_amean",
]

DOWNLOAD_ATTEMPTS = 3      # 유튜브 다운로드 시도 횟수 (간헐적 HTTP 403 대비)
DOWNLOAD_RETRY_WAIT = 10   # 재시도 대기(초) × 시도 번호

# 오디오 피처 추출기
#   "opensmile": 실제 openSMILE ComParE_2016 LLD의 곡 평균 — DEAM 학습 피처와 같은 도구 (기본)
#   "librosa"  : 원본 프로젝트의 librosa/parselmouth 근사 — DEAM과 스케일이 크게 달라(도메인 시프트) 비교용으로만 유지
AUDIO_EXTRACTOR = "opensmile"
# 음량 정규화: 곡마다 openSMILE RMS 에너지 평균을 DEAM 1802곡의 중앙값에 맞춘 뒤 피처를 뽑는다.
# 요즘 음원은 DEAM 클립보다 1.7~3.7배(평균 +8.5dB) 커서 에너지 계열 피처가 DEAM 범위 끝에 몰렸고,
# 그 결과 Arousal 예측이 좁게 압축됐다(docs/phase2_분석.md 4절). openSMILE 추출기에서만 적용.
LOUDNESS_NORMALIZE = True
LOUDNESS_REF_FEATURE = "pcm_RMSenergy_sma_amean"

# openSMILE ComParE_2016 LLD 이름 ↔ DEAM 피처 이름(_amean 제거, MFCC는 'pcm_fftMag_' 접두어 없음)
OPENSMILE_RENAME = {"pcm_fftMag_mfcc_sma[1]": "mfcc_sma[1]", "pcm_fftMag_mfcc_sma[4]": "mfcc_sma[4]"}

# librosa/parselmouth 추출 파라미터 (openSMILE 근사, AUDIO_EXTRACTOR="librosa"일 때)
AUDIO_SR = 16000
AUDIO_HOP = 160
AUDIO_WIN = 400
AUDIO_N_FFT = 512


# ─────────────────────────────────────────────
# 3단계: 딥러닝 (DEAM 학습 → Valence/Arousal 예측)
# ─────────────────────────────────────────────
SEED = 42
DEAM_JOIN_LEFT = "FileName"
DEAM_JOIN_RIGHT = "song_id"
TARGET_COLS = ["valence_mean", "arousal_mean"]
PRED_COLS = ["valence_mean_pred", "arousal_mean_pred"]

N_FOLDS = 5
EPOCHS = 300
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
HIDDEN_UNITS = (128, 64, 32)
DROPOUTS = (0.2, 0.15)
EARLY_STOP_PATIENCE = 25
LR_PLATEAU_PATIENCE = 10
LR_PLATEAU_FACTOR = 0.5
# 예측값 척도
#   "deam"  : 모델 출력(DEAM 1~9 척도) 그대로 — 같은 곡은 항상 같은 결과 (기본, openSMILE 피처 전제)
#   "minmax": 원본 scale_to_10 — 플레이리스트 안에서 min-max로 0~10 재조정(상대 순위)
PRED_SCALING = "deam"
SCALE_MAX = 10          # minmax일 때 최댓값
PRED_ROUND = 4


# ─────────────────────────────────────────────
# 4단계: 가사 핵심 키워드
# ─────────────────────────────────────────────
MIN_WORD_LENGTH = 2       # 지나치게 짧은 단어 제외
MIN_COUNT_IN_SONG = 2     # 곡 안에서 너무 적게 나온 단어 제외
MAX_GLOBAL_RATIO = 0.7    # (원본에서 비활성화된 필터, 값만 보관)
SUMMARY_TOP_N = 20
SPLIT_CHUNK_WORDS = 15    # 줄·문장 구분이 없는 가사를 나누는 단어 수
ALLOWED_POS = {"noun", "verb", "adjective"}
SPACY_MODEL = "en_core_web_sm"

LYRICS_STOPWORDS = {
    # Korean
    "나", "너", "우리", "그", "그녀", "저", "이", "것", "거",
    "진짜", "그냥", "너무", "정말", "약간", "음", "아",
    "그게", "와", "하", "사람", "남자", "여자", "아이",
    # English
    "yeah", "oh", "ooh", "uh", "ah", "la", "hey", "na", "da", "ra",
    "aha", "really", "um", "ha", "I", "me", "you", "us", "he", "she",
    "they", "her", "his", "them", "our", "this", "that", "such", "so",
    "girl", "boy", "man", "woman", "guy", "something", "whoa",
}


# ─────────────────────────────────────────────
# 5단계: 감정 → 색
# ─────────────────────────────────────────────
EMOTION_COLS = ["joy", "sadness", "anger", "fear", "trust", "disgust", "surprise", "anticipation"]
SENTIMENT_COLS = ["positive", "negative"]
SCORE_COLS = EMOTION_COLS + SENTIMENT_COLS
KOREAN_PREDICATE_POS = {"verb", "adjective"}   # 사전 조회 시 '어간+다' 기본형으로 찾는 품사

# 감정 → (Hue 0~360, Saturation, Value)
EMOTION_HSV_MAP: dict[str, tuple[float, float, float]] = {
    "joy": (49, 0.90, 0.90),
    "sadness": (230, 0.94, 0.85),
    "anger": (0, 0.88, 0.85),
    "fear": (279, 0.90, 0.80),
    "trust": (131, 1.00, 0.75),
    "disgust": (43, 0.63, 0.65),
    "surprise": (27, 1.00, 0.90),
    "anticipation": (334, 0.65, 0.90),
    "unmatched": (0, 0.00, 0.60),
}

EMOTION_COLOR_NAME_MAP: dict[str, str] = {
    "joy": "노란색",
    "sadness": "파란색",
    "anger": "빨간색",
    "fear": "보라색",
    "trust": "초록색",
    "disgust": "갈색",
    "surprise": "주황색",
    "anticipation": "분홍색",
    "unmatched": "회색",
}

SENTIMENT_VALUE_ADJUST = 0.10   # 긍부정 비율(-1~1)에 따른 V(명도) 보정 폭: V + 0.10 × ratio

# 감정 점수가 동점일 때: 곡의 긍부정 비율이 양수면 긍정 감정, 음수면 부정 감정을 먼저 고른다
# (0이거나 같은 그룹 안에서는 EMOTION_COLS 순서). surprise는 중립.
POSITIVE_EMOTIONS = ("joy", "trust", "anticipation")
NEGATIVE_EMOTIONS = ("sadness", "anger", "fear", "disgust")
SENTIMENT_LABEL_THRESHOLD = 0.35
EMOTION_COMBO_TOP_N = 3


# ─────────────────────────────────────────────
# 6단계: 형용사 + 시각화
# ─────────────────────────────────────────────
VA_CENTER = 5             # (x - 5) / 5 로 -1~1 변환 (그래프 좌표)
VA_HALF_RANGE = 5
# 형용사 임계값
#   "calibrated" (기본): 학습 시 DEAM 검증(OOF) 예측을 3등분하는 값 (models/train_meta.json에 저장).
#       모델 예측이 평균 쪽으로 압축되어(R² 0.4 수준) 원래 ±0.2(=4~6) 기준으로는 DEAM에서도 70%가
#       '포근한'이 됐음 → 모델 출력 분포에 맞춘 고정 임계값. 플레이리스트와 무관한 절대 기준.
#   "fixed": 원본 규칙, (x-5)/5 > +0.2 / > -0.2
LABEL_THRESHOLD_MODE = "calibrated"
LABEL_QUANTILES = (1 / 3, 2 / 3)
LABEL_THRESHOLD = 0.2     # fixed 모드

FAILED_LABEL = "분석 실패"   # 예측값(음원) 또는 가사 색이 없는 곡
FAILED_COLOR = "#555566"
VALENCE_LABELS = ("밝은", "담담한", "우울한")    # > +0.2 / > -0.2 / 나머지
AROUSAL_LABELS = ("활기찬", "포근한", "차분한")

AROUSAL_COLORS = {"활기찬": "#FF4C4C", "포근한": "#4CA3FF", "차분한": "#A0A0C8"}
VALENCE_COLORS = {"밝은": "#FFD700", "담담한": "#87CEEB", "우울한": "#708090"}
VALENCE_MARKERS = {"밝은": "o", "담담한": "D", "우울한": "s"}
VALENCE_SYMBOLS = {"밝은": "●", "담담한": "◆", "우울한": "■"}

# 가사 색감 9가지 (시각화용 실제 색)
LYRICS_COLOR_MAP = {
    "노란색": "#FFD700",
    "파란색": "#4CA3FF",
    "빨간색": "#FF4C4C",
    "보라색": "#9B59B6",
    "초록색": "#7EC8A4",
    "갈색": "#8B6347",
    "주황색": "#FF9933",
    "분홍색": "#FF69B4",
    "회색": "#A0A0A0",
}
