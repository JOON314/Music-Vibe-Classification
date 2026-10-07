"""프로젝트 전체 설정: 경로·파일명·피처 목록·임계값·색 매핑을 한 곳에 모은다.

결과에 영향을 주는 값(피처 순서, 모델 하이퍼파라미터, 임계값, 색 매핑)은
CLAUDE.md의 도메인 규칙이므로 승인 없이 바꾸지 않는다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
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
    # 유튜브 음원 임시 다운로드 위치 (원본 동작: 프로젝트 루트)
    audio_tmp_dir: Path = field(default=ROOT_DIR)

    # ── 학습 데이터·사전 (data/)
    @property
    def deam_features_csv(self) -> Path:
        return self.data_dir / "deam_features_mean.csv"

    @property
    def deam_labels_csv(self) -> Path:
        return self.data_dir / "deam_labels_avg_mood.csv"

    @property
    def lexicon_csv(self) -> Path:
        return self.data_dir / "emotion_lexicon.csv"

    # ── 단계별 산출물 (outputs/)
    @property
    def lyrics_csv(self) -> Path:            # 1단계
        return self.output_dir / "01_lyrics.csv"

    @property
    def audio_features_csv(self) -> Path:    # 2단계
        return self.output_dir / "02_audio_features.csv"

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

    @property
    def final_csv(self) -> Path:             # 6단계
        return self.output_dir / "final_result.csv"

    @property
    def figure_png(self) -> Path:            # 6단계
        return self.output_dir / "playlist_analysis.png"

    def ensure_dirs(self) -> None:
        """출력 폴더가 없으면 만든다."""
        for d in (self.output_dir, self.models_dir):
            d.mkdir(parents=True, exist_ok=True)


CSV_ENCODING = "utf-8-sig"


# ─────────────────────────────────────────────
# 1단계: 멜론 가사 크롤링
# ─────────────────────────────────────────────
MELON_URL = "https://www.melon.com"
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

# librosa/parselmouth 추출 파라미터 (openSMILE 근사)
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
SCALE_MAX = 10          # scale_to_10: 플레이리스트 내 min-max → 0~10
PRED_ROUND = 4


# ─────────────────────────────────────────────
# 4단계: 가사 핵심 키워드
# ─────────────────────────────────────────────
MIN_WORD_LENGTH = 2       # 지나치게 짧은 단어 제외
MIN_COUNT_IN_SONG = 2     # 곡 안에서 너무 적게 나온 단어 제외
MAX_GLOBAL_RATIO = 0.7    # (원본에서 비활성화된 필터, 값만 보관)
SUMMARY_TOP_N = 20
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

SENTIMENT_VALUE_ADJUST = 0.10   # 감정 비율에 따른 V 보정 폭
SENTIMENT_LABEL_THRESHOLD = 0.35
EMOTION_COMBO_TOP_N = 3


# ─────────────────────────────────────────────
# 6단계: 형용사 + 시각화
# ─────────────────────────────────────────────
VA_CENTER = 5             # (x - 5) / 5 로 -1~1 변환
VA_HALF_RANGE = 5
LABEL_THRESHOLD = 0.2

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
