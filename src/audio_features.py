"""2단계: 유튜브에서 음원을 받아 openSMILE 근사 오디오 피처 24개를 추출한다.

입력: input/Playlist.csv
출력: outputs/02_audio_features.csv (song_id, title, artist, FEAT_COLS...)

학습 데이터(DEAM)는 openSMILE 피처, 여기서는 librosa/parselmouth로 같은 이름의 값을 근사한다.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import config
from src.common import get_logger, load_playlist, write_csv

log = get_logger(__name__)

_TMP_EXTS = (".wav", ".webm", ".m4a", ".mp4")


def download_audio(title: str, artist: str, out_path: Path) -> Path:
    """유튜브 검색 첫 결과를 wav로 받는다. out_path는 확장자 없는 경로."""
    import yt_dlp

    query = f"{artist}-{title}"
    wav_file = Path(f"{out_path}.wav")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": f"{out_path}.%(ext)s",
        "postprocessors": [{"key": "FFmpegExtractAudio",
                            "preferredcodec": "wav",
                            "preferredquality": "192"}],
        "quiet": True, "no_warnings": True,
    }
    log.info("[다운로드] %s - %s", artist, title)
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([f"ytsearch1:{query}"])
    if not wav_file.exists():
        raise FileNotFoundError(wav_file)
    return wav_file


def _nanmean(x) -> float:
    """NaN을 뺀 평균, 값이 없으면 0.0."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    return float(x.mean()) if len(x) else 0.0


def extract_features(wav_path: Path) -> dict[str, float]:
    """wav 한 곡에서 FEAT_COLS 24개 값을 계산한다(곡 전체 프레임 평균)."""
    import librosa
    import parselmouth
    from parselmouth.praat import call

    wav_path = str(wav_path)
    y, sr = librosa.load(wav_path, sr=config.AUDIO_SR, mono=True)
    hop, win, n_fft = config.AUDIO_HOP, config.AUDIO_WIN, config.AUDIO_N_FFT

    D = librosa.stft(y, n_fft=n_fft, hop_length=hop, win_length=win)
    S_mag = np.abs(D)
    S_pow = S_mag ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    n_frames = S_mag.shape[1]
    times = librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=hop)

    # ── Praat: F0 / voicing / jitter / shimmer / HNR
    snd = parselmouth.Sound(wav_path)
    pitch = call(snd, "To Pitch", 0.0, 75, 600)
    f0_vals = np.array([pitch.get_value_at_time(t) for t in times], dtype=float)
    voiced = (~np.isnan(f0_vals)) & (f0_vals > 0)
    f0_voiced = f0_vals[voiced]

    pp = call(snd, "To PointProcess (periodic, cc)", 75, 600)
    jitter_local = call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3) or 0.0
    shimmer_local = call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6) or 0.0

    harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    hnr_vals = np.array([harmonicity.get_value(t) for t in times], dtype=float)

    # ── RMS / ZCR
    rms = librosa.feature.rms(y=y, frame_length=win, hop_length=hop)[0]
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=win, hop_length=hop)[0]

    # ── 26밴드 auditory spectrum (비정규화, openSMILE 정의에 맞춤)
    mel_fb = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=26, fmin=20, fmax=sr / 2, norm=None)
    aud_spec = mel_fb @ S_pow
    audspec_l1 = aud_spec.sum(axis=0)

    # 프레임 길이 정렬
    n = min(n_frames, len(rms), len(zcr), len(audspec_l1), len(hnr_vals))
    S_m = S_mag[:, :n]
    S_p = S_pow[:, :n]
    rms, zcr, audspec_l1, hnr_vals = rms[:n], zcr[:n], audspec_l1[:n], hnr_vals[:n]

    total_pow = S_p.sum(axis=0) + 1e-10
    prob = S_p / total_pow

    def band_energy(flo: float, fhi: float) -> np.ndarray:
        mask = (freqs >= flo) & (freqs < fhi)
        return S_p[mask].sum(axis=0)

    def rolloff(ratio: float) -> np.ndarray:
        cum = np.cumsum(S_p, axis=0)
        idx = np.argmax(cum >= ratio * total_pow, axis=0)
        return freqs[idx]

    fband_250_650 = band_energy(250, 650)
    fband_1k_4k = band_energy(1000, 4000)
    ro25, ro50, ro75, ro90 = rolloff(0.25), rolloff(0.50), rolloff(0.75), rolloff(0.90)

    centroid = (freqs[:, None] * S_p).sum(axis=0) / total_pow
    dev = freqs[:, None] - centroid[None, :]
    variance = (dev ** 2 * S_p).sum(axis=0) / total_pow
    std_s = np.sqrt(variance + 1e-10)
    skew = (dev ** 3 * S_p).sum(axis=0) / (total_pow * std_s ** 3 + 1e-10)
    kurt = (dev ** 4 * S_p).sum(axis=0) / (total_pow * std_s ** 4 + 1e-10)
    entropy = -np.sum(prob * np.log(prob + 1e-10), axis=0)
    x = freqs - freqs.mean()
    slope = (x[:, None] * S_m).sum(axis=0) / (np.sum(x ** 2) + 1e-10)

    # Flux: 비정규화 magnitude 차이의 L2 (openSMILE 정의)
    flux = np.concatenate([[0], np.sqrt(np.sum(np.diff(S_m, axis=1) ** 2, axis=0))])

    bark = 13 * np.arctan(0.00076 * freqs) + 3.5 * np.arctan((freqs / 7500) ** 2)
    sharp = (bark[:, None] * S_p).sum(axis=0) / total_pow

    # MFCC (openSMILE 호환에 가깝게: htk + lifter)
    mfcc_all = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=15, n_fft=n_fft,
                                    hop_length=hop, win_length=win,
                                    htk=True, lifter=22)[:, :n]

    feats = {
        "F0final_sma_amean": float(f0_voiced.mean()) if len(f0_voiced) else 0.0,
        "voicingFinalUnclipped_sma_amean": float(voiced[:n].mean()),
        "jitterLocal_sma_amean": float(jitter_local),
        "shimmerLocal_sma_amean": float(shimmer_local),
        "logHNR_sma_amean": _nanmean(hnr_vals),
        "audspec_lengthL1norm_sma_amean": _nanmean(audspec_l1),
        "pcm_RMSenergy_sma_amean": _nanmean(rms),
        "pcm_zcr_sma_amean": _nanmean(zcr),
        "pcm_fftMag_fband250-650_sma_amean": _nanmean(fband_250_650),
        "pcm_fftMag_fband1000-4000_sma_amean": _nanmean(fband_1k_4k),
        "pcm_fftMag_spectralRollOff25.0_sma_amean": _nanmean(ro25),
        "pcm_fftMag_spectralRollOff50.0_sma_amean": _nanmean(ro50),
        "pcm_fftMag_spectralRollOff75.0_sma_amean": _nanmean(ro75),
        "pcm_fftMag_spectralRollOff90.0_sma_amean": _nanmean(ro90),
        "pcm_fftMag_spectralFlux_sma_amean": _nanmean(flux),
        "pcm_fftMag_spectralCentroid_sma_amean": _nanmean(centroid),
        "pcm_fftMag_spectralEntropy_sma_amean": _nanmean(entropy),
        "pcm_fftMag_spectralVariance_sma_amean": _nanmean(variance),
        "pcm_fftMag_spectralSkewness_sma_amean": _nanmean(skew),
        "pcm_fftMag_spectralKurtosis_sma_amean": _nanmean(kurt),
        "pcm_fftMag_spectralSlope_sma_amean": _nanmean(slope),
        "pcm_fftMag_psySharpness_sma_amean": _nanmean(sharp),
        "pcm_fftMag_mfcc_sma[1]_amean": _nanmean(mfcc_all[1]),
        "pcm_fftMag_mfcc_sma[4]_amean": _nanmean(mfcc_all[4]),
    }
    assert list(feats) == config.FEAT_COLS, "피처 순서가 config.FEAT_COLS와 다릅니다"
    return feats


def _cleanup(stem: Path) -> None:
    """다운로드 임시 파일 삭제."""
    for ext in _TMP_EXTS:
        p = Path(f"{stem}{ext}")
        if p.exists():
            p.unlink()


def run(cfg: config.Config) -> pd.DataFrame:
    """플레이리스트 곡별로 다운로드 → 피처 추출 후 02_audio_features.csv로 저장한다."""
    warnings.filterwarnings("ignore")
    playlist = load_playlist(cfg.input_csv)
    rows: list[dict] = []

    for i, song in enumerate(playlist.itertuples(index=False), 1):
        title, artist = song.title, song.artist
        if not isinstance(title, str) or not isinstance(artist, str) or not title or not artist:
            continue
        log.info("[%d/%d] %s - %s", i, len(playlist), artist, title)
        stem = Path(cfg.audio_tmp_dir) / f"{artist}_{title}".replace(" ", "_")
        try:
            wav = download_audio(title, artist, stem)
            rows.append({"song_id": song.song_id, "title": title, "artist": artist,
                         **extract_features(wav)})
            _cleanup(stem)
        except Exception as e:  # noqa: BLE001 - 곡 단위 실패는 건너뜀(원본 동작)
            log.error("  [에러] %s", e)
            continue

    if not rows:
        log.warning("피처를 추출한 곡이 없습니다. 파일을 쓰지 않습니다.")
        return pd.DataFrame(columns=["song_id", "title", "artist", *config.FEAT_COLS])

    out = pd.DataFrame(rows)[["song_id", "title", "artist", *config.FEAT_COLS]]
    write_csv(out, cfg.audio_features_csv, log)
    return out
