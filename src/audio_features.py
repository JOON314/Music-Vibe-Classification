"""2단계: 유튜브에서 음원을 받아 오디오 피처 24개(DEAM 학습 피처와 같은 이름)를 추출한다.

입력: input/Playlist.csv
출력: outputs/02_audio_features.csv (song_id, title, artist, FEAT_COLS...)
      outputs/02_audio_failed.csv (실패한 곡과 사유 — 실패 곡은 3단계 이후 예측값이 비어 있음)
캐시: outputs/cache/audio_features_cache.csv — 이미 피처를 뽑은 (title, artist)는 다시 받지 않음
     (피처 계산 코드를 바꿨다면 --no-cache로 새로 뽑을 것)

추출기(config.AUDIO_EXTRACTOR)
  - opensmile (기본): DEAM과 같은 openSMILE ComParE_2016 LLD의 곡 전체 평균
  - librosa: 원본 프로젝트의 근사 구현. DEAM과 스케일이 수백~수천 σ 달라(docs/phase2_분석.md) 비교용
"""
from __future__ import annotations

import os
import shutil
import tempfile
import warnings
from contextlib import nullcontext
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

import config
from src.common import get_logger, load_playlist, read_csv, write_csv

log = get_logger(__name__)

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
        "quiet": True, "no_warnings": True, "noprogress": True,
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


@lru_cache(maxsize=1)
def _opensmile_extractor():
    """openSMILE ComParE_2016 LLD 추출기 (한 번만 생성).

    opensmile 패키지는 설정 파일 경로를 ASCII로만 C 라이브러리에 넘겨서, 프로젝트 경로에
    한글('캡스톤')이 있으면 UnicodeEncodeError가 난다 → 설정 폴더를 ASCII 임시 경로로 복사해 사용.
    """
    import opensmile

    src = os.path.join(os.path.dirname(opensmile.__file__), "core", "config")
    dst = os.path.join(tempfile.gettempdir(), f"opensmile_config_{opensmile.__version__}")
    if not os.path.isdir(dst):
        shutil.copytree(src, dst)
    if not dst.isascii():
        raise RuntimeError(f"openSMILE 설정 경로에 ASCII가 아닌 문자가 있습니다: {dst} (TMPDIR을 영문 경로로 설정하세요)")

    class _Smile(opensmile.Smile):
        @property
        def default_config_root(self) -> str:
            return dst

    return _Smile(feature_set=opensmile.FeatureSet.ComParE_2016,
                  feature_level=opensmile.FeatureLevel.LowLevelDescriptors)


def extract_features_opensmile(wav_path: Path) -> dict[str, float]:
    """openSMILE ComParE_2016 LLD를 곡 전체 프레임 평균 → FEAT_COLS 24개.

    DEAM 피처(output.csv)도 openSMILE 프레임 값을 곡별로 평균한 것(참고자료/Data process.py)이라 같은 방식.
    """
    lld = _opensmile_extractor().process_file(str(wav_path))
    feats = {}
    for col in config.FEAT_COLS:
        name = col.removesuffix("_amean")
        name = config.OPENSMILE_RENAME.get(name, name)
        feats[col] = float(np.nanmean(lld[name].to_numpy(dtype=float)))
    return feats


def extract_features(wav_path: Path) -> dict[str, float]:
    """config.AUDIO_EXTRACTOR에 따라 피처 24개를 추출한다."""
    if config.AUDIO_EXTRACTOR == "opensmile":
        return extract_features_opensmile(wav_path)
    if config.AUDIO_EXTRACTOR == "librosa":
        return extract_features_librosa(wav_path)
    raise ValueError(f"알 수 없는 AUDIO_EXTRACTOR: {config.AUDIO_EXTRACTOR}")


def extract_features_librosa(wav_path: Path) -> dict[str, float]:
    """(원본 방식) librosa/parselmouth로 openSMILE 피처를 근사한다(곡 전체 프레임 평균)."""
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
    """한 곡의 다운로드 임시 파일(확장자 무관) 삭제."""
    for p in stem.parent.glob(f"{stem.name}.*"):
        p.unlink(missing_ok=True)


def _load_cache(cfg: config.Config) -> dict[tuple[str, str], dict[str, float]]:
    """(title, artist) → 피처 dict. 캐시 미사용·파일 없음·피처 목록 불일치면 빈 dict."""
    if not cfg.use_cache or not cfg.audio_cache_csv.exists():
        return {}
    df = read_csv(cfg.audio_cache_csv, dtype={"title": str, "artist": str})
    if list(df.columns) != ["title", "artist", *config.FEAT_COLS]:
        log.warning("오디오 캐시의 피처 목록이 config와 달라 무시합니다: %s", cfg.audio_cache_csv)
        return {}
    return {(r["title"], r["artist"]): {c: r[c] for c in config.FEAT_COLS}
            for r in df.to_dict("records")}


def _save_cache(cfg: config.Config, cache: dict[tuple[str, str], dict[str, float]]) -> None:
    rows = [{"title": t, "artist": a, **f} for (t, a), f in cache.items()]
    write_csv(pd.DataFrame(rows, columns=["title", "artist", *config.FEAT_COLS]), cfg.audio_cache_csv)


def run(cfg: config.Config) -> pd.DataFrame:
    """플레이리스트 곡별로 다운로드 → 피처 추출 후 02_audio_features.csv로 저장한다."""
    warnings.filterwarnings("ignore")
    playlist = load_playlist(cfg.input_csv)
    rows: list[dict] = []
    failed: list[dict] = []
    cache = _load_cache(cfg)
    n_cached = 0

    tmp_ctx = (nullcontext(str(cfg.audio_tmp_dir)) if cfg.audio_tmp_dir
               else tempfile.TemporaryDirectory(prefix="playlist_audio_"))
    with tmp_ctx as tmp_dir:
        Path(tmp_dir).mkdir(parents=True, exist_ok=True)
        for i, song in enumerate(playlist.itertuples(index=False), 1):
            title, artist = song.title, song.artist
            ids = {"song_id": song.song_id, "title": title, "artist": artist}
            if not isinstance(title, str) or not isinstance(artist, str) or not title or not artist:
                failed.append({**ids, "reason": "제목/아티스트 없음"})
                continue
            if (title, artist) in cache:
                rows.append({**ids, **cache[(title, artist)]})
                n_cached += 1
                continue
            log.info("[%d/%d] %s - %s", i, len(playlist), artist, title)
            stem = Path(tmp_dir) / f"song_{song.song_id}"  # 곡 이름의 특수문자(/ 등) 문제 회피
            try:
                wav = download_audio(title, artist, stem)
                feats = extract_features(wav)
                rows.append({**ids, **feats})
                cache[(title, artist)] = feats
            except Exception as e:  # noqa: BLE001 - 곡 단위 실패는 기록하고 다음 곡으로
                log.error("  [실패] %s - %s: %s", artist, title, e)
                failed.append({**ids, "reason": str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__})
            finally:
                _cleanup(stem)

    if n_cached:
        log.info("오디오 캐시 사용: %d곡", n_cached)
    if cfg.use_cache:
        _save_cache(cfg, cache)

    failed_df = pd.DataFrame(failed, columns=["song_id", "title", "artist", "reason"])
    write_csv(failed_df, cfg.audio_failed_csv, log)
    if failed:
        log.warning("오디오 피처 실패 %d곡: %s", len(failed), ", ".join(f"{t}" for t in failed_df["title"]))

    # 성공 0곡이어도 파일은 새로 씀 (원본은 쓰지 않아 이전 실행의 오래된 파일이 3단계에 쓰였음)
    out = pd.DataFrame(rows, columns=["song_id", "title", "artist", *config.FEAT_COLS])
    if out.empty:
        log.warning("피처를 추출한 곡이 없습니다.")
    write_csv(out, cfg.audio_features_csv, log)
    return out
