# %%
import os, numpy as np, pandas as pd, librosa, parselmouth
from parselmouth.praat import call
import warnings
warnings.filterwarnings("ignore")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "input", "Playlist.csv")

# ─────────────────────────────────────────────
def download_audio(title, artist, out_path="downloaded_audio"):
    import yt_dlp
    query = f"{artist}-{title}"
    wav_file = f"{out_path}.wav"
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": f"{out_path}.%(ext)s",
        "postprocessors": [{"key": "FFmpegExtractAudio",
                            "preferredcodec": "wav",
                            "preferredquality": "192"}],
        "quiet": True, "no_warnings": True,
    }
    print(f"[다운로드] {artist} - {title}")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([f"ytsearch1:{query}"])
    if not os.path.exists(wav_file):
        raise FileNotFoundError(wav_file)
    return wav_file

# ─────────────────────────────────────────────
SELECTED = [
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

def nanmean(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    return float(x.mean()) if len(x) else 0.0

def extract_features(wav_path, title, artist):
    y, sr = librosa.load(wav_path, sr=16000, mono=True)
    hop, win, n_fft = 160, 400, 512

    D = librosa.stft(y, n_fft=n_fft, hop_length=hop, win_length=win)
    S_mag = np.abs(D)
    S_pow = S_mag ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    n_frames = S_mag.shape[1]
    times = librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=hop)

    # ── Praat: F0 / voicing / jitter / shimmer / HNR ──
    snd = parselmouth.Sound(wav_path)
    pitch = call(snd, "To Pitch", 0.0, 75, 600)
    f0_vals = np.array([pitch.get_value_at_time(t) for t in times], dtype=float)
    voiced = (~np.isnan(f0_vals)) & (f0_vals > 0)
    f0_voiced = f0_vals[voiced]

    pp = call(snd, "To PointProcess (periodic, cc)", 75, 600)
    jitter_local  = call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3) or 0.0
    shimmer_local = call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6) or 0.0

    harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    hnr_vals = np.array([harmonicity.get_value(t) for t in times], dtype=float)

    # ── RMS / ZCR ──
    rms  = librosa.feature.rms(y=y, frame_length=win, hop_length=hop)[0]
    zcr  = librosa.feature.zero_crossing_rate(y, frame_length=win, hop_length=hop)[0]

    # ── 26-band Bark/Mel auditory spectrum (비정규화, 원본 openSMILE 정의에 맞춤) ──
    mel_fb = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=26, fmin=20, fmax=sr/2, norm=None)
    aud_spec = mel_fb @ S_pow                  # (26, n_frames)
    audspec_l1 = aud_spec.sum(axis=0)          # L1 norm 그 자체 (정규화 X)

    # 프레임 길이 정렬
    n = min(n_frames, len(rms), len(zcr), len(audspec_l1), len(hnr_vals))
    S_m = S_mag[:, :n]; S_p = S_pow[:, :n]
    rms = rms[:n]; zcr = zcr[:n]; audspec_l1 = audspec_l1[:n]
    hnr_vals = hnr_vals[:n]

    total_pow = S_p.sum(axis=0) + 1e-10
    prob = S_p / total_pow

    def band_energy(flo, fhi):
        mask = (freqs >= flo) & (freqs < fhi)
        return S_p[mask].sum(axis=0)
    fband_250_650 = band_energy(250, 650)
    fband_1k_4k   = band_energy(1000, 4000)

    def rolloff(ratio):
        cum = np.cumsum(S_p, axis=0)
        idx = np.argmax(cum >= ratio * total_pow, axis=0)
        return freqs[idx]
    ro25, ro50, ro75, ro90 = rolloff(0.25), rolloff(0.50), rolloff(0.75), rolloff(0.90)

    centroid = (freqs[:, None] * S_p).sum(axis=0) / total_pow
    variance = ((freqs[:, None] - centroid[None, :]) ** 2 * S_p).sum(axis=0) / total_pow
    std_s    = np.sqrt(variance + 1e-10)
    skew     = ((freqs[:, None] - centroid[None, :]) ** 3 * S_p).sum(axis=0) / (total_pow * std_s**3 + 1e-10)
    kurt     = ((freqs[:, None] - centroid[None, :]) ** 4 * S_p).sum(axis=0) / (total_pow * std_s**4 + 1e-10)
    entropy  = -np.sum(prob * np.log(prob + 1e-10), axis=0)
    x = freqs - freqs.mean()
    slope    = (x[:, None] * S_m).sum(axis=0) / (np.sum(x**2) + 1e-10)

    # Flux: 비정규화 magnitude 차이의 L2 (openSMILE 정의)
    flux = np.concatenate([[0], np.sqrt(np.sum(np.diff(S_m, axis=1)**2, axis=0))])

    bark  = 13 * np.arctan(0.00076 * freqs) + 3.5 * np.arctan((freqs / 7500) ** 2)
    sharp = (bark[:, None] * S_p).sum(axis=0) / total_pow

    # MFCC (openSMILE 호환에 가깝게: htk + lifter)
    mfcc_all = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=15, n_fft=n_fft,
                                    hop_length=hop, win_length=win,
                                    htk=True, lifter=22)
    mfcc_all = mfcc_all[:, :n]

    row = {"title": title, "artist": artist}
    row["F0final_sma_amean"]                          = float(f0_voiced.mean()) if len(f0_voiced) else 0.0
    row["voicingFinalUnclipped_sma_amean"]            = float(voiced[:n].mean())
    row["jitterLocal_sma_amean"]                      = float(jitter_local)
    row["shimmerLocal_sma_amean"]                     = float(shimmer_local)
    row["logHNR_sma_amean"]                           = nanmean(hnr_vals)
    row["audspec_lengthL1norm_sma_amean"]             = nanmean(audspec_l1)
    row["pcm_RMSenergy_sma_amean"]                    = nanmean(rms)
    row["pcm_zcr_sma_amean"]                          = nanmean(zcr)
    row["pcm_fftMag_fband250-650_sma_amean"]          = nanmean(fband_250_650)
    row["pcm_fftMag_fband1000-4000_sma_amean"]        = nanmean(fband_1k_4k)
    row["pcm_fftMag_spectralRollOff25.0_sma_amean"]   = nanmean(ro25)
    row["pcm_fftMag_spectralRollOff50.0_sma_amean"]   = nanmean(ro50)
    row["pcm_fftMag_spectralRollOff75.0_sma_amean"]   = nanmean(ro75)
    row["pcm_fftMag_spectralRollOff90.0_sma_amean"]   = nanmean(ro90)
    row["pcm_fftMag_spectralFlux_sma_amean"]          = nanmean(flux)
    row["pcm_fftMag_spectralCentroid_sma_amean"]      = nanmean(centroid)
    row["pcm_fftMag_spectralEntropy_sma_amean"]       = nanmean(entropy)
    row["pcm_fftMag_spectralVariance_sma_amean"]      = nanmean(variance)
    row["pcm_fftMag_spectralSkewness_sma_amean"]      = nanmean(skew)
    row["pcm_fftMag_spectralKurtosis_sma_amean"]      = nanmean(kurt)
    row["pcm_fftMag_spectralSlope_sma_amean"]         = nanmean(slope)
    row["pcm_fftMag_psySharpness_sma_amean"]          = nanmean(sharp)
    row["pcm_fftMag_mfcc_sma[1]_amean"]               = nanmean(mfcc_all[1])
    row["pcm_fftMag_mfcc_sma[4]_amean"]               = nanmean(mfcc_all[4])

    cols = ["title", "artist"] + SELECTED
    return pd.DataFrame([row])[cols]

# ─────────────────────────────────────────────
def main():
    df_list = pd.read_csv(csv_path, encoding="utf-8-sig")
    results = []
    for i in range(len(df_list)):
        title  = str(df_list.iloc[i]["title"]).strip()
        artist = str(df_list.iloc[i]["artist"]).strip()
        if not title or not artist:
            continue
        print(f"\n[{i+1}/{len(df_list)}] {artist} - {title}")
        try:
            safe = f"{artist}_{title}".replace(" ", "_")
            wav = download_audio(title, artist, out_path=safe)
            results.append(extract_features(wav, title, artist))
            for ext in [".wav", ".webm", ".m4a", ".mp4"]:
                tmp = safe + ext
                if os.path.exists(tmp):
                    os.remove(tmp)
        except Exception as e:
            print(f"  [에러] {e}")
            continue
    if results:
        out_dir = os.path.join(BASE_DIR, "dataset")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "playlist_features.csv")

        pd.concat(results, ignore_index=True).to_csv(
            out_path, index=False, encoding="utf-8-sig"
        )
        print(f"\n✅ 저장 완료: {out_path}")

if __name__ == "__main__":
    main()
