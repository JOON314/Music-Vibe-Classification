import os
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
os.makedirs(DATASET_DIR, exist_ok=True)  # 혹시 없을 경우 대비

SEED = 42
tf.random.set_seed(SEED); np.random.seed(SEED)

FEAT_COLS = [
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
N_FEAT = len(FEAT_COLS)

df_x    = pd.read_csv(os.path.join(DATASET_DIR, "output.csv"))
df_y    = pd.read_csv(os.path.join(DATASET_DIR, "Avg_mood.csv"))
df_test = pd.read_csv(os.path.join(DATASET_DIR, "playlist_features.csv"))
for d in (df_x, df_y, df_test):
    d.columns = d.columns.str.strip()

merged = pd.merge(df_x, df_y, left_on="FileName", right_on="song_id", how="inner")
print(f"[데이터] 매핑 완료: {len(merged)}곡, feature {N_FEAT}개")

X = merged[FEAT_COLS].values.astype(np.float32)
Y = merged[["valence_mean", "arousal_mean"]].values.astype(np.float32)

# NaN/Inf 마스킹 (X, Y 동시에)
mask = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
X, Y = X[mask], Y[mask]

test_features = df_test[FEAT_COLS].values.astype(np.float32)
test_mask = np.isfinite(test_features).all(axis=1)
test_features_clean = test_features.copy()
test_features_clean[~np.isfinite(test_features_clean)] = 0.0

def build_model():
    init = tf.keras.initializers.GlorotUniform(seed=SEED)
    m = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(N_FEAT,)),
        tf.keras.layers.Dense(128, activation="relu", kernel_initializer=init),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2, seed=SEED),
        tf.keras.layers.Dense(64, activation="relu", kernel_initializer=init),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.15, seed=SEED),
        tf.keras.layers.Dense(32, activation="relu", kernel_initializer=init),
        tf.keras.layers.Dense(2, activation="linear"),
    ])
    m.compile(loss="mse",
              optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              metrics=["mae"])
    return m

kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
fold_maes, fold_r2s = [], []
all_test_preds = []

for fold, (tr, va) in enumerate(kf.split(X), 1):
    X_tr, X_va = X[tr], X[va]
    Y_tr, Y_va = Y[tr], Y[va]
    sx, sy = StandardScaler(), StandardScaler()
    X_tr_s, Y_tr_s = sx.fit_transform(X_tr), sy.fit_transform(Y_tr)
    X_va_s, Y_va_s = sx.transform(X_va), sy.transform(Y_va)
    test_s = sx.transform(test_features_clean)

    model = build_model()
    cbs = [
        tf.keras.callbacks.EarlyStopping(patience=25, restore_best_weights=True, monitor="val_loss"),
        tf.keras.callbacks.ReduceLROnPlateau(patience=10, factor=0.5, verbose=0),
    ]
    model.fit(X_tr_s, Y_tr_s, epochs=300, batch_size=32,
              validation_data=(X_va_s, Y_va_s),
              callbacks=cbs, verbose=0)

    pred_va = sy.inverse_transform(model.predict(X_va_s, verbose=0))
    mae = mean_absolute_error(Y_va, pred_va)
    r2v = r2_score(Y_va[:, 0], pred_va[:, 0])
    r2a = r2_score(Y_va[:, 1], pred_va[:, 1])
    fold_maes.append(mae); fold_r2s.append((r2v, r2a))
    print(f"  Fold {fold} | MAE {mae:.3f} | R²(V) {r2v:.3f} | R²(A) {r2a:.3f}")

    pred_test = sy.inverse_transform(model.predict(test_s, verbose=0))
    all_test_preds.append(pred_test)

print(f"\n[결과] MAE {np.mean(fold_maes):.3f}±{np.std(fold_maes):.3f}")
print(f"[결과] R²(Valence) {np.mean([r[0] for r in fold_r2s]):.3f}")
print(f"[결과] R²(Arousal) {np.mean([r[1] for r in fold_r2s]):.3f}")

ens = np.mean(all_test_preds, axis=0)

def scale_to_10(pred_col):
    p_min, p_max = pred_col.min(), pred_col.max()
    scaled = (pred_col - p_min) / (p_max - p_min + 1e-10) * 10
    return np.clip(scaled, 0, 10)

result = df_test[["title", "artist"]].copy()
result["valence_mean_pred"] = scale_to_10(ens[:, 0]).round(4)
result["arousal_mean_pred"] = scale_to_10(ens[:, 1]).round(4)

print("\n===== 예측 결과 (앙상블) =====")
print(result.to_string(index=False))

try:
    orig = pd.read_csv(os.path.join(DATASET_DIR, "Playlists_lyrics_mood.csv"), encoding="utf-8-sig")
    for c in ("title", "artist"):
        orig[c] = orig[c].str.strip(); result[c] = result[c].str.strip()
    merged_out = pd.merge(orig, result, on=["title", "artist"], how="left")
    merged_out.to_csv(os.path.join(DATASET_DIR, "Playlists_lyrics_mood.csv"), index=False, encoding="utf-8-sig")
    print("\n✅ Playlists_lyrics_mood.csv 업데이트 완료")
except FileNotFoundError:
    result.to_csv(os.path.join(DATASET_DIR, "prediction_result.csv"), index=False, encoding="utf-8-sig")
    print("\n✅ prediction_result.csv 저장 완료")
