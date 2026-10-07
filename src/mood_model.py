"""3단계: DEAM 데이터로 Valence/Arousal 회귀 모델을 학습하고 플레이리스트 곡을 예측한다.

입력: data/deam_features_mean.csv(X), data/deam_labels_avg_mood.csv(Y), outputs/02_audio_features.csv
출력: outputs/03_va_predictions.csv (song_id, title, artist, valence_mean_pred, arousal_mean_pred)
      models/fold_{1..5}.keras, models/scalers.joblib, models/train_meta.json

models/에 학습 결과가 있으면 불러와서 예측만 하고, 없거나 --retrain이면 학습 후 저장한다.

모델: Dense 128-64-32-2, 5-Fold 앙상블, SEED=42 (CLAUDE.md 도메인 규칙).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

import config
from src.common import ID_COLS, get_logger, read_csv, write_csv

log = get_logger(__name__)


@dataclass
class FoldModel:
    """한 Fold의 학습 결과: 모델 + 입력/출력 스케일러 + 검증 지표."""

    model: object
    scaler_x: object
    scaler_y: object
    mae: float
    r2_valence: float
    r2_arousal: float


def set_seeds(seed: int = config.SEED) -> None:
    """TensorFlow·NumPy 난수 시드 고정."""
    import tensorflow as tf

    tf.random.set_seed(seed)
    np.random.seed(seed)


def load_training_data(cfg: config.Config) -> tuple[np.ndarray, np.ndarray]:
    """DEAM 피처와 라벨을 FileName == song_id로 조인하고 NaN/Inf 행을 제거한다."""
    df_x = pd.read_csv(cfg.deam_features_csv)
    df_y = pd.read_csv(cfg.deam_labels_csv)
    for d in (df_x, df_y):
        d.columns = d.columns.str.strip()

    merged = pd.merge(df_x, df_y, left_on=config.DEAM_JOIN_LEFT,
                      right_on=config.DEAM_JOIN_RIGHT, how="inner")
    log.info("[데이터] 매핑 완료: %d곡, feature %d개", len(merged), len(config.FEAT_COLS))

    X = merged[config.FEAT_COLS].values.astype(np.float32)
    Y = merged[config.TARGET_COLS].values.astype(np.float32)
    mask = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
    return X[mask], Y[mask]


def build_model(n_feat: int):
    """Dense 128-64-32-2 회귀 모델."""
    import tensorflow as tf

    init = tf.keras.initializers.GlorotUniform(seed=config.SEED)
    h1, h2, h3 = config.HIDDEN_UNITS
    d1, d2 = config.DROPOUTS
    m = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_feat,)),
        tf.keras.layers.Dense(h1, activation="relu", kernel_initializer=init),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(d1, seed=config.SEED),
        tf.keras.layers.Dense(h2, activation="relu", kernel_initializer=init),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(d2, seed=config.SEED),
        tf.keras.layers.Dense(h3, activation="relu", kernel_initializer=init),
        tf.keras.layers.Dense(2, activation="linear"),
    ])
    m.compile(loss="mse",
              optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
              metrics=["mae"])
    return m


def train(X: np.ndarray, Y: np.ndarray) -> list[FoldModel]:
    """5-Fold 교차검증으로 Fold별 모델을 학습하고 검증 지표를 기록한다."""
    import tensorflow as tf
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler

    kf = KFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED)
    folds: list[FoldModel] = []

    for fold, (tr, va) in enumerate(kf.split(X), 1):
        sx, sy = StandardScaler(), StandardScaler()
        X_tr_s, Y_tr_s = sx.fit_transform(X[tr]), sy.fit_transform(Y[tr])
        X_va_s, Y_va_s = sx.transform(X[va]), sy.transform(Y[va])

        model = build_model(X.shape[1])
        cbs = [
            tf.keras.callbacks.EarlyStopping(patience=config.EARLY_STOP_PATIENCE,
                                             restore_best_weights=True, monitor="val_loss"),
            tf.keras.callbacks.ReduceLROnPlateau(patience=config.LR_PLATEAU_PATIENCE,
                                                 factor=config.LR_PLATEAU_FACTOR, verbose=0),
        ]
        model.fit(X_tr_s, Y_tr_s, epochs=config.EPOCHS, batch_size=config.BATCH_SIZE,
                  validation_data=(X_va_s, Y_va_s), callbacks=cbs, verbose=0)

        Y_va = Y[va]
        pred_va = sy.inverse_transform(model.predict(X_va_s, verbose=0))
        fm = FoldModel(
            model=model, scaler_x=sx, scaler_y=sy,
            mae=float(mean_absolute_error(Y_va, pred_va)),
            r2_valence=float(r2_score(Y_va[:, 0], pred_va[:, 0])),
            r2_arousal=float(r2_score(Y_va[:, 1], pred_va[:, 1])),
        )
        folds.append(fm)
        log.info("  Fold %d | MAE %.3f | R²(V) %.3f | R²(A) %.3f",
                 fold, fm.mae, fm.r2_valence, fm.r2_arousal)

    maes = [f.mae for f in folds]
    log.info("[결과] MAE %.3f±%.3f", np.mean(maes), np.std(maes))
    log.info("[결과] R²(Valence) %.3f", np.mean([f.r2_valence for f in folds]))
    log.info("[결과] R²(Arousal) %.3f", np.mean([f.r2_arousal for f in folds]))
    return folds


def predict(folds: list[FoldModel], features: np.ndarray) -> np.ndarray:
    """Fold 모델 예측의 평균(앙상블). NaN/Inf 피처는 스케일링 전에 0으로 채운다(원본 동작)."""
    clean = features.astype(np.float32).copy()
    clean[~np.isfinite(clean)] = 0.0
    preds = [
        f.scaler_y.inverse_transform(f.model.predict(f.scaler_x.transform(clean), verbose=0))
        for f in folds
    ]
    return np.mean(preds, axis=0)


def save_models(folds: list[FoldModel], cfg: config.Config) -> None:
    """Fold 모델(.keras)·스케일러(joblib)·학습 메타데이터(json)를 models/에 저장한다."""
    import joblib
    import tensorflow as tf

    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    for k, f in enumerate(folds, 1):
        f.model.save(cfg.fold_model_path(k))
    joblib.dump([(f.scaler_x, f.scaler_y) for f in folds], cfg.scalers_path)
    meta = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "tensorflow": tf.__version__,
        "seed": config.SEED,
        "n_folds": len(folds),
        "feat_cols": config.FEAT_COLS,
        "folds": [{"fold": k, "mae": f.mae, "r2_valence": f.r2_valence, "r2_arousal": f.r2_arousal}
                  for k, f in enumerate(folds, 1)],
    }
    cfg.model_meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("모델 저장: %s", cfg.models_dir)


def has_saved_models(cfg: config.Config) -> bool:
    """저장된 모델 세트가 완전히 있는지."""
    paths = [cfg.scalers_path, cfg.model_meta_path,
             *(cfg.fold_model_path(k) for k in range(1, config.N_FOLDS + 1))]
    return all(p.exists() for p in paths)


def load_models(cfg: config.Config) -> list[FoldModel]:
    """저장된 Fold 모델을 불러온다. 피처 목록이 현재 config와 다르면 오류."""
    import joblib
    import tensorflow as tf

    meta = json.loads(cfg.model_meta_path.read_text(encoding="utf-8"))
    if meta["feat_cols"] != config.FEAT_COLS:
        raise ValueError("저장된 모델의 피처 목록이 config.FEAT_COLS와 다릅니다. --retrain으로 다시 학습하세요.")
    scalers = joblib.load(cfg.scalers_path)
    folds = [
        FoldModel(model=tf.keras.models.load_model(cfg.fold_model_path(m["fold"])),
                  scaler_x=sx, scaler_y=sy,
                  mae=m["mae"], r2_valence=m["r2_valence"], r2_arousal=m["r2_arousal"])
        for m, (sx, sy) in zip(meta["folds"], scalers)
    ]
    log.info("저장된 모델 사용: %s (학습 %s, MAE %.3f)", cfg.models_dir, meta["created"],
             np.mean([f.mae for f in folds]))
    return folds


def scale_to_10(pred_col: np.ndarray) -> np.ndarray:
    """플레이리스트 안에서 min-max로 0~10 재조정한다(원본 동작, Phase 2 검토 대상)."""
    p_min, p_max = pred_col.min(), pred_col.max()
    scaled = (pred_col - p_min) / (p_max - p_min + 1e-10) * config.SCALE_MAX
    return np.clip(scaled, 0, config.SCALE_MAX)


def run(cfg: config.Config) -> pd.DataFrame:
    """(필요하면 학습·저장) → 플레이리스트 예측 → 03_va_predictions.csv 저장."""
    df_test = read_csv(cfg.audio_features_csv)
    df_test.columns = df_test.columns.str.strip()
    test_features = df_test[config.FEAT_COLS].values
    if len(df_test) == 0:
        log.warning("예측할 곡이 없습니다 (02_audio_features.csv가 비어 있음).")
        result = pd.DataFrame(columns=[*ID_COLS, *config.PRED_COLS])
        write_csv(result, cfg.va_predictions_csv, log)
        return result

    if cfg.retrain or not has_saved_models(cfg):
        set_seeds()
        X, Y = load_training_data(cfg)
        folds = train(X, Y)
        save_models(folds, cfg)
    else:
        folds = load_models(cfg)
    ens = predict(folds, test_features)

    result = df_test[ID_COLS].copy()
    for i, col in enumerate(config.PRED_COLS):
        result[col] = scale_to_10(ens[:, i]).round(config.PRED_ROUND)

    log.info("예측 결과 (앙상블)\n%s", result.to_string(index=False))
    write_csv(result, cfg.va_predictions_csv, log)
    return result
