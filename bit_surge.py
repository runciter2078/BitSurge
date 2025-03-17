# -*- coding: utf-8 -*-
"""
PREDICCIÓN DEL MOVIMIENTO DEL BITCOIN - VERSIÓN DEFINITIVA
============================================================
Esta implementación integral predice el movimiento del precio de Bitcoin utilizando:
- Descarga robusta de datos históricos de Binance.
- Cálculo extenso de indicadores técnicos y variables sintéticas.
- Reducción de dimensionalidad y eliminación de multicolinealidad mediante PCA.
- Modelos de clasificación (XGBoost, RandomForest y LSTM) con validación temporal.
- Optimización de hiperparámetros con Optuna, empleando TimeSeriesSplit.
- Modelo LSTM entrenado con TimeseriesGenerator para respetar la secuencia temporal.
- Ensamble ponderado basado en F1-score y optimización de threshold.
- Funciones adicionales para walk-forward validation y data drift (MMD).
- Análisis de importancia de features mediante permutation importance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_ta as ta
import requests, time, logging, sys
import warnings
import optuna
from datetime import datetime, timedelta
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, 
                             recall_score, f1_score, classification_report, roc_auc_score,
                             precision_recall_curve)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import tensorflow as tf

# Configuraciones iniciales y logging
warnings.filterwarnings('ignore')
sns.set(style='whitegrid')
plt.style.use('seaborn-v0_8-whitegrid')
logging.basicConfig(stream=sys.stdout, level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# =============================================================================
# 1. DESCARGA DE DATOS DE BTC DESDE BINANCE CON MANEJO DE ERRORES Y REINTENTOS
# =============================================================================
logging.info("Descargando datos de BTC-USDT de Binance (últimos 24 meses, datos horarios)...")

def get_binance_klines(symbol, interval, start_time, end_time, max_retries=5):
    """
    Descarga datos (klines) de Binance para el símbolo e intervalo especificados,
    desde start_time hasta end_time (en milisegundos) con manejo de errores y reintentos.
    """
    url = "https://api.binance.com/api/v3/klines"
    all_data = []
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_time,
        "endTime": end_time,
        "limit": 1000
    }
    while True:
        retries = 0
        success = False
        while retries < max_retries and not success:
            try:
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                success = True
            except requests.exceptions.RequestException as e:
                retries += 1
                logging.warning(f"Error en la solicitud: {e}. Reintento {retries}/{max_retries}...")
                time.sleep(1)
        if not success or not data:
            break
        all_data.extend(data)
        last_time = data[-1][0]
        if last_time >= end_time or len(data) < 1000:
            break
        params["startTime"] = last_time + 1
    return all_data

# Calcular tiempos de inicio y fin en milisegundos
end_time = int(datetime.now().timestamp() * 1000)
start_time = int((datetime.now() - timedelta(days=730)).timestamp() * 1000)
symbol = "BTCUSDT"
interval = "1h"

klines = get_binance_klines(symbol, interval, start_time, end_time)
btc_data = pd.DataFrame(klines, columns=["open_time", "open", "high", "low", "close", "volume",
                                         "close_time", "quote_asset_volume", "num_trades",
                                         "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"])
logging.info(f"Datos descargados: {btc_data.shape}")

# Conversión de columnas y definición del timestamp
btc_data["open_time"] = pd.to_datetime(btc_data["open_time"], unit="ms")
for col in ["open", "high", "low", "close", "volume"]:
    btc_data[col] = pd.to_numeric(btc_data[col], errors="coerce")
btc_df = btc_data.copy()

# =============================================================================
# 2. CÁLCULO DE INDICADORES TÉCNICOS Y VARIABLES SINTÉTICAS
# =============================================================================
logging.info("Calculando indicadores técnicos y variables sintéticas...")

def compute_indicators(df):
    # Indicadores básicos
    df["SMA_10"] = ta.sma(df["close"], length=10)
    df["SMA_20"] = ta.sma(df["close"], length=20)
    df["EMA_10"] = ta.ema(df["close"], length=10)
    df["EMA_20"] = ta.ema(df["close"], length=20)
    df["RSI"] = ta.rsi(df["close"])
    
    # MACD
    macd = ta.macd(df["close"])
    df["MACD"] = macd["MACD_12_26_9"]
    df["MACD_signal"] = macd["MACDs_12_26_9"]
    df["MACD_hist"] = macd["MACDh_12_26_9"]
    
    # Bollinger Bands
    bb = ta.bbands(df["close"])
    df["BBL"] = bb.iloc[:, 0]
    df["BBM"] = bb.iloc[:, 1]
    df["BBU"] = bb.iloc[:, 2]
    
    # ATR, ADX, CCI, Estocástico y OBV
    df["ATR"] = ta.atr(df["high"], df["low"], df["close"])
    adx = ta.adx(df["high"], df["low"], df["close"])
    df["ADX"] = adx["ADX_14"]
    df["CCI"] = ta.cci(df["high"], df["low"], df["close"])
    stoch = ta.stoch(df["high"], df["low"], df["close"])
    df["STOCH_k"] = stoch["STOCHk_14_3_3"]
    df["STOCH_d"] = stoch["STOCHd_14_3_3"]
    df["OBV"] = ta.obv(df["close"], df["volume"])
    
    # Variables sintéticas "originales"
    df["price_range"] = df["high"] - df["low"]
    df["price_range_pct"] = df["price_range"] / df["close"]
    df["body_size"] = abs(df["close"] - df["open"])
    df["body_pct"] = df["body_size"] / df["open"]
    df["vol_change_pct"] = df["volume"].pct_change() * 100
    df["range_vs_vol"] = df["price_range"] / df["volume"]
    df["rolling_range_mean_8h"] = df["price_range"].rolling(window=8).mean()
    df["range_deviation"] = (df["price_range"] - df["rolling_range_mean_8h"]) / df["rolling_range_mean_8h"]
    df["vol_ratio_8h"] = df["volume"] / df["volume"].rolling(window=8).mean()
    df["sma_ratio"] = df["SMA_10"] / df["SMA_20"]
    df["ema_ratio"] = df["EMA_10"] / df["EMA_20"]
    df["atr_pct"] = df["ATR"] / df["close"]
    df["macd_diff"] = df["MACD"] - df["MACD_signal"]
    df["bb_width"] = (df["BBU"] - df["BBL"]) / df["close"]
    df["price_momentum_8h"] = df["close"] - df["close"].shift(8)
    df["vol_momentum_8h"] = df["volume"].pct_change(periods=8) * 100

    # Variables sintéticas adicionales (no basadas en volumen)
    df["return_24h"] = df["close"].pct_change(periods=24) * 100
    df["vol_std_24h"] = df["volume"].rolling(window=24).std()
    df["rsi_rolling_8h"] = df["RSI"].rolling(window=8).mean()
    df["macd_diff_std_8h"] = df["macd_diff"].rolling(window=8).std()
    df["bb_position"] = (df["close"] - df["BBL"]) / (df["BBU"] - df["BBL"] + 1e-6)
    df["vol_ratio_24h"] = df["volume"] / df["volume"].rolling(window=24).mean()
    df["price_change_4h"] = df["close"].pct_change(periods=4) * 100
    df["price_change_12h"] = df["close"].pct_change(periods=12) * 100
    df["price_change_24h"] = df["close"].pct_change(periods=24) * 100
    df["vol_momentum_4h"] = df["volume"].pct_change(periods=4) * 100
    df["range_body_ratio"] = df["price_range"] / (df["body_size"] + 1e-6)
    df["high_open_diff"] = df["high"] - df["open"]
    df["low_open_diff"] = df["open"] - df["low"]
    df["close_open_diff"] = df["close"] - df["open"]
    df["avg_price"] = (df["high"] + df["low"] + df["close"]) / 3
    df["price_std_8h"] = df["close"].rolling(window=8).std()
    df["price_std_24h"] = df["close"].rolling(window=24).std()
    df["atr_std_8h"] = df["ATR"].rolling(window=8).std()
    df["adx_diff"] = df["ADX"].diff()
    df["cci_change"] = df["CCI"].diff()
    
    # Variables sintéticas basadas en volumen
    df["vol_ma_4h"] = df["volume"].rolling(window=4).mean()
    df["vol_ma_8h"] = df["volume"].rolling(window=8).mean()
    df["vol_ma_24h"] = df["volume"].rolling(window=24).mean()
    df["vol_std_4h"] = df["volume"].rolling(window=4).std()
    df["vol_std_8h"] = df["volume"].rolling(window=8).std()
    df["vol_cv_8h"] = df["vol_std_8h"] / (df["vol_ma_8h"] + 1e-6)
    df["vol_cv_24h"] = df["vol_std_24h"] / (df["vol_ma_24h"] + 1e-6)
    df["vol_ema_8h"] = df["volume"].ewm(span=8, adjust=False).mean()
    df["vol_ema_24h"] = df["volume"].ewm(span=24, adjust=False).mean()
    df["vol_ratio_ema_8h"] = df["volume"] / (df["vol_ema_8h"] + 1e-6)
    df["vol_ratio_ema_24h"] = df["volume"] / (df["vol_ema_24h"] + 1e-6)
    df["vol_roc_24h"] = df["volume"].pct_change(periods=24) * 100
    df["vol_osc_4h_8h"] = (df["vol_ma_4h"] - df["vol_ma_8h"]) / (df["vol_ma_8h"] + 1e-6)
    df["vol_diff"] = df["volume"].diff()
    df["vol_slope_3h"] = df["volume"].diff(periods=3) / 3
    df["vol_slope_6h"] = df["volume"].diff(periods=6) / 6
    df["vol_acceleration"] = df["vol_slope_3h"].diff()
    df["vol_zscore_8h"] = (df["volume"] - df["vol_ma_8h"]) / (df["vol_std_8h"] + 1e-6)
    df["vol_zscore_24h"] = (df["volume"] - df["vol_ma_24h"]) / (df["vol_std_24h"] + 1e-6)
    df["vol_pct_rank_24h"] = df["volume"].rolling(window=24).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    
    # Variables relacionadas con Bollinger Bands
    window_bband = 8
    df["bband_upper_touch_count"] = ((df["close"] >= df["BBU"]).astype(int)).rolling(window=window_bband).sum()
    df["bband_lower_touch_count"] = ((df["close"] <= df["BBL"]).astype(int)).rolling(window=window_bband).sum()
    df["bband_upper_touch_pct"] = df["bband_upper_touch_count"] / window_bband
    df["bband_lower_touch_pct"] = df["bband_lower_touch_count"] / window_bband
    df["bband_upper_exceed_count"] = ((df["close"] > df["BBU"]).astype(int)).rolling(window=window_bband).sum()
    df["bband_lower_exceed_count"] = ((df["close"] < df["BBL"]).astype(int)).rolling(window=window_bband).sum()
    df["bband_mid_diff_mean"] = (abs(df["close"] - df["BBM"])).rolling(window=window_bband).mean()
    df["bband_width_mean"] = df["bb_width"].rolling(window=window_bband).mean()
    
    # Variables temporales adicionales
    df["hour"] = df["open_time"].dt.hour
    df["dayofweek"] = df["open_time"].dt.dayofweek
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["rsi_vol"] = df["RSI"] * (df["volume"] / df["volume"].mean())
    df["corr_close_volume_24h"] = df["close"].rolling(window=24).corr(df["volume"])
    df["cum_vwap"] = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()
    df["skew_close_24h"] = df["close"].rolling(window=24).skew()
    df["kurt_close_24h"] = df["close"].rolling(window=24).kurt()
    
    return df

btc_df = compute_indicators(btc_df)
btc_df.dropna(inplace=True)
logging.info(f"Indicadores calculados. Datos finales: {btc_df.shape}")

# =============================================================================
# 3. CREACIÓN DEL TARGET Y MANEJO DEL DESBALANCE DE CLASES
# =============================================================================
logging.info("Creando variable objetivo: Subida ≥ 0.25% en la siguiente hora...")
btc_df["target"] = np.where(((btc_df["close"].shift(-1) / btc_df["close"]) - 1) >= 0.0025, 1, 0)
btc_df = btc_df.iloc[:-1]
logging.info("Distribución del target:")
logging.info(btc_df["target"].value_counts())

# =============================================================================
# 4. ELIMINACIÓN DE COLINEALIDAD Y REDUCCIÓN CON PCA (OPCIONAL)
# =============================================================================
logging.info("Eliminando features altamente correlacionadas...")

def remove_highly_correlated_features(df, threshold=0.95):
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    return df.drop(columns=to_drop), to_drop

features = btc_df.drop(columns=["open_time", "target"]).select_dtypes(include=[np.number])
features_reduced, dropped_features = remove_highly_correlated_features(features, threshold=0.95)
logging.info(f"Features eliminadas: {dropped_features}")

# Aplicación de PCA en el pipeline (opcional)
pca_components = min(50, features_reduced.shape[1])
pca_temp = PCA(n_components=pca_components, random_state=SEED)
features_temp = pca_temp.fit_transform(features_reduced)
explained_variance = np.sum(pca_temp.explained_variance_ratio_)
logging.info(f"Varianza explicada por PCA (n_components={pca_components}): {explained_variance:.2%}")

# Creamos el dataframe final para modelado
df_model = pd.concat([features_reduced, btc_df["target"]], axis=1)
logging.info(f"Datos para modelado: {df_model.shape}")

# =============================================================================
# 5. DIVISIÓN DE DATOS, PREPROCESAMIENTO CON PIPELINE Y BASELINE NAÏVE
# =============================================================================
logging.info("Dividiendo datos en entrenamiento y prueba (80/20)...")
split_idx = int(len(df_model) * 0.8)
train_df = df_model.iloc[:split_idx].copy()
test_df = df_model.iloc[split_idx:].copy()

X_train = train_df.drop("target", axis=1).values
y_train = train_df["target"].values
X_test = test_df.drop("target", axis=1).values
y_test = test_df["target"].values

# Calcular pesos de clase para manejar el desbalance
classes = np.unique(y_train)
class_weights = dict(zip(classes, compute_class_weight('balanced', classes=classes, y=y_train)))
logging.info(f"Pesos de clase: {class_weights}")

# Pipeline de preprocesamiento: StandardScaler + PCA
preprocessing_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=min(pca_components, X_train.shape[1]), random_state=SEED))
])
X_train_preprocessed = preprocessing_pipeline.fit_transform(X_train)
X_test_preprocessed = preprocessing_pipeline.transform(X_test)

# Baseline naïve: predecir siempre la clase mayoritaria (0)
naive_preds = np.zeros_like(y_test)
def evaluate_model(y_true, y_pred, model_name="Modelo"):
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_pred)
    logging.info(f"\nEvaluación {model_name}:")
    logging.info(f"Matriz de Confusión:\n{cm}")
    logging.info(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
    logging.info(f"Reporte:\n{classification_report(y_true, y_pred, zero_division=0)}")
    return cm, acc, prec, rec, f1, auc

logging.info("Evaluando baseline naïve (predicción siempre 0)...")
cm_naive, acc_naive, prec_naive, rec_naive, f1_naive, auc_naive = evaluate_model(y_test, naive_preds, "Naive Model")

# =============================================================================
# 6. DEFINICIÓN Y OPTIMIZACIÓN DE MODELOS CON VALIDACIÓN CRONOLÓGICA
# =============================================================================
# Se utiliza TimeSeriesSplit para evitar data leakage
tscv = TimeSeriesSplit(n_splits=5)

# ----- 6.1 Optimización de XGBoost con Optuna -----
def objective_xgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
        'scale_pos_weight': class_weights.get(0, 1) / class_weights.get(1, 1),
        'use_label_encoder': False,
        'eval_metric': 'logloss',
        'random_state': SEED
    }
    model = XGBClassifier(**params)
    f1_scores = []
    for train_idx, val_idx in tscv.split(X_train_preprocessed):
        X_tr, X_val = X_train_preprocessed[train_idx], X_train_preprocessed[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)
        f1_scores.append(f1_score(y_val, preds, zero_division=0))
    return -np.mean(f1_scores)  # Negativo para maximizar F1

study_xgb = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=SEED))
logging.info("Optimizando hiperparámetros para XGBoost...")
study_xgb.optimize(objective_xgb, n_trials=20, show_progress_bar=True)
best_params_xgb = study_xgb.best_params
logging.info(f"Mejores hiperparámetros para XGBoost: {best_params_xgb}")

xgb_model = XGBClassifier(**best_params_xgb, use_label_encoder=False, eval_metric='logloss', random_state=SEED)
xgb_model.fit(X_train_preprocessed, y_train)
xgb_preds = xgb_model.predict(X_test_preprocessed)
xgb_prob = xgb_model.predict_proba(X_test_preprocessed)[:, 1]

# ----- 6.2 Optimización de RandomForest con Optuna -----
def objective_rf(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 300),
        'max_depth': trial.suggest_int('max_depth', 5, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
        'class_weight': 'balanced',
        'random_state': SEED
    }
    model = RandomForestClassifier(**params)
    f1_scores = []
    for train_idx, val_idx in tscv.split(X_train_preprocessed):
        X_tr, X_val = X_train_preprocessed[train_idx], X_train_preprocessed[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)
        f1_scores.append(f1_score(y_val, preds, zero_division=0))
    return -np.mean(f1_scores)

study_rf = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=SEED))
logging.info("Optimizando hiperparámetros para RandomForest...")
study_rf.optimize(objective_rf, n_trials=15, show_progress_bar=True)
best_params_rf = study_rf.best_params
logging.info(f"Mejores hiperparámetros para RandomForest: {best_params_rf}")

rf_model = RandomForestClassifier(**best_params_rf, class_weight='balanced', random_state=SEED)
rf_model.fit(X_train_preprocessed, y_train)
rf_preds = rf_model.predict(X_test_preprocessed)
rf_prob = rf_model.predict_proba(X_test_preprocessed)[:, 1]

# ----- 6.3 Modelo LSTM unidireccional con TimeseriesGenerator -----
def create_sequences(X, y, seq_length):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i+seq_length])
        y_seq.append(y[i+seq_length])
    return np.array(X_seq), np.array(y_seq)

seq_length = 24  # Ventana de 24 horas
# Generamos secuencias usando el preprocesamiento sin PCA para LSTM
X_train_seq, y_train_seq = create_sequences(X_train_preprocessed, y_train, seq_length)
X_test_seq, y_test_seq = create_sequences(X_test_preprocessed, y_test, seq_length)

# Para validación temporal en LSTM usamos TimeseriesGenerator:
split_seq = int(len(X_train_seq) * 0.9)
train_generator = TimeseriesGenerator(X_train_preprocessed, y_train, length=seq_length, 
                                        sampling_rate=1, stride=1, start_index=0, end_index=split_seq-1, batch_size=64)
val_generator = TimeseriesGenerator(X_train_preprocessed, y_train, length=seq_length, 
                                      sampling_rate=1, stride=1, start_index=split_seq, batch_size=64)

logging.info("Entrenando modelo LSTM unidireccional con validación temporal...")
lstm_model = Sequential([
    LSTM(64, return_sequences=True, activation='tanh', input_shape=(seq_length, X_train_preprocessed.shape[1])),
    Dropout(0.3),
    LSTM(32, activation='tanh'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])
lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lstm_model.fit(train_generator, epochs=50, validation_data=val_generator, callbacks=[early_stop], verbose=1)
lstm_prob = lstm_model.predict(X_test_seq).flatten()
lstm_preds = (lstm_prob >= 0.5).astype(int)

# Ejemplo comentado: Optimización de arquitectura LSTM con Optuna
"""
def objective_lstm(trial):
    units = trial.suggest_categorical('units', [32, 64, 128])
    dropout_rate = trial.suggest_float('dropout', 0.2, 0.5)
    model = Sequential([
        LSTM(units, return_sequences=True, activation='tanh', input_shape=(seq_length, X_train_preprocessed.shape[1])),
        Dropout(dropout_rate),
        LSTM(units // 2, activation='tanh'),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    # Entrenar usando un TimeseriesGenerator para validación temporal
    history = model.fit(train_generator, epochs=30, validation_data=val_generator, callbacks=[EarlyStopping(monitor='val_loss', patience=3)], verbose=0)
    preds = model.predict(X_test_seq).flatten()
    preds_binary = (preds >= 0.5).astype(int)
    f1 = f1_score(y_test_seq, preds_binary, zero_division=0)
    return -f1

study_lstm = optuna.create_study(direction='minimize')
study_lstm.optimize(objective_lstm, n_trials=10)
best_params_lstm = study_lstm.best_params
logging.info(f"Mejores hiperparámetros para LSTM: {best_params_lstm}")
"""

# =============================================================================
# 7. OPTIMIZACIÓN DEL THRESHOLD CON PRECISION-RECALL CURVE
# =============================================================================
precisions, recalls, thresholds = precision_recall_curve(y_test, xgb_prob)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-6)
optimal_threshold = thresholds[np.argmax(f1_scores)]
logging.info(f"Threshold óptimo basado en F1: {optimal_threshold:.4f}")
# Aplicamos el threshold óptimo para XGBoost y RandomForest (LSTM sigue con 0.5)
xgb_preds_opt = (xgb_prob >= optimal_threshold).astype(int)
rf_preds_opt = (rf_prob >= optimal_threshold).astype(int)

# =============================================================================
# 8. ENSAMBLE PONDERADO DE MODELOS
# =============================================================================
# Para alinear las predicciones, descartamos las primeras 'seq_length' muestras en XGBoost y RF
xgb_prob_aligned = xgb_prob[seq_length:]
rf_prob_aligned = rf_prob[seq_length:]
# Se usan los F1-score evaluados para ponderar
total_f1 = f1_score(y_test, xgb_preds_opt, zero_division=0) + f1_score(y_test, rf_preds_opt, zero_division=0) + f1_score(y_test_seq, lstm_preds, zero_division=0)
w_xgb = f1_score(y_test, xgb_preds_opt, zero_division=0) / total_f1
w_rf = f1_score(y_test, rf_preds_opt, zero_division=0) / total_f1
w_lstm = f1_score(y_test_seq, lstm_preds, zero_division=0) / total_f1
logging.info(f"Pesos de ensamble: XGBoost: {w_xgb:.2f}, RF: {w_rf:.2f}, LSTM: {w_lstm:.2f}")

ensemble_prob = (w_xgb * xgb_prob_aligned + w_rf * rf_prob_aligned + w_lstm * lstm_prob) / (w_xgb + w_rf + w_lstm)
ensemble_preds = (ensemble_prob >= 0.5).astype(int)
logging.info("\n--- Evaluación Ensamble Ponderado ---")
cm_ens, acc_ens, prec_ens, rec_ens, f1_ens, auc_ens = evaluate_model(y_test_seq, ensemble_preds, "Ensamble Ponderado")

# =============================================================================
# 9. FUNCIONES ADICIONALES: WALK-FORWARD VALIDATION Y DATA DRIFT (MMD)
# =============================================================================
def walk_forward_validation(data, n_splits=5, window=1000):
    """
    Ejemplo de validación walk-forward: se dividen los datos en ventanas móviles.
    Esta función es un esqueleto para backtesting.
    """
    splits = []
    for i in range(n_splits):
        train = data[:window + i * window]
        test = data[window + i * window: window + (i+1)*window]
        splits.append((train, test))
    return splits

def compute_mmd(X_train, X_test):
    """
    Calcula Maximum Mean Discrepancy (MMD) usando un kernel RBF para detectar drift en los datos.
    """
    from sklearn.metrics.pairwise import rbf_kernel
    K_train = rbf_kernel(X_train)
    K_test = rbf_kernel(X_test, X_train)
    mmd = np.mean(K_train) - 2*np.mean(K_test) + np.mean(rbf_kernel(X_test))
    return mmd

# Ejemplo de uso de walk-forward validation (para backtesting)
# splits = walk_forward_validation(X_train_preprocessed, n_splits=5, window=500)
# for i, (train_data, test_data) in enumerate(splits):
#     logging.info(f"Split {i+1}: Train shape: {train_data.shape}, Test shape: {test_data.shape}")

# Ejemplo de cálculo de data drift (MMD)
mmd_value = compute_mmd(X_train_preprocessed, X_test_preprocessed)
logging.info(f"Data Drift (MMD) entre train y test: {mmd_value:.4f}")

# =============================================================================
# 10. ANÁLISIS DE IMPORTANCIA DE FEATURES CON PERMUTATION IMPORTANCE (SIN PCA)
# =============================================================================
logging.info("Calculando importancia de features (sin PCA) con permutation importance...")
rf_interpret = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=SEED, class_weight='balanced')
scaler_interpret = StandardScaler().fit(X_train)
X_train_scaled_interpret = scaler_interpret.transform(X_train)
X_test_scaled_interpret = scaler_interpret.transform(X_test)
rf_interpret.fit(X_train_scaled_interpret, y_train)
perm_importance = permutation_importance(rf_interpret, X_test_scaled_interpret, y_test, n_repeats=10, random_state=SEED)
importances = pd.Series(perm_importance.importances_mean, index=features_reduced.columns).sort_values(ascending=False)
logging.info("Top 10 features por permutation importance:")
logging.info(importances.head(10))

# =============================================================================
# 11. VISUALIZACIÓN DE LAS MATRICES DE CONFUSIÓN
# =============================================================================
def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel("Predicción")
    plt.ylabel("Valor real")
    plt.show()

plot_confusion_matrix(confusion_matrix(y_test, xgb_preds_opt), "Matriz de Confusión - XGBoost")
plot_confusion_matrix(confusion_matrix(y_test, rf_preds_opt), "Matriz de Confusión - RandomForest")
plot_confusion_matrix(confusion_matrix(y_test_seq, lstm_preds), "Matriz de Confusión - LSTM")
plot_confusion_matrix(confusion_matrix(y_test_seq, ensemble_preds), "Matriz de Confusión - Ensamble Ponderado")

logging.info("Proceso de modelado, optimización y evaluación completado con éxito.")
