# --- LOG DE ERRORES ---
def log_error(message):
    log_file = "error_log.txt"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {message}\n")
    except Exception as e:
        print(f"No se pudo escribir en el log de errores: {e}")

# --- LIBRERÍAS ---
import time
import os
import json
import hmac
import base64
import hashlib
import requests
import warnings
from urllib.parse import urlencode
from datetime import datetime, timedelta
import pandas as pd
import pandas_ta as ta
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import optuna
import csv
import joblib
from backtesting import Backtest, Strategy
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
# ========================================================
# --- CONFIGURACIÓN Y ESTADO GLOBAL ---
# ========================================================
# IMPORTANTE: Si usas cuenta DEMO, pon USE_DEMO=True y usa claves API de SIMULACIÓN
# Si usas cuenta REAL, pon USE_DEMO=False y usa claves API de PRODUCCIÓN
USE_DEMO = True  # <<< CAMBIAR A True SOLO SI TIENES CLAVES DE CUENTA SIMULADA
API_KEY = "bg_3f4f817facf5756582de753cebab9b7b"
SECRET_KEY = "f07325cf8cbafc0d5d81ace4211839da3bb6f99400e411a9b14aee3408698b5e"
PASSPHRASE = "WYXShy7q26dYaUgng6AY4"
USE_SIGNED_PASSPHRASE = True  # True = passphrase firmada (requerido para cuentas demo)
PRODUCT_TYPE = "USDT-FUTURES"
TOTAL_CAPITAL_USDT = 10.0
RISK_PER_TRADE_PERC = 5.0   # 1% de 10 USDT ≈ 0.10 USDT de pérdida objetivo al tocar SL
CYCLE_INTERVAL_MINUTES = 5

# --- MODO EXPERIMENTO: Invertir señales y cerrar a +10% ---
# Cuando el modelo vea LONG, se hace SHORT y viceversa.
# El take-profit se fija al 10% desde el precio de entrada.
EXPERIMENT_MODE = True
EXPERIMENT_10P_TP_PERC = 0.5  # 1% de ganancia = 0.1 USD en 10 USD
# MEJORA: Parámetros de la estrategia - Versión "Táctica" (Equilibrio entre Calidad y Frecuencia)
STRATEGY_PARAMS = {
    # --- PARÁMETROS CENTRALES (más permisivos) ---
    "CONFIDENCE_THRESHOLD": 0.20,       # Acepta señales con menor confianza
    "SL_ATR_MULTIPLIER": 12.0,          # Stop loss más grande (menos liquidaciones)
    "MIN_SL_PERC": 0.25,                # Piso adicional: SL al menos 0.25% del precio
    # --- FILTROS DE MERCADO (más laxos) ---
    "ADX_TREND_STRENGTH": 5,            # Permite operar en mercados con poca tendencia
    "MIN_BB_WIDTH_PERC": 0.5,           # Acepta menor volatilidad
    # --- PARÁMETROS BASE ---
    "MIN_ATR_PERC": 0.05,
    "MAX_ATR_PERC": 25.0,
    "MAX_HOLD_TIME_MINUTES": 1440,
    "PREDICTION_HORIZON": 10,
    "MOVEMENT_THRESHOLD": 0.0005,
    "TP_RR_RATIO": 1.1,                 # TP más cercano para no descartar setups
    # --- FILTRO DE AGOTAMIENTO ---
    "RSI_EXHAUST_LONG": 75.0,           # Más permisivo: solo filtra sobrecompra extrema
    "RSI_EXHAUST_SHORT": 25.0           # Más permisivo: solo filtra sobreventa extrema
}
STATE_FILE = "bot_state.json"
MODEL_FILE = "lgbm_trading_model.joblib"
FEATURES_LIST_FILE = "model_features.json"
DEFAULT_POSITION_STATE = {"in_position": False, "symbol": None, "type": None, "entry_price": 0.0, "stop_loss": 0.0, "take_profit": 0.0, "position_size_asset": 0.0, "position_size_usd": 0.0, "initial_sl_distance": 0.0, "pnl": 0.0, "entry_timestamp": None}
position_state = {}
# ========================================================
# --- PERSISTENCIA Y API ---
# ========================================================
def get_current_market_data(symbol):
    status, res = signed_request("GET", ENDPOINTS["ticker"], params={"symbol": symbol, "productType": PRODUCT_TYPE})
    if status == 200 and res.get('data'):
        ticker_data = res['data'][0]
        price = float(ticker_data.get('lastPr', 0))
        if price > 0: return {"price": price, "funding_rate": float(ticker_data.get('fundingRate', 0))}
    return None

def log_trade_to_csv(trade_details):
    file_path, fieldnames = "trade_history.csv", ["entry_timestamp", "exit_timestamp", "symbol", "type", "entry_price", "exit_price", "position_size_asset", "position_size_usd", "close_reason", "gross_pnl_usd", "total_commission_usd", "net_pnl_usd"]
    file_exists = os.path.exists(file_path)
    try:
        with open(file_path, mode='a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists: writer.writeheader()
            writer.writerow(trade_details)
        print(f"*** Operación registrada con éxito en {file_path} ***")
    except Exception as e: print(f"!!! ERROR al escribir en el archivo CSV: {e} !!!")

def log_attempt_to_csv(attempt_details):
    file_path, fieldnames = "trade_attempts.csv", [
        "timestamp", "symbol", "model_signal", "confidence", "current_price", "decision", "reason"
    ]
    file_exists = os.path.exists(file_path)
    try:
        with open(file_path, mode='a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(attempt_details)
    except Exception as e:
        print(f"!!! ERROR al escribir intento en CSV: {e} !!!")

def fetch_history_candles(symbol, granularity, start_time_ms, end_time_ms):
    """Descarga velas entre un rango [start_time_ms, end_time_ms] inclusive.
    Pagina hacia atrás usando endTime hasta cubrir el rango solicitado.
    """
    print(f"  -> Iniciando descarga para {symbol} ({granularity}) entre {pd.to_datetime(start_time_ms, unit='ms')} y {pd.to_datetime(end_time_ms, unit='ms')}...")
    all_candles = []
    limit_per_call = 100  # Máximo por llamada de la API
    current_end_time = end_time_ms
    call_count = 0

    while current_end_time > start_time_ms:
        call_count += 1
        params = {
            "symbol": symbol,
            "productType": PRODUCT_TYPE,
            "granularity": granularity,
            "limit": str(limit_per_call),
            "endTime": str(current_end_time),
            "startTime": str(start_time_ms)
        }
        try:
            status, res = signed_request("GET", ENDPOINTS["candles"], params=params)
            print(f"     Llamada {call_count}... ", end="")
            if status != 200 or not isinstance(res, dict) or res.get('code') != '00000':
                print(f"\n     !!! Fallo en la petición a la API: {res} !!!")
                log_error(f"API error en fetch_history_candles: status={status}, res={res}")
                break
            candles = res.get("data", [])
            if not candles:
                print("OK. No se recibieron más velas. Descarga completada.")
                break
            print(f"OK. {len(candles)} velas recibidas. Timestamp más antiguo: {pd.to_datetime(candles[-1][0], unit='ms')}")
            all_candles.extend(candles)
            # Ajustamos el siguiente endTime para la próxima iteración
            oldest_ts = int(candles[-1][0])
            current_end_time = oldest_ts - 1
            if oldest_ts <= start_time_ms:
                break
            time.sleep(1.0)  # 1 segundo entre peticiones reduce carga
        except Exception as e:
            log_error(f"Excepción en fetch_history_candles: {e}")
            break

    if not all_candles:
        print(f"!!! ALERTA: No se pudo descargar ninguna vela para {symbol} ({granularity}) en el rango especificado. !!!")
        return None

    df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume", "quoteVolume"])
    df.drop_duplicates(subset=["timestamp"], inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.astype({col: float for col in ["open", "high", "low", "close", "volume", "quoteVolume"]})
    df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}, inplace=True)

    # Filtrar exactamente al rango solicitado por seguridad
    start_dt = pd.to_datetime(start_time_ms, unit='ms')
    end_dt = pd.to_datetime(end_time_ms, unit='ms')
    df = df[(df['timestamp'] >= start_dt) & (df['timestamp'] <= end_dt)]
    return df.sort_values('timestamp', ascending=True).set_index('timestamp')

def fetch_public_candles(symbol, granularity, start_time_ms, end_time_ms):
    """Descarga velas usando endpoint público (sin firma)."""
    params = {
        "symbol": symbol,
        "productType": PRODUCT_TYPE,
        "granularity": granularity,
        "limit": "100",
        "endTime": str(end_time_ms),
        "startTime": str(start_time_ms)
    }
    url = BASE_URL + "/api/v2/mix/market/candles?" + urlencode(params)
    
    try:
        resp = requests.get(url, timeout=15)
        return resp.status_code, resp.json()
    except Exception as e:
        return 500, {"error": str(e)}
# ========================================================
def save_state_to_file(state):
    try:
        with open(STATE_FILE, 'w', encoding='utf-8') as f: json.dump(state, f)
    except Exception as e: print(f"!!! ERROR guardando estado: {e} !!!")

def load_state_from_file():
    if not os.path.exists(STATE_FILE): return DEFAULT_POSITION_STATE.copy()
    try:
        with open(STATE_FILE, 'r', encoding='utf-8') as f: loaded = json.load(f)
        state = DEFAULT_POSITION_STATE.copy()
        state.update(loaded if isinstance(loaded, dict) else {})
        return state
    except Exception as e:
        print(f"!!! ERROR cargando estado, usando por defecto: {e} !!!")
        return DEFAULT_POSITION_STATE.copy()

# URL base (Bitget V2 usa la misma URL para demo y producción)
BASE_URL = "https://api.bitget.com"

ENDPOINTS = {
    "candles": "/api/v2/mix/market/candles",
    "place_order": "/api/v2/mix/order/place-order",
    "close_all_positions": "/api/v2/mix/order/close-positions",
    "ticker": "/api/v2/mix/market/ticker",
    "open_interest": "/api/v2/mix/market/open-interest"
}

def _timestamp_ms(): return str(int(time.time() * 1000))

def _sign(secret, prehash):
    h = hmac.new(secret.encode('utf-8'), prehash.encode('utf-8'), hashlib.sha256).digest()
    return base64.b64encode(h).decode()

def signed_request(method, path, params=None, body=None, timeout=15, max_retries=3):
    """Cliente autenticado para Bitget V2 con reintentos automáticos."""
    
    for attempt in range(max_retries):
        timestamp = _timestamp_ms()
        body_str = json.dumps(body) if body else ""
        
        # Construir path con parámetros para GET
        path_with_params = path
        if method.upper() == "GET" and params:
            query_string = '&'.join([f"{key}={value}" for key, value in sorted(params.items())])
            path_with_params += '?' + query_string
        
        # Construir prehash EXACTAMENTE como espera Bitget
        prehash = timestamp + method.upper() + path_with_params + body_str
        signature = _sign(SECRET_KEY, prehash)
        
        headers = {
            "ACCESS-KEY": API_KEY,
            "ACCESS-SIGN": signature,
            "ACCESS-TIMESTAMP": timestamp,
            "ACCESS-PASSPHRASE": PASSPHRASE,  # NO firmar la passphrase
            "Content-Type": "application/json",
            "locale": "en-US"
        }
        
        # CRÍTICO: Header para modo demo
        if USE_DEMO:
            headers["paptrading"] = "1"
        
        url = f"{BASE_URL}{path_with_params if method.upper() == 'GET' else path}"
        
        try:
            if method.upper() == "GET":
                r = requests.get(url, headers=headers, timeout=timeout)
            else:
                r = requests.post(url, headers=headers, data=body_str.encode('utf-8'), timeout=timeout)
            
            # Manejo de errores 502/503 con reintentos
            if r.status_code in [502, 503, 504]:
                wait_time = 2 ** attempt
                print(f"⚠️ Error {r.status_code}. Reintento {attempt+1}/{max_retries} en {wait_time}s...")
                time.sleep(wait_time)
                continue
            
            if r.status_code == 200:
                return r.status_code, r.json()
            else:
                return r.status_code, {"error": "HTTP Error", "message": r.text}
                
        except requests.exceptions.Timeout:
            print(f"⏱️ Timeout en intento {attempt+1}/{max_retries}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
        except Exception as e:
            print(f"❌ Error: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
    
    return 502, {"error": "Max retries exceeded"}
# ========================================================
# --- CREACIÓN DE FEATURES Y MODELO ---
# ========================================================
def create_features_and_target(df_1m, df_5m, df_btc_15m, create_target=True):
    print("  Creando indicadores y features...")
    # Evitar modificar los originales
    df_1m = df_1m.copy()
    df_5m = df_5m.copy()
    
    # --- Indicadores en 1 minuto ---
    df_1m.ta.rsi(append=True)
    df_1m.ta.macd(append=True)
    bbands = df_1m.ta.bbands(length=20, std=2)
    if isinstance(bbands, pd.DataFrame):
        df_1m = pd.concat([df_1m, bbands], axis=1)
        upper_col = next((c for c in df_1m.columns if c.startswith('BBU_')), None)
        lower_col = next((c for c in df_1m.columns if c.startswith('BBL_')), None)
        if upper_col and lower_col:
            df_1m['bb_width'] = (df_1m[upper_col] - df_1m[lower_col]) / df_1m['Close']
        else:
            df_1m['bb_width'] = 0.0
    else:
        df_1m['bb_width'] = 0.0

    df_1m['return_5m'] = df_1m['Close'].pct_change(5)
    
    # --- Indicadores en 5 minutos ---
    # EMA 21 y EMA 200 (MACRO)
    df_5m['EMA_21_5m'] = ta.ema(df_5m['Close'], length=21)
    df_5m['EMA_MACRO'] = ta.ema(df_5m['Close'], length=200) # Necesita muchas velas
    
    atr_result = ta.atr(high=df_5m['High'], low=df_5m['Low'], close=df_5m['Close'], length=14)
    df_5m['ATR_14_5m'] = atr_result
    df_5m['ATRr_14_5m'] = (df_5m['ATR_14_5m'] / df_5m['Close']) * 100
    df_5m.ta.adx(length=14, append=True)
    df_5m['RSI_14_5m'] = ta.rsi(df_5m['Close'])

    # --- Fusión de datos ---
    # Seleccionamos columnas clave de 5m
    cols_5m = ['EMA_21_5m', 'EMA_MACRO', 'ATR_14_5m', 'ATRr_14_5m', 'ADX_14', 'RSI_14_5m']
    # Filtramos solo las que existen
    cols_5m = [c for c in cols_5m if c in df_5m.columns]
    
    features_5m = df_5m[cols_5m].sort_index()
    
    # Merge asof
    df_combined = pd.merge_asof(
        df_1m.sort_index(), 
        features_5m, 
        left_index=True, 
        right_index=True, 
        direction='backward',
        tolerance=pd.Timedelta("10min")
    )

    df_features = df_combined
    
    # --- Limpieza de datos (CRÍTICO) ---
    print(f"  [DIAGNÓSTICO] Filas totales: {len(df_features)}")
    
    # Columnas que NO pueden ser NaN
    cols_criticas = ['Close', 'EMA_MACRO', 'ATR_14_5m', 'ADX_14', 'bb_width']
    
    # Verificar si falta data crítica antes de borrar
    if df_features['EMA_MACRO'].isna().all():
        print("!!! ERROR CRÍTICO: 'EMA_MACRO' son todos NaN. Faltan datos históricos (mínimo 200 velas de 5m).")
        return pd.DataFrame() # Retorno vacío controlado

    df_features.dropna(subset=cols_criticas, inplace=True)
    print(f"  [DIAGNÓSTICO] Filas útiles después de limpieza: {len(df_features)}")

    if df_features.empty:
        return pd.DataFrame()

    if create_target:
        p = STRATEGY_PARAMS
        future_returns = df_features['Close'].pct_change(periods=p["PREDICTION_HORIZON"]).shift(-p["PREDICTION_HORIZON"])
        df_features['target'] = 0
        df_features.loc[future_returns > p["MOVEMENT_THRESHOLD"], 'target'] = 1
        df_features.loc[future_returns < -p["MOVEMENT_THRESHOLD"], 'target'] = -1
        df_features.dropna(inplace=True)

    return df_features

def objective(trial, X_train, y_train, X_test, y_test):
    """Optimiza hiperparámetros usando binary_logloss y retorna accuracy."""
    param = {
        'objective': 'binary',
        'metric': 'binary_logloss',  # Métrica estándar para clasificación binaria
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'class_weight': 'balanced',
        'n_estimators': trial.suggest_int('n_estimators', 100, 400),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 20, 80),
        'max_depth': trial.suggest_int('max_depth', -1, 10),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
    }
    model = lgb.LGBMClassifier(**param)
    # La llamada a fit ahora funcionará porque la métrica está definida en los parámetros
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=[lgb.early_stopping(15, verbose=False)])

    # Devolvemos el accuracy para que Optuna lo optimice
    return model.score(X_test, y_test)

def train_and_save_model(df_train_data, model_path=MODEL_FILE, features_path=FEATURES_LIST_FILE):
    print("--- Entrenando un nuevo modelo de clasificación ---")
    ### MEJORA CLAVE: Simplificar el problema a una clasificación binaria (LONG vs SHORT) ###
    df_train_data = df_train_data[df_train_data['target'] != 0].copy()
    print(f"  Filtrando datos de 'HOLD'. Quedan {len(df_train_data)} muestras para el entrenamiento binario.")

    if len(df_train_data) < 500:
        print(f"!!! Datos insuficientes para entrenamiento. Se requieren al menos 500 muestras, pero solo se encontraron {len(df_train_data)}. !!!")
        return None, None

    features = [col for col in df_train_data.columns if col not in ['Open', 'High', 'Low', 'Close', 'target', 'quoteVolume', 'Volume', 'timestamp']]
    X, y = df_train_data[features], df_train_data['target']

    # El mapeo de y a 0 y 1 es manejado internamente por LGBM si las clases son -1 y 1.

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    print(f"Set de entrenamiento: {len(X_train)} muestras. Set de prueba: {len(X_test)} muestras.")

    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_test, y_test), n_trials=50)
    best_params = study.best_params
    best_params['class_weight'] = 'balanced'
    best_params['verbosity'] = -1
    best_model = lgb.LGBMClassifier(**best_params).fit(X_train, y_train)

    print("\n--- Reporte de Clasificación en Datos de Prueba ---")
    # Ahora los target_names son solo SHORT y LONG
    print(classification_report(y_test, best_model.predict(X_test), target_names=['SHORT (-1)', 'LONG (1)'], zero_division=0))
    joblib.dump(best_model, model_path)
    with open(features_path, 'w') as f: json.dump(features, f)
    print(f"*** Modelo guardado exitosamente en {model_path} ***")
    return best_model, features

def load_model_and_predict(df_latest_features):
    # PROTECCIÓN CONTRA DATAFRAME VACÍO
    if df_latest_features is None or df_latest_features.empty:
        print("!!! Advertencia: DataFrame vacío al intentar predecir. Retornando HOLD.")
        return {"model_signal": "HOLD", "confidence": 0.0}

    if not os.path.exists(MODEL_FILE): 
        return {"model_signal": "HOLD", "confidence": 0.0}
    
    try:
        model = joblib.load(MODEL_FILE)
        with open(FEATURES_LIST_FILE, 'r') as f: model_features = json.load(f)
        
        # Rellenar columnas faltantes con 0 para evitar error de shape
        last_row = df_latest_features.iloc[[-1]].copy()
        for feat in model_features:
            if feat not in last_row.columns:
                last_row[feat] = 0
        
        # Filtrar solo las columnas que el modelo espera
        X_pred = last_row[model_features]
        
        predicted_signal = model.predict(X_pred)[0]
        probabilities = model.predict_proba(X_pred)[0]
        
        # Mapeo estándar: 1=LONG, -1=SHORT
        signal_map = {1: "LONG", -1: "SHORT", 0: "HOLD"}
        # Obtener el índice de la clase para la probabilidad correcta
        class_idx = list(model.classes_).index(predicted_signal)
        
        return {
            "model_signal": signal_map.get(predicted_signal, "HOLD"), 
            "confidence": probabilities[class_idx]
        }
    except Exception as e:
        print(f"Error en predicción: {e}")
        return {"model_signal": "HOLD", "confidence": 0.0}

# ========================================================
# --- LÓGICA DE TRADING Y ANÁLISIS ---
# ========================================================
def get_market_analysis(symbol, df_features):
    print("Iniciando análisis de mercado sobre datos actualizados...")
    market_data = get_current_market_data(symbol)
    if market_data is None:
        print("!!! No se pudo obtener el precio actual. !!!"); return None
    current_price = market_data["price"]
    if df_features.empty:
        print("!!! El DataFrame proporcionado está vacío. !!!"); return None
    model_data = load_model_and_predict(df_features)
    analysis = {**model_data, "current_price": current_price, "decision": "HOLD", "reason": ""}
    last_atr_5m = df_features['ATR_14_5m'].iloc[-1]
    atr_perc = df_features['ATRr_14_5m'].iloc[-1]
    is_uptrend_5m = current_price > df_features['EMA_21_5m'].iloc[-1]
    try:
        ema_macro_val = df_features['EMA_MACRO'].iloc[-1]
        is_macro_bullish = current_price > ema_macro_val
    except (KeyError, IndexError):
        print("[WARN] No se encontró EMA_MACRO en df_features. Usando fallback.")
        is_macro_bullish = current_price > df_features['Close'].iloc[-1]
    last_adx = df_features['ADX_14'].iloc[-1]
    last_bb_width = df_features['bb_width'].iloc[-1] * 100
    try:
        last_rsi_5m = df_features['RSI_14_5m'].iloc[-1]
    except (KeyError, IndexError):
        last_rsi_5m = None
    print("\n--- INICIANDO DIAGNÓSTICO DE SEÑAL ---")
    print(f"1. Señal Bruta del Modelo: {analysis['model_signal']} (Confianza: {analysis['confidence']:.2%})")
    # --- Guardar intento de trading en CSV ---
    log_attempt_to_csv({
        "timestamp": datetime.utcnow().isoformat(),
        "symbol": symbol,
        "model_signal": analysis["model_signal"],
        "confidence": analysis["confidence"],
        "current_price": current_price,
        "decision": analysis["decision"],
        "reason": analysis["reason"]
    })
    if analysis["model_signal"] != "HOLD":
        p = STRATEGY_PARAMS
        passed_conf = analysis["confidence"] >= p["CONFIDENCE_THRESHOLD"]
        passed_vol = p["MIN_ATR_PERC"] < atr_perc < p["MAX_ATR_PERC"]
        trend_5m_ok = (analysis["model_signal"] == "LONG" and is_uptrend_5m) or (analysis["model_signal"] == "SHORT" and not is_uptrend_5m)
        macro_ok = (analysis["model_signal"] == "LONG" and is_macro_bullish) or (analysis["model_signal"] == "SHORT" and not is_macro_bullish)
        passed_trend_strength = last_adx > p["ADX_TREND_STRENGTH"]
        passed_bb_width = last_bb_width > p["MIN_BB_WIDTH_PERC"]
        exhausted_move = False
        if last_rsi_5m is not None:
            exhausted_move = (analysis["model_signal"] == "LONG" and last_rsi_5m >= p["RSI_EXHAUST_LONG"]) or \
                             (analysis["model_signal"] == "SHORT" and last_rsi_5m <= p["RSI_EXHAUST_SHORT"])
        passed_exhaustion = not exhausted_move
        print(f"  - Confianza (> {p['CONFIDENCE_THRESHOLD']:.0%}): {'✓ PASA' if passed_conf else '✗ FALLA'} ({analysis['confidence']:.2%})")
        print(f"  - Volatilidad ATR% ({p['MIN_ATR_PERC']}% < ATR < {p['MAX_ATR_PERC']}%): {'✓ PASA' if passed_vol else '✗ FALLA'} (ATR: {atr_perc:.2f}%)")
        print(f"  - Tendencia 5m OK?: {'✓ PASA' if trend_5m_ok else '✗ FALLA'}")
        print(f"  - Tendencia Macro 1H OK?: {'✓ PASA' if macro_ok else '✗ FALLA'}")
        print(f"  - Fuerza de Tendencia (ADX > {p['ADX_TREND_STRENGTH']}): {'✓ PASA' if passed_trend_strength else '✗ FALLA'} (ADX: {last_adx:.2f})")
        print(f"  - Filtro de Rango (BBW > {p['MIN_BB_WIDTH_PERC']}%): {'✓ PASA' if passed_bb_width else '✗ FALLA'} (BBW: {last_bb_width:.2f}%)")
        print(f"  - RSI agotamiento (LONG<{p['RSI_EXHAUST_LONG']}, SHORT>{p['RSI_EXHAUST_SHORT']}): {'✓ PASA' if passed_exhaustion else '✗ FALLA'} ({last_rsi_5m:.2f} RSI)" if last_rsi_5m is not None else "N/D")
        if passed_conf and passed_vol and macro_ok and trend_5m_ok and passed_trend_strength and passed_bb_width and passed_exhaustion:
            # INVERTIR SEÑAL: Si es LONG, hacer SHORT y viceversa
            original_signal = analysis["model_signal"]
            analysis["decision"] = "SHORT" if original_signal == "LONG" else "LONG"
            sl_atr = last_atr_5m * p["SL_ATR_MULTIPLIER"]
            sl_floor = current_price * (p.get("MIN_SL_PERC", 0) / 100.0)
            sl_dist = max(sl_atr, sl_floor)
            analysis["stop_loss"] = current_price - sl_dist if analysis["decision"] == "LONG" else current_price + sl_dist
            # MODO EXPERIMENTO: TP fijo al +10%
            if EXPERIMENT_MODE:
                tp_perc = EXPERIMENT_10P_TP_PERC / 100.0
                analysis["take_profit"] = current_price * (1 + tp_perc) if analysis["decision"] == "LONG" else current_price * (1 - tp_perc)
            else:
                analysis["take_profit"] = current_price + (sl_dist * p["TP_RR_RATIO"]) if analysis["decision"] == "LONG" else current_price - (sl_dist * p["TP_RR_RATIO"])
            analysis["sl_distance"] = sl_dist
            print(f"-> DECISIÓN FINAL: {analysis['decision']} ✓✓✓ (Invertido de {original_signal}). Todos los filtros pasaron.")
        else:
            print("-> DECISIÓN FINAL: HOLD. Señal rechazada por filtros.")
    else:
        print("-> Decisión: HOLD (Modelo no generó señal de entrada)")
    print("--- FIN DEL DIAGNÓSTICO ---\n")
    return analysis
# ========================================================
# --- GESTIÓN DE ÓRDENES Y POSICIÓN ---
# ========================================================
def place_order(symbol, size, side):
    final_side = "buy" if side == "LONG" else "sell"
    client_oid = f"bot_{int(time.time() * 1000)}"
    body = {
        "symbol": symbol,
        "productType": PRODUCT_TYPE,
        "marginCoin": "USDT",
        "marginMode": "isolated",
        "size": str(round(size, 3)),
        "side": final_side,
        "tradeSide": "open",
        "orderType": "market",
        "clientOid": client_oid
    }
    # NOTA: NO agregar "simulated" - Bitget detecta demo por las claves API
    return signed_request("POST", ENDPOINTS["place_order"], body=body)

def execute_flash_close(symbol, exit_price, reason):
    global position_state
    print(f"\n--- INTENTANDO CIERRE RÁPIDO para {symbol} por {reason} ---")
    TAKER_FEE_PERC = 0.06
    hold_side = "long" if position_state.get("type") == "LONG" else "short"
    body = {"symbol": symbol, "productType": PRODUCT_TYPE, "marginCoin": "USDT", "holdSide": hold_side}
    status, response = signed_request("POST", ENDPOINTS["close_all_positions"], body=body)

    if status == 200 and response.get('code') in ['00000', '0']:
        print("¡¡¡ POSICIÓN CERRADA CON ÉXITO !!!"); is_closed_successfully = True
    elif response.get('code') in ['22002', '40541']:
        print("CIERRE CONFIRMADO: La posición ya no existía. Corrigiendo estado."); is_closed_successfully = True; reason += " (State Desync)"
    else: is_closed_successfully = False

    if is_closed_successfully:
        closed_trade_state = position_state.copy()
        if closed_trade_state.get("entry_price") == 0.0:
             position_state = DEFAULT_POSITION_STATE.copy(); save_state_to_file(position_state); return True

        pnl_calc = (exit_price - closed_trade_state["entry_price"]) if closed_trade_state["type"] == "LONG" else (closed_trade_state["entry_price"] - exit_price)
        gross_pnl = pnl_calc * closed_trade_state["position_size_asset"]
        commission = (closed_trade_state["position_size_usd"] + (exit_price * closed_trade_state["position_size_asset"])) * (TAKER_FEE_PERC / 100.0)
        net_pnl = gross_pnl - commission

        log_trade_to_csv({
            "entry_timestamp": closed_trade_state.get("entry_timestamp", "N/A"), "exit_timestamp": datetime.utcnow().isoformat(),
            "symbol": closed_trade_state["symbol"], "type": closed_trade_state["type"], "entry_price": f"{closed_trade_state['entry_price']:.5f}",
            "exit_price": f"{exit_price:.5f}", "position_size_asset": f"{closed_trade_state['position_size_asset']:.4f}",
            "position_size_usd": f"{closed_trade_state['position_size_usd']:.2f}", "close_reason": reason,
            "gross_pnl_usd": f"{gross_pnl:.4f}", "total_commission_usd": f"{commission:.4f}", "net_pnl_usd": f"{net_pnl:.4f}"
        })

        position_state = DEFAULT_POSITION_STATE.copy(); save_state_to_file(position_state); return True

    print(f"--- FALLO EL CIERRE RÁPIDO: {response.get('msg', str(response))} ---"); return False

def calculate_position_size(capital, risk_perc, entry_price, sl_price):
    risk_amount_usd = capital * (risk_perc / 100.0)
    sl_distance_usd = abs(entry_price - sl_price)
    if sl_distance_usd == 0: return 0, 0
    size_asset = risk_amount_usd / sl_distance_usd
    size_usd = size_asset * entry_price
    # Límite de tamaño: no exceder el capital total para evitar posiciones enormes por SL muy corto
    if size_usd > capital:
        size_usd = capital
        size_asset = size_usd / entry_price
    return size_asset, size_usd

def open_new_position(trade_plan, symbol):
    global position_state
    size_asset, size_usd = calculate_position_size(TOTAL_CAPITAL_USDT, RISK_PER_TRADE_PERC, trade_plan['current_price'], trade_plan['stop_loss'])
    if size_asset <= 0:
        print("!!! Error: Tamaño de posición calculado es cero. !!!"); return
    status, response = place_order(symbol, size_asset, trade_plan['decision'])
    print(f"Respuesta de la API al abrir posición: {status}, {response}")
    if status == 200 and response.get('code') == '00000':
        print(f"\n¡¡¡ NUEVA POSICIÓN ABIERTA: {trade_plan['decision']} | Tamaño: {size_usd:.2f} USD !!!\n")
        
        # OBTENER PRECIO REAL DE EJECUCIÓN
        order_data = response.get('data', {})
        actual_entry_price = float(order_data.get('fillPrice', trade_plan['current_price']))
        
        # Recalcular SL/TP con precio real
        sl_dist = trade_plan['sl_distance']
        if trade_plan['decision'] == "LONG":
            actual_sl = actual_entry_price - sl_dist
            if EXPERIMENT_MODE:
                actual_tp = actual_entry_price * (1 + (EXPERIMENT_10P_TP_PERC / 100.0))
            else:
                actual_tp = actual_entry_price + (sl_dist * STRATEGY_PARAMS['TP_RR_RATIO'])
        else:
            actual_sl = actual_entry_price + sl_dist
            if EXPERIMENT_MODE:
                actual_tp = actual_entry_price * (1 - (EXPERIMENT_10P_TP_PERC / 100.0))
            else:
                actual_tp = actual_entry_price - (sl_dist * STRATEGY_PARAMS['TP_RR_RATIO'])
        
        position_state = { "in_position": True, "symbol": symbol, "type": trade_plan['decision'], "entry_price": actual_entry_price, "stop_loss": actual_sl, "take_profit": actual_tp, "position_size_asset": size_asset, "position_size_usd": size_usd, "initial_sl_distance": sl_dist, "pnl": 0.0, "entry_timestamp": datetime.utcnow().isoformat() }
        save_state_to_file(position_state)
    else: print(f"--- ERROR AL ABRIR LA POSICIÓN: {response.get('msg', 'Error desconocido')} ---")

def manage_position_with_candle_price(current_price):
    global position_state
    if not position_state.get("in_position"): return
    pnl = (current_price - position_state["entry_price"]) * position_state["position_size_asset"] if position_state["type"] == "LONG" else (position_state["entry_price"] - current_price) * position_state["position_size_asset"]
    entry_time = datetime.fromisoformat(position_state["entry_timestamp"])
    duration_min = (datetime.utcnow() - entry_time).total_seconds() / 60
    print(f"--- POSICIÓN ACTIVA [{position_state['type']}] | Duración: {duration_min:.1f} min --- PnL: {'\033[92m' if pnl >= 0 else '\033[91m'}{pnl:+.4f} USD{'\033[0m'}")
    closed, reason = False, ""
    # Cierre por ganancia del 10% del tamaño nocional
    target_pnl = 0.1  # Ganancia fija de 0.1 USD
    if pnl >= target_pnl:
        closed, reason = True, "ROE-10%-Target"
    # Cierre por TP/SL de precio
    elif (position_state["type"] == "LONG" and (current_price <= position_state["stop_loss"] or (position_state["take_profit"] is not None and current_price >= position_state["take_profit"]))) or \
       (position_state["type"] == "SHORT" and (current_price >= position_state["stop_loss"] or (position_state["take_profit"] is not None and current_price <= position_state["take_profit"]))):
        closed, reason = True, "Take-Profit" if pnl > 0 else "Stop-Loss"
    # Cierre por tiempo máximo
    elif duration_min >= STRATEGY_PARAMS["MAX_HOLD_TIME_MINUTES"]:
        closed, reason = True, "Time-Stop"

    if closed: execute_flash_close(position_state["symbol"], current_price, reason); return
    
    if pnl > 0 and abs(current_price - position_state["entry_price"]) >= position_state["initial_sl_distance"]:
        new_sl = position_state["entry_price"]
        if (position_state["type"] == "LONG" and new_sl > position_state["stop_loss"]) or \
           (position_state["type"] == "SHORT" and new_sl < position_state["stop_loss"]):
            position_state["stop_loss"] = new_sl
            print(f"*** TRAILING STOP (BE): SL movido a {new_sl:.5f} ***"); save_state_to_file(position_state)
# ========================================================
# --- LÓGICA DE BACKTESTING ---
# ========================================================
class MLStrategy(Strategy):
    model, features = None, None
    trade_on_close = False
    def init(self):
        print("\n--- Pre-calculando datos para un backtest preciso... ---")
        df = self.data.df.copy()
        X = df[self.features]
        df['signal'] = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        classes = list(self.model.classes_)
        class_to_idx = {c: i for i, c in enumerate(classes)}
        def map_signal(sig): return class_to_idx.get(sig, 0)
        sig_idx = df['signal'].apply(map_signal).to_numpy()
        df['confidence'] = probabilities[range(len(df)), sig_idx]
        self.signal = self.I(lambda: df['signal'])
        self.confidence = self.I(lambda: df['confidence'])
        self.atr_perc = self.I(lambda: df['ATRr_14_5m'])
        self.ema_21_5m = self.I(lambda: df['EMA_21_5m'])
        
        # --- CORRECCIÓN IMPORTANTE: Asegúrate de que la columna se llame 'EMA_MACRO' ---
        # El backtest debe usar la misma columna que creamos en run_backtest.
        if 'EMA_MACRO' in df.columns:
            macro_series = df['EMA_MACRO']
        else:
            # Si por alguna razón no está presente, calculamos la EMA(200) sobre velas de 5 minutos
            try:
                close_5m = df['Close'].resample('5T').last()
                ema_5m = ta.ema(close_5m, length=200)
                ema_5m = ema_5m.reindex(df.index, method='ffill')
                macro_series = ema_5m
            except Exception:
                # Último recurso: usar una EMA(50) sobre 1m
                macro_series = ta.ema(df['Close'], length=50)
        self.macro_ema = self.I(lambda: macro_series)
        
        self.atr_14_5m = self.I(lambda: df['ATR_14_5m'])
        self.adx = self.I(lambda: df['ADX_14'])
        self.bb_width = self.I(lambda: df['bb_width'])
        self.rsi_5m = self.I(lambda: df['RSI_14_5m'])
        # --- NUEVO: Variable para guardar la señal pendiente ---
        self.pending_signal = None

    def next(self):
        p = STRATEGY_PARAMS
        price = self.data.Close[-1]
        # --- Parte 1: Gestión de Posición Activa (NUEVA LÓGICA MANUAL DE TSL) ---
        if self.position:
            if self.position.is_long:
                new_sl = self.data.Low[-1] - (self.atr_14_5m[-1] * p['SL_ATR_MULTIPLIER'])
                if new_sl > self.trades[-1].sl:
                    self.trades[-1].sl = new_sl
            else:
                new_sl = self.data.High[-1] + (self.atr_14_5m[-1] * p['SL_ATR_MULTIPLIER'])
                if new_sl < self.trades[-1].sl:
                    self.trades[-1].sl = new_sl
            return
        # --- Parte 2: Lógica de Entrada con Confirmación (Opción 2: Confirmación de Precio) ---
        if self.pending_signal:
            signal_type = self.pending_signal['type']
            entry_price = self.pending_signal['entry_price']
            sl_dist = self.atr_14_5m[-1] * p['SL_ATR_MULTIPLIER']
            
            if signal_type == "LONG":
                # INVERTIDO: Entra si el precio CAE 0.05% desde la señal (SHORT en vez de LONG)
                if self.data.Close[-1] <= entry_price * 0.9995:
                    self.sell(sl=price + sl_dist)
                    print(f"✓✓✓ CONFIRMACIÓN LONG→SHORT en {self.data.index[-1]}. Entrando SHORT.")
                    self.pending_signal = None
                # O cancela si sube 0.1%
                elif self.data.Close[-1] >= entry_price * 1.001:
                    print(f"✗ Señal LONG→SHORT cancelada por movimiento adverso en {self.data.index[-1]}")
                    self.pending_signal = None
                    
            elif signal_type == "SHORT":
                # INVERTIDO: Entra si el precio SUBE 0.05% desde la señal (LONG en vez de SHORT)
                if self.data.Close[-1] >= entry_price * 1.0005:
                    self.buy(sl=price - sl_dist)
                    print(f"✓✓✓ CONFIRMACIÓN SHORT→LONG en {self.data.index[-1]}. Entrando LONG.")
                    self.pending_signal = None
                # O cancela si cae 0.1%
                elif self.data.Close[-1] <= entry_price * 0.999:
                    print(f"✗ Señal SHORT→LONG cancelada por movimiento adverso en {self.data.index[-1]}")
                    self.pending_signal = None
            return
        # --- Parte 3: Búsqueda de Nuevas Señales (sin cambios) ---
        sig = self.signal[-1]
        if sig == 0:
            return
        conf = self.confidence[-1]
        atr_p = self.atr_perc[-1]
        adx_strength = self.adx[-1]
        is_up_5m = price > self.ema_21_5m[-1]
        is_macro_bull = price > self.macro_ema[-1]
        current_bb_width = self.bb_width[-1] * 100
        cond_conf = conf >= p["CONFIDENCE_THRESHOLD"]
        cond_atr = p["MIN_ATR_PERC"] < atr_p < p["MAX_ATR_PERC"]
        cond_adx = adx_strength > p["ADX_TREND_STRENGTH"]
        cond_bbw = current_bb_width > p["MIN_BB_WIDTH_PERC"]
        sig_str = "LONG" if sig == 1 else "SHORT"
        cond_trend_5m = (sig_str == "LONG" and is_up_5m) or (sig_str == "SHORT" and not is_up_5m)
        cond_macro = (sig_str == "LONG" and is_macro_bull) or (sig_str == "SHORT" and not is_macro_bull)
        rsi_val = self.rsi_5m[-1]
        cond_exhaust = True
        if not pd.isna(rsi_val):
            cond_exhaust = not ((sig_str == "LONG" and rsi_val >= p["RSI_EXHAUST_LONG"]) or (sig_str == "SHORT" and rsi_val <= p["RSI_EXHAUST_SHORT"]))
        if cond_conf and cond_atr and cond_trend_5m and cond_macro and cond_adx and cond_bbw and cond_exhaust:
            print(f"→ SEÑAL {sig_str} DETECTADA en {self.data.index[-1]}. Esperando confirmación...")
            self.pending_signal = {
                'type': sig_str,
                'entry_price': price
            }
        else:
            print(f"✗ Señal {sig_str} en {self.data.index[-1]} RECHAZADA por filtros.")

def run_backtest(df, model, features):
    print("\n" + "="*60 + "\n--- INICIANDO BACKTEST DE LA ESTRATEGIA ---\n" + "="*60)
    # --- MEJORA: Calcular la EMA macro sobre velas de 5 minutos (EMA_200 sobre 5m) ---
    # Esto alinea la lógica entre backtest y el análisis en vivo (Solution 2).
    try:
        close_5m = df['Close'].resample('5T').last()
        ema_5m = ta.ema(close_5m, length=200)
        # Reindex back to 1m timestamps using forward-fill so every 1m row has the corresponding 5m EMA
        ema_5m = ema_5m.reindex(df.index, method='ffill')
        df['EMA_MACRO'] = ema_5m
    except Exception as e:
        print(f"[WARN] No se pudo calcular EMA_MACRO 5m: {e}. Intentando EMA(200) directa sobre 1m...")
        df['EMA_MACRO'] = ta.ema(df['Close'], length=200)

    # Asegurarse de que no queden NaNs en EMA_MACRO (llenado hacia adelante)
    df['EMA_MACRO'].fillna(method='ffill', inplace=True)
    df.dropna(subset=['EMA_MACRO'], inplace=True)  # Eliminamos las primeras filas donde la EMA no se puede calcular

    if df.empty:
        print("!!! ERROR: El dataframe para backtest quedó vacío después del preprocesamiento. Revisar los datos de entrada. !!!")
        return

    MLStrategy.model, MLStrategy.features = model, features
    bt = Backtest(df, MLStrategy, cash=10000, commission=.0006, exclusive_orders=True)
    stats = bt.run(trade_on_close=True)

    with open("backtest_results.txt", "w", encoding="utf-8") as f:
        f.write("="*60 + "\nRESULTADOS DEL BACKTEST\n" + "="*60 + "\n\n" + str(stats))

    print("\n--- RESULTADOS DEL BACKTEST ---\n", stats, "\n" + "-"*29)
    print("\n*** Resultados guardados en backtest_results.txt ***\n")
    print("Presiona ENTER para ver el gráfico interactivo...")
    input()
    bt.plot()
# ========================================================
# --- BUCLE PRINCIPAL ---
# ========================================================
if __name__ == "__main__":
    SYMBOL = "TRXUSDT"
    RUN_BACKTEST_ONLY = False  # True = Backtest, False = Dinero Real
    BACKTEST_FILE = "backtest_data.pkl"
    if RUN_BACKTEST_ONLY:
        print(f"\n--- MODO BACKTEST ACTIVADO ({SYMBOL}) ---")
        
        df_features = None

        # 1. INTENTAR CARGAR DATOS DEL ARCHIVO (VELOCIDAD MÁXIMA)
        if os.path.exists(BACKTEST_FILE):
            print(f"-> ¡Archivo de caché encontrado! Cargando {BACKTEST_FILE}...")
            try:
                df_features = pd.read_pickle(BACKTEST_FILE)
                print(f"✓ Datos cargados al instante. ({len(df_features)} velas)")
            except Exception as e:
                print(f"!!! Error leyendo caché: {e}. Se descargarán nuevos datos.")
                df_features = None

        # 2. SI NO HAY CACHÉ, DESCARGAR DE LA API (SOLO UNA VEZ)
        if df_features is None:
            print("-> Descargando datos nuevos de la API...")

            # CONFIGURACIÓN PARA 90 días de historial (igual que el otro script)
            DAYS_OF_DATA = 8
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=DAYS_OF_DATA)
            end_ms = int(end_date.timestamp() * 1000)
            start_ms = int(start_date.timestamp() * 1000)
            print(f"   Descargando datos para un período de {DAYS_OF_DATA} días ({start_date.date()} -> {end_date.date()})...")

            df_1m_hist = fetch_history_candles(SYMBOL, "60", start_ms, end_ms)
            df_5m_hist = fetch_history_candles(SYMBOL, "300", start_ms, end_ms)
            df_btc_15m_hist = fetch_history_candles("BTCUSDT", "900", start_ms, end_ms)

            if any(d is None or d.empty for d in [df_1m_hist, df_5m_hist, df_btc_15m_hist]):
                print("!!! Error crítico: Faltan datos de la API. !!!")
                exit(1)

            print("✓ Datos descargados. Calculando indicadores...")
            df_features = create_features_and_target(df_1m_hist, df_5m_hist, df_btc_15m_hist, create_target=True)

            # --- GUARDAMOS EL ARCHIVO PARA EL FUTURO ---
            df_features.to_pickle(BACKTEST_FILE)
            print(f"✓ Archivo {BACKTEST_FILE} creado y guardado. La próxima vez será instantáneo.")

        # 3. GESTIÓN DEL MODELO
        if not os.path.exists(MODEL_FILE) or not os.path.exists(FEATURES_LIST_FILE):
            print("-> Entrenando nuevo modelo...")
            # Usamos el 70% para entrenar
            train_size = int(len(df_features) * 0.7)
            df_train = df_features.iloc[:train_size]
            model, features = train_and_save_model(df_train)
            # El resto para backtest
            df_backtest = df_features.iloc[train_size:]
        else:
            print("-> Usando modelo existente...")
            model = joblib.load(MODEL_FILE)
            with open(FEATURES_LIST_FILE, 'r') as f:
                features = json.load(f)
            
            # Usamos el último 30% de los datos cargados para el backtest
            # (O ajusta esto si quieres usar todo el dataset)
            cutoff = int(len(df_features) * 0.7)
            df_backtest = df_features.iloc[cutoff:]

        # 4. EJECUTAR BACKTEST
        if df_backtest.empty:
            print("!!! Error: No hay datos suficientes para el backtest. !!!")
        else:
            run_backtest(df_backtest, model, features)

    else:
        # ========================================================
        # MODO LIVE TRADING (OPTIMIZADO CON BUFFER)
        # ========================================================
        print("\n--- MODO LIVE TRADING ACTIVADO (Latencia Baja) ---")
        position_state = load_state_from_file()
        
        # --- BUFFERS DE MEMORIA ---
        # Aquí guardaremos el historial para no descargarlo siempre
        buffer_1m = None
        buffer_5m = None
        
        while True:
            now_utc = datetime.utcnow()
            # Log ligero para no ensuciar tanto la consola
            print(f"\n[{now_utc.strftime('%H:%M:%S')}] ", end="")

            try:
                # ---------------------------------------------------------
                # PASO 1: GESTIÓN DE DATOS (INCREMENTAL)
                # ---------------------------------------------------------
                end_ms = int(now_utc.timestamp() * 1000)
                
                # A) SI ES LA PRIMERA VEZ (ARRANQUE): DESCARGA COMPLETA (3 DÍAS)
                if buffer_1m is None or buffer_5m is None:
                    print("Inicializando caché (3 días)... ", end="")
                    start_ms = int((now_utc - timedelta(days=3)).timestamp() * 1000)
                    
                    buffer_1m = fetch_history_candles(SYMBOL, "60", start_ms, end_ms)
                    buffer_5m = fetch_history_candles(SYMBOL, "300", start_ms, end_ms)
                    
                    if buffer_1m is None or buffer_5m is None:
                        print("❌ Error inicializando datos. Reintentando en 10s...")
                        time.sleep(10)
                        continue
                    print("✅ Listo.")

                # B) SI YA TENEMOS DATOS: DESCARGA SOLO LO NUEVO
                else:
                    # Empezamos 1 ms después de la última vela que tenemos
                    last_ts_1m = buffer_1m.index[-1]
                    start_ms_update = int(last_ts_1m.timestamp() * 1000) + 1
                    
                    # Si ha pasado menos de 1 minuto, no descargamos nada para ahorrar API
                    if (now_utc - last_ts_1m).total_seconds() < 60:
                         print("Esperando cierre de vela...", end="")
                    else:
                        print(f"Actualizando... ", end="")
                        new_1m = fetch_history_candles(SYMBOL, "60", start_ms_update, end_ms)
                        
                        # Para 5m, revisamos su propia última fecha
                        last_ts_5m = buffer_5m.index[-1]
                        start_ms_5m = int(last_ts_5m.timestamp() * 1000) + 1
                        new_5m = fetch_history_candles(SYMBOL, "300", start_ms_5m, end_ms)

                        # Actualizar buffers (Concatenar y quitar duplicados)
                        if new_1m is not None and not new_1m.empty:
                            buffer_1m = pd.concat([buffer_1m, new_1m])
                            # Mantenemos solo duplicados 'last' y limitamos tamaño para no explotar RAM
                            buffer_1m = buffer_1m[~buffer_1m.index.duplicated(keep='last')].tail(5000) 
                        
                        if new_5m is not None and not new_5m.empty:
                            buffer_5m = pd.concat([buffer_5m, new_5m])
                            buffer_5m = buffer_5m[~buffer_5m.index.duplicated(keep='last')].tail(1000)

                # ---------------------------------------------------------
                # PASO 2: PROCESAMIENTO RÁPIDO
                # ---------------------------------------------------------
                # Calculamos indicadores sobre el buffer en memoria (muy rápido, <0.05s)
                # Pasamos 'False' a create_target para ahorrar tiempo
                df_live_data = create_features_and_target(buffer_1m, buffer_5m, None, create_target=False)
                
                if df_live_data is not None and not df_live_data.empty:
                    # ---------------------------------------------------------
                    # PASO 3: TRADING
                    # ---------------------------------------------------------
                    if position_state.get("in_position", False):
                        # Gestión de posición (SL/TP)
                        market_data = get_current_market_data(SYMBOL)
                        if market_data:
                            manage_position_with_candle_price(market_data['price'])
                    else:
                        # Buscar entrada
                        analysis = get_market_analysis(SYMBOL, df_live_data)
                        if analysis and analysis.get("decision") in ["LONG", "SHORT"]:
                            open_new_position(analysis, SYMBOL)
                else:
                    print("⚠️ Datos insuficientes tras procesamiento.")

            except Exception as e:
                print(f"\n❌ Error en ciclo: {e}")
                # No detenemos el bot, solo esperamos un poco
                time.sleep(5)

            # ---------------------------------------------------------
            # CONTROL DE TIEMPO
            # ---------------------------------------------------------
            # Calcular tiempo para el siguiente ciclo de 5 minutos exactos
            base_time = datetime.utcnow()
            # Truco: Si ya procesamos, esperamos 10 segundos antes de chequear de nuevo
            # para dar sensación de "casi tiempo real" sin saturar la API, 
            # o usamos la lógica de ciclo completo.
            
            # Opción Rápida: Chequear cada minuto si hay vela nueva
            sleep_sec = 60 - base_time.second
            if sleep_sec < 5: sleep_sec += 60
            
            # Descomenta esta línea si quieres ver el tiempo de espera
            # print(f" [Esperando {sleep_sec}s]") 
            time.sleep(sleep_sec)
