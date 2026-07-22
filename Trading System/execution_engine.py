# type: ignore

import asyncio
import aiohttp
import json
import os
import logging
import math
import time
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from binance import AsyncClient, BinanceSocketManager
from binance.exceptions import BinanceAPIException
import warnings


# تنظیمات لاگر پایدار سیستم
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.FileHandler("live_futures_trading.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

API_KEY = "youer_api_key_from_binance_testnet"                   # get API Key on the --> https://demo.binance.com/en/my/settings/api-management
API_SECRET = "youer_SECRET_api_key_from_binance_testnet"         # get secret API Key
MODEL_PATH = "MODELv2.6.joblib"


class CustomTradingEnsemble:
    def __init__(self, xgb_model, cat_model, lgb_model, scaler, up_threshold, down_threshold, class_0_weight=2.5, class_2_weight=2.5):
        self.xgb_model = xgb_model
        self.cat_model = cat_model
        self.lgb_model = lgb_model
        self.scaler = scaler
        self.up_threshold = up_threshold
        self.down_threshold = down_threshold
        self.class_0_weight = class_0_weight  # وزن بهینه کلاس Down
        self.class_2_weight = class_2_weight  # وزن بهینه کلاس Up

    def predict_proba(self, X_raw):
        # مرتب‌سازی ستون‌ها بر اساس چیدمان زمان آموزش اسکیلر
        X_raw = X_raw[self.scaler.feature_names_in_]
        X_scaled = self.scaler.transform(X_raw)
        
        # ۱. ترکیب احتمالات مدل‌ها (XGB:1, LGB:2, CAT:3)
        xgb_probs = self.xgb_model.predict_proba(X_scaled)
        cat_probs = self.cat_model.predict_proba(X_scaled)
        lgb_probs = self.lgb_model.predict_proba(X_scaled)
        ensemble_probs = (xgb_probs * 1 + lgb_probs * 2 + cat_probs * 3) / 6.0
        
        # ۲. اعمال وزن‌های بهینه کلاس‌ها (Post-Training Shift)
        final_probs = ensemble_probs.copy()
        final_probs[:, 0] *= self.class_0_weight
        final_probs[:, 2] *= self.class_2_weight
        
        # نرمال‌سازی مجدد احتمالات ردیف‌ها تا جمع آن‌ها ۱ شود
        row_sums = final_probs.sum(axis=1, keepdims=True)
        final_probs = np.divide(final_probs, row_sums, out=final_probs, where=row_sums > 0)
        
        return final_probs

    def predict(self, X_raw):
        final_probs = self.predict_proba(X_raw)
        prob_down = final_probs[:, 0]
        prob_up = final_probs[:, 2]
        
        # ساختار برداری سریع برای اعمال آستانه‌ها
        preds = np.ones(len(X_raw), dtype=int)  # پیش‌فرض: Sideway (1)
        
        up_cond = (prob_up > self.up_threshold) & (prob_up > prob_down)
        preds[up_cond] = 2
        
        down_cond = (prob_down > self.down_threshold) & (prob_down > prob_up) & (~up_cond)
        preds[down_cond] = 0
        
        return preds


class FuturesLiveTradingSystem:
    def __init__(self, symbol: str, leverage: int = 3, risk_per_trade: float = 0.10):
        self.symbol = symbol.upper()
        self.state_file = f"futures_state_{self.symbol}.json"
        self._state_lock = asyncio.Lock()  # قفل async برای thread-safe ساختن مدیریت state
        
        self.client = None
        self.bsm = None
        self.is_running = False
        
        # پارامترهای منطبق بر بک‌تست
        self.leverage = leverage
        self.risk_per_trade = risk_per_trade
        self.atr_multiplier = 3.0
        self.tp_multiplier = 2.0
        self.regime_sma_period = 70
        self.regime_filter_enabled = True
        self.counter_trend_size_mult = 0.0  # 0.0 یعنی پوزیشن خلاف روند کامل رد شود
        
        # مشخصات اعشار صرافی فیوچرز
        self.price_precision = 2
        self.qty_precision = 3
        self.min_notional = 5.0
        
        # لیست دقیق فیچرهای مدل (۳۸ فیچر)
        self.feature_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume', 'RSI_14', 'ATR_14',
            'MACD_Line', 'MACD_Signal', 'MACD_Histogram', 'BB_Position_20',
            'FearGreedIndex', 'HashRate', 'ActiveAddresses', 'TxVolumeUSD',
            'RSI_Signal_Neutral', 'RSI_Signal_Overbought (Sell)', 'RSI_Signal_Oversold (Buy)',
            'BB_Signal_Above Upper Band (Sell)', 'BB_Signal_Below Lower Band (Buy)', 'BB_Signal_Neutral',
            'Dist_EMA7_pct', 'Dist_EMA14_pct', 'Dist_EMA21_pct', 'Dist_EMA50_pct',
            'Dist_EMA200_pct', 'EMA_7_21_ratio', 'EMA_21_50_ratio', 'EMA_50_200_ratio',
            'EMA_21_slope', 'EMA_200_slope', 'ATR_pct', 'Return_5d', 'Return_10d',
            'Return_20d', 'Dist_from_20d_high', 'Dist_from_20d_low', 'ADX_14'
        ]
        
        logger.info(f"Loading Scaled Model from {MODEL_PATH}...")
        self.model = joblib.load(MODEL_PATH)
        
        self.history_df = pd.DataFrame()
        self.state = {
            "bot_status": "IDLE",  # IDLE, LONG, SHORT
            "entry_price": 0.0,
            "qty": 0.0,
            "trailing_stop": 0.0,
            "take_profit": 0.0
        }
        self.last_kline_time = None
        
        # کش دیتای خارجی (مدت زمان انقضا: ۴ ساعت = 14400 ثانیه)
        self.ext_cache = {"data": (50.0, 4e8, 9e5, 1e10), "last_update": 0}

    async def _load_state(self) -> dict:
        """بارگذاری فایل state با قفل asyncio"""
        async with self._state_lock:
            if os.path.exists(self.state_file):
                try:
                    with open(self.state_file, 'r') as f:
                        return json.load(f)
                except Exception as e:
                    logger.error(f"State load error: {e}")
            return {
                "bot_status": "IDLE",
                "entry_price": 0.0,
                "qty": 0.0,
                "trailing_stop": 0.0,
                "take_profit": 0.0
            }

    async def _save_state(self):
        """ذخیره فایل state با قفل asyncio"""
        async with self._state_lock:
            try:
                with open(self.state_file, 'w') as f:
                    json.dump(self.state, f, indent=4)
            except Exception as e:
                logger.error(f"State save error: {e}")

    async def init_client(self):
        self.state = await self._load_state()
        self.client = await AsyncClient.create(
            api_key=API_KEY, 
            api_secret=API_SECRET, 
            testnet=True
        )
        self.client.FUTURES_URL = 'https://testnet.binancefuture.com'
        self.bsm = BinanceSocketManager(self.client)
        
        await self._configure_futures_account()
        await self._prefill_historical_data()
        await self._sync_position_with_exchange()

        logger.info("Binance Futures Client Initialized (Testnet).")

    async def _configure_futures_account(self):
        try:
            await self.client.futures_change_leverage(symbol=self.symbol, leverage=self.leverage)
            logger.info(f"Leverage set to {self.leverage}x")
            try:
                await self.client.futures_change_margin_type(symbol=self.symbol, marginType='ISOLATED')
                logger.info("Margin type set to ISOLATED")
            except Exception as e:
                if 'No need to change margin type' not in str(e):
                    logger.warning(f"Margin type warning: {e}")
            
            info = await self.client.futures_exchange_info()
            for s in info['symbols']:
                if s['symbol'] == self.symbol:
                    for f in s['filters']:
                        if f['filterType'] == 'PRICE_FILTER':
                            self.price_precision = int(round(-math.log10(float(f['tickSize']))))
                        elif f['filterType'] == 'LOT_SIZE':
                            self.qty_precision = int(round(-math.log10(float(f['stepSize']))))
                        elif f['filterType'] == 'MIN_NOTIONAL':
                            self.min_notional = float(f['notional'])
            logger.info(f"Rules: PricePrec={self.price_precision}, QtyPrec={self.qty_precision}, MinNotional={self.min_notional}")
        except Exception as e:
            logger.error(f"Failed to configure futures account: {e}")

    async def _sync_position_with_exchange(self):
        """همگام‌سازی پوزیشن لوکال با صرافی (Position Validation)"""
        try:
            positions = await self.client.futures_position_information(symbol=self.symbol)
            if positions:
                pos = positions[0]
                position_amt = float(pos['positionAmt'])
                entry_price = float(pos['entryPrice'])
                
                if position_amt > 0:
                    real_status = "LONG"
                elif position_amt < 0:
                    real_status = "SHORT"
                else:
                    real_status = "IDLE"

                if self.state["bot_status"] != real_status:
                    logger.warning(f"⚠️ Desync Detected! Local State: {self.state['bot_status']} | Exchange State: {real_status}")
                    self.state["bot_status"] = real_status
                    self.state["qty"] = abs(position_amt)
                    self.state["entry_price"] = entry_price
                    if real_status == "IDLE":
                        self.state["trailing_stop"] = 0.0
                        self.state["take_profit"] = 0.0
                    await self._save_state()
                    logger.info("State synced with exchange successfully.")
        except Exception as e:
            logger.error(f"Error syncing position with exchange: {e}")

    async def _prefill_historical_data(self):
        klines = await self.client.futures_historical_klines(self.symbol, AsyncClient.KLINE_INTERVAL_1DAY, "601 days ago UTC")
        records = []
        for k in klines[:-1]:
            records.append({
                "Open": float(k[1]), "High": float(k[2]), 
                "Low": float(k[3]), "Close": float(k[4]), "Volume": float(k[5])
            })
        self.history_df = pd.DataFrame(records)
        self.last_kline_time = klines[-1][0]

    async def _fetch_external_features(self) -> tuple:
        """دریافت داده‌های آن‌چین با کش ۴ ساعته (14400 ثانیه)"""
        current_time = time.time()
        if current_time - self.ext_cache["last_update"] < 14400:
            return self.ext_cache["data"]

        fng_val, hash_val, addr_val, vol_val = self.ext_cache["data"]
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("https://api.alternative.me/fng/?limit=1") as res:
                    fng_val = float((await res.json())['data'][0]['value'])
                async with session.get("https://api.blockchain.info/charts/hash-rate?timespan=1days&format=json") as res:
                    hash_val = float((await res.json())['values'][-1]['y'])
                async with session.get("https://api.blockchain.info/charts/n-unique-addresses?timespan=1days&format=json") as res:
                    addr_val = float((await res.json())['values'][-1]['y'])
                async with session.get("https://api.blockchain.info/charts/estimated-transaction-volume-usd?timespan=1days&format=json") as res:
                    vol_val = float((await res.json())['values'][-1]['y'])
                    
            self.ext_cache = {"data": (fng_val, hash_val, addr_val, vol_val), "last_update": current_time}
            logger.info("External features refreshed (4-hour cache updated).")
        except Exception as e:
            logger.warning(f"External API error, using cached data. Error: {e}")
            
        return self.ext_cache["data"]

    async def _calculate_live_features(self) -> pd.Series:
        df = self.history_df.copy()
        
        # ۱. محاسبه EMAها
        df["EMA_7"] = df["Close"].ewm(span=7, adjust=False).mean()
        df["EMA_14"] = df["Close"].ewm(span=14, adjust=False).mean()
        df["EMA_21"] = df["Close"].ewm(span=21, adjust=False).mean()
        df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()
        df["EMA_200"] = df["Close"].ewm(span=200, adjust=False).mean()
        
        # ۲. محاسبه محاسبه is_bull_regime (رژیم بازار)            
        df["SMA_70"] = df["Close"].rolling(window=self.regime_sma_period).mean()
        if df["SMA_70"].iloc[-1:].isna().any():
            logger.error("Insufficient history for regime calculation. Skipping signal.")

        df["is_bull_regime"] = np.where(df["Close"] > df["SMA_70"], 1, 0)

        # ۳. محاسبه RSI_14
        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0.0)).ewm(alpha=1/14, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1/14, adjust=False).mean()
        rs = gain / (loss + 1e-10)
        df["RSI_14"] = 100 - (100 / (1 + rs))

        # ۴. محاسبه ATR_14
        tr1 = df["High"] - df["Low"]
        tr2 = abs(df["High"] - df["Close"].shift(1))
        tr3 = abs(df["Low"] - df["Close"].shift(1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        df["ATR_14"] = tr.ewm(alpha=1/14, adjust=False).mean()
        
        # ۵. محاسبه MACD
        ema12 = df["Close"].ewm(span=12, adjust=False).mean()
        ema26 = df["Close"].ewm(span=26, adjust=False).mean()
        df["MACD_Line"] = ema12 - ema26
        df["MACD_Signal"] = df["MACD_Line"].ewm(span=9, adjust=False).mean()
        df["MACD_Histogram"] = df["MACD_Line"] - df["MACD_Signal"]
        
        # ۶. محاسبه Bollinger Bands
        sma_20 = df["Close"].rolling(window=20).mean()
        std_20 = df["Close"].rolling(window=20).std()
        bb_upper = sma_20 + (2 * std_20)
        bb_lower = sma_20 - (2 * std_20)
        df["BB_Position_20"] = (df["Close"] - bb_lower) / (bb_upper - bb_lower + 1e-10)
        
        # ۷. سیگنال‌های One-Hot
        df["RSI_Signal_Overbought (Sell)"] = np.where(df["RSI_14"] > 70, 1, 0)
        df["RSI_Signal_Oversold (Buy)"] = np.where(df["RSI_14"] < 30, 1, 0)
        df["RSI_Signal_Neutral"] = np.where((df["RSI_14"] >= 30) & (df["RSI_14"] <= 70), 1, 0)
        df["BB_Signal_Above Upper Band (Sell)"] = np.where(df["Close"] > bb_upper, 1, 0)
        df["BB_Signal_Below Lower Band (Buy)"] = np.where(df["Close"] < bb_lower, 1, 0)
        df["BB_Signal_Neutral"] = np.where((df["Close"] <= bb_upper) & (df["Close"] >= bb_lower), 1, 0)
        
        # ۸. نسبت‌های انحراف و شیپ
        df["Dist_EMA7_pct"] = (df["Close"] - df["EMA_7"]) / df["Close"]
        df["Dist_EMA14_pct"] = (df["Close"] - df["EMA_14"]) / df["Close"]
        df["Dist_EMA21_pct"] = (df["Close"] - df["EMA_21"]) / df["Close"]
        df["Dist_EMA50_pct"] = (df["Close"] - df["EMA_50"]) / df["Close"]
        df["Dist_EMA200_pct"] = (df["Close"] - df["EMA_200"]) / df["Close"]
        
        df["EMA_7_21_ratio"] = df["EMA_7"] / df["EMA_21"] - 1
        df["EMA_21_50_ratio"] = df["EMA_21"] / df["EMA_50"] - 1
        df["EMA_50_200_ratio"] = df["EMA_50"] / df["EMA_200"] - 1
        df["EMA_21_slope"] = df["EMA_21"].pct_change(5)
        df["EMA_200_slope"] = df["EMA_200"].pct_change(20)
        
        # ۹. محاسبه ADX_14
        up_move = df["High"] - df["High"].shift(1)
        down_move = df["Low"].shift(1) - df["Low"]
        df["+DM"] = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        df["-DM"] = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        tr_smooth = tr.ewm(alpha=1/14, adjust=False).mean()
        plus_di = 100 * (df["+DM"].ewm(alpha=1/14, adjust=False).mean() / (tr_smooth + 1e-10))
        minus_di = 100 * (df["-DM"].ewm(alpha=1/14, adjust=False).mean() / (tr_smooth + 1e-10))
        dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10))
        df["ADX_14"] = dx.ewm(alpha=1/14, adjust=False).mean()
        
        # ۱۰. فاکتورهای مومنتوم
        df["ATR_pct"] = df["ATR_14"] / df["Close"]
        df["Return_5d"] = df["Close"].pct_change(5)
        df["Return_10d"] = df["Close"].pct_change(10)
        df["Return_20d"] = df["Close"].pct_change(20)
        df["Dist_from_20d_high"] = (df["Close"] - df["High"].rolling(20).max()) / df["Close"]
        df["Dist_from_20d_low"] = (df["Close"] - df["Low"].rolling(20).min()) / df["Close"]
        
        # ۱۱. داده‌های آن‌چین
        fng, hshr, addr, tx_vol = await self._fetch_external_features()
        df["FearGreedIndex"] = fng
        df["HashRate"] = hshr
        df["ActiveAddresses"] = addr
        df["TxVolumeUSD"] = tx_vol
        
        return df.iloc[-1]

    async def start_trading_loop(self):
        self.is_running = True
        logger.info("Live Futures Stream (1D) Activated.")
        last_minute = -1
        
        while self.is_running:
            try:
                kline_socket = self.bsm.kline_socket(symbol=self.symbol, interval=AsyncClient.KLINE_INTERVAL_1DAY)
                async with kline_socket as stream:
                    while True:
                        msg = await stream.recv()
                        if not msg or 'k' not in msg :
                            continue
                        
                        kline = msg['k']
                        kline_start = int(kline['t'])
                        is_candle_closed = kline['x']
                        
                        live_day_row = {
                            "Open": float(kline['o']), "High": float(kline['h']),
                            "Low": float(kline['l']), "Close": float(kline['c']), "Volume": float(kline['v'])
                        }
                        
                        if self.last_kline_time is None or kline_start > self.last_kline_time:
                            self.history_df = pd.concat([self.history_df, pd.DataFrame([live_day_row])], ignore_index=True).iloc[-600:]
                            self.last_kline_time = kline_start
                        else:
                            self.history_df.iloc[-1] = pd.Series(live_day_row)
                            
                        current_price = float(kline['c'])

                        # پردازش سیگنال راس هر دقیقه
                        current_minute = datetime.now().minute
                        if current_minute != last_minute:
                            last_minute = current_minute
                            asyncio.create_task(self._process_live_signal(current_price))
                            
                        # آپدیت روزانه استاپ متحرک
                        if is_candle_closed and self.state['bot_status'] != "IDLE":
                            await self._update_trailing_stop(current_price)
                            
            except Exception as e:
                logger.error(f"WebSocket Error: {e}. Reconnecting in 5s...")
                await asyncio.sleep(5)

    async def _update_trailing_stop(self, close_price: float):
        latest_features = await self._calculate_live_features()
        atr_curr = float(latest_features['ATR_14'])
        
        if self.state['bot_status'] == 'LONG':
            new_stop = close_price - (atr_curr * self.atr_multiplier)
            self.state['trailing_stop'] = max(self.state['trailing_stop'], new_stop)
            logger.info(f"Trailing Stop Updated (LONG): {self.state['trailing_stop']}")
        elif self.state['bot_status'] == 'SHORT':
            new_stop = close_price + (atr_curr * self.atr_multiplier)
            self.state['trailing_stop'] = min(self.state['trailing_stop'], new_stop)
            logger.info(f"Trailing Stop Updated (SHORT): {self.state['trailing_stop']}")
            
        await self._save_state()

    async def _process_live_signal(self, current_price: float):
        # همگام‌سازی قبل از پردازش سیگنال
        await self._sync_position_with_exchange()
        
        latest_features = await self._calculate_live_features()
        
        # --- 3. Feature Validation ---
        missing_cols = [c for c in self.feature_columns if c not in latest_features.index]
        if missing_cols:
            logger.error(f"❌ Feature Validation Failed! Missing columns: {missing_cols}. Skipping signal processing.")
            return

        input_vector = pd.DataFrame([latest_features[self.feature_columns]])
        
        if input_vector.isna().sum().sum() > 0 or np.isinf(input_vector.values).any():
            logger.error("❌ Feature Validation Failed! Found NaN or Inf values in features. Skipping signal processing.")
            return
            
        ml_signal = int(self.model.predict(input_vector)[0])
        status = self.state['bot_status']
        
        # لاگ‌های دقیقه‌ای قیمت به DEBUG منتقل می‌شوند (در فایل اصلی لاگ نمی‌شوند مگر سطح رو DEBUG باشه)
        logger.debug(f"Price: {current_price} | Signal: {ml_signal} | State: {status}")

        # فقط اگر وضعیت پوزیشن تغییر کرد یا سیگنال جدید آمد، INFO بدهد
        if getattr(self, 'last_state', None) != status:
            logger.info(f"🔔 State Changed: {getattr(self, 'last_state', 'NONE')} ➔ {status} | Price: {current_price}")
            self.last_state = status
        
        # ۱. چک کردن خروج‌ها
        if status != "IDLE":
            trade_closed = False
            exit_reason = ""
            
            ts = self.state['trailing_stop']
            tp = self.state['take_profit']
            
            if status == "LONG":
                if current_price <= ts:
                    exit_reason, trade_closed = "Trailing Stop", True
                elif current_price >= tp:
                    exit_reason, trade_closed = "Take Profit", True
                elif ml_signal == 0:
                    exit_reason, trade_closed = "Signal Flip", True

            elif status == "SHORT":
                if current_price >= ts:
                    exit_reason, trade_closed = "Trailing Stop", True
                elif current_price <= tp:
                    exit_reason, trade_closed = "Take Profit", True
                elif ml_signal == 2:
                    exit_reason, trade_closed = "Signal Flip", True

            if trade_closed:
                await self._execute_market_exit(status, exit_reason)
                status = "IDLE"
                
        # ۲. چک کردن ورودها
        if status == "IDLE" and ml_signal in [0, 2]:
            acc = await self.client.futures_account()
            usdt_balance = next((float(b['availableBalance']) for b in acc['assets'] if b['asset'] == 'USDT'), 0.0)
            
            if usdt_balance < 10:
                logger.warning(f"Insufficient balance: {usdt_balance} USDT")
                return

            atr_prev = float(latest_features['ATR_14'])
            is_bull_regime_prev = bool(latest_features['is_bull_regime'])
            
            sl_distance = atr_prev * self.atr_multiplier
            sl_pct = sl_distance / current_price
            
            risk_amount = usdt_balance * self.risk_per_trade
            desired_pos_size = risk_amount / sl_pct
            max_allowed = usdt_balance * self.leverage
            
            proposed_side = "LONG" if ml_signal == 2 else "SHORT"
            is_counter_trend = False
            
            if self.regime_filter_enabled:
                if proposed_side == "SHORT" and is_bull_regime_prev:
                    is_counter_trend = True
                elif proposed_side == "LONG" and not is_bull_regime_prev:
                    is_counter_trend = True
                
            if is_counter_trend:
                logger.info(f"Counter Trend Detected. Multiplier: {self.counter_trend_size_mult}")
                desired_pos_size *= self.counter_trend_size_mult
                
            position_size = min(desired_pos_size, max_allowed)
            
            if position_size <= 0:
                logger.info("Trade rejected by Regime/Trend Filter.")
                return

            qty = round(position_size / current_price, self.qty_precision)
            
            # --- 5. Quantity Validation Post-Rounding ---
            actual_notional = qty * current_price
            if actual_notional < self.min_notional or qty <= 0:
                logger.warning(f"Post-rounding quantity validation failed! Calculated Notional: {actual_notional:.2f}$ < Min Notional: {self.min_notional}$. Skipping order.")
                return

            await self._execute_smart_entry(proposed_side, qty, current_price, atr_prev)

    async def _execute_smart_entry(self, side: str, qty: float, current_price: float, atr_prev: float):
        try:
            binance_side = 'BUY' if side == "LONG" else 'SELL'
            logger.info(f"Executing MARKET {binance_side} for {qty} BTC")
            
            order = await self.client.futures_create_order(
                symbol=self.symbol, side=binance_side, type='MARKET', quantity=qty
            )
            
            executed_price = float(order['avgPrice']) if 'avgPrice' in order and float(order['avgPrice']) > 0 else current_price
            sl_dist = atr_prev * self.atr_multiplier
            tp_dist = atr_prev * self.tp_multiplier
            
            if side == "LONG":
                ts = executed_price - sl_dist
                tp = executed_price + tp_dist
            else:
                ts = executed_price + sl_dist
                tp = executed_price - tp_dist
                
            self.state.update({
                "bot_status": side, "entry_price": executed_price, 
                "qty": qty, "trailing_stop": ts, "take_profit": tp
            })
            await self._save_state()
            logger.info(f"Entered {side} at {executed_price}. TS: {ts}, TP: {tp}")
            
        except BinanceAPIException as e:
            logger.error(f"Entry Error: {e.message}")

    async def _execute_market_exit(self, side: str, reason: str):
        pos_qty = self.state['qty']
        logger.info(f"Closing {side} position. Reason: {reason}")
        try:
            close_side = 'SELL' if side == "LONG" else 'BUY'
            
            await self.client.futures_create_order(
                symbol=self.symbol, side=close_side, type='MARKET', 
                quantity=pos_qty, reduceOnly=True
            )
            
            self.state.update({"bot_status": "IDLE", "entry_price": 0.0, "qty": 0.0, "trailing_stop": 0.0, "take_profit": 0.0})
            await self._save_state()
            logger.info("Position closed successfully.")
            
        except BinanceAPIException as e:
            logger.error(f"Exit Error: {e.message}")

    async def shutdown(self):
        self.is_running = False
        if self.client: 
            await self.client.close_connection()
        logger.info("System shutting down.")

async def main():
    bot = FuturesLiveTradingSystem(symbol="BTCUSDT")
    await bot.init_client()
    try:
        await bot.start_trading_loop()
    except KeyboardInterrupt:
        pass
    finally:
        await bot.shutdown()

if __name__ == "__main__":
    try:
        asyncio.run(main())

    except KeyboardInterrupt:
        logger.info('bot closed')
