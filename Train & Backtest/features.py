# features.py
import numpy as np
import pandas as pd

def calculate_adx(df, period=14):
    df = df.copy()
    high, low, prev_close = df['High'], df['Low'], df['Close'].shift(1)
    
    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)
    df['TR'] = np.maximum(tr1, np.maximum(tr2, tr3))
    
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    df['+DM'] = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    df['-DM'] = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    alpha = 1 / period
    df['TR_smooth'] = df['TR'].ewm(alpha=alpha, adjust=False).mean()
    df['+DM_smooth'] = df['+DM'].ewm(alpha=alpha, adjust=False).mean()
    df['-DM_smooth'] = df['-DM'].ewm(alpha=alpha, adjust=False).mean()
    
    df['+DI'] = 100 * (df['+DM_smooth'] / df['TR_smooth'])
    df['-DI'] = 100 * (df['-DM_smooth'] / df['TR_smooth'])
    df['DX'] = 100 * (abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI']))
    df['ADX_14'] = df['DX'].ewm(alpha=alpha, adjust=False).mean()
    
    df.drop(['TR', '+DM', '-DM', 'TR_smooth', '+DM_smooth', '-DM_smooth', 'DX', '+DI', '-DI'], axis=1, inplace=True)
    return df

def generate_custom_features(df):
    df['Dist_EMA7_pct']   = (df['Close'] - df['EMA_7'])   / df['Close']
    df['Dist_EMA14_pct']  = (df['Close'] - df['EMA_14'])  / df['Close']
    df['Dist_EMA21_pct']  = (df['Close'] - df['EMA_21'])  / df['Close']
    df['Dist_EMA50_pct']  = (df['Close'] - df['EMA_50'])  / df['Close']
    df['Dist_EMA200_pct'] = (df['Close'] - df['EMA_200']) / df['Close']
    
    df['EMA_7_21_ratio']   = df['EMA_7']  / df['EMA_21'] - 1
    df['EMA_21_50_ratio']  = df['EMA_21'] / df['EMA_50'] - 1
    df['EMA_50_200_ratio'] = df['EMA_50'] / df['EMA_200'] - 1
    
    df['EMA_21_slope'] = df['EMA_21'].pct_change(5)
    df['EMA_200_slope'] = df['EMA_200'].pct_change(20)
    
    df['ATR_pct'] = df['ATR_14'] / df['Close']
    df['Return_5d'] = df['Close'].pct_change(5)
    df['Return_10d'] = df['Close'].pct_change(10)
    df['Return_20d'] = df['Close'].pct_change(20)
    
    df['Dist_from_20d_high'] = (df['Close'] - df['High'].rolling(20).max()) / df['Close']
    df['Dist_from_20d_low']  = (df['Close'] - df['Low'].rolling(20).min())  / df['Close']
    
    df = calculate_adx(df, period=14)
    df = df.drop(columns=['EMA_7','EMA_14','EMA_21','EMA_50','EMA_200'])
    return df
