# Data Paths
DATA_PATHS = {
    'price': 'data/bitcoin_price_.csv',
    'fg': 'data/fear_greed.csv',
    'onchain': 'data/onchain.csv'
}

# Timeline and Splitting Parameters
START_CUTOFF = "2018-02-01"
END_CUTOFF = "2025-12-29"
TRAIN_SPLIT_INDEX = 2400
TARGET_THRESHOLD = 0.02

# Feature Selection
COLS_TO_DROP = [
    "SMA_7", "SMA_14", "SMA_21", "SMA_50", "SMA_200",
    "RSI_7", "RSI_21", "ATR_21",
    "BB_Upper_20", "BB_Lower_20", "BB_Middle_20",
    "BB_Upper_50", "BB_Middle_50", "BB_Lower_50",
    "MACD_Signal_Cross", 'RSI_Signal_Neutral',
    'RSI_Signal_Overbought (Sell)', 'RSI_Signal_Oversold (Buy)',
    'BB_Signal_Above Upper Band (Sell)', 'BB_Signal_Below Lower Band (Buy)'
]

EXCLUDE_COLS_FOR_TRAIN = [
    "Target", "Next_Day_Return", "Open", "High", "Low", "Close", "Volume"
]
