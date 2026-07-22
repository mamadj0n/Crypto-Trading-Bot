# data_pipeline.py
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from config import DATA_PATHS, START_CUTOFF, END_CUTOFF, TARGET_THRESHOLD, COLS_TO_DROP, EXCLUDE_COLS_FOR_TRAIN, TRAIN_SPLIT_INDEX
from features import generate_custom_features

class DataPipeline:
    def __init__(self):
        self.scaler = MinMaxScaler()
        
    def load_and_merge(self):
        price = pd.read_csv(DATA_PATHS['price'], parse_dates=["Date"])
        fg = pd.read_csv(DATA_PATHS['fg'], parse_dates=["Date"])
        onchain = pd.read_csv(DATA_PATHS['onchain'], parse_dates=["Date"])
        
        start_date = min(price["Date"].min(), fg["Date"].min(), onchain["Date"].min())
        end_date = max(price["Date"].max(), fg["Date"].max(), onchain["Date"].max())
        df_timeline = pd.DataFrame({"Date": pd.date_range(start=start_date, end=end_date, freq="D")})
        
        df = pd.merge(df_timeline, price, on="Date", how="left")
        df = pd.merge(df, fg, on="Date", how="left")
        df = pd.merge(df, onchain, on="Date", how="left")
        
        df = df.sort_values(by="Date").reset_index(drop=True)
        numerical_cols = df.columns.difference(["Date"])
        df[numerical_cols] = df[numerical_cols].interpolate(method="linear")
        
        df_filtered = df[(df["Date"] >= START_CUTOFF) & (df["Date"] <= END_CUTOFF)].copy()
        df = df_filtered.sort_values(by="Date").reset_index(drop=True).bfill()
        
        fill_cols = ["HashRate", "ActiveAddresses", "TxVolumeUSD", "BB_Signal", "MACD_Signal_Cross", "RSI_Signal"]
        df[fill_cols] = df[fill_cols].ffill().bfill()
        
        return df

    def create_targets_and_clean(self, df):
        df = generate_custom_features(df)
        df = pd.get_dummies(df, columns=['RSI_Signal', 'BB_Signal'], dtype=int)
        df = df.set_index('Date').drop(columns=COLS_TO_DROP)
        
        df["Next_Day_Return"] = df["Close"].shift(-1) / df["Close"] - 1
        df["Target"] = 1  
        df.loc[df["Next_Day_Return"] > TARGET_THRESHOLD, "Target"] = 2  
        df.loc[df["Next_Day_Return"] < -TARGET_THRESHOLD, "Target"] = 0  
        
        return df.dropna(subset=["Next_Day_Return"]).copy()

    def split_and_scale(self, df):
        y = df["Target"]
        X = df.drop(columns=EXCLUDE_COLS_FOR_TRAIN)
        
        X_train, Y_train = X.iloc[:TRAIN_SPLIT_INDEX], y.iloc[:TRAIN_SPLIT_INDEX]
        X_test, Y_test = X.iloc[TRAIN_SPLIT_INDEX:], y.iloc[TRAIN_SPLIT_INDEX:]
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, Y_train, Y_test, X_train.columns
