import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple, Any, Union
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ===================================================================
# کلاس پایه (BaseBacktester)
# شامل متدهای مشترک برای محاسبه معیارها و رسم نمودارها
# ===================================================================
class BaseBacktester:
    """
    کلاس پایه برای بک‌تست‌های تک‌دارایی و چند-دارایی.
    زیرکلاس‌ها باید متدهای زیر را پیاده‌سازی کنند:
        - run()
        - _get_symbols() -> List[str]
        - _get_dataframe(symbol: Optional[str] = None) -> pd.DataFrame
        - _get_price_series(symbol: Optional[str] = None) -> pd.Series
        - _get_dates() -> Union[pd.DatetimeIndex, pd.RangeIndex]
    """
    def __init__(self,
                 initial_capital: float = 1000.0,
                 leverage: float = 3.0,
                 taker_fee: float = 0.0005,
                 daily_funding_rate: float = 0.0001,
                 atr_multiplier: float = 3.0,
                 tp_multiplier: float = 2.0,
                 risk_per_trade: float = 0.07,
                 regime_sma_period: int = 70,
                 regime_filter_enabled: bool = True,
                 counter_trend_size_mult: float = 0.0):
        
        self.initial_capital = initial_capital
        self.leverage = leverage
        self.taker_fee = taker_fee
        self.daily_funding_rate = daily_funding_rate
        self.atr_multiplier = atr_multiplier
        self.tp_multiplier = tp_multiplier
        self.risk_per_trade = risk_per_trade
        self.regime_sma_period = regime_sma_period
        self.regime_filter_enabled = regime_filter_enabled
        self.counter_trend_size_mult = counter_trend_size_mult

        # متغیرهای مشترک
        self.equity_curve = []
        self.trades_log = []
        self.cash = initial_capital
        self.positions = {}  # برای چند-دارایی استفاده می‌شود (در تک‌دارایی خالی می‌ماند)
        self.benchmark_symbol = None  # در زیرکلاس تنظیم شود

    # ==================== متدهای انتزاعی (باید در زیرکلاس پیاده‌سازی شوند) ====================
    def run(self):
        raise NotImplementedError("Subclasses must implement run()")

    def _get_symbols(self) -> List[str]:
        raise NotImplementedError

    def _get_dataframe(self, symbol: Optional[str] = None) -> pd.DataFrame:
        """بازگرداندن DataFrame مربوط به یک نماد (در تک‌دارایی symbol=None)"""
        raise NotImplementedError

    def _get_price_series(self, symbol: Optional[str] = None) -> pd.Series:
        """بازگرداندن سری قیمت Close برای یک نماد"""
        df = self._get_dataframe(symbol)
        return df['Close']

    def _get_dates(self) -> Union[pd.DatetimeIndex, pd.RangeIndex]:
        """بازگرداندن ایندکس زمانی (از اولین DataFrame)"""
        raise NotImplementedError

    # ==================== متدهای کمکی مشترک ====================
    def _format_xaxis(self, ax, dates):
        if isinstance(dates, pd.DatetimeIndex):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.tick_params(colors='white', labelsize=9)

    def _get_total_equity(self, current_prices: Dict[str, float]) -> float:
        """محاسبه کل حقوق صاحبان سهام (برای چند-دارایی)"""
        unrealized_sum = 0.0
        for sym, pos in self.positions.items():
            price = current_prices.get(sym, np.nan)
            if np.isnan(price):
                continue
            if pos['direction'] == 1:
                unrealized = pos['position_size_usd'] * (price - pos['entry_price']) / pos['entry_price']
            else:
                unrealized = pos['position_size_usd'] * (pos['entry_price'] - price) / pos['entry_price']
            unrealized_sum += unrealized
        return self.cash + unrealized_sum

    # ==================== متدهای محاسبه معیارها ====================
    def calculate_alpha_beta(self, market_benchmark: Optional[pd.Series] = None):
        if not self.equity_curve:
            print("❌ No equity data.")
            return None, None

        # تعیین بنچمارک: اگر داده نشده، از اولین نماد استفاده کن
        if market_benchmark is None:
            benchmark = self._get_price_series(self._get_symbols()[0])
        else:
            benchmark = market_benchmark

        equity_series = pd.Series(self.equity_curve)
        # هم‌ترازسازی طول
        if len(equity_series) == len(benchmark) - 1:
            benchmark = benchmark.iloc[1:]
        elif len(equity_series) != len(benchmark):
            min_len = min(len(equity_series), len(benchmark))
            benchmark = benchmark.iloc[:min_len]
            equity_series = equity_series.iloc[:min_len]

        strategy_returns = equity_series.pct_change().fillna(0)
        benchmark_returns = benchmark.pct_change().fillna(0)

        cov = np.cov(strategy_returns, benchmark_returns)[0, 1]
        var = np.var(benchmark_returns)
        beta = cov / var if var != 0 else 0
        avg_strat = strategy_returns.mean() * 365
        avg_bench = benchmark_returns.mean() * 365
        alpha = avg_strat - beta * avg_bench

        print(f"📊 Alpha: {alpha:+.4f}, Beta: {beta:.4f}")
        return alpha, beta

    def calculate_professional_metrics(self, market_benchmark: Optional[pd.Series] = None):
        """محاسبه معیارهای عملکردی (قابل override در زیرکلاس‌ها)"""
        if not self.equity_curve:
            print("❌ No equity data.")
            return None

        trades_df = pd.DataFrame(self.trades_log)
        equity_series = pd.Series(self.equity_curve)
        daily_returns = equity_series.pct_change().fillna(0)

        total_return = (equity_series.iloc[-1] / self.initial_capital - 1) * 100
        total_days = len(self.equity_curve)
        annualized_return = ((equity_series.iloc[-1] / self.initial_capital) ** (365 / total_days) - 1) * 100

        rolling_max = equity_series.cummax()
        drawdowns = (equity_series - rolling_max) / rolling_max
        max_dd = drawdowns.min() * 100

        std_dev = daily_returns.std()
        sharpe = (np.sqrt(365) * daily_returns.mean() / std_dev) if std_dev != 0 else 0

        negative_returns = daily_returns[daily_returns < 0]
        downside_std = negative_returns.std()
        sortino = (np.sqrt(365) * daily_returns.mean() / downside_std) if downside_std != 0 else 0

        calmar = annualized_return / abs(max_dd) if max_dd != 0 else np.inf

        # معیارهای مرتبط با معاملات
        if len(trades_df) > 0:
            exposure_days = trades_df['Holding_Days'].sum()  # تقریبی
            exposure_pct = (exposure_days / total_days) * 100
            gross_profits = trades_df[trades_df['PnL_USD'] > 0]['PnL_USD'].sum()
            gross_losses = abs(trades_df[trades_df['PnL_USD'] < 0]['PnL_USD'].sum())
            profit_factor = gross_profits / gross_losses if gross_losses != 0 else np.inf
            win_rate = (len(trades_df[trades_df['PnL_USD'] > 0]) / len(trades_df)) * 100
            avg_win = trades_df[trades_df['PnL_USD'] > 0]['PnL_USD'].mean() if len(trades_df[trades_df['PnL_USD'] > 0]) > 0 else 0
            avg_loss = abs(trades_df[trades_df['PnL_USD'] < 0]['PnL_USD'].mean()) if len(trades_df[trades_df['PnL_USD'] < 0]) > 0 else 0
            expectancy = ((win_rate/100) * avg_win) - (((100 - win_rate)/100) * avg_loss)
            avg_holding = trades_df['Holding_Days'].mean()
        else:
            exposure_pct = 0
            profit_factor = 0
            win_rate = 0
            expectancy = 0
            avg_holding = 0
            exposure_days = 0

        alpha, beta = self.calculate_alpha_beta(market_benchmark)

        # چاپ گزارش
        print("\n" + "="*70)
        print("        🏆  INSTITUTIONAL QUANT PERFORMANCE REPORT  🏆")
        print("="*70)
        print(f"💰 Initial Capital         : ${self.initial_capital:,.2f}")
        print(f"📈 Final Portfolio Value   : ${equity_series.iloc[-1]:,.2f} ({total_return:+.2f}%)")
        print(f"📊 Annualized Return       : {annualized_return:+.2f}%")
        print(f"⏱️  Market Exposure Time    : {exposure_pct:.2f}% ({exposure_days} out of {total_days} days)")
        print("-"*70)
        print(f"🔄 Total Trades            : {len(trades_df)}")
        print(f"🎯 Win Rate                : {win_rate:.2f}%")
        print(f"📈 Avg Holding Duration    : {avg_holding:.1f} Days" if len(trades_df) > 0 else "📈 Avg Holding Duration    : N/A")
        print(f"⚖️ Profit Factor           : {profit_factor:.2f}")
        print(f"📊 System Expectancy       : ${expectancy:+.2f} per trade")
        print("-"*70)
        print(f"🔴 Max Drawdown (MDD)      : {max_dd:.2f}%")
        print(f"⚡ Annualized Sharpe Ratio  : {sharpe:.2f}")
        print(f"📉 Annualized Sortino Ratio: {sortino:.2f}")
        print(f"📐 Calmar Ratio            : {calmar:.2f}")
        print("-"*70)
        print(f"📈 Alpha (vs Benchmark)    : {alpha:+.4f}")
        print(f"📉 Beta  (vs Benchmark)    : {beta:.4f}")
        print("="*70 + "\n")

        return trades_df

    # ==================== متدهای رسم نمودار ====================
    def plot_results(self, figsize: Tuple[int, int] = (18, 14),
                     save_path: Optional[str] = None) -> None:
        """داشبورد کامل شامل Equity، Drawdown، توزیع بازده و Heatmap ماهانه"""
        trades_df = pd.DataFrame(self.trades_log)
        if len(trades_df) == 0:
            print("⚠️ No trades to plot!")
            return

        plt.style.use('dark_background')
        sns.set_palette("husl")

        fig = plt.figure(figsize=figsize, facecolor='#0a0a0a', constrained_layout=True)
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.25,
                              top=0.92, bottom=0.08, left=0.08, right=0.92)

        dates = self._get_dates()
        equity_series = pd.Series(self.equity_curve)

        # ========== 1. Equity Curve ==========
        ax1 = fig.add_subplot(gs[0:2, 0:2], facecolor='#1a1a2e')
        ax1.plot(dates, equity_series, color='#00ff88', linewidth=2.5, label='Portfolio Equity')
        ax1.fill_between(dates, self.initial_capital, equity_series,
                         where=(equity_series >= self.initial_capital),
                         color='#00ff88', alpha=0.15, interpolate=True)
        ax1.fill_between(dates, self.initial_capital, equity_series,
                         where=(equity_series < self.initial_capital),
                         color='#ff4444', alpha=0.15, interpolate=True)
        ax1.axhline(y=self.initial_capital, color='white', linestyle='--',
                    linewidth=1, alpha=0.5, label=f'Initial Capital: ${self.initial_capital:,.0f}')

        # نمایش نقاط ورود/خروج (اگر trade log موجود باشد)
        if len(trades_df) > 0:
            entry_indices = []
            for entry_date in trades_df['Entry_Date']:
                if isinstance(entry_date, pd.Timestamp):
                    try:
                        idx = dates.get_loc(entry_date)
                        entry_indices.append(idx)
                    except:
                        closest_idx = np.argmin(np.abs(dates - entry_date))
                        entry_indices.append(closest_idx)
                else:
                    entry_indices.append(int(entry_date) - 1)

            win_mask = trades_df['PnL_USD'] > 0
            loss_mask = trades_df['PnL_USD'] < 0
            win_indices = [entry_indices[i] for i in range(len(entry_indices)) if win_mask.iloc[i]]
            loss_indices = [entry_indices[i] for i in range(len(entry_indices)) if loss_mask.iloc[i]]

            if win_indices:
                win_dates = [dates[idx] for idx in win_indices if idx < len(dates)]
                ax1.scatter(win_dates, [self.initial_capital]*len(win_dates),
                            color='#00ff88', s=80, alpha=0.7, marker='^',
                            edgecolors='white', linewidth=1,
                            label=f'Win Trades ({len(win_dates)})')
            if loss_indices:
                loss_dates = [dates[idx] for idx in loss_indices if idx < len(dates)]
                ax1.scatter(loss_dates, [self.initial_capital]*len(loss_dates),
                            color='#ff4444', s=80, alpha=0.7, marker='v',
                            edgecolors='white', linewidth=1,
                            label=f'Loss Trades ({len(loss_dates)})')

        ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:,.0f}'))
        self._format_xaxis(ax1, dates)
        ax1.grid(True, alpha=0.15, linestyle='--')
        ax1.set_title('Portfolio Equity & Trade Performance', fontsize=14, fontweight='bold', color='white')
        ax1.set_ylabel('Portfolio Value ($)', fontsize=11, color='white')
        ax1.legend(loc='upper left', facecolor='#1a1a2e', edgecolor='white', fontsize=9)

        # ========== 2. Drawdown ==========
        ax2 = fig.add_subplot(gs[2, 0:2], facecolor='#1a1a2e')
        rolling_max = equity_series.cummax()
        drawdown_series = (equity_series - rolling_max) / rolling_max * 100
        ax2.fill_between(dates, 0, drawdown_series, color='#ff4444', alpha=0.6)
        ax2.plot(dates, drawdown_series, color='#ff6666', linewidth=1.5)
        ax2.axhline(y=0, color='white', linestyle='-', linewidth=0.5, alpha=0.5)
        max_dd_idx = drawdown_series.idxmin()
        max_dd_val = drawdown_series.min()
        ax2.scatter([max_dd_idx], [max_dd_val], color='yellow', s=100,
                    zorder=5, label=f'Max DD: {max_dd_val:.1f}%')
        self._format_xaxis(ax2, dates)
        ax2.grid(True, alpha=0.15, linestyle='--')
        ax2.set_title('Drawdown', fontsize=13, fontweight='bold', color='white')
        ax2.set_ylabel('Drawdown (%)', fontsize=11, color='white')
        ax2.legend(loc='lower left', facecolor='#1a1a2e', edgecolor='white', fontsize=9)
        ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.0f}%'))

        # ========== 3. توزیع بازده ==========
        ax3 = fig.add_subplot(gs[0:2, 2], facecolor='#1a1a2e')
        returns = trades_df['Return_Pct'].dropna()
        if len(returns) > 0:
            bins = min(20, len(np.unique(returns)))
            n, bins_patch, patches = ax3.hist(returns, bins=bins, color='#00aaff',
                                              alpha=0.7, edgecolor='white', linewidth=0.5)
            for i, (patch, bin_edge) in enumerate(zip(patches, bins_patch[:-1])):
                if bin_edge >= 0:
                    patch.set_facecolor('#00ff88')
                    patch.set_alpha(0.8)
                else:
                    patch.set_facecolor('#ff4444')
                    patch.set_alpha(0.8)
            mean_return = returns.mean()
            ax3.axvline(x=mean_return, color='yellow', linestyle='--', linewidth=2,
                        label=f'Mean: {mean_return:.1f}%')
        ax3.tick_params(colors='white', labelsize=10)
        ax3.grid(True, alpha=0.15, linestyle='--', axis='x')
        ax3.set_title('Return Distribution', fontsize=13, fontweight='bold', color='white')
        ax3.set_xlabel('Return per Trade (%)', fontsize=11, color='white')
        ax3.set_ylabel('Frequency', fontsize=11, color='white')
        ax3.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.0f}%'))
        ax3.legend(loc='upper right', facecolor='#1a1a2e', edgecolor='white', fontsize=9)

        # ========== 4. Cumulative Return ==========
        ax4 = fig.add_subplot(gs[2, 2], facecolor='#1a1a2e')
        cum_returns = (equity_series / self.initial_capital - 1) * 100
        ax4.plot(dates, cum_returns, color='#ffaa00', linewidth=2, label='Cumulative Return')
        ax4.fill_between(dates, 0, cum_returns, color='#ffaa00', alpha=0.2)
        ax4.axhline(y=0, color='white', linestyle='-', linewidth=0.5, alpha=0.5)
        ax4.tick_params(colors='white', labelsize=9)
        ax4.grid(True, alpha=0.15, linestyle='--')
        ax4.set_title('Cumulative Return', fontsize=12, fontweight='bold', color='white')
        ax4.set_ylabel('Return (%)', fontsize=10, color='white')
        self._format_xaxis(ax4, dates)
        ax4.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.0f}%'))
        ax4.legend(loc='upper left', facecolor='#1a1a2e', edgecolor='white', fontsize=8)

        # ========== 5. Monthly Heatmap ==========
        if isinstance(dates, pd.DatetimeIndex):
            equity_ts = pd.Series(equity_series.values, index=dates)
            monthly_returns = equity_ts.resample('M').last().pct_change().dropna()
            if len(monthly_returns) > 0:
                monthly_df = pd.DataFrame({
                    'Year': monthly_returns.index.year,
                    'Month': monthly_returns.index.month,
                    'Return': monthly_returns.values * 100
                })
                heatmap_data = monthly_df.pivot(index='Year', columns='Month', values='Return')
                if not heatmap_data.empty:
                    ax_heat = fig.add_subplot(gs[3, :], facecolor='#1a1a2e')
                    im = ax_heat.imshow(heatmap_data.values, cmap='RdYlGn', aspect='auto',
                                        interpolation='nearest', vmin=-10, vmax=10)
                    ax_heat.set_xticks(np.arange(12))
                    ax_heat.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                             'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                                            color='white', fontsize=9)
                    years = heatmap_data.index
                    n_years = len(years)
                    if n_years > 10:
                        step = max(1, n_years // 10)
                        tick_indices = np.arange(0, n_years, step)
                        ax_heat.set_yticks(tick_indices)
                        ax_heat.set_yticklabels([years[i] for i in tick_indices], color='white', fontsize=8)
                    else:
                        ax_heat.set_yticks(np.arange(n_years))
                        ax_heat.set_yticklabels(years, color='white', fontsize=9)
                    ax_heat.set_title('Monthly Returns Heatmap (%)', fontsize=13,
                                      fontweight='bold', color='white', pad=15)
                    for i in range(len(heatmap_data.index)):
                        for j in range(12):
                            if not np.isnan(heatmap_data.iloc[i, j]):
                                ax_heat.text(j, i, f'{heatmap_data.iloc[i, j]:.0f}',
                                             ha='center', va='center',
                                             color='black' if abs(heatmap_data.iloc[i, j]) > 5 else 'white',
                                             fontsize=8, fontweight='bold')
                    cbar = plt.colorbar(im, ax=ax_heat, shrink=0.8, pad=0.02)
                    cbar.ax.yaxis.set_tick_params(color='white', labelsize=9)
                    plt.setp(plt.getp(cbar.ax, 'yticklabels'), color='white')
                    cbar.set_label('Return (%)', color='white', fontsize=10)

        # ========== 6. متن اطلاعات ==========
        total_return = (equity_series.iloc[-1] / self.initial_capital - 1) * 100
        fig.text(0.02, 0.01,
                 f"Total Trades: {len(trades_df)} | Win Rate: {len(trades_df[trades_df['PnL_USD'] > 0])/len(trades_df)*100:.1f}% | "
                 f"Total Return: {total_return:+.1f}% | Final Equity: ${equity_series.iloc[-1]:,.2f}",
                 color='white', fontsize=10, alpha=0.7)

        plt.suptitle('Institutional Backtest Analysis Dashboard',
                     fontsize=18, fontweight='bold', color='white', y=0.98)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#0a0a0a')
            print(f"Chart saved to: {save_path}")
        else:
            plt.show()
        plt.close()

    def plot_equity_only(self, figsize: Tuple[int, int] = (14, 8),
                         save_path: Optional[str] = None) -> None:
        """نمودار ساده Equity و Drawdown"""
        if not self.equity_curve:
            print("⚠️ No equity data to plot.")
            return

        plt.style.use('dark_background')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize,
                                       gridspec_kw={'height_ratios': [2, 1]},
                                       facecolor='#0a0a0a')

        dates = self._get_dates()
        equity_series = pd.Series(self.equity_curve)

        ax1.plot(dates, equity_series, color='#00ff88', linewidth=2.5)
        ax1.fill_between(dates, self.initial_capital, equity_series,
                         where=(equity_series >= self.initial_capital),
                         color='#00ff88', alpha=0.2)
        ax1.fill_between(dates, self.initial_capital, equity_series,
                         where=(equity_series < self.initial_capital),
                         color='#ff4444', alpha=0.2)
        ax1.axhline(y=self.initial_capital, color='white', linestyle='--', alpha=0.5)
        ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:,.0f}'))
        self._format_xaxis(ax1, dates)
        ax1.grid(True, alpha=0.15)
        ax1.set_title('Portfolio Equity Curve', fontsize=16, fontweight='bold', color='white')
        ax1.set_ylabel('Portfolio Value ($)', fontsize=12, color='white')

        rolling_max = equity_series.cummax()
        drawdown_series = (equity_series - rolling_max) / rolling_max * 100
        ax2.fill_between(dates, 0, drawdown_series, color='#ff4444', alpha=0.6)
        ax2.plot(dates, drawdown_series, color='#ff6666', linewidth=1.5)
        self._format_xaxis(ax2, dates)
        ax2.grid(True, alpha=0.15)
        ax2.set_title('Drawdown', fontsize=14, fontweight='bold', color='white')
        ax2.set_ylabel('Drawdown (%)', fontsize=12, color='white')
        ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.1f}%'))

        plt.suptitle('Backtest Results', fontsize=18, fontweight='bold', color='white', y=0.98)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#0a0a0a')
            print(f"Chart saved to: {save_path}")
        else:
            plt.show()
        plt.close()

    def plot_trade_analysis(self, figsize: Tuple[int, int] = (14, 8),
                            save_path: Optional[str] = None) -> None:
        """نمودارهای تحلیلی معاملات (PnL cumulative، PnL per trade، توزیع مدت، دلایل خروج)"""
        trades_df = pd.DataFrame(self.trades_log)
        if len(trades_df) == 0:
            print("No trades to plot!")
            return

        plt.style.use('dark_background')
        fig, axes = plt.subplots(2, 2, figsize=figsize, facecolor='#0a0a0a')
        fig.suptitle('Trade Analysis Dashboard', fontsize=18, fontweight='bold',
                     color='white', y=0.98)

        # Cumulative PnL
        cum_pnl = trades_df['PnL_USD'].cumsum()
        ax = axes[0, 0]
        ax.plot(range(len(cum_pnl)), cum_pnl, color='#00aaff', linewidth=2.5)
        ax.fill_between(range(len(cum_pnl)), 0, cum_pnl, alpha=0.3, color='#00aaff')
        ax.axhline(y=0, color='white', linestyle='--', alpha=0.5)
        ax.tick_params(colors='white', labelsize=10)
        ax.grid(True, alpha=0.15)
        ax.set_title('Cumulative PnL', fontsize=13, fontweight='bold', color='white')
        ax.set_xlabel('Trade Number', fontsize=10, color='white')
        ax.set_ylabel('PnL ($)', fontsize=10, color='white')
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:,.0f}'))

        # PnL per Trade
        ax = axes[0, 1]
        colors = ['#00ff88' if x > 0 else '#ff4444' for x in trades_df['PnL_USD']]
        ax.bar(range(len(trades_df)), trades_df['PnL_USD'], color=colors, alpha=0.8)
        ax.axhline(y=0, color='white', linestyle='-', linewidth=0.5)
        ax.tick_params(colors='white', labelsize=10)
        ax.grid(True, alpha=0.15, axis='y')
        ax.set_title('PnL per Trade', fontsize=13, fontweight='bold', color='white')
        ax.set_xlabel('Trade Number', fontsize=10, color='white')
        ax.set_ylabel('PnL ($)', fontsize=10, color='white')
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:,.0f}'))

        # Holding Days
        ax = axes[1, 0]
        holding_days = trades_df['Holding_Days']
        ax.hist(holding_days, bins=15, color='#ff66aa', alpha=0.7, edgecolor='white')
        ax.axvline(x=holding_days.mean(), color='yellow', linestyle='--', linewidth=2,
                   label=f'Mean: {holding_days.mean():.1f} days')
        ax.tick_params(colors='white', labelsize=10)
        ax.grid(True, alpha=0.15, axis='y')
        ax.set_title('Holding Days Distribution', fontsize=13, fontweight='bold', color='white')
        ax.set_xlabel('Days', fontsize=10, color='white')
        ax.set_ylabel('Frequency', fontsize=10, color='white')
        ax.legend(loc='upper right', facecolor='#1a1a2e', edgecolor='white', fontsize=9)

        # Exit Reasons
        ax = axes[1, 1]
        exit_reasons = trades_df['Reason'].value_counts()
        colors_pie = ['#00ff88', '#ffaa00', '#ff4444', '#00aaff', '#ff66aa']
        wedges, texts, autotexts = ax.pie(exit_reasons.values,
                                          labels=exit_reasons.index,
                                          autopct='%1.1f%%',
                                          colors=colors_pie[:len(exit_reasons)],
                                          startangle=90,
                                          textprops={'color': 'white', 'fontsize': 10})
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        ax.set_title('Exit Reasons', fontsize=13, fontweight='bold', color='white')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#0a0a0a')
            print(f"Chart saved to: {save_path}")
        else:
            plt.show()
        plt.close()

    def plot_price_chart_with_trades(self, symbol: Optional[str] = None,
                                     figsize: Tuple[int, int] = (16, 12),
                                     show_sl_tp: bool = True,
                                     save_path: Optional[str] = None) -> None:
        """رسم چارت قیمتی با نمایش معاملات (برای تک‌دارایی یا یک نماد خاص در چند-دارایی)"""
        trades_df = pd.DataFrame(self.trades_log)
        if len(trades_df) == 0:
            print("⚠️ No trades to plot!")
            return

        # تعیین نماد مورد نظر
        symbols = self._get_symbols()
        if symbol is None:
            symbol = symbols[0]
        if symbol not in symbols:
            print(f"⚠️ Symbol {symbol} not found. Available: {symbols}")
            return

        # فیلتر معاملات آن نماد
        if 'Symbol' in trades_df.columns:
            sym_trades = trades_df[trades_df['Symbol'] == symbol]
        else:
            sym_trades = trades_df  # در تک‌دارایی، همه معاملات مربوط به همان نماد است

        if len(sym_trades) == 0:
            print(f"⚠️ No trades for {symbol}.")
            return

        df = self._get_dataframe(symbol)
        dates = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.RangeIndex(len(df))
        price_data = df['Close']

        # استخراج تاریخ ورود/خروج
        entry_dates = []
        exit_dates = []
        entry_prices = []
        exit_prices = []
        trade_types = []
        trade_pnls = []

        for _, row in sym_trades.iterrows():
            entry_date = row['Entry_Date']
            exit_date = row['Exit_Date']
            if isinstance(entry_date, pd.Timestamp):
                entry_idx = dates.get_loc(entry_date) if entry_date in dates else None
            else:
                entry_idx = int(entry_date)
            if isinstance(exit_date, pd.Timestamp):
                exit_idx = dates.get_loc(exit_date) if exit_date in dates else None
            else:
                exit_idx = int(exit_date)

            if entry_idx is not None and exit_idx is not None:
                entry_dates.append(dates[entry_idx])
                exit_dates.append(dates[exit_idx])
                entry_prices.append(price_data.iloc[entry_idx])
                exit_prices.append(price_data.iloc[exit_idx])
                trade_types.append(row['Type'])
                trade_pnls.append(row['PnL_USD'])

        plt.style.use('dark_background')
        fig = plt.figure(figsize=figsize, facecolor='#0a0a0a')
        gs = fig.add_gridspec(3, 1, height_ratios=[2.5, 1, 1], hspace=0.3,
                              top=0.92, bottom=0.08, left=0.08, right=0.92)

        # ========== 1. نمودار قیمت ==========
        ax1 = fig.add_subplot(gs[0, 0], facecolor='#1a1a2e')
        ax1.plot(dates, price_data, color='#88ccff', linewidth=1.5, label='Close Price')

        for i, (ed, ep, typ) in enumerate(zip(entry_dates, entry_prices, trade_types)):
            if typ == 'Long':
                marker = '^'
                color = '#00ff88'
                label = 'Long Entry' if i == 0 else ""
            else:
                marker = 'v'
                color = '#ffaa00'
                label = 'Short Entry' if i == 0 else ""
            ax1.scatter(ed, ep, marker=marker, s=120, color=color, edgecolors='white',
                        linewidth=1.5, zorder=5, label=label)

            exit_color = '#ff4444' if trade_pnls[i] < 0 else '#00ff88'
            ax1.scatter(exit_dates[i], exit_prices[i], marker='o', s=100,
                        facecolors='none', edgecolors=exit_color, linewidth=2,
                        zorder=5, label='Exit' if i == 0 else "")

            if show_sl_tp:
                entry_idx = dates.get_loc(ed)
                atr_entry = df['ATR_14'].iloc[entry_idx]
                open_price = df['Open'].iloc[entry_idx]
                if typ == 'Long':
                    sl_price = open_price - (atr_entry * self.atr_multiplier)
                    tp_price = open_price + (atr_entry * self.tp_multiplier)
                    sl_color = '#ff6666'
                    tp_color = '#66ff66'
                else:
                    sl_price = open_price + (atr_entry * self.atr_multiplier)
                    tp_price = open_price - (atr_entry * self.tp_multiplier)
                    sl_color = '#ff6666'
                    tp_color = '#66ff66'

                exit_idx = dates.get_loc(exit_dates[i])
                x_range = dates[entry_idx:exit_idx+1]
                ax1.hlines(y=sl_price, xmin=x_range[0], xmax=x_range[-1],
                           colors=sl_color, linestyles='--', linewidth=1, alpha=0.7)
                ax1.hlines(y=tp_price, xmin=x_range[0], xmax=x_range[-1],
                           colors=tp_color, linestyles='--', linewidth=1, alpha=0.7)

        ax1.set_title(f'Price Chart with Trades - {symbol}', fontsize=14,
                      fontweight='bold', color='white')
        ax1.set_ylabel('Price ($)', fontsize=11, color='white')
        ax1.tick_params(colors='white', labelsize=10)
        ax1.grid(True, alpha=0.15, linestyle='--')
        ax1.legend(loc='upper left', facecolor='#1a1a2e', edgecolor='white', fontsize=9)

        # ========== 2. Equity Comparison (Strategy vs Buy&Hold) ==========
        ax2 = fig.add_subplot(gs[1, 0], facecolor='#1a1a2e')
        # محاسبه Buy & Hold برای این نماد
        buy_hold_equity = self.initial_capital * (price_data / price_data.iloc[0])
        equity_dates = dates[1:] if len(dates) == len(self.equity_curve) + 1 else dates
        equity_series = pd.Series(self.equity_curve)

        # اگر چند-دارایی است، equity_curve کل است، اما برای مقایسه با یک نماد، بهتر است equity سهم آن نماد را داشته باشیم
        # در چند-دارایی، equity_by_symbol موجود نیست در Base، اما می‌توان از آن استفاده کرد اگر در زیرکلاس تعریف شده باشد
        # در اینجا از equity_curve کل استفاده می‌کنیم (برای تک‌دارایی دقیق است، برای چند-دارایی تقریبی)
        ax2.plot(equity_dates, equity_series, color='#00ff88', linewidth=2.5,
                 label='Strategy Equity (Total)')
        ax2.plot(equity_dates, buy_hold_equity.iloc[1:] if len(dates) == len(self.equity_curve)+1 else buy_hold_equity,
                 color='#ffaa00', linewidth=2, linestyle='--', label='Buy & Hold')
        ax2.axhline(y=self.initial_capital, color='white', linestyle=':',
                    linewidth=1, alpha=0.5, label='Initial Capital')
        ax2.set_title('Equity Comparison: Strategy vs Buy & Hold', fontsize=13,
                      fontweight='bold', color='white')
        ax2.set_ylabel('Portfolio Value ($)', fontsize=11, color='white')
        ax2.tick_params(colors='white', labelsize=10)
        ax2.grid(True, alpha=0.15, linestyle='--')
        ax2.legend(loc='upper left', facecolor='#1a1a2e', edgecolor='white', fontsize=9)
        ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:,.0f}'))

        # ========== 3. Risk per Trade ==========
        ax3 = fig.add_subplot(gs[2, 0], facecolor='#1a1a2e')
        risks = []
        trade_labels = []
        for i, (ed, ep, typ) in enumerate(zip(entry_dates, entry_prices, trade_types)):
            entry_idx = dates.get_loc(ed)
            atr_entry = df['ATR_14'].iloc[entry_idx]
            risk_pct = (atr_entry * self.atr_multiplier) / ep * 100
            risks.append(risk_pct)
            trade_labels.append(f'T{i+1}')

        bars = ax3.bar(trade_labels, risks, color='#ff66aa', alpha=0.7, edgecolor='white')
        ax3.axhline(y=np.mean(risks), color='yellow', linestyle='--', linewidth=2,
                    label=f'Avg Risk: {np.mean(risks):.2f}%')
        ax3.set_title('Risk per Trade (Stop-Loss Distance from Entry)', fontsize=13,
                      fontweight='bold', color='white')
        ax3.set_ylabel('Risk (%)', fontsize=11, color='white')
        ax3.tick_params(colors='white', labelsize=9)
        ax3.grid(True, alpha=0.15, linestyle='--', axis='y')
        ax3.legend(loc='upper right', facecolor='#1a1a2e', edgecolor='white', fontsize=9)

        for bar, risk in zip(bars, risks):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                     f'{risk:.1f}%', ha='center', va='bottom',
                     color='white', fontsize=8)

        plt.suptitle(f'Price Chart with Trade Details - {symbol}',
                     fontsize=18, fontweight='bold', color='white', y=0.98)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#0a0a0a')
            print(f"Chart saved to: {save_path}")
        else:
            plt.show()
        plt.close()


# ===================================================================
# Class SingleBacktester 
# ===================================================================
class SingleBacktester(BaseBacktester):
    def __init__(self, df: pd.DataFrame, **kwargs):
        super().__init__(**kwargs)
        self.df = df.copy()
        self.benchmark_symbol = 'asset'  # فقط یک نماد

        # محاسبه رژیم (اگر فعال باشد)
        if self.regime_filter_enabled:
            if 'SMA_regime' not in self.df.columns:
                self.df['SMA_regime'] = self.df['Close'].rolling(
                    window=self.regime_sma_period, min_periods=self.regime_sma_period // 2
                ).mean()
            self.df['is_bull_regime'] = self.df['Close'] > self.df['SMA_regime']
            self.df['is_bull_regime'] = self.df['is_bull_regime'].fillna(False)
        else:
            self.df['is_bull_regime'] = False

        # متغیرهای مربوط به پوزیشن
        self.position = 0
        self.entry_price = 0.0
        self.position_size_usd = 0.0
        self.trailing_stop = 0.0
        self.take_profit = 0.0
        self.entry_idx = 0

    def _get_symbols(self) -> List[str]:
        return ['asset']

    def _get_dataframe(self, symbol: Optional[str] = None) -> pd.DataFrame:
        return self.df

    def _get_dates(self) -> Union[pd.DatetimeIndex, pd.RangeIndex]:
        if isinstance(self.df.index, pd.DatetimeIndex):
            return self.df.index[1:] if len(self.df) == len(self.equity_curve) + 1 else self.df.index
        else:
            return pd.RangeIndex(len(self.equity_curve))

    def run(self):
        if len(self.df) < 2:
            self.equity_curve = [self.initial_capital]
            print("⚠️ Not enough data to run backtest (need at least 2 rows).")
            return self.equity_curve, pd.DataFrame(self.trades_log)

        self.cash = self.initial_capital
        self.position = 0
        self.equity_curve = []
        self.trades_log = []
        current_trade = None

        for i in range(1, len(self.df)):
            t_prev = i - 1
            t_curr = i

            signal_prev = self.df['Signal'].iloc[t_prev]
            open_price = self.df['Open'].iloc[t_curr]
            high_price = self.df['High'].iloc[t_curr]
            low_price = self.df['Low'].iloc[t_curr]
            close_price = self.df['Close'].iloc[t_curr]
            atr_prev = self.df['ATR_14'].iloc[t_prev]
            is_bull_regime_prev = self.df['is_bull_regime'].iloc[t_prev]
            date_curr = self.df.index[t_curr] if isinstance(self.df.index, pd.DatetimeIndex) else t_curr

            trade_closed = False
            exit_reason = ""
            exit_price = 0.0

            # --- بررسی خروج ---
            if self.position == 1:
                if low_price <= self.trailing_stop and high_price >= self.take_profit:
                    exit_price = min(open_price, self.trailing_stop)
                    trade_closed = True
                    exit_reason = "Trailing Stop (Dual Touch)"
                elif low_price <= self.trailing_stop:
                    exit_price = min(open_price, self.trailing_stop)
                    trade_closed = True
                    exit_reason = "Trailing Stop"
                elif high_price >= self.take_profit:
                    exit_price = max(open_price, self.take_profit)
                    trade_closed = True
                    exit_reason = "Take Profit"
            elif self.position == -1:
                if high_price >= self.trailing_stop and low_price <= self.take_profit:
                    exit_price = max(open_price, self.trailing_stop)
                    trade_closed = True
                    exit_reason = "Trailing Stop (Dual Touch)"
                elif high_price >= self.trailing_stop:
                    exit_price = max(open_price, self.trailing_stop)
                    trade_closed = True
                    exit_reason = "Trailing Stop"
                elif low_price <= self.take_profit:
                    exit_price = min(open_price, self.take_profit)
                    trade_closed = True
                    exit_reason = "Take Profit"

            if not trade_closed and self.position != 0:
                if (self.position == 1 and signal_prev == 0) or (self.position == -1 and signal_prev == 2):
                    exit_price = open_price
                    trade_closed = True
                    exit_reason = "Signal Flip"

            if trade_closed:
                if self.position == 1:
                    price_return = (exit_price - self.entry_price) / self.entry_price
                else:
                    price_return = (self.entry_price - exit_price) / self.entry_price

                pnl_usd = self.position_size_usd * price_return
                exit_fee = self.position_size_usd * (exit_price / self.entry_price) * self.taker_fee
                self.cash += (pnl_usd - exit_fee)

                if current_trade is not None:
                    trade_record = {
                        'Symbol': 'asset',
                        'Entry_Date': self.df.index[self.entry_idx] if isinstance(self.df.index, pd.DatetimeIndex) else self.entry_idx,
                        'Exit_Date': date_curr,
                        'Type': 'Long' if self.position == 1 else 'Short',
                        'Entry_Price': self.entry_price,
                        'Exit_Price': exit_price,
                        'Holding_Days': t_curr - self.entry_idx,
                        'PnL_USD': pnl_usd - exit_fee,
                        'Return_Pct': (pnl_usd - exit_fee) / (self.position_size_usd / self.leverage) * 100,
                        'Reason': exit_reason,
                        'Position_Size_USD': self.position_size_usd,
                        'Leverage': self.leverage,
                        'Entry_Fee': current_trade.get('entry_fee', 0.0),
                        'Exit_Fee': exit_fee,
                        'Total_Fees': current_trade.get('entry_fee', 0.0) + exit_fee,
                        'Stop_Loss': self.trailing_stop,
                        'Take_Profit': self.take_profit,
                        'Risk_Amount_USD': current_trade.get('risk_amount', 0.0),
                        'MAE_pct': current_trade.get('MAE', 0.0),
                        'MFE_pct': current_trade.get('MFE', 0.0),
                        'Return_on_Risk': (pnl_usd - exit_fee) / current_trade.get('risk_amount', 1.0) if current_trade.get('risk_amount', 0) != 0 else 0,
                        'Equity_at_Entry': current_trade.get('equity_at_entry', self.initial_capital),
                        'Equity_at_Exit': self.cash,
                        'Regime_at_Entry': current_trade.get('regime_at_entry', 'Unknown'),
                    }
                    self.trades_log.append(trade_record)
                    current_trade = None

                self.position = 0
                self.position_size_usd = 0.0

            # --- ورود به پوزیشن جدید ---
            if self.position == 0 and self.cash > 0:
                if signal_prev in [0, 2]:
                    sl_distance_usd = atr_prev * self.atr_multiplier
                    sl_pct = sl_distance_usd / open_price

                    risk_amount_usd = self.cash * self.risk_per_trade
                    desired_pos_size_usd = risk_amount_usd / sl_pct
                    max_allowed_pos_usd = self.cash * self.leverage

                    proposed_side = 1 if signal_prev == 2 else -1
                    is_counter_trend = False
                    if self.regime_filter_enabled:
                        if proposed_side == -1 and is_bull_regime_prev:
                            is_counter_trend = True
                        elif proposed_side == 1 and not is_bull_regime_prev:
                            is_counter_trend = True

                    if is_counter_trend:
                        if self.counter_trend_size_mult <= 0.0:
                            desired_pos_size_usd = 0.0
                        else:
                            desired_pos_size_usd *= self.counter_trend_size_mult
                            risk_amount_usd *= self.counter_trend_size_mult

                    self.position_size_usd = min(desired_pos_size_usd, max_allowed_pos_usd)

                    if self.position_size_usd > 0:
                        entry_fee = self.position_size_usd * self.taker_fee
                        self.cash -= entry_fee

                        self.position = proposed_side
                        self.entry_price = open_price
                        self.entry_idx = t_curr

                        if self.position == 1:
                            self.trailing_stop = open_price - sl_distance_usd
                            self.take_profit = open_price + (atr_prev * self.tp_multiplier)
                        else:
                            self.trailing_stop = open_price + sl_distance_usd
                            self.take_profit = open_price - (atr_prev * self.tp_multiplier)

                        current_trade = {
                            'risk_amount': risk_amount_usd,
                            'equity_at_entry': self.cash + entry_fee,
                            'entry_fee': entry_fee,
                            'MAE': 0.0,
                            'MFE': 0.0,
                            'regime_at_entry': 'Bull' if is_bull_regime_prev else 'Bear',
                        }

            # --- به‌روزرسانی trailing stop و funding ---
            if self.position != 0:
                atr_curr = self.df['ATR_14'].iloc[t_curr]
                if self.position == 1:
                    new_stop = close_price - (atr_curr * self.atr_multiplier)
                    self.trailing_stop = max(self.trailing_stop, new_stop)
                    funding_fee = self.position_size_usd * self.daily_funding_rate
                    self.cash -= funding_fee

                    if current_trade is not None:
                        mfe_candidate = (high_price - self.entry_price) / self.entry_price * 100
                        mae_candidate = (self.entry_price - low_price) / self.entry_price * 100
                        current_trade['MFE'] = max(current_trade['MFE'], mfe_candidate)
                        current_trade['MAE'] = min(current_trade['MAE'], -mae_candidate)

                else:
                    new_stop = close_price + (atr_curr * self.atr_multiplier)
                    self.trailing_stop = min(self.trailing_stop, new_stop)
                    simulated_funding_sign = 1 if (i % 10 < 7) else -1
                    funding_fee = self.position_size_usd * self.daily_funding_rate * simulated_funding_sign
                    self.cash += funding_fee

                    if current_trade is not None:
                        mfe_candidate = (self.entry_price - low_price) / self.entry_price * 100
                        mae_candidate = (high_price - self.entry_price) / self.entry_price * 100
                        current_trade['MFE'] = max(current_trade['MFE'], mfe_candidate)
                        current_trade['MAE'] = min(current_trade['MAE'], -mae_candidate)

            # --- محاسبه equity روزانه ---
            if self.position == 1:
                unrealized_return = (close_price - self.entry_price) / self.entry_price
                day_equity = self.cash + (self.position_size_usd * unrealized_return)
            elif self.position == -1:
                unrealized_return = (self.entry_price - close_price) / self.entry_price
                day_equity = self.cash + (self.position_size_usd * unrealized_return)
            else:
                day_equity = self.cash

            self.equity_curve.append(max(0.0, day_equity))

        return self.equity_curve, pd.DataFrame(self.trades_log)


# ===================================================================
# Class MultiBacktester
# ===================================================================
class MultiBacktester(BaseBacktester):
    def __init__(self, dataframes: Dict[str, pd.DataFrame], **kwargs):
        super().__init__(**kwargs)
        self.dataframes = {sym: df.copy() for sym, df in dataframes.items()}
        self.symbols = list(self.dataframes.keys())
        self.benchmark_symbol = self.symbols[0]

        # محاسبه رژیم برای هر دارایی
        if self.regime_filter_enabled:
            for sym, df in self.dataframes.items():
                if 'SMA_regime' not in df.columns:
                    df['SMA_regime'] = df['Close'].rolling(
                        window=self.regime_sma_period, min_periods=self.regime_sma_period // 2
                    ).mean()
                df['is_bull_regime'] = df['Close'] > df['SMA_regime']
                df['is_bull_regime'] = df['is_bull_regime'].fillna(False)
        else:
            for sym, df in self.dataframes.items():
                df['is_bull_regime'] = False

        # متغیرهای مربوط به چند-دارایی
        self.equity_by_symbol = {sym: [] for sym in self.symbols}
        self.daily_exposure = {sym: [] for sym in self.symbols}
        self._validate_indices()

    def _validate_indices(self):
        if len(self.symbols) == 0:
            raise ValueError("حداقل یک DataFrame باید ارائه شود.")
        base_index = self.dataframes[self.symbols[0]].index
        for sym, df in self.dataframes.items():
            if not df.index.equals(base_index):
                print(f"⚠️ ایندکس {sym} با سایرین همخوانی ندارد.")

    def _get_symbols(self) -> List[str]:
        return self.symbols

    def _get_dataframe(self, symbol: Optional[str] = None) -> pd.DataFrame:
        if symbol is None:
            symbol = self.symbols[0]
        return self.dataframes[symbol]

    def _get_dates(self) -> Union[pd.DatetimeIndex, pd.RangeIndex]:
        df0 = self.dataframes[self.symbols[0]]
        if isinstance(df0.index, pd.DatetimeIndex):
            return df0.index[1:] if len(df0) == len(self.equity_curve) + 1 else df0.index
        else:
            return pd.RangeIndex(len(self.equity_curve))

    def _close_position(self, symbol: str, exit_price: float, exit_idx: int,
                        exit_reason: str, current_prices: Dict[str, float]) -> Dict:
        """بستن پوزیشن برای یک نماد (مطابق نسخه چند-دارایی)"""
        pos = self.positions.pop(symbol, None)
        if pos is None:
            return None

        if pos['direction'] == 1:
            price_return = (exit_price - pos['entry_price']) / pos['entry_price']
        else:
            price_return = (pos['entry_price'] - exit_price) / pos['entry_price']

        gross_pnl = pos['position_size_usd'] * price_return
        exit_fee = pos['position_size_usd'] * (exit_price / pos['entry_price']) * self.taker_fee
        net_pnl_reported = gross_pnl - exit_fee

        self.cash += net_pnl_reported  # مارجین قبلاً کسر شده است

        entry_date = self.dataframes[symbol].index[pos['entry_idx']] if isinstance(self.dataframes[symbol].index, pd.DatetimeIndex) else pos['entry_idx']
        exit_date = self.dataframes[symbol].index[exit_idx] if isinstance(self.dataframes[symbol].index, pd.DatetimeIndex) else exit_idx

        trade_record = {
            'Symbol': symbol,
            'Entry_Date': entry_date,
            'Exit_Date': exit_date,
            'Type': 'Long' if pos['direction'] == 1 else 'Short',
            'Entry_Price': pos['entry_price'],
            'Exit_Price': exit_price,
            'Holding_Days': exit_idx - pos['entry_idx'],
            'PnL_USD': net_pnl_reported,
            'Return_Pct': (net_pnl_reported / (pos['position_size_usd'] / self.leverage)) * 100 if (pos['position_size_usd'] / self.leverage) > 0 else 0,
            'Reason': exit_reason,
            'Position_Size_USD': pos['position_size_usd'],
            'Leverage': self.leverage,
            'Entry_Fee': pos['entry_fee'],
            'Exit_Fee': exit_fee,
            'Total_Fees': pos['entry_fee'] + exit_fee,
            'Stop_Loss': pos['initial_stop'],
            'Take_Profit': pos['take_profit'],
            'Risk_Amount_USD': pos['risk_amount'],
            'MAE_pct': pos['mae'],
            'MFE_pct': pos['mfe'],
            'Return_on_Risk': net_pnl_reported / pos['risk_amount'] if pos['risk_amount'] != 0 else 0,
            'Equity_at_Entry': None,
            'Equity_at_Exit': self.cash + sum(self._calc_unrealized(s, current_prices) for s in self.positions),
            'Regime_at_Entry': pos['regime_at_entry'],
        }
        self.trades_log.append(trade_record)
        return trade_record

    def _calc_unrealized(self, symbol: str, current_prices: Dict[str, float]) -> float:
        pos = self.positions.get(symbol)
        if pos is None:
            return 0.0
        price = current_prices.get(symbol, np.nan)
        if np.isnan(price):
            return 0.0
        if pos['direction'] == 1:
            return pos['position_size_usd'] * (price - pos['entry_price']) / pos['entry_price']
        else:
            return pos['position_size_usd'] * (pos['entry_price'] - price) / pos['entry_price']

    def run(self):
        df0 = self.dataframes[self.symbols[0]]
        total_rows = len(df0)
        if total_rows < 2:
            print("⚠️ داده کافی نیست.")
            self.equity_curve = [self.initial_capital]
            return self.equity_curve, pd.DataFrame()

        self.cash = self.initial_capital
        self.positions = {}
        self.equity_curve = []
        self.equity_by_symbol = {sym: [] for sym in self.symbols}
        self.trades_log = []
        self.daily_exposure = {sym: [] for sym in self.symbols}

        for idx in tqdm(range(1, total_rows), desc="Running Multi-Asset Backtest"):
            current_prices = {}
            atr_values = {}
            signals_prev = {}
            bull_regime_prev = {}
            for sym, df in self.dataframes.items():
                current_prices[sym] = df['Close'].iloc[idx]
                atr_values[sym] = df['ATR_14'].iloc[idx]
                signals_prev[sym] = df['Signal'].iloc[idx-1] if idx > 0 else 0
                bull_regime_prev[sym] = df['is_bull_regime'].iloc[idx-1] if idx > 0 else False

            # --- به‌روزرسانی پوزیشن‌های باز ---
            for sym, pos in list(self.positions.items()):
                df = self.dataframes[sym]
                close_price = df['Close'].iloc[idx]
                atr_curr = df['ATR_14'].iloc[idx]
                high = df['High'].iloc[idx]
                low = df['Low'].iloc[idx]

                if pos['direction'] == 1:
                    new_stop = close_price - (atr_curr * self.atr_multiplier)
                    pos['trailing_stop'] = max(pos['trailing_stop'], new_stop)
                    funding_fee = pos['position_size_usd'] * self.daily_funding_rate
                    self.cash -= funding_fee
                else:
                    new_stop = close_price + (atr_curr * self.atr_multiplier)
                    pos['trailing_stop'] = min(pos['trailing_stop'], new_stop)
                    simulated_funding_sign = 1 if (idx % 10 < 7) else -1
                    funding_fee = pos['position_size_usd'] * self.daily_funding_rate * simulated_funding_sign
                    self.cash += funding_fee

                if pos['direction'] == 1:
                    mfe_candidate = (high - pos['entry_price']) / pos['entry_price'] * 100
                    mae_candidate = (pos['entry_price'] - low) / pos['entry_price'] * 100
                else:
                    mfe_candidate = (pos['entry_price'] - low) / pos['entry_price'] * 100
                    mae_candidate = (high - pos['entry_price']) / pos['entry_price'] * 100
                pos['mfe'] = max(pos['mfe'], mfe_candidate)
                pos['mae'] = min(pos['mae'], -mae_candidate)

            # --- بررسی خروج ---
            to_close = []
            for sym, pos in self.positions.items():
                df = self.dataframes[sym]
                open_price = df['Open'].iloc[idx]
                high = df['High'].iloc[idx]
                low = df['Low'].iloc[idx]
                signal_prev = signals_prev[sym]

                exit_price = None
                reason = None
                should_exit = False

                if pos['direction'] == 1:
                    if low <= pos['trailing_stop'] and high >= pos['take_profit']:
                        exit_price = min(open_price, pos['trailing_stop'])
                        reason = "Trailing Stop (Dual Touch)"
                        should_exit = True
                    elif low <= pos['trailing_stop']:
                        exit_price = min(open_price, pos['trailing_stop'])
                        reason = "Trailing Stop"
                        should_exit = True
                    elif high >= pos['take_profit']:
                        exit_price = max(open_price, pos['take_profit'])
                        reason = "Take Profit"
                        should_exit = True
                    elif signal_prev == 0:
                        exit_price = open_price
                        reason = "Signal Flip"
                        should_exit = True
                else:
                    if high >= pos['trailing_stop'] and low <= pos['take_profit']:
                        exit_price = max(open_price, pos['trailing_stop'])
                        reason = "Trailing Stop (Dual Touch)"
                        should_exit = True
                    elif high >= pos['trailing_stop']:
                        exit_price = max(open_price, pos['trailing_stop'])
                        reason = "Trailing Stop"
                        should_exit = True
                    elif low <= pos['take_profit']:
                        exit_price = min(open_price, pos['take_profit'])
                        reason = "Take Profit"
                        should_exit = True
                    elif signal_prev == 2:  # اصلاح شده برای شورت
                        exit_price = open_price
                        reason = "Signal Flip"
                        should_exit = True

                if should_exit:
                    to_close.append((sym, exit_price, reason))

            for sym, exit_price, reason in to_close:
                self._close_position(sym, exit_price, idx, reason, current_prices)

            # --- باز کردن پوزیشن‌های جدید ---
            for sym, df in self.dataframes.items():
                if sym in self.positions:
                    continue

                signal_prev = signals_prev[sym]
                if signal_prev not in [0, 2]:
                    continue

                current_used_margin = sum(p['position_size_usd'] / self.leverage for p in self.positions.values())
                available_cash = self.cash - current_used_margin
                if available_cash <= 0:
                    continue

                open_price = df['Open'].iloc[idx]
                atr_prev = df['ATR_14'].iloc[idx-1] if idx > 0 else df['ATR_14'].iloc[idx]
                sl_distance = atr_prev * self.atr_multiplier
                sl_pct = sl_distance / open_price

                risk_amount = available_cash * self.risk_per_trade
                desired_pos_size = risk_amount / sl_pct
                max_allowed = available_cash * self.leverage
                position_size = min(desired_pos_size, max_allowed)

                if position_size <= 0:
                    continue

                proposed_side = 1 if signal_prev == 2 else -1
                is_counter_trend = False
                if self.regime_filter_enabled:
                    bull_regime = bull_regime_prev[sym]
                    if proposed_side == -1 and bull_regime:
                        is_counter_trend = True
                    elif proposed_side == 1 and not bull_regime:
                        is_counter_trend = True

                if is_counter_trend:
                    if self.counter_trend_size_mult <= 0.0:
                        continue
                    else:
                        position_size *= self.counter_trend_size_mult
                        risk_amount *= self.counter_trend_size_mult

                entry_fee = position_size * self.taker_fee
                margin_required = position_size / self.leverage
                if margin_required + entry_fee > available_cash:
                    continue

                self.cash -= entry_fee

                if proposed_side == 1:
                    initial_stop = open_price - sl_distance
                    take_profit = open_price + (atr_prev * self.tp_multiplier)
                    trailing_stop = initial_stop
                else:
                    initial_stop = open_price + sl_distance
                    take_profit = open_price - (atr_prev * self.tp_multiplier)
                    trailing_stop = initial_stop

                self.positions[sym] = {
                    'direction': proposed_side,
                    'entry_price': open_price,
                    'entry_idx': idx,
                    'position_size_usd': position_size,
                    'initial_stop': initial_stop,
                    'take_profit': take_profit,
                    'trailing_stop': trailing_stop,
                    'entry_fee': entry_fee,
                    'risk_amount': risk_amount,
                    'regime_at_entry': 'Bull' if bull_regime_prev[sym] else 'Bear',
                    'mfe': 0.0,
                    'mae': 0.0,
                }

            # --- محاسبه equity و ثبت ---
            total_equity = self._get_total_equity(current_prices)
            self.equity_curve.append(total_equity)

            for sym in self.symbols:
                if sym in self.positions:
                    equity_contrib = self._calc_unrealized(sym, current_prices)
                else:
                    equity_contrib = 0.0
                self.equity_by_symbol[sym].append(equity_contrib)

            for sym in self.symbols:
                pos = self.positions.get(sym)
                if pos:
                    exposure = (pos['position_size_usd'] / self.leverage) / total_equity * 100 if total_equity > 0 else 0
                else:
                    exposure = 0.0
                self.daily_exposure[sym].append(exposure)

        trades_df = pd.DataFrame(self.trades_log)
        return self.equity_curve, trades_df

    # ==================== متدهای اضافی برای چند-دارایی ====================
    def calculate_professional_metrics(self, market_benchmark: Optional[pd.Series] = None):
        """Override برای اضافه کردن بخش breakdown به تفکیک دارایی"""
        trades_df = super().calculate_professional_metrics(market_benchmark)
        if trades_df is not None and len(trades_df) > 0:
            print("\n" + "="*70)
            print("        📊  ASSET-BY-ASSET BREAKDOWN  📊")
            print("="*70)
            for sym in self.symbols:
                sym_trades = trades_df[trades_df['Symbol'] == sym]
                if len(sym_trades) == 0:
                    print(f"\n{sym}: No trades")
                    continue
                total_pnl = sym_trades['PnL_USD'].sum()
                win_rate_sym = (len(sym_trades[sym_trades['PnL_USD'] > 0]) / len(sym_trades)) * 100
                avg_hold = sym_trades['Holding_Days'].mean()
                profit_factor_sym = sym_trades[sym_trades['PnL_USD'] > 0]['PnL_USD'].sum() / abs(sym_trades[sym_trades['PnL_USD'] < 0]['PnL_USD'].sum()) if len(sym_trades[sym_trades['PnL_USD'] < 0]) > 0 else np.inf
                long_trades = sym_trades[sym_trades['Type'] == 'Long']
                short_trades = sym_trades[sym_trades['Type'] == 'Short']
                print(f"\n{sym}:")
                print(f"  Trades: {len(sym_trades)} | Net PnL: ${total_pnl:+.2f} | Win Rate: {win_rate_sym:.1f}% | Avg Hold: {avg_hold:.1f} days | PF: {profit_factor_sym:.2f}")
                if len(long_trades) > 0:
                    print(f"    Longs: {len(long_trades)} | Win Rate: {len(long_trades[long_trades['PnL_USD']>0])/len(long_trades)*100:.1f}% | Avg Return: {long_trades['Return_Pct'].mean():+.2f}%")
                if len(short_trades) > 0:
                    print(f"    Shorts: {len(short_trades)} | Win Rate: {len(short_trades[short_trades['PnL_USD']>0])/len(short_trades)*100:.1f}% | Avg Return: {short_trades['Return_Pct'].mean():+.2f}%")
            print("="*70 + "\n")
        return trades_df

    def plot_results(self, figsize: Tuple[int, int] = (20, 16),
                     save_path: Optional[str] = None) -> None:
        """Override برای اضافه کردن بخش Equity Contribution و Allocation"""
        if not self.equity_curve:
            print("⚠️ No data to plot. Run backtest first.")
            return

        plt.style.use('dark_background')
        sns.set_palette("husl")

        fig = plt.figure(figsize=figsize, facecolor='#0a0a0a')
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.25,
                              top=0.92, bottom=0.08, left=0.08, right=0.92)

        dates = self._get_dates()
        equity_series = pd.Series(self.equity_curve)

        # ========== 1. Equity Curve کل ==========
        ax1 = fig.add_subplot(gs[0, :2], facecolor='#1a1a2e')
        ax1.plot(dates, equity_series, color='#00ff88', linewidth=2.5, label='Portfolio Equity')
        ax1.fill_between(dates, self.initial_capital, equity_series,
                         where=(equity_series >= self.initial_capital),
                         color='#00ff88', alpha=0.15)
        ax1.fill_between(dates, self.initial_capital, equity_series,
                         where=(equity_series < self.initial_capital),
                         color='#ff4444', alpha=0.15)
        ax1.axhline(y=self.initial_capital, color='white', linestyle='--',
                    linewidth=1, alpha=0.5, label='Initial Capital')
        ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:,.0f}'))
        self._format_xaxis(ax1, dates)
        ax1.grid(True, alpha=0.15, linestyle='--')
        ax1.set_title('Portfolio Equity Curve', fontsize=14, fontweight='bold', color='white')
        ax1.set_ylabel('Portfolio Value ($)', fontsize=11, color='white')
        ax1.legend(loc='upper left', facecolor='#1a1a2e', edgecolor='white', fontsize=9)

        # ========== 2. Equity Contribution by Asset ==========
        ax2 = fig.add_subplot(gs[0, 2], facecolor='#1a1a2e')
        for sym, eq_list in self.equity_by_symbol.items():
            if len(eq_list) == len(dates):
                ax2.plot(dates, eq_list, label=sym, linewidth=1.5)
        ax2.axhline(y=0, color='white', linestyle='--', alpha=0.3)
        ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:,.0f}'))
        self._format_xaxis(ax2, dates)
        ax2.grid(True, alpha=0.15, linestyle='--')
        ax2.set_title('Equity Contribution by Asset', fontsize=13, fontweight='bold', color='white')
        ax2.set_ylabel('Unrealized PnL ($)', fontsize=10, color='white')
        ax2.legend(loc='upper left', facecolor='#1a1a2e', edgecolor='white', fontsize=8)

        # ========== 3. Drawdown کل ==========
        ax3 = fig.add_subplot(gs[1, :2], facecolor='#1a1a2e')
        rolling_max = equity_series.cummax()
        drawdown_series = (equity_series - rolling_max) / rolling_max * 100
        ax3.fill_between(dates, 0, drawdown_series, color='#ff4444', alpha=0.6)
        ax3.plot(dates, drawdown_series, color='#ff6666', linewidth=1.5)
        ax3.axhline(y=0, color='white', linestyle='-', linewidth=0.5, alpha=0.5)
        max_dd_idx = drawdown_series.idxmin()
        max_dd_val = drawdown_series.min()
        ax3.scatter([max_dd_idx], [max_dd_val], color='yellow', s=100,
                    zorder=5, label=f'Max DD: {max_dd_val:.1f}%')
        self._format_xaxis(ax3, dates)
        ax3.grid(True, alpha=0.15, linestyle='--')
        ax3.set_title('Portfolio Drawdown', fontsize=13, fontweight='bold', color='white')
        ax3.set_ylabel('Drawdown (%)', fontsize=11, color='white')
        ax3.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.0f}%'))
        ax3.legend(loc='lower left', facecolor='#1a1a2e', edgecolor='white', fontsize=9)

        # ========== 4. Capital Allocation ==========
        ax4 = fig.add_subplot(gs[1, 2], facecolor='#1a1a2e')
        exposure_data = {}
        for sym, exp_list in self.daily_exposure.items():
            if len(exp_list) == len(dates):
                exposure_data[sym] = exp_list
        if exposure_data:
            df_exp = pd.DataFrame(exposure_data, index=dates)
            df_exp.fillna(0, inplace=True)
            df_exp.plot.area(ax=ax4, alpha=0.7, linewidth=0.5)
            ax4.set_ylabel('Exposure (%)', fontsize=10, color='white')
            ax4.set_title('Capital Allocation Over Time', fontsize=13, fontweight='bold', color='white')
            ax4.grid(True, alpha=0.15, linestyle='--')
            ax4.legend(loc='upper left', facecolor='#1a1a2e', edgecolor='white', fontsize=8)
            ax4.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.0f}%'))

        # ========== 5. Trade Return Distribution ==========
        trades_df = pd.DataFrame(self.trades_log)
        if len(trades_df) > 0:
            ax5 = fig.add_subplot(gs[2, 0], facecolor='#1a1a2e')
            returns = trades_df['Return_Pct'].dropna()
            if len(returns) > 0:
                bins = min(20, len(np.unique(returns)))
                n, bins_patch, patches = ax5.hist(returns, bins=bins, color='#00aaff',
                                                  alpha=0.7, edgecolor='white', linewidth=0.5)
                for i, (patch, bin_edge) in enumerate(zip(patches, bins_patch[:-1])):
                    if bin_edge >= 0:
                        patch.set_facecolor('#00ff88')
                        patch.set_alpha(0.8)
                    else:
                        patch.set_facecolor('#ff4444')
                        patch.set_alpha(0.8)
                mean_return = returns.mean()
                ax5.axvline(x=mean_return, color='yellow', linestyle='--', linewidth=2,
                            label=f'Mean: {mean_return:.1f}%')
            ax5.grid(True, alpha=0.15, linestyle='--', axis='x')
            ax5.set_title('Trade Return Distribution', fontsize=13, fontweight='bold', color='white')
            ax5.set_xlabel('Return per Trade (%)', fontsize=10, color='white')
            ax5.set_ylabel('Frequency', fontsize=10, color='white')
            ax5.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.0f}%'))
            ax5.legend(loc='upper right', facecolor='#1a1a2e', edgecolor='white', fontsize=9)

            # ========== 6. Cumulative PnL ==========
            ax6 = fig.add_subplot(gs[2, 1], facecolor='#1a1a2e')
            cum_pnl = trades_df['PnL_USD'].cumsum()
            ax6.plot(range(len(cum_pnl)), cum_pnl, color='#00aaff', linewidth=2.5)
            ax6.fill_between(range(len(cum_pnl)), 0, cum_pnl, alpha=0.3, color='#00aaff')
            ax6.axhline(y=0, color='white', linestyle='--', alpha=0.5)
            ax6.grid(True, alpha=0.15, linestyle='--')
            ax6.set_title('Cumulative PnL (All Trades)', fontsize=13, fontweight='bold', color='white')
            ax6.set_xlabel('Trade Number', fontsize=10, color='white')
            ax6.set_ylabel('Cumulative PnL ($)', fontsize=10, color='white')
            ax6.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:,.0f}'))

            # ========== 7. Net PnL by Asset ==========
            ax7 = fig.add_subplot(gs[2, 2], facecolor='#1a1a2e')
            pnl_by_sym = trades_df.groupby('Symbol')['PnL_USD'].sum().sort_values()
            colors_bar = ['#00ff88' if x > 0 else '#ff4444' for x in pnl_by_sym.values]
            ax7.barh(pnl_by_sym.index, pnl_by_sym.values, color=colors_bar, alpha=0.8)
            ax7.axvline(x=0, color='white', linestyle='-', linewidth=0.5, alpha=0.5)
            ax7.grid(True, alpha=0.15, linestyle='--', axis='x')
            ax7.set_title('Net PnL by Asset', fontsize=13, fontweight='bold', color='white')
            ax7.set_xlabel('Net PnL ($)', fontsize=10, color='white')
            ax7.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:,.0f}'))

        plt.suptitle('Multi-Asset Portfolio Backtest Dashboard',
                     fontsize=18, fontweight='bold', color='white', y=0.98)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#0a0a0a')
            print(f"Chart saved to: {save_path}")
        else:
            plt.show()
        plt.close()
