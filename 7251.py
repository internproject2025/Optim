import streamlit as st
import datetime
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats.mstats import gmean
from pyqubo import Array, Constraint, Placeholder
import requests
from io import StringIO
import os, time, pickle
import socketio

st.title("Optimal Portfolio")
# app.py

# --- Streamlit Page Configuration and Styling ---
st.set_page_config(page_title="QUBO Portfolio Optimizer", layout="wide")

# Embed custom CSS for minimalist design
st.markdown("""
<style>
    /* Set background to white */
    .reportview-container .main {
        background-color: #ffffff;
    }
    /* Sidebar background */
    .sidebar .sidebar-content {
        background-color: #f8f9fa; /* Very light gray */
    }
    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        color: #333333; /* Soft black */
    }
    /* Metric labels and values */
    .stMetricLabel {
        font-size: 1rem;
        color: #666666; /* Muted text */
    }
    .stMetricValue {
        font-size: 1.5rem;
        color: #222222; /* Slightly darker for value */
    }
    /* Dataframe styling */
    .stDataFrame {
        border: 1px solid #e0e0e0; /* Light border */
        border-radius: 4px;
    }
    /* Button styling */
    .stButton>button {
        background-color: #0d6efd; /* Muted/Soft Blue */
        color: white;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1rem;
    }
    .stButton>button:hover {
        background-color: #0b5ed7; /* Slightly darker on hover */
    }
    /* Spinner color */
    .stSpinner > div {
        color: #0d6efd; /* Match button color */
    }
    /* Reduce padding slightly for a cleaner look */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 5%;
        padding-right: 5%;
    }
</style>
""", unsafe_allow_html=True)

# --- Core Logic Functions ---

# 超級電腦 API 設定
SERVER_URI = "http://60.250.149.247:8080"
current_solution = None
job_received = False

# SocketIO 客戶端設定
sio = socketio.Client()


# --- SocketIO Event Handlers ---
@sio.on("return solution")
def handle_solution(data):
    global current_solution, job_received
    try:
        result = pickle.loads(data)
        current_solution = np.array(result["sampleset"].record[0][0], dtype=np.float32)
        print(f"🧠 收到解，長度 = {len(current_solution)}")
        job_received = True
    except Exception as e:
        print(f"⚠️ 解析解失敗: {e}")
        current_solution = None
        job_received = True


@sio.on("job finished")
def handle_job_finished(data_dict):
    job_token = data_dict.get("job_token", "N/A")
    print(f"📥 Job finished! Token = {job_token}")
    sio.emit("request solution", job_token)


@sio.on("job error")
def handle_job_error(data):
    global current_solution, job_received
    print(f"⚠️ Quantix 錯誤：{data.get('message', '未知錯誤')}")
    current_solution = None
    job_received = True
    sio.disconnect()


@sio.on("return message")
def handle_message(data):
    try:
        msg = pickle.loads(data)
        print("📩 回應訊息：", msg)
    except Exception as e:
        print(f"⚠️ 回應訊息反序列化失敗：{e}")
    finally:
        sio.disconnect()


# --- Helper Functions ---

def build_qubo_for_server(qubo_dict, timeout_ms=10000):
    return pickle.dumps({
        "qubo": qubo_dict,
        "offset": 0,
        "timeout": timeout_ms
    }, protocol=2)



    
def get_top_10_stocks():
    """
    Returns a curated list of the 10 most valuable and heavily traded US stocks.
    """
    return [
        'AAPL',  
        'MSFT',  
        'AMZN',  
        'NVDA',  
        'GOOGL', 
        'META',  
        'TSLA',  
        'BRK.B',
        'KO',   
        'V', 
        'DIS'
    ]
    
def get_stock_list(etf_type, selected_tickers=None):
    
    if etf_type == 'DOW30':
        # Hard-coded list of current Dow Jones 30 components
        stock_list = [
            'AAPL', 'AMGN', 'AXP', 'BA', 'CAT', 'CRM', 'CSCO', 'CVX', 'DOW', 'GS',
            'HD', 'HON', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM', 'MRK',
            'MSFT', 'NKE', 'PG', 'TRV', 'UNH', 'V', 'VZ', 'WBA', 'WMT', 'DIS'
        ]
    elif etf_type == 'self-picked':
        if selected_tickers and len(selected_tickers) > 0:
            return selected_tickers
        else:
            return ['AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL']  # Fallback: top 5
    elif etf_type == 'QQQ':
        url = "https://www.invesco.com/us/financial-products/etfs/holdings/main/holdings/0?audienceType=Investor&action=download&ticker=QQQ"
        response = requests.get(url)
        response.raise_for_status()
        holdings_df = pd.read_csv(StringIO(response.text))
        stock_list = holdings_df['Holding Ticker'].dropna().tolist()
        stock_list = [ticker.strip() for ticker in stock_list]
    elif etf_type == '0050':
        stock_list = [
    '2330.TW',  # 台積電
    '2317.TW',  # 鴻海
    '2454.TW',  # 聯發科
    '2308.TW',  # 台達電
    '2382.TW',  # 廣達
    '2881.TW',  # 富邦金
    '2891.TW',  # 中信金
    '2882.TW',  # 國泰金
    '2303.TW',  # 聯電
    '2412.TW',  # 中華電
    '2884.TW',  # 玉山金
    '2886.TW',  # 兆豐金
    '3711.TW',  # 日月光投控
    '2357.TW',  # 華碩
    '1216.TW',  # 統一
    '2885.TW',  # 元大金
    '2345.TW',  # 智邦
    '3231.TW',  # 緯創
    '3034.TW',  # 聯詠
    '2892.TW',  # 第一金
    '2379.TW',  # 瑞昱
    '6669.TW',  # 緯穎
    '2890.TW',  # 永豐金
    '5880.TW',  # 合庫金
    '2383.TW',  # 台光電
    '2880.TW',  # 華南金
    '3661.TW',  # 世芯-KY
    '3017.TW',  # 奇鋐
    '2883.TW',  # 凱基金
    '3008.TW',  # 大立光
    '2301.TW',  # 光寶科
    '2603.TW',  # 長榮
    '2002.TW',  # 中鋼
    '5871.TW',  # 中租-KY
    '2327.TW',  # 國巨
    '2887.TW',  # 台新金
    '5876.TW',  # 上海商銀
    '1101.TW',  # 台泥
    '3045.TW',  # 台灣大
    '4938.TW',  # 和碩
    '4904.TW',  # 遠傳
    '2395.TW',  # 研華
    '2207.TW',  # 和泰車
    '1301.TW',  # 台塑
    '2912.TW',  # 統一超
    '6446.TW',  # 藥華藥
    '1303.TW',  # 南亞
    '2609.TW',  # 陽明
    '2615.TW',  # 萬海
    '6505.TW'   # 台塑化
]
    else:
        raise ValueError("Invalid ETF type. Choose from 'SPY500', 'self-picked', 'QQQ'.")
    return stock_list
    



def calculate_max_drawdown(cumulative_values):
    """
    Calculates the Maximum Drawdown (MDD) from a time series of cumulative values.
    """
    if len(cumulative_values) == 0:
        return 0.0

    cumulative_values = np.asarray(cumulative_values)
    running_max = np.maximum.accumulate(cumulative_values)
    drawdowns = (cumulative_values - running_max) / running_max
    drawdowns = np.nan_to_num(drawdowns, nan=0.0, posinf=0.0, neginf=0.0)
    max_dd = np.min(drawdowns)
    return -max_dd


def get_market_performance(start_date, end_date, spy_data):
    """Calculate market performance using SPY data"""
    if not spy_data.empty and len(spy_data) >= 2:
        spy_start = spy_data.iloc[0]
        spy_end = spy_data.iloc[-1]

        if isinstance(spy_start, (pd.Series, pd.DataFrame)):
            spy_start = spy_start.iloc[0] if len(spy_start) > 0 else np.nan
        if isinstance(spy_end, (pd.Series, pd.DataFrame)):
            spy_end = spy_end.iloc[0] if len(spy_end) > 0 else np.nan

        try:
            spy_start = float(spy_start)
            spy_end = float(spy_end)
        except (ValueError, TypeError):
            print("⚠️ SPY data contains non-numeric values.")
            return np.nan

        market_performance = (spy_end / spy_start) - 1
        market_performance = float(market_performance)
        print("✅ Get SPY benchmark")
        return market_performance
    else:
        print("⚠️ SPY data is empty or has insufficient data.")
        return np.nan


def calculate_sharpe_ratio_metrics(selected_indices, selected_weights, returns_df, list_price_start, rf_rate=0.0):
    """
    Calculates Sharpe Ratio and related metrics.
    Returns: (sharpe_ratio, mu_annual, sigma_annual) or (np.nan, np.nan, np.nan) if undefined.
    """
    try:
        selected_indices = np.asarray(selected_indices, dtype=int)
    except (ValueError, TypeError) as e:
        print(f"⚠️ selected_indices 無法轉換為整數陣列: {e}")
        return np.nan, np.nan, np.nan

    if len(selected_indices) == 0:
        print("⚠️ Sharpe ratio undefined (no selected stocks).")
        return np.nan, np.nan, np.nan

    dollar_positions = selected_weights[selected_indices] * list_price_start[selected_indices]

    if dollar_positions.sum() == 0:
        print("⚠️ Sharpe ratio undefined (empty portfolio or zero investment).")
        return np.nan, np.nan, np.nan

    w_frac = dollar_positions / dollar_positions.sum()

    try:
        selected_returns_df = returns_df.iloc[:, selected_indices]
    except IndexError as e:
        print(f"⚠️ 從 returns_df 選取列時索引錯誤: {e}")
        return np.nan, np.nan, np.nan

    port_ret = selected_returns_df.dot(w_frac)
    mu_daily = port_ret.mean()
    sigma_daily = port_ret.std(ddof=1)

    if sigma_daily == 0 or np.isnan(sigma_daily):
        print("⚠️ Sharpe ratio undefined (zero or NaN volatility).")
        return np.nan, np.nan, np.nan

    mu_annual = mu_daily * 252
    sigma_annual = sigma_daily * (252 ** 0.5)
    sharpe = (mu_annual - rf_rate) / sigma_annual if sigma_annual and not np.isnan(sigma_annual) else np.nan
    return sharpe, mu_annual, sigma_annual



def optimize_portfolio(etf_type, start_date, end_date, p_value, q_value, server_uri=None, use_supercomputer=False,
                       selected_tickers=None):  # ← Changed parameter name
    """Optimizes portfolio and returns results dictionary."""
    global SERVER_URI, current_solution, job_received
    # Reset global state for each run
    current_solution = None
    job_received = False

    if server_uri:
        SERVER_URI = server_uri

    try:
        # --- 1. Get Stock List ---
        if etf_type == 'self-picked':
            stock_list = get_stock_list(etf_type, selected_tickers)
        else:
            stock_list = get_stock_list(etf_type)

        st.info(f"Downloading data for {len(stock_list)} stocks...")

        # --- 2. Download Data ---
        data = yf.download(stock_list, start=start_date, end=end_date, auto_adjust=True)["Close"]
        # Filter out tickers that failed to download or have insufficient data
        downloaded_tickers = [col for col in data.columns if data[col].first_valid_index() is not None]
        if not downloaded_tickers:
            st.error("❌ No valid stock data could be downloaded.")
            return None

        data = data[downloaded_tickers]
        data_normalized = data / data.iloc[0]

        spy_data = yf.download('SPY', start=start_date, end=end_date, auto_adjust=True)["Close"]
        spy_data = spy_data.dropna()

        # --- 3. Calculate Returns ---
        returns_data = {}
        for sc in downloaded_tickers:
            prices = data_normalized[sc].dropna()
            if len(prices) > 1:
                returns_data[sc] = np.log(prices / prices.shift(1)).dropna()
        returns_df = pd.DataFrame(returns_data)
        returns_df = returns_df.fillna(0)

        rates = [returns_df[col].values for col in returns_df.columns]
        downloaded_tickers = list(returns_df.columns)
        N = len(downloaded_tickers)

        if N == 0:
            st.error("❌ No stocks available after processing returns.")
            return None

        # Recalculate list_price_start and list_price_end based on the aligned data
        list_price_start = data.loc[returns_df.index[0], downloaded_tickers].values
        list_price_end = data.loc[returns_df.index[-1], downloaded_tickers].values

        # --- 4. Calculate Geometric Mean Returns ---
        w_1 = np.array([gmean(r + 1) for r in rates])
        w = w_1 - 1

        # --- 5. Set up QUBO Model ---
        num_bits = 5
        x = Array.create("x", shape=(N, num_bits), vartype="BINARY")
        weights = np.array([2 ** j for j in range(num_bits)])

        # Define cost functions
        cost_return = sum([-w[i] * sum(weights[j] * x[i, j] for j in range(num_bits)) for i in range(N)])
        cost_risk = sum([
            sum(weights[b1] * x[i, b1] for b1 in range(num_bits)) *
            sum(weights[b2] * x[j, b2] for b2 in range(num_bits)) * np.cov(rates[i], rates[j])[0][1]
            for i in range(N) for j in range(N)
        ])

        p_placeholder = Placeholder('p')
        q_placeholder = Placeholder('q')
        cost_func = p_placeholder * cost_return + q_placeholder * cost_risk
        model = cost_func.compile()

        # Use provided p and q values
        feed_dict = {'p': p_value, 'q': q_value}
        qubo, offset = model.to_qubo(feed_dict=feed_dict)

        # --- 6. Solve QUBO ---
        best_sample_array = None

        if use_supercomputer and SERVER_URI:
            # --- Use Supercomputer ---
            st.info("📤 正在使用超級電腦 API 求解 QUBO...")
            qubo_data = build_qubo_for_server(qubo, timeout_ms=10000)
            current_solution = None
            job_received = False

            try:
                st.write(f"🔗 連接到超級電腦 API: {SERVER_URI}")
                sio.connect(SERVER_URI)
                sio.emit("upload qubo", qubo_data)

                # Wait for solution
                max_wait = 120  # seconds
                wait_interval = 10  # seconds
                for i in range(int(max_wait / wait_interval)):
                    if job_received:
                        break
                    time.sleep(wait_interval)
                    st.write(f"⏳ 等待解答中... ({(i + 1) * wait_interval}s elapsed)")

                if job_received and current_solution is not None:
                    best_sample_array = current_solution.reshape((N, num_bits))
                    st.success("✅ 成功從超級電腦 API 獲得解")
                else:
                    raise Exception("未在時間內從超級電腦 API 收到有效解")

            except Exception as e:
                st.error(f"❌ 超級電腦 API 錯誤：{e}")
                st.info("🔄 回退到本地 OpenJij 求解器...")
                use_supercomputer = False
            finally:
                try:
                    sio.disconnect()
                except:
                    pass

        # --- Local Solver (OpenJij) ---
        if not use_supercomputer or best_sample_array is None:
            st.info("⚠️ 使用本地 OpenJij 求解器...")
            try:
                import openjij as oj
                sampler = oj.SASampler()
                # Convert QUBO for OpenJij
                h = {i: v for (i, j), v in qubo.items() if i == j}
                J = {(i, j): v for (i, j), v in qubo.items() if i != j}
                # Ensure J is upper triangular
                J_upper = {}
                for (i, j), v in J.items():
                    if i <= j:
                        J_upper[(i, j)] = v
                    else:
                        J_upper[(j, i)] = v

                response = sampler.sample_ising(h, J, num_reads=100)
                best_sample_dict = response.first.sample
                # Convert back to array format
                best_sample_array = np.array([
                    [best_sample_dict.get(f'x[{i}][{j}]', 0) for j in range(num_bits)]
                    for i in range(N)
                ])
                st.success("✅ 本地求解器完成")
            except ImportError:
                st.error("❌ OpenJij 未安裝。請運行 'pip install openjij'。無法進行本地計算。")
                return None
            except Exception as e:
                st.error(f"❌ 本地求解器錯誤: {e}")
                import traceback
                st.text(traceback.format_exc())
                return None

        if best_sample_array is None:
            st.error("❌ 無法獲得有效解。")
            return None

        # --- 7. Decode Solution ---
        selected_weights = np.array([
            sum(best_sample_array[i][j] * weights[j] for j in range(num_bits))
            for i in range(N)
        ])
        selected_indices = [i for i in range(N) if selected_weights[i] > 0]
        selected_stocks = [downloaded_tickers[i] for i in selected_indices]

        # --- 8. Calculate Portfolio Performance Metrics ---
        annual_returns = (list_price_end / list_price_start) - 1
        total_investment = sum(
            list_price_start[i] * selected_weights[i] for i in range(N) if selected_weights[i] > 0)

        if total_investment > 0 and selected_indices:
            total_expected_return = sum(
                (list_price_start[i] * selected_weights[i] / total_investment) * annual_returns[i]
                for i in selected_indices
            )
        else:
            total_expected_return = 0.0

        # --- 9. Calculate Sharpe Ratio ---
        sharpe_ratio, mu_annual, sigma_annual = calculate_sharpe_ratio_metrics(
            np.array(selected_indices, dtype=int), selected_weights, returns_df, list_price_start, rf_rate=0.037
        )

        # --- 10. Calculate Maximum Drawdowns (MDD) ---
        # --- 10a. Portfolio MDD ---
        portfolio_mdd = np.nan
        if selected_indices and total_investment > 0:
            try:
                selected_tickers = [downloaded_tickers[i] for i in selected_indices]
               
                selected_prices_df = data[selected_tickers].reindex(returns_df.index).ffill()

                selected_dollar_positions = np.array([list_price_start[i] for i in selected_indices])
                selected_weights_fractional = selected_dollar_positions / selected_dollar_positions.sum()

                portfolio_value_normalized_ts = selected_prices_df.dot(selected_weights_fractional)

                initial_norm_value = portfolio_value_normalized_ts.iloc[0]
                if pd.notna(initial_norm_value) and initial_norm_value != 0:
                    portfolio_value_ts = portfolio_value_normalized_ts * (total_investment / initial_norm_value)
                else:
                    portfolio_value_ts = portfolio_value_normalized_ts * total_investment
                    print("⚠️ Portfolio initial normalized value is zero or NaN, using alternative scaling.")

                portfolio_mdd = calculate_max_drawdown(portfolio_value_ts)
                print(f"✅ Portfolio MDD calculated: {portfolio_mdd:.2%}")
            except Exception as e:
                print(f"⚠️ Error calculating portfolio MDD: {e}")
                import traceback
                traceback.print_exc()
                portfolio_mdd = np.nan
        else:
            print("⚠️ No stocks selected or zero investment, cannot calculate portfolio MDD.")
            portfolio_mdd = np.nan

        # --- 10b. SPY MDD ---
        spy_mdd = np.nan
        if not spy_data.empty:
            try:
                spy_mdd = calculate_max_drawdown(spy_data)
                print(f"✅ SPY MDD calculated: {spy_mdd:.2%}")
            except Exception as e:
                print(f"⚠️ Error calculating SPY MDD: {e}")
                import traceback
                traceback.print_exc()
                spy_mdd = np.nan
        else:
            print("⚠️ SPY data is empty, cannot calculate SPY MDD.")

        # --- 11. Calculate Market Performance ---
        market_performance = get_market_performance(start_date, end_date, spy_data)

        # --- 12. Prepare Weights Data for Display ---
        weights_data = []
        if total_investment > 0 and selected_indices:
            for i in selected_indices:
                weight_pct = (list_price_start[i] * selected_weights[i]) / total_investment * 100
                weights_data.append({
                    'Ticker': downloaded_tickers[i],
                    'Units': selected_weights[i],
                    'Weight %': f"{weight_pct:.2f}%"
                })

        # --- 13. Return Results Dictionary ---
        return {
            'selected_stocks': selected_stocks,
            'num_selected': len(selected_stocks),
            'total_investment': total_investment,
            'expected_return': mu_annual,
            'volatility': sigma_annual,
            'sharpe_ratio': sharpe_ratio,
            'market_return': market_performance,
            'weights_data': weights_data,
            'p': p_value,
            'q': q_value,
            'portfolio_mdd': portfolio_mdd,
            'spy_mdd': spy_mdd,
            'portfolio_value_ts': portfolio_value_ts if not np.isnan(portfolio_mdd) else pd.Series(dtype=float),
            'spy_price_ts': spy_data,
        }

    except Exception as e:
        st.error(f"❌ 優化過程中發生錯誤: {e}")
        import traceback
        st.text(traceback.format_exc())
        return None


def plot_portfolio_vs_spy(portfolio_series, spy_series, title="Portfolio vs SPY Performance"):
    """Version 2: Uses Matplotlib for enhanced styling"""
    import matplotlib.pyplot as plt

    # --- Data Validation ---
    if portfolio_series.empty or spy_series.empty:
        st.warning("⚠️ Empty data - cannot plot.")
        return

    # --- Data Processing ---
    try:
        # Align and normalize data
        df = pd.concat([portfolio_series, spy_series], axis=1, join='inner')
        df.columns = ['Portfolio', 'SPY']
        df_normalized = df / df.iloc[0]

        # Create figure with transparent background
        fig, ax = plt.subplots(figsize=(10, 6), facecolor='none')
        ax.set_facecolor('none')

        # Plot smooth lines
        df_normalized['Portfolio'].plot(
            ax=ax,
            label='My Portfolio',
            linewidth=1.8,
            color='#ff7f0e',
            alpha=0.8,
            antialiased=True
        )

        df_normalized['SPY'].plot(
            ax=ax,
            label='SPY',
            linewidth=1.8,
            color='#1f77b4',
            alpha=0.8,
            antialiased=True
        )

        # Styling
        ax.set_title(title, fontsize=14, pad=20, color='#333333')
        ax.set_xlabel("Date", fontsize=10, color='#666666')
        ax.set_ylabel("Normalized Value", fontsize=10, color='#666666')
        ax.tick_params(colors='#666666')

        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.grid(True, linestyle='--', alpha=0.1, color='gray')

        legend = ax.legend(framealpha=0.5, facecolor='white')
        for text in legend.get_texts():
            text.set_color('#333333')

        # Display
        st.pyplot(fig, transparent=True)
        st.write(f"**Portfolio Final Value**: ${df_normalized['Portfolio'].iloc[-1]:.2f}")
        st.write(f"**SPY Final Value**: ${df_normalized['SPY'].iloc[-1]:.2f}")

    except Exception as e:
        st.error(f"❌ Enhanced plotting failed: {e}")



def main():
    st.title("📈 QUBO Portfolio Optimizer")
    st.markdown("Optimize your investment portfolio using Quantum-inspired computing (QUBO).")

    # --- Sidebar Inputs ---
    st.sidebar.header("Configuration")
    etf_type = st.sidebar.selectbox("Select ETF ", ['QQQ', 'DOW30', '0050','self-picked'], index=0)
    # --- Conditional Input for Self-Picked Tickers ---
    selected_tickers = None
    if etf_type == 'self-picked':
        top_10_tickers = get_top_10_stocks()
        st.sidebar.info("Select stocks from the top 10 most valuable and traded companies:")

        selected_tickers = st.sidebar.multiselect(
            "Choose stocks to include:",
            options=top_10_tickers,
            default=top_10_tickers[:5],  # Default: first 5 (AAPL, MSFT, AMZN, NVDA, GOOGL)
            help="Select one or more stocks from the top 10 largest companies by market cap and trading volume."
        )

        if not selected_tickers:
            st.sidebar.warning("⚠️ Please select at least one stock.")
    start_date = st.sidebar.date_input("Start Date", value=datetime.date(2024, 1, 1))
    end_date = st.sidebar.date_input("End Date", value=datetime.date(2025, 1, 1))

    st.sidebar.subheader("QUBO Parameters")
    p_value = st.sidebar.number_input("Return Weight (p)", min_value=0.01, value=1.0, step=0.1)
    q_value = st.sidebar.number_input("Risk Weight (q)", min_value=0.01, value=1.0, step=0.1)
    use_supercomputer = st.sidebar.checkbox("Use Supercomputer API", value=False)
    #use_supercomputer = False  # Hardcoded for simplicity

    # --- Run Button ---
    run_button = st.sidebar.button("🚀 Optimize Portfolio")

    # --- Results Display ---
    if run_button:
        st.header("Optimization Results")
        # Ensure end_date is after start_date
        if end_date <= start_date:
            st.error("❌ End date must be after start date.")
            return

        # Prepare arguments for optimize_portfolio
        optimize_kwargs = {
            'etf_type': etf_type,
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d'),
            'p_value': p_value,
            'q_value': q_value,
            'use_supercomputer': use_supercomputer
        }
        if etf_type == 'self-picked':
            optimize_kwargs['selected_tickers'] = selected_tickers

        with st.spinner("Running optimization..."):
            result = optimize_portfolio(**optimize_kwargs)

        if result:
            st.success("✅ Optimization completed!")

            # --- 1. Key Metrics ---
            st.subheader("Portfolio Summary")
            # --- First Row of Metrics ---
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Expected Annual Return",
                        f"{result['expected_return']:.2%}" if not np.isnan(result['expected_return']) else "N/A")
            col2.metric("Annualized Volatility",
                        f"{result['volatility']:.2%}" if not np.isnan(result['volatility']) else "N/A")
            col3.metric("Sharpe Ratio",
                        f"{result['sharpe_ratio']:.3f}" if not np.isnan(result['sharpe_ratio']) else "N/A")
            col4.metric("Selected Stocks", result['num_selected'])

            # --- Second Row of Metrics (Including MDD) ---
            col5, col6, col7, col8 = st.columns(4)
            col5.metric("Portfolio MDD",
                        f"{result['portfolio_mdd']:.2%}" if not np.isnan(result['portfolio_mdd']) else "N/A")
            col6.metric("SPY MDD", f"{result['spy_mdd']:.2%}" if not np.isnan(result['spy_mdd']) else "N/A")
            col7.metric("Total Investment", f"${result['total_investment']:.2f}")
            col8.metric("P/Q Ratio", f"{result['p']:.1f}/{result['q']:.1f}")

            # --- 2. Portfolio Weights ---
            st.subheader("Portfolio Weights")
            if result['weights_data']:
                weights_df = pd.DataFrame(result['weights_data'])
                st.dataframe(weights_df, hide_index=True, use_container_width=True)
            else:
                st.info("No weights to display.")

            # --- 3. Benchmark Comparison ---
            st.subheader("Benchmark Comparison")
            if not all(np.isnan([result['market_return'], result['portfolio_mdd'], result['spy_mdd']])):
                # --- Return Comparison ---
                if not np.isnan(result['market_return']):
                    bm_col1, bm_col2 = st.columns(2)
                    bm_col1.metric("Market Return (SPY)", f"{result['market_return']:.2%}")
                    diff = result['expected_return'] - result['market_return']
                    diff_str = f"{'+' if diff > 0 else ''}{diff:.2%}"
                    bm_col2.metric("Return Difference", diff_str, delta=diff_str, delta_color="normal")
                else:
                    st.write("Market Return (SPY): N/A")

                # --- MDD Comparison ---
                if not np.isnan(result['portfolio_mdd']) and not np.isnan(result['spy_mdd']):
                    mdd_col1, mdd_col2 = st.columns(2)
                    mdd_col1.metric("Portfolio MDD", f"{result['portfolio_mdd']:.2%}")
                    mdd_diff = result['portfolio_mdd'] - result['spy_mdd']
                    mdd_diff_str = f"{'+' if mdd_diff > 0 else ''}{mdd_diff:.2%}"
                    mdd_col2.metric("MDD vs SPY", mdd_diff_str, delta=mdd_diff_str, delta_color="inverse")
                else:
                    st.write("MDD Comparison: N/A")
            else:
                st.warning("Market benchmark data could not be calculated.")
            # --- 4. Portfolio vs SPY Plot ---
            st.subheader("Portfolio vs SPY Performance")
            if not result['portfolio_value_ts'].empty and not result['spy_price_ts'].empty:
                plot_portfolio_vs_spy(result['portfolio_value_ts'], result['spy_price_ts'])

            else:
                st.warning("⚠️ Unable to display plot: Portfolio or SPY data is empty.")

        else:
            st.error("❌ Optimization failed. Please check the logs or inputs.")


if __name__ == "__main__":
    main()

# First version: for picking stocks
# fixed p, q
# 8/2 update, graph visualization


