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
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

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
    """將 QUBO 字典轉換為超級電腦 API 所需的格式"""
    return pickle.dumps({
        "qubo": qubo_dict,
        "offset": 0,
        "timeout": timeout_ms
    }, protocol=2)


def get_stock_list(etf_type, user_tickers_input=None):
    """
    Fetches the list of stock tickers based on the ETF type.
    For 'self-picked', uses the provided user_tickers_input.
    """
    if etf_type == 'DOW30':
        
        stock_list = stock_list = [
            'NVDA', 'MSFT', 'AAPL', 'AMZN', 'WMT', 'JPM', 'V', 'JNJ', 'HD', 'PG',
            'CVX', 'KO', 'UNH', 'CSCO', 'CRM', 'IBM', 'MCD', 'AXP', 'GS', 'MRK',
            'DIS', 'CAT', 'VZ', 'BA', 'AMGN', 'HON', 'NKE', 'SHW', 'MMM', 'TRV'
        ]
        
    elif etf_type == 'self-picked':
        # Handle user input for self-picked stocks
        if user_tickers_input:
            # Split by comma, strip whitespace, and filter out empty strings
            stock_list = [ticker.strip().upper() for ticker in user_tickers_input.split(',') if ticker.strip()]
            if not stock_list:
                # Fallback if input is empty or only commas
                st.warning("無輸入有效股票代碼，使用預設清單。")
                stock_list = ['GSG', 'TLT', 'GLD', 'NVDA', 'AAPL', 'MSFT']
        else:
            # Fallback if function is called without input for self-picked
            st.warning("無輸入股票代碼，使用預設清單。")
            stock_list = ['GSG', 'TLT', 'GLD', 'NVDA', 'AAPL', 'MSFT']
    elif etf_type == 'QQQ':
        url = "https://www.invesco.com/us/financial-products/etfs/holdings/main/holdings/0?audienceType=Investor&action=download&ticker=QQQ"
        response = requests.get(url)
        response.raise_for_status()
        holdings_df = pd.read_csv(StringIO(response.text))
        stock_list = holdings_df['Holding Ticker'].dropna().tolist()
        stock_list = [ticker.strip() for ticker in stock_list]
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


def get_market_performance(spy_data):
    """Calculate market performance using SPY data.
       Note: start_date and end_date are passed for context/logging if needed,
       but the calculation relies solely on the provided spy_data Series."""
    # 直接使用 spy_data 進行計算，假設它已經是正確時間範圍內的數據
    if not spy_data.empty and len(spy_data) >= 2:
        spy_start = spy_data.iloc[0]
        spy_end = spy_data.iloc[-1]

        # 處理可能的 Series/DataFrame 單元素情況
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
        market_performance = float(market_performance)  # 確保是 Python float
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


def plot_portfolio_vs_spy(portfolio_series, spy_series, title="Portfolio vs SPY Performance"):
    # 檢查輸入
    if portfolio_series.empty or spy_series.empty:
        st.warning("圖表數據為空。")
        return
    try:
        df = pd.concat([portfolio_series, spy_series], axis=1, join="inner")
        df.columns = ["Portfolio", "SPY"]
        # 正常化到 1
        if df.iloc[0, 0] != 0 and df.iloc[0, 1] != 0:
            df = df / df.iloc[0]
        else:
            st.warning("初始值為零，無法正規化。")
            return
        st.subheader(title)
        st.line_chart(df)
    except Exception as e:
        st.error(f"畫圖時出錯: {e}")


def prepare_data(etf_type, start_date, end_date, user_tickers_input=None):
    """Prepare stock and SPY data once for reuse."""
    try:
        # --- 1. Get Stock List ---
        if etf_type == 'self-picked':
            stock_list = get_stock_list(etf_type, user_tickers_input)
        else:
            stock_list = get_stock_list(etf_type)
        st.info(f"Downloading data for {len(stock_list)} stocks...")
        # --- 2. Download Data ---
        data = yf.download(stock_list, start=start_date, end=end_date, auto_adjust=True)["Close"]
        # Filter out tickers that failed to download or have insufficient data
        downloaded_tickers = [col for col in data.columns if data[col].first_valid_index() is not None]
        if not downloaded_tickers:
            st.error("❌ No valid stock data could be downloaded.")
            return None, None
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
            return None, None
        # Recalculate list_price_start and list_price_end based on the aligned data
        list_price_start = data.loc[returns_df.index[0], downloaded_tickers].values
        list_price_end = data.loc[returns_df.index[-1], downloaded_tickers].values

        # Package data for reuse
        prepared_data = {
            'data': data,
            'returns_df': returns_df,
            'rates': rates,
            'downloaded_tickers': downloaded_tickers,
            'list_price_start': list_price_start,
            'list_price_end': list_price_end,
            'N': N
        }
        return prepared_data, spy_data
    except Exception as e:
        st.error(f"❌ Error preparing data: {e}")
        import traceback
        st.text(traceback.format_exc())
        return None, None


def optimize_portfolio(prepared_data, spy_data, p_value, q_value, SERVER_URI=None, use_supercomputer=False):
    """Optimizes portfolio using pre-downloaded data and returns results dictionary."""
    global current_solution, job_received
    # Reset global state for each run
    current_solution = None
    job_received = False

    # If SERVER_URI is None, use the global default value
    if SERVER_URI is None:
        SERVER_URI = "http://60.250.149.247:8080"
    try:
        # Unpack prepared data
        data = prepared_data['data']
        returns_df = prepared_data['returns_df']
        rates = prepared_data['rates']
        downloaded_tickers = prepared_data['downloaded_tickers']
        list_price_start = prepared_data['list_price_start']
        list_price_end = prepared_data['list_price_end']
        N = prepared_data['N']

        # --- 4. Calculate Geometric Mean Returns ---
        w_1 = np.array([gmean(r + 1) for r in rates])
        w = w_1 - 1
        # --- 5. Set up QUBO Model ---
        num_bits = 4
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
            st.info("📤 正在使用仁寶 API 求解 QUBO...")
            qubo_data = build_qubo_for_server(qubo, timeout_ms=30000)
            current_solution = None
            job_received = False
            try:
                st.write(f"🔗 連接到超級電腦 API: {SERVER_URI}")
                start_connect_time = time.time()  # 開始計時
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
                total_time = time.time() - start_connect_time  # 計總時間
                st.write(f"⏱️  超級電腦總耗時: {total_time:.2f} 秒")
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

        # --- Local Solver (moved inside try, after finally) ---
        if not use_supercomputer or best_sample_array is None:
            # st.info("⚠️ 使用本地 OpenJij 求解器...")
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
                response = sampler.sample_ising(h, J, num_reads=1000)  # Increased for better quality
                best_sample_dict = response.first.sample
                # Convert back to array format
                best_sample_array = np.array([
                    [best_sample_dict.get(f'x[{i}][{j}]', 0) for j in range(num_bits)]
                    for i in range(N)
                ])
                # st.success("✅ 本地求解器完成")
            except ImportError:
                st.error("❌ OpenJij 未安裝。請運行 'pip install openjij'。無法進行本地計算。")
                return None
            except Exception as e:
                st.error(f"❌ 本地求解器錯誤: {e}")
                import traceback
                st.text(traceback.format_exc())
                return None

        # Final check for valid solution
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
        portfolio_value_ts = pd.Series(dtype=float)  # Initialize here
        if selected_indices and total_investment > 0:
            try:
                selected_tickers = [downloaded_tickers[i] for i in selected_indices]
                selected_prices_df = data[selected_tickers].reindex(returns_df.index, method='pad').fillna(
                    method='pad')
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
        market_performance = get_market_performance(spy_data)  # --- 12. Prepare Weights Data for Display ---
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
            'portfolio_value_ts': portfolio_value_ts,  # This will be a Series (empty or with data)
            'spy_price_ts': spy_data,
        }
    except Exception as e:
        st.error(f"❌ 優化過程中發生錯誤: {e}")
    import traceback
    st.text(traceback.format_exc())
    return None


def optimize_portfolio_grid(etf_type, start_date, end_date, p_range, q_range, num_divisions, SERVER_URI=None,
                            use_supercomputer=False, user_tickers_input=None):
    """
    Optimize portfolios for a grid of p and q values using pre-downloaded data and return results.
    """
    # --- 1. Prepare data once ---
    prepared_data, spy_data = prepare_data(etf_type, start_date, end_date, user_tickers_input)
    if prepared_data is None or spy_data is None:
        st.error("❌ Failed to prepare data for grid search.")
        return pd.DataFrame()

    results = []
    total_combinations = num_divisions * num_divisions
    progress_bar = st.progress(0)
    status_text = st.empty()

    p_values = np.linspace(p_range[0], p_range[1], num_divisions)
    q_values = np.linspace(q_range[0], q_range[1], num_divisions)

    counter = 0
    for p in p_values:
        for q in q_values:
            status_text.text(f"Optimizing for p={p:.2f}, q={q:.2f} ({counter + 1}/{total_combinations})")
            # --- 2. Optimize using prepared data ---
            result = optimize_portfolio(
                prepared_data=prepared_data,
                spy_data=spy_data,
                p_value=p,
                q_value=q,
                SERVER_URI=SERVER_URI,
                use_supercomputer=use_supercomputer  # Will be False for grid search
            )
            if result:
                results.append({
                    'p': p,
                    'q': q,
                    'expected_return': result['expected_return'],
                    'volatility': result['volatility'],
                    'sharpe_ratio': result['sharpe_ratio']
                })
            counter += 1
            progress_bar.progress(counter / total_combinations)

    status_text.empty()
    progress_bar.empty()
    return pd.DataFrame(results)


def optimize_portfolio_search(etf_type, start_date, end_date, p_range, q_range, num_points, search_type,
                              SERVER_URI=None,
                              use_supercomputer=False, user_tickers_input=None):
    """
    Optimize portfolios for a set of p and q values using either Grid Search or Random Search.
    """
    # --- 1. Prepare data once ---
    prepared_data, spy_data = prepare_data(etf_type, start_date, end_date, user_tickers_input)
    if prepared_data is None or spy_data is None:
        st.error("❌ Failed to prepare data for search.")
        return pd.DataFrame()
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    # --- 2. Generate p and q values based on search type ---
    if search_type == "Grid Search":
        # Create a grid
        num_divisions = int(np.ceil(np.sqrt(num_points)))  # Aim for roughly num_points total
        p_values = np.linspace(p_range[0], p_range[1], num_divisions)
        q_values = np.linspace(q_range[0], q_range[1], num_divisions)
        # Cartesian product
        pq_combinations = [(p, q) for p in p_values for q in q_values]
        total_combinations = len(pq_combinations)
        st.info(f"Performing Grid Search with ~{num_divisions}x{num_divisions} = {total_combinations} points.")
    elif search_type == "Random Search":
        # Generate random points
        np.random.seed(42)  # For reproducibility, optional
        p_values = np.random.uniform(p_range[0], p_range[1], num_points)
        q_values = np.random.uniform(q_range[0], q_range[1], num_points)
        pq_combinations = list(zip(p_values, q_values))
        total_combinations = len(pq_combinations)
        st.info(f"Performing Random Search with {total_combinations} points.")
    else:
        st.error(f"Unknown search type: {search_type}")
        return pd.DataFrame()
    if total_combinations == 0:
        st.error("No parameter combinations to evaluate.")
        return pd.DataFrame()
    counter = 0
    for p, q in pq_combinations:
        status_text.text(f"Optimizing for p={p:.3f}, q={q:.3f} ({counter + 1}/{total_combinations})")
        # --- 3. Optimize using prepared data ---
        # *** 關鍵修改點 ***
        # 傳遞 use_supercomputer 個參數落去 optimize_portfolio
        result = optimize_portfolio(
            prepared_data=prepared_data,
            spy_data=spy_data,
            p_value=p,
            q_value=q,
            SERVER_URI=SERVER_URI,
            use_supercomputer=use_supercomputer  # <-- 改呢行，用傳入嘅參數
        )
        # *** 修改完 ***
        if result:
            results.append({
                'p': p,
                'q': q,
                'expected_return': result['expected_return'],
                'volatility': result['volatility'],
                'sharpe_ratio': result['sharpe_ratio']
            })
        counter += 1
        progress_bar.progress(counter / total_combinations)

    status_text.empty()
    progress_bar.empty()
    return pd.DataFrame(results)


def main():
    st.title("📈 QUBO Portfolio Optimizer")
    st.markdown("Optimize your investment portfolio using Quantum-inspired computing (QUBO).")
    # --- Sidebar Inputs ---
    st.sidebar.header("Configuration")
    etf_type = st.sidebar.selectbox("Select ETF Universe", ['QQQ', 'DOW30', 'self-picked'], index=0)
    # --- Conditional Input for Self-Picked Tickers ---
    user_tickers_input = None
    if etf_type == 'self-picked':
        default_tickers = "GSG,TLT,GLD,NVDA,AAPL,MSFT"
        user_tickers_input = st.sidebar.text_area(
            "Enter your stock tickers (comma-separated):",
            value=default_tickers,
            height=100,
            help="e.g., AAPL, MSFT, GOOGL",
            key="user_tickers"
        )
        st.sidebar.info("Please ensure ticker symbols are correct. At least one ticker is required.")
    start_date = st.sidebar.date_input("Start Date", value=datetime.date(2024, 1, 1))
    end_date = st.sidebar.date_input("End Date", value=datetime.date(2025, 1, 1))
    # --- New Inputs for p and q Search ---
    st.sidebar.subheader("Parameter Search")
    p_min = st.sidebar.number_input("Minimum Return Weight (p)", min_value=0.01, value=0.1, step=0.1, format="%.2f")
    p_max = st.sidebar.number_input("Maximum Return Weight (p)", min_value=0.01, value=2.0, step=0.1, format="%.2f")
    q_min = st.sidebar.number_input("Minimum Risk Weight (q)", min_value=0.01, value=0.1, step=0.1, format="%.2f")
    q_max = st.sidebar.number_input("Maximum Risk Weight (q)", min_value=0.01, value=2.0, step=0.1, format="%.2f")
    # --- Search Type Selection ---
    search_type = st.sidebar.radio("Search Type", ("Grid Search", "Random Search"))
    # --- Number of Points/Divisions Input ---
    if search_type == "Grid Search":
        num_divisions = st.sidebar.slider("Number of Divisions per Axis", min_value=2, max_value=20, value=5)
        # Calculate approximate total points for grid search
        total_points = num_divisions * num_divisions
        st.sidebar.write(f"Approximate total evaluations: {total_points}")
        num_points = total_points  # Use total points for the function call
    elif search_type == "Random Search":
        num_points = st.sidebar.slider("Number of Random Points", min_value=5, max_value=400, value=25)
        st.sidebar.write(f"Total evaluations: {num_points}")
        num_divisions = None  # Not used for random search

    # *** 新增輸入: 是否喺搜索中使用超級電腦 ***
    use_supercomputer_in_search = st.sidebar.checkbox("Use Supercomputer During Search?", value=False)

    # --- Run Button ---
    run_button = st.sidebar.button("🚀 Run Parameter Search")
    # --- Results Display ---
    if run_button:
        st.header("Optimization Results")
        # Ensure end_date is after start_date
        if end_date <= start_date:
            st.error("❌ End date must be after start date.")
            return
        # Prepare arguments for optimize_portfolio_search
        optimize_kwargs = {
            'etf_type': etf_type,
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d'),
            'p_range': (p_min, p_max),
            'q_range': (q_min, q_max),
            'num_points': num_points,  # Pass the determined number of points
            'search_type': search_type,  # Pass the chosen search type
            # *** 修改點: 傳遞 use_supercomputer 值 ***
            'use_supercomputer': use_supercomputer_in_search,  # <-- 改呢行
            # *** 修改點完 ***
            'SERVER_URI': SERVER_URI
        }
        if etf_type == 'self-picked':
            optimize_kwargs['user_tickers_input'] = user_tickers_input
        with st.spinner(f"Running {search_type}..."):
            df_results = optimize_portfolio_search(**optimize_kwargs)
        if not df_results.empty:
            st.success("✅ Search completed!")
            # --- Visualize Results ---
            st.subheader("Sharpe Ratio Scatter Plot")
            # 使用 Plotly 繪製散點圖，顏色代表夏普比率
            fig = px.scatter(
                df_results,
                x='volatility',
                y='expected_return',
                color='sharpe_ratio',
                color_continuous_scale='Viridis',
                title=f"Return vs Volatility ({search_type})",
                labels={
                    'expected_return': 'Expected Annual Return',
                    'volatility': 'Annualized Volatility',
                    'sharpe_ratio': 'Sharpe Ratio'
                },
                hover_data=['p', 'q']  # 滑鼠懸停時顯示 p, q 值
            )
            fig.update_layout(
                xaxis=dict(title="Volatility"),
                yaxis=dict(title="Return"),
                coloraxis_colorbar=dict(title="Sharpe Ratio")
            )
            st.plotly_chart(fig, use_container_width=True)
            # --- Find and Display Best Result ---
            best_idx = df_results['sharpe_ratio'].idxmax()
            best_result = df_results.loc[best_idx]
            st.subheader("Best Portfolio Found")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Sharpe Ratio", f"{best_result['sharpe_ratio']:.3f}")
            col2.metric("Return", f"{best_result['expected_return']:.2%}")
            col3.metric("Volatility", f"{best_result['volatility']:.2%}")
            col4.metric("p, q", f"{best_result['p']:.2f}, {best_result['q']:.2f}")

            # *** 新增功能: 用最佳參數重新計算 (可選擇用超級電腦) ***
            st.subheader("Re-calculate with Best Parameters")
            recalculate_with_supercomputer = st.checkbox("Use Supercomputer for Re-calculation?", value=False)
            if st.button("🔄 Re-calculate Best Portfolio"):
                with st.spinner("Re-calculating with best parameters..."):
                    # Prepare data again for the single final calculation
                    # (或者你可以儲存 prepare_data 嘅結果，避免重新下載)
                    prepared_data_final, spy_data_final = prepare_data(etf_type, start_date.strftime('%Y-%m-%d'),
                                                                       end_date.strftime('%Y-%m-%d'),
                                                                       user_tickers_input if etf_type == 'self-picked' else None)
                    if prepared_data_final is None or spy_data_final is None:
                        st.error("❌ Failed to prepare data for final calculation.")
                    else:
                        final_result = optimize_portfolio(
                            prepared_data=prepared_data_final,
                            spy_data=spy_data_final,
                            p_value=best_result['p'],
                            q_value=best_result['q'],
                            SERVER_URI=SERVER_URI,
                            use_supercomputer=recalculate_with_supercomputer  # <-- 用 checkbox 嘅值
                        )
                        if final_result:
                            st.success("✅ Final calculation completed!")
                            # 顯示最終結果 (可以複製 optimize_portfolio 嘅顯示邏輯)
                            st.write(f"**Selected Stocks:** {', '.join(final_result['selected_stocks'])}")
                            st.write(f"**Number of Stocks Selected:** {final_result['num_selected']}")
                            st.write(f"**Total Investment:** ${final_result['total_investment']:.2f}")
                            st.write(f"**Expected Annual Return:** {final_result['expected_return']:.2%}")
                            st.write(f"**Annualized Volatility:** {final_result['volatility']:.2%}")
                            st.write(f"**Sharpe Ratio:** {final_result['sharpe_ratio']:.3f}")
                            st.write(f"**Market (SPY) Return:** {final_result['market_return']:.2%}")
                            st.write(f"**Portfolio Max Drawdown (MDD):** {final_result['portfolio_mdd']:.2%}")
                            st.write(f"**SPY Max Drawdown (MDD):** {final_result['spy_mdd']:.2%}")
                            # 顯示權重表
                            if final_result['weights_data']:
                                st.write("**Portfolio Weights:**")
                                st.dataframe(pd.DataFrame(final_result['weights_data']), hide_index=True)
                            # 畫圖
                            plot_portfolio_vs_spy(final_result['portfolio_value_ts'], final_result['spy_price_ts'],
                                                  "Final Portfolio vs SPY Performance")
                        else:
                            st.error("❌ Final calculation failed.")
            # *** 新增功能完 ***

            # --- Show Table of All Results ---
            st.subheader("All Search Results")
            # 重新排列 DataFrame 列的順序以便顯示
            columns_order = ['p', 'q', 'expected_return', 'volatility', 'sharpe_ratio']
            df_display = df_results[columns_order].copy()
            # 格式化百分比
            df_display['expected_return'] = df_display['expected_return'].apply(lambda x: f"{x:.2%}")
            df_display['volatility'] = df_display['volatility'].apply(lambda x: f"{x:.2%}")
            st.dataframe(df_display)
        else:
            st.warning("No results found for the given parameters.")


if __name__ == "__main__":
    main()

# model for finding p,q
# Self-picked function still have problem
# 3/8 change to be not showing ⚠️ 使用本地 OpenJij 求解器... + ✅ 本地求解器完成 every time
# time for 3x3=9 points,QQQ = 4min
