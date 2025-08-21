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
        .reportview-container .main { background-color: #ffffff; }
        .sidebar .sidebar-content { background-color: #f8f9fa; }
        h1, h2, h3, h4, h5, h6 { color: #333333; }
        .stMetricLabel { font-size: 1rem; color: #666666; }
        .stMetricValue { font-size: 1.5rem; color: #222222; }
        .stDataFrame { border: 1px solid #e0e0e0; border-radius: 4px; }
        .stButton>button {
            background-color: #0d6efd; color: white; border: none; border-radius: 4px; padding: 0.5rem 1rem;
        }
        .stButton>button:hover { background-color: #0b5ed7; }
        .stSpinner > div { color: #0d6efd; }
        .block-container {
            padding-top: 2rem; padding-bottom: 2rem; padding-left: 5%; padding-right: 5%;
        }
    </style>
    """, unsafe_allow_html=True)

# --- Core Logic Functions ---
SERVER_URI = "http://60.250.149.247:8080"
current_solution = None
job_received = False
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

def get_stock_list(etf_type, user_tickers_input=None):
    if etf_type == 'DOW30' :
       stock_list = [
            'NVDA', 'MSFT', 'AAPL', 'AMZN', 'WMT', 'JPM', 'V', 'JNJ', 'HD', 'PG',
            'CVX', 'KO', 'UNH', 'CSCO', 'CRM', 'IBM', 'MCD', 'AXP', 'GS', 'MRK',
            'DIS', 'CAT', 'VZ', 'BA', 'AMGN', 'HON', 'NKE', 'SHW', 'MMM', 'TRV'
        ]
    elif etf_type == 'self-picked':
        if user_tickers_input:
            stock_list = [ticker.strip().upper() for ticker in user_tickers_input.split(',') if ticker.strip()]
            if not stock_list:
                st.warning("無輸入有效股票代碼，使用預設清單。")
                stock_list = ['GSG', 'TLT', 'GLD', 'NVDA', 'AAPL', 'MSFT']
        else:
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
    if len(cumulative_values) == 0:
        return 0.0
    cumulative_values = np.asarray(cumulative_values)
    running_max = np.maximum.accumulate(cumulative_values)
    drawdowns = (cumulative_values - running_max) / running_max
    drawdowns = np.nan_to_num(drawdowns, nan=0.0, posinf=0.0, neginf=0.0)
    max_dd = np.min(drawdowns)
    return -max_dd

def get_market_performance(spy_data):
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
            return np.nan
        market_performance = (spy_end / spy_start) - 1
        return float(market_performance)
    return np.nan

def calculate_sharpe_ratio_metrics(selected_indices, selected_weights, returns_df, list_price_start, rf_rate=0.037):
    try:
        selected_indices = np.asarray(selected_indices, dtype=int)
    except (ValueError, TypeError):
        return np.nan, np.nan, np.nan
    if len(selected_indices) == 0:
        return np.nan, np.nan, np.nan
    dollar_positions = selected_weights[selected_indices] * list_price_start[selected_indices]
    if dollar_positions.sum() == 0:
        return np.nan, np.nan, np.nan
    w_frac = dollar_positions / dollar_positions.sum()
    try:
        selected_returns_df = returns_df.iloc[:, selected_indices]
    except IndexError:
        return np.nan, np.nan, np.nan
    port_ret = selected_returns_df.dot(w_frac)
    mu_daily = port_ret.mean()
    sigma_daily = port_ret.std(ddof=1)
    if sigma_daily == 0 or np.isnan(sigma_daily):
        return np.nan, np.nan, np.nan
    mu_annual = mu_daily * 252
    sigma_annual = sigma_daily * (252 ** 0.5)
    sharpe = (mu_annual - rf_rate) / sigma_annual if sigma_annual else np.nan
    return sharpe, mu_annual, sigma_annual

def plot_portfolio_vs_spy(portfolio_series, spy_series, title="Portfolio vs SPY Performance"):
    if portfolio_series.empty or spy_series.empty:
        st.warning("圖表數據為空。")
        return
    try:
        df = pd.concat([portfolio_series, spy_series], axis=1, join="inner")
        df.columns = ["Portfolio", "SPY"]
        if df.iloc[0, 0] != 0 and df.iloc[0, 1] != 0:
            df = df / df.iloc[0]
        else:
            st.warning("初始值為零，無法正規化。")
            return
        st.subheader(title)
        st.line_chart(df)
    except Exception as e:
        st.error(f"畫圖時出錯: {e}")

def prepare_data(etf_type, start_date, end_date, user_tickers_input=None, full_data=None):
    try:
        if full_data is not None:
            data = full_data['data'].loc[start_date:end_date]
            spy_data = full_data['spy_data'].loc[start_date:end_date]
            if len(data) < 2 or len(spy_data) < 2:
                return None, None
            data_normalized = data / data.iloc[0]
            returns_data = {}
            for sc in data.columns:
                prices = data_normalized[sc].dropna()
                if len(prices) > 1:
                    returns_data[sc] = np.log(prices / prices.shift(1)).dropna()
            returns_df = pd.DataFrame(returns_data).fillna(0)
            rates = [returns_df[col].values for col in returns_df.columns]
            downloaded_tickers = list(returns_df.columns)
            N = len(downloaded_tickers)
            if N == 0:
                return None, None
            list_price_start = data.loc[returns_df.index[0], downloaded_tickers].values if len(returns_df) > 0 else np.zeros(N)
            list_price_end = data.loc[returns_df.index[-1], downloaded_tickers].values if len(returns_df) > 0 else np.zeros(N)
            prepared_data = {
                'data': data, 'returns_df': returns_df, 'rates': rates, 'downloaded_tickers': downloaded_tickers,
                'list_price_start': list_price_start, 'list_price_end': list_price_end, 'N': N
            }
            return prepared_data, spy_data

        # Original full download
        if etf_type == 'self-picked':
            stock_list = get_stock_list(etf_type, user_tickers_input)
        else:
            stock_list = get_stock_list(etf_type)
        st.info(f"Downloading data for {len(stock_list)} stocks...")
        data = yf.download(stock_list, start=start_date, end=end_date, auto_adjust=True)["Close"]
        downloaded_tickers = [col for col in data.columns if data[col].first_valid_index() is not None]
        if not downloaded_tickers:
            st.error("❌ No valid stock data could be downloaded.")
            return None, None
        data = data[downloaded_tickers]
        data_normalized = data / data.iloc[0]
        spy_data = yf.download('SPY', start=start_date, end=end_date, auto_adjust=True)["Close"].dropna()
        returns_data = {}
        for sc in downloaded_tickers:
            prices = data_normalized[sc].dropna()
            if len(prices) > 1:
                returns_data[sc] = np.log(prices / prices.shift(1)).dropna()
        returns_df = pd.DataFrame(returns_data).fillna(0)
        rates = [returns_df[col].values for col in returns_df.columns]
        downloaded_tickers = list(returns_df.columns)
        N = len(downloaded_tickers)
        if N == 0:
            st.error("❌ No stocks available after processing returns.")
            return None, None
        list_price_start = data.loc[returns_df.index[0], downloaded_tickers].values
        list_price_end = data.loc[returns_df.index[-1], downloaded_tickers].values
        prepared_data = {
            'data': data, 'returns_df': returns_df, 'rates': rates, 'downloaded_tickers': downloaded_tickers,
            'list_price_start': list_price_start, 'list_price_end': list_price_end, 'N': N
        }
        return prepared_data, spy_data
    except Exception as e:
        st.error(f"❌ Error preparing data: {e}")
        return None, None

def optimize_portfolio(prepared_data, spy_data, p_value, q_value, SERVER_URI=None, use_supercomputer=False):
    global current_solution, job_received
    current_solution = None
    job_received = False
    if SERVER_URI is None:
        SERVER_URI = "http://60.250.149.247:8080"
    try:
        data = prepared_data['data']
        returns_df = prepared_data['returns_df']
        rates = prepared_data['rates']
        downloaded_tickers = prepared_data['downloaded_tickers']
        list_price_start = prepared_data['list_price_start']
        list_price_end = prepared_data['list_price_end']
        N = prepared_data['N']

        w_1 = np.array([gmean(r + 1) for r in rates])
        w = w_1 - 1

        num_bits = 4
        x = Array.create("x", shape=(N, num_bits), vartype="BINARY")
        weights = np.array([2 ** j for j in range(num_bits)])
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
        feed_dict = {'p': p_value, 'q': q_value}
        qubo, offset = model.to_qubo(feed_dict=feed_dict)

        best_sample_array = None
        if use_supercomputer and SERVER_URI:
            st.info("📤 正在使用仁寶 API 求解 QUBO...")
            qubo_data = build_qubo_for_server(qubo, timeout_ms=30000)
            current_solution = None
            job_received = False
            try:
                start_connect_time = time.time()
                sio.connect(SERVER_URI)
                sio.emit("upload qubo", qubo_data)
                max_wait = 120
                for i in range(int(max_wait / 10)):
                    if job_received:
                        break
                    time.sleep(10)
                    st.write(f"⏳ 等待解答中... ({(i + 1) * 10}s elapsed)")
                total_time = time.time() - start_connect_time
                st.write(f"⏱️  超級電腦總耗時: {total_time:.2f} 秒")
                if job_received and current_solution is not None:
                    best_sample_array = current_solution.reshape((N, num_bits))
                    st.success("✅ 成功從超級電腦 API 獲得解")
                else:
                    raise Exception("未在時間內收到有效解")
            except Exception as e:
                st.error(f"❌ 超級電腦 API 錯誤：{e}")
                st.info("🔄 回退到本地 OpenJij 求解器...")
                use_supercomputer = False
            finally:
                try:
                    sio.disconnect()
                except:
                    pass

        if not use_supercomputer or best_sample_array is None:
            try:
                import openjij as oj
                sampler = oj.SASampler()
                h = {i: v for (i, j), v in qubo.items() if i == j}
                J = {(i, j): v for (i, j), v in qubo.items() if i != j}
                J_upper = {}
                for (i, j), v in J.items():
                    if i <= j:
                        J_upper[(i, j)] = v
                    else:
                        J_upper[(j, i)] = v
                response = sampler.sample_ising(h, J, num_reads=1000)
                best_sample_dict = response.first.sample
                best_sample_array = np.array([
                    [best_sample_dict.get(f'x[{i}][{j}]', 0) for j in range(num_bits)]
                    for i in range(N)
                ])
            except ImportError:
                st.error("❌ OpenJij 未安裝。請運行 'pip install openjij'。無法進行本地計算。")
                return None
            except Exception as e:
                st.error(f"❌ 本地求解器錯誤: {e}")
                return None

        if best_sample_array is None:
            st.error("❌ 無法獲得有效解。")
            return None

        selected_weights = np.array([
            sum(best_sample_array[i][j] * weights[j] for j in range(num_bits))
            for i in range(N)
        ])
        selected_indices = [i for i in range(N) if selected_weights[i] > 0]
        selected_stocks = [downloaded_tickers[i] for i in selected_indices]

        total_investment = sum(
            list_price_start[i] * selected_weights[i] for i in range(N) if selected_weights[i] > 0)
        total_expected_return = sum(
            (list_price_start[i] * selected_weights[i] / total_investment) * ((list_price_end[i] / list_price_start[i]) - 1)
            for i in selected_indices
        ) if total_investment > 0 and selected_indices else 0.0

        sharpe_ratio, mu_annual, sigma_annual = calculate_sharpe_ratio_metrics(
            np.array(selected_indices, dtype=int), selected_weights, returns_df, list_price_start
        )

        portfolio_mdd = np.nan
        portfolio_value_ts = pd.Series(dtype=float)
        if selected_indices and total_investment > 0:
            try:
                selected_tickers = [downloaded_tickers[i] for i in selected_indices]
                selected_prices_df = data[selected_tickers].reindex(returns_df.index, method='pad').fillna(method='pad')
                selected_dollar_positions = np.array([list_price_start[i] for i in selected_indices])
                selected_weights_fractional = selected_dollar_positions / selected_dollar_positions.sum()
                portfolio_value_normalized_ts = selected_prices_df.dot(selected_weights_fractional)
                initial_norm_value = portfolio_value_normalized_ts.iloc[0]
                portfolio_value_ts = portfolio_value_normalized_ts * (total_investment / initial_norm_value) if pd.notna(initial_norm_value) and initial_norm_value != 0 else portfolio_value_normalized_ts * total_investment
                portfolio_mdd = calculate_max_drawdown(portfolio_value_ts)
            except:
                portfolio_mdd = np.nan

        spy_mdd = calculate_max_drawdown(spy_data) if not spy_data.empty else np.nan
        market_performance = get_market_performance(spy_data)

        weights_data = []
        if total_investment > 0 and selected_indices:
            for i in selected_indices:
                weight_pct = (list_price_start[i] * selected_weights[i]) / total_investment * 100
                weights_data.append({
                    'Ticker': downloaded_tickers[i],
                    'Units': selected_weights[i],
                    'Weight %': f"{weight_pct:.2f}%"
                })

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
            'portfolio_value_ts': portfolio_value_ts,
            'spy_price_ts': spy_data,
        }
    except Exception as e:
        st.error(f"❌ 優化過程中發生錯誤: {e}")
        return None

# --- Rolling Window Backtest ---
def rolling_window_backtest(etf_type, start_date, end_date, p_value, q_value,
                          training_window=252, testing_window=21,
                          SERVER_URI=None, use_supercomputer=False,
                          user_tickers_input=None):
    st.info(f"Downloading full dataset for backtest from {start_date} to {end_date}...")
    full_data, full_spy_data = prepare_data(etf_type, start_date, end_date, user_tickers_input)
    if full_data is None or full_spy_data is None:
        return None

    all_dates = full_data['data'].index
    if len(all_dates) < training_window + testing_window:
        st.error(f"Not enough data for rolling window (need at least {training_window + testing_window} days)")
        return None

    portfolio_values = []
    spy_values = []
    portfolio_dates = []
    selected_stocks_history = []
    weights_history = []

    all_dates_str = [d.strftime('%Y-%m-%d') if isinstance(d, (pd.Timestamp, datetime.datetime)) else d for d in all_dates]

    for i in range(0, len(all_dates_str) - training_window - testing_window + 1, testing_window):
        train_start = all_dates_str[i]
        train_end = all_dates_str[i + training_window - 1]
        test_start = all_dates_str[i + training_window]
        test_end = all_dates_str[min(i + training_window + testing_window - 1, len(all_dates_str) - 1)]

        st.write(f"Backtesting: Training {train_start} to {train_end}, Testing {test_start} to {test_end}")

        train_data, train_spy = prepare_data(etf_type, train_start, train_end, user_tickers_input,
            full_data={'data': full_data['data'], 'spy_data': full_spy_data})
        if train_data is None or train_data['N'] < 1:
            continue

        result = optimize_portfolio(prepared_data=train_data, spy_data=train_spy, p_value=p_value, q_value=q_value,
                                  SERVER_URI=SERVER_URI, use_supercomputer=use_supercomputer)
        if not result or not result['selected_stocks'] or result['total_investment'] <= 0:
            continue

        selected_stocks_history.append({'date': test_start, 'stocks': result['selected_stocks']})
        weights_history.append({'date': test_start, 'weights': result['weights_data']})

        test_data, test_spy = prepare_data(etf_type, test_start, test_end, user_tickers_input,
            full_data={'data': full_data['data'], 'spy_data': full_spy_data})
        if test_data is None or test_data['data'].empty:
            continue

        try:
            selected_indices = [i for i, ticker in enumerate(test_data['downloaded_tickers']) if ticker in result['selected_stocks']]
            if not selected_indices:
                continue
            selected_weights = np.zeros(len(test_data['downloaded_tickers']))
            for weight_data in result['weights_data']:
                if weight_data['Ticker'] in test_data['downloaded_tickers']:
                    idx = test_data['downloaded_tickers'].index(weight_data['Ticker'])
                    selected_weights[idx] = float(weight_data['Weight %'].rstrip('%')) / 100
            selected_weights = selected_weights / selected_weights.sum() if selected_weights.sum() > 0 else selected_weights

            prices = test_data['data'].values
            portfolio_value = np.dot(prices, selected_weights)
            if len(portfolio_value) > 0 and portfolio_value[0] > 0:
                portfolio_value = portfolio_value / portfolio_value[0]
                portfolio_values.extend(portfolio_value)
                portfolio_dates.extend(test_data['data'].index)
        except:
            continue

        try:
            if not test_spy.empty and test_spy.iloc[0] > 0:
                spy_test = test_spy / test_spy.iloc[0]
                spy_values.extend(spy_test.values)
        except:
            continue

    if portfolio_dates and portfolio_values:
        portfolio_series = pd.Series(portfolio_values, index=portfolio_dates)
        spy_series = pd.Series(spy_values[:len(portfolio_dates)], index=portfolio_dates) if len(spy_values) >= len(portfolio_dates) else None
        total_return = portfolio_values[-1] / portfolio_values[0] - 1 if len(portfolio_values) > 0 else 0

        return {
            'portfolio_series': portfolio_series,
            'spy_series': spy_series,
            'total_return': total_return,
            'selected_stocks_history': selected_stocks_history,
            'weights_history': weights_history,
            'dates': portfolio_dates
        }
    return None

def optimize_portfolio_search(etf_type, start_date, end_date, p_range, q_range, num_points, search_type,
                              SERVER_URI=None, use_supercomputer=False, user_tickers_input=None):
    prepared_data, spy_data = prepare_data(etf_type, start_date, end_date, user_tickers_input)
    if prepared_data is None or spy_data is None:
        st.error("❌ Failed to prepare data for search.")
        return pd.DataFrame()

    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    if search_type == "Grid Search":
        num_divisions = int(np.ceil(np.sqrt(num_points)))
        p_values = np.linspace(p_range[0], p_range[1], num_divisions)
        q_values = np.linspace(q_range[0], q_range[1], num_divisions)
        pq_combinations = [(p, q) for p in p_values for q in q_values]
        total_combinations = len(pq_combinations)
        st.info(f"Performing Grid Search with ~{num_divisions}x{num_divisions} = {total_combinations} points.")
    elif search_type == "Random Search":
        np.random.seed(42)
        p_values = np.random.uniform(p_range[0], p_range[1], num_points)
        q_values = np.random.uniform(q_range[0], q_range[1], num_points)
        pq_combinations = list(zip(p_values, q_values))
        total_combinations = len(pq_combinations)
        st.info(f"Performing Random Search with {total_combinations} points.")
    else:
        st.error(f"Unknown search type: {search_type}")
        return pd.DataFrame()

    counter = 0
    for p, q in pq_combinations:
        status_text.text(f"Optimizing for p={p:.3f}, q={q:.3f} ({counter + 1}/{total_combinations})")
        result = optimize_portfolio(
            prepared_data=prepared_data,
            spy_data=spy_data,
            p_value=p,
            q_value=q,
            SERVER_URI=SERVER_URI,
            use_supercomputer=use_supercomputer
        )
        if result:
            results.append({
                'p': p, 'q': q, 'expected_return': result['expected_return'],
                'volatility': result['volatility'], 'sharpe_ratio': result['sharpe_ratio']
            })
        counter += 1
        progress_bar.progress(counter / total_combinations)

    status_text.empty()
    progress_bar.empty()
    return pd.DataFrame(results)

# --- Main App ---
def main():
    st.title("📈 QUBO Portfolio Optimizer")
    st.markdown("Optimize your investment portfolio using Quantum-inspired computing (QUBO).")

    # --- Sidebar Inputs ---
    st.sidebar.header("Configuration")
    etf_type = st.sidebar.selectbox("Select ETF Universe", ['QQQ', 'DOW30', 'self-picked'], index=0)
    user_tickers_input = None
    if etf_type == 'self-picked':
        default_tickers = "GSG,TLT,GLD,NVDA,AAPL,MSFT"
        user_tickers_input = st.sidebar.text_area(
            "Enter your stock tickers (comma-separated):",
            value=default_tickers, height=100,
            help="e.g., AAPL, MSFT, GOOGL", key="user_tickers"
        )
        st.sidebar.info("Please ensure ticker symbols are correct. At least one ticker is required.")

    start_date = st.sidebar.date_input("Start Date", value=datetime.date(2024, 1, 1))
    end_date = st.sidebar.date_input("End Date", value=datetime.date(2025, 1, 1))

    # --- Parameter Search ---
    st.sidebar.subheader("Parameter Search")
    p_min = st.sidebar.number_input("Minimum Return Weight (p)", min_value=0.01, value=0.1, step=0.1, format="%.2f")
    p_max = st.sidebar.number_input("Maximum Return Weight (p)", min_value=0.01, value=2.0, step=0.1, format="%.2f")
    q_min = st.sidebar.number_input("Minimum Risk Weight (q)", min_value=0.01, value=0.1, step=0.1, format="%.2f")
    q_max = st.sidebar.number_input("Maximum Risk Weight (q)", min_value=0.01, value=2.0, step=0.1, format="%.2f")
    search_type = st.sidebar.radio("Search Type", ("Grid Search", "Random Search"))

    if search_type == "Grid Search":
        num_divisions = st.sidebar.slider("Number of Divisions per Axis", min_value=2, max_value=20, value=5)
        total_points = num_divisions * num_divisions
        st.sidebar.write(f"Approximate total evaluations: {total_points}")
        num_points = total_points
    else:
        num_points = st.sidebar.slider("Number of Random Points", min_value=5, max_value=400, value=25)
        st.sidebar.write(f"Total evaluations: {num_points}")

    use_supercomputer_in_search = st.sidebar.checkbox("Use Supercomputer During Search?", value=False)

    # --- Rolling Window Backtest ---
    st.sidebar.subheader("Rolling Window Backtest")
    do_rolling_backtest = st.sidebar.checkbox("Perform Rolling Window Backtest", value=False)
    if do_rolling_backtest:
        training_window = st.sidebar.number_input("Training Window (days)", min_value=30, value=252, step=10)
        testing_window = st.sidebar.number_input("Testing Window (days)", min_value=5, value=21, step=5)
        use_best_params = st.sidebar.checkbox("Use Best Parameters from Search", value=True)
        if not use_best_params:
            p_value = st.sidebar.number_input("Return Weight (p)", min_value=0.01, value=1.0, step=0.1, format="%.2f")
            q_value = st.sidebar.number_input("Risk Weight (q)", min_value=0.01, value=1.0, step=0.1, format="%.2f")

    run_button = st.sidebar.button("🚀 Run Parameter Search")
    run_backtest_button = st.sidebar.button("📊 Run Rolling Backtest") if do_rolling_backtest else False

    # --- Main Logic ---
    if run_button:
        if end_date <= start_date:
            st.error("❌ End date must be after start date.")
            return
        with st.spinner(f"Running {search_type}..."):
            df_results = optimize_portfolio_search(
                etf_type=etf_type, start_date=start_date.strftime('%Y-%m-%d'), end_date=end_date.strftime('%Y-%m-%d'),
                p_range=(p_min, p_max), q_range=(q_min, q_max), num_points=num_points, search_type=search_type,
                use_supercomputer=use_supercomputer_in_search, SERVER_URI=SERVER_URI,
                user_tickers_input=user_tickers_input if etf_type == 'self-picked' else None
            )
        if not df_results.empty:
            st.success("✅ Search completed!")
            st.session_state['saved_search_results'] = df_results
            fig = px.scatter(df_results, x='volatility', y='expected_return', color='sharpe_ratio',
                           color_continuous_scale='Viridis', title=f"Return vs Volatility ({search_type})",
                           labels={'expected_return': 'Expected Annual Return', 'volatility': 'Annualized Volatility'},
                           hover_data=['p', 'q'])
            st.plotly_chart(fig, use_container_width=True)
            best_idx = df_results['sharpe_ratio'].idxmax()
            best_result = df_results.loc[best_idx]
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Sharpe Ratio", f"{best_result['sharpe_ratio']:.3f}")
            col2.metric("Return", f"{best_result['expected_return']:.2%}")
            col3.metric("Volatility", f"{best_result['volatility']:.2%}")
            col4.metric("p, q", f"{best_result['p']:.2f}, {best_result['q']:.2f}")

            st.subheader("Re-calculate with Best Parameters")
            recalculate_with_supercomputer = st.checkbox("Use Supercomputer for Re-calculation?", value=False)
            if st.button("🔄 Re-calculate Best Portfolio"):
                with st.spinner("Re-calculating with best parameters..."):
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
                            use_supercomputer=recalculate_with_supercomputer
                        )
                        if final_result:
                            st.success("✅ Final calculation completed!")
                            st.write(f"**Selected Stocks:** {', '.join(final_result['selected_stocks'])}")
                            st.write(f"**Number of Stocks Selected:** {final_result['num_selected']}")
                            st.write(f"**Total Investment:** ${final_result['total_investment']:.2f}")
                            st.write(f"**Expected Annual Return:** {final_result['expected_return']:.2%}")
                            st.write(f"**Annualized Volatility:** {final_result['volatility']:.2%}")
                            st.write(f"**Sharpe Ratio:** {final_result['sharpe_ratio']:.3f}")
                            st.write(f"**Market (SPY) Return:** {final_result['market_return']:.2%}")
                            st.write(f"**Portfolio MDD:** {final_result['portfolio_mdd']:.2%}")
                            st.write(f"**SPY MDD:** {final_result['spy_mdd']:.2%}")
                            if final_result['weights_data']:
                                st.write("**Portfolio Weights:**")
                                st.dataframe(pd.DataFrame(final_result['weights_data']), hide_index=True)
                            plot_portfolio_vs_spy(final_result['portfolio_value_ts'], final_result['spy_price_ts'],
                                                "Final Portfolio vs SPY Performance")
                        else:
                            st.error("❌ Final calculation failed.")

            st.subheader("All Search Results")
            df_display = df_results[['p', 'q', 'expected_return', 'volatility', 'sharpe_ratio']].copy()
            df_display['expected_return'] = df_display['expected_return'].apply(lambda x: f"{x:.2%}")
            df_display['volatility'] = df_display['volatility'].apply(lambda x: f"{x:.2%}")
            st.dataframe(df_display)

    # --- Rolling Backtest Execution ---
    if do_rolling_backtest and run_backtest_button:
        st.header("Rolling Window Backtest Results")
        p_value_final = None
        q_value_final = None

        # 重要！檢查 st.session_state，唔好檢查 'df_results' in locals()
        if use_best_params and 'saved_search_results' in st.session_state and not st.session_state[
            'saved_search_results'].empty:
            best_idx = st.session_state['saved_search_results']['sharpe_ratio'].idxmax()
            p_value_final = st.session_state['saved_search_results'].loc[best_idx, 'p']
            q_value_final = st.session_state['saved_search_results'].loc[best_idx, 'q']
            st.info(f"Using best parameters from search: p={p_value_final:.2f}, q={q_value_final:.2f}")

        elif not use_best_params:
            p_value_final = p_value
            q_value_final = q_value
        else:
            st.warning("No search results available. Using default parameters.")
            p_value_final = 1.0
            q_value_final = 1.0

        with st.spinner("Running rolling window backtest..."):
            backtest_results = rolling_window_backtest(
                etf_type=etf_type,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                p_value=p_value_final,
                q_value=q_value_final,
                training_window=training_window,
                testing_window=testing_window,
                SERVER_URI=SERVER_URI,
                use_supercomputer=use_supercomputer_in_search,
                user_tickers_input=user_tickers_input if etf_type == 'self-picked' else None
            )

        if backtest_results:
            st.success("✅ Backtest completed!")
            st.subheader("Cumulative Performance")
            if backtest_results['spy_series'] is not None:
                plot_portfolio_vs_spy(backtest_results['portfolio_series'], backtest_results['spy_series'],
                                    "Rolling Window Portfolio vs SPY Performance")
            else:
                st.line_chart(backtest_results['portfolio_series'])

            st.subheader("Performance Metrics")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Return", f"{backtest_results['total_return']:.2%}")

            if len(backtest_results['dates']) > 1:
                years = (backtest_results['dates'][-1] - backtest_results['dates'][0]).days / 365.25
                if years > 0:
                    cagr = (1 + backtest_results['total_return']) ** (1/years) - 1
                    col2.metric("CAGR", f"{cagr:.2%}")

            portfolio_mdd = calculate_max_drawdown(backtest_results['portfolio_series'])
            col3.metric("Portfolio MDD", f"{portfolio_mdd:.2%}")

            if backtest_results['spy_series'] is not None:
                spy_mdd = calculate_max_drawdown(backtest_results['spy_series'])
                col4.metric("SPY MDD", f"{spy_mdd:.2%}")

            if len(backtest_results['portfolio_series']) > 1:
                daily_returns = backtest_results['portfolio_series'].pct_change().dropna()
                if len(daily_returns) > 0:
                    annualized_return = (backtest_results['portfolio_series'].iloc[-1] /
                                       backtest_results['portfolio_series'].iloc[0]) ** (252 / len(daily_returns)) - 1
                    annualized_vol = daily_returns.std() * np.sqrt(252)
                    sharpe_ratio = annualized_return / annualized_vol if annualized_vol > 0 else 0
                    st.metric("Backtest Sharpe Ratio", f"{sharpe_ratio:.3f}")

            st.subheader("Portfolio Turnover")
            turnover_rates = []
            for i in range(1, len(backtest_results['selected_stocks_history'])):
                prev = set(backtest_results['selected_stocks_history'][i-1]['stocks'])
                curr = set(backtest_results['selected_stocks_history'][i]['stocks'])
                turnover = len(prev.symmetric_difference(curr)) / len(prev.union(curr)) if prev else 0
                turnover_rates.append(turnover)
            if turnover_rates:
                st.metric("Average Turnover Rate", f"{np.mean(turnover_rates):.2%}")
                turnover_series = pd.Series(turnover_rates, index=[item['date'] for item in backtest_results['selected_stocks_history'][1:]])
                st.line_chart(turnover_series)

            st.subheader("Portfolio Composition Over Time")
            if backtest_results['weights_history']:
                stocks_list = sorted(set().union(*[set(w['Ticker'] for w in item['weights']) for item in backtest_results['weights_history']]))
                composition_data = []
                for window in backtest_results['weights_history']:
                    row = {'date': window['date']}
                    for stock in stocks_list:
                        weight = next((float(w['Weight %'].rstrip('%'))/100 for w in window['weights'] if w['Ticker'] == stock), 0)
                        row[stock] = weight
                    composition_data.append(row)
                composition_df = pd.DataFrame(composition_data).set_index('date')
                st.dataframe(composition_df.style.format("{:.1%}"))

if __name__ == "__main__":
    main()
