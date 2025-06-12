import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from ta import add_all_ta_features
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.trend import MACD
import warnings

warnings.filterwarnings('ignore')

# Mapping of ticker symbols to professional display names
TICKER_NAMES = {
    # Technology
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "NVDA": "NVIDIA",
    "AVGO": "Broadcom",
    "ASML": "ASML Holding",
    "TSM": "Taiwan Semi.",
    "ADBE": "Adobe",
    "CSCO": "Cisco",
    "ACN": "Accenture",
    "CRM": "Salesforce",
    "ORCL": "Oracle",
    "SAP": "SAP",
    "INTU": "Intuit",
    "AMD": "AMD",
    "INTC": "Intel",
    "QCOM": "Qualcomm",
    "TXN": "Texas Instruments",
    "AMAT": "Applied Materials",
    "LRCX": "Lam Research",
    "KLAC": "KLA Corp",
    "SNOW": "Snowflake",
    "PLTR": "Palantir",
    "U": "Unity",
    "DDOG": "Datadog",
    "ZS": "Zscaler",
    "CRWD": "CrowdStrike",
    "NET": "Cloudflare",
    "MDB": "MongoDB",
    "NOW": "ServiceNow",
    "TEAM": "Atlassian",
    "SHOP": "Shopify",
    "SQ": "Block",
    "PYPL": "PayPal",
    "DOCU": "DocuSign",
    "ZM": "Zoom",
    "OKTA": "Okta",
    "TWLO": "Twilio",
    "FSLY": "Fastly",
    "PINS": "Pinterest",
    "SPOT": "Spotify",
    # Finance
    "JPM": "JPMorgan Chase",
    "BAC": "Bank of America",
    "WFC": "Wells Fargo",
    "C": "Citigroup",
    "HSBC": "HSBC Holdings",
    "GS": "Goldman Sachs",
    "MS": "Morgan Stanley",
    "BLK": "BlackRock",
    "SCHW": "Charles Schwab",
    "AXP": "American Express",
    "V": "Visa",
    "MA": "Mastercard",
    "COIN": "Coinbase",
    "MUFG": "Mitsubishi UFJ",
    "RY": "Royal Bank of Canada",
    "TD": "Toronto-Dominion",
    "BNPQY": "BNP Paribas",
    "ING": "ING Groep",
    "USB": "US Bancorp",
    "PNC": "PNC Financial",
    "TFC": "Truist",
    "BK": "Bank of NY Mellon",
    "STT": "State Street",
    "DFS": "Discover",
    "ALLY": "Ally Financial",
    "CBOE": "Cboe Global Markets",
    "MCO": "Moody's",
    "SPGI": "S&P Global",
    # Consumer
    "AMZN": "Amazon",
    "WMT": "Walmart",
    "COST": "Costco",
    "TGT": "Target",
    "HD": "Home Depot",
    "LOW": "Lowe's",
    "NKE": "Nike",
    "MCD": "McDonald's",
    "SBUX": "Starbucks",
    "PEP": "PepsiCo",
    "KO": "Coca-Cola",
    "PG": "Procter & Gamble",
    "UL": "Unilever",
    "NSRGY": "Nestle",
    "EL": "Estee Lauder",
    "LVMUY": "LVMH",
    "KHC": "Kraft Heinz",
    "PM": "Philip Morris",
    "MO": "Altria",
    "BUD": "AB InBev",
    "DIS": "Disney",
    "NFLX": "Netflix",
    "CMCSA": "Comcast",
    "T": "AT&T",
    "VZ": "Verizon",
    "CHTR": "Charter Comm.",
    "TMUS": "T-Mobile",
    "DISH": "DISH Network",
    "FOXA": "Fox Corp",
    "TWX": "Time Warner",
    # Healthcare
    "JNJ": "Johnson & Johnson",
    "PFE": "Pfizer",
    "ABBV": "AbbVie",
    "LLY": "Eli Lilly",
    "MRK": "Merck",
    "NVS": "Novartis",
    "AZN": "AstraZeneca",
    "UNH": "UnitedHealth",
    "DHR": "Danaher",
    "TMO": "Thermo Fisher",
    "ISRG": "Intuitive Surgical",
    "SYK": "Stryker",
    "BDX": "Becton Dickinson",
    "BSX": "Boston Scientific",
    "MDT": "Medtronic",
    "ZTS": "Zoetis",
    "VRTX": "Vertex Pharma",
    "REGN": "Regeneron",
    "GILD": "Gilead",
    "BMY": "Bristol Myers",
    "AMGN": "Amgen",
    "BIIB": "Biogen",
    "CVS": "CVS Health",
    "WBA": "Walgreens Boots",
    "CI": "Cigna",
    "HUM": "Humana",
    "ANTM": "Anthem",
    "DGX": "Quest Diagnostics",
    "LH": "LabCorp",
    "IQV": "IQVIA",
    # Energy & Industrials
    "XOM": "Exxon Mobil",
    "CVX": "Chevron",
    "SHEL": "Shell",
    "TTE": "TotalEnergies",
    "BP": "BP",
    "ENB": "Enbridge",
    "COP": "ConocoPhillips",
    "EOG": "EOG Resources",
    "BHP": "BHP Group",
    "RIO": "Rio Tinto",
    "CAT": "Caterpillar",
    "DE": "Deere & Co.",
    "HON": "Honeywell",
    "GE": "General Electric",
    "BA": "Boeing",
    "RTX": "RTX Corp",
    "LMT": "Lockheed Martin",
    "NOC": "Northrop Grumman",
    "GD": "General Dynamics",
    "MMM": "3M",
    "UPS": "UPS",
    "FDX": "FedEx",
    "UNP": "Union Pacific",
    "CSX": "CSX Corp",
    "NSC": "Norfolk Southern",
    "EMR": "Emerson",
    "ITW": "Illinois Tool Works",
    "ETN": "Eaton",
    "WM": "Waste Management",
    "RSG": "Republic Services",
    # Emerging Markets, Crypto, EV, Commodities, etc.
    "BABA": "Alibaba",
    "TCEHY": "Tencent",
    "JD": "JD.com",
    "PDD": "Pinduoduo",
    "BIDU": "Baidu",
    "NTES": "NetEase",
    "005930.KS": "Samsung Elec.",
    "000660.KS": "SK Hynix",
    "0688.HK": "China Shenhua",
    "3690.HK": "Meituan",
    "601318.SS": "Ping An Ins.",
    "600519.SS": "Kweichow Moutai",
    "601288.SS": "Agri. Bank China",
    "RELIANCE.NS": "Reliance Ind.",
    "TATASTEEL.NS": "Tata Steel",
    "INFY": "Infosys",
    "HDB": "HDFC Bank",
    "ICICIY": "ICICI Bank",
    "ITUB": "Itau Unibanco",
    "VALE": "Vale S.A.",
    "PBR": "Petrobras",
    "BSBR": "Banco Santander Brasil",
    "BBD": "Banco Bradesco",
    "FMX": "Femsa",
    "AMX": "America Movil",
    "GRMN": "Garmin", # Example, often in other categories too
    "ASR": "Grupo Aeroportuario del Sureste",
    "GFNORTEO.MX": "GFNorte",
    "BAP": "Credicorp",
    # Crypto & Blockchain
    "MARA": "Marathon Digital",
    "RIOT": "Riot Platforms",
    "MSTR": "MicroStrategy",
    "HUT": "Hut 8 Mining",
    "BITF": "Bitfarms",
    "CLSK": "CleanSpark",
    "BTBT": "Bit Digital",
    "MOGO": "Mogo Inc.",
    "SI": "Silvergate Capital", # Note: Check current status
    "CAN": "Canaan Inc.",
    "HIVE": "HIVE Blockchain",
    "BTG": "B2Gold Corp.", # Often in materials/mining too
    "ARBK": "Argo Blockchain",
    "CIFR": "Cipher Mining",
    "DMGI": "DMG Blockchain",
    "GLXY": "Galaxy Digital",
    "BITI": "ProShares Short Bitcoin ETF",
    "BITO": "ProShares Bitcoin Strategy ETF",
    "BTCC": "Purpose Bitcoin ETF",
    # EV & Clean Energy
    "TSLA": "Tesla",
    "NIO": "NIO Inc.",
    "LI": "Li Auto",
    "XPEV": "XPeng",
    "RIVN": "Rivian",
    "LCID": "Lucid Motors",
    "FSR": "Fisker", # Note: Check current status
    "PLUG": "Plug Power",
    "FCEL": "FuelCell Energy",
    "BE": "Bloom Energy",
    "ENPH": "Enphase Energy",
    "SEDG": "SolarEdge",
    "FSLR": "First Solar",
    "RUN": "Sunrun",
    "SPWR": "SunPower",
    "NEE": "NextEra Energy",
    "DQ": "Daqo New Energy",
    "JKS": "JinkoSolar",
    "CSIQ": "Canadian Solar",
    "NOVA": "Sunnova Energy",
    "MAXN": "Maxeon Solar",
    "ARRY": "Array Technologies",
    "SHLS": "Shoals Technologies",
    "STEM": "Stem Inc.",
    "BLDP": "Ballard Power",
    "BEEM": "Beam Global",
    "SPI": "SPI Energy",
    "ENVX": "Enovix",
    # Commodities & ETFs
    "GLD": "Gold ETF (SPDR)",
    "SLV": "Silver ETF (iShares)",
    "USO": "US Oil Fund",
    "UNG": "US Natural Gas Fund",
    "DBC": "Commodities Index Fund (Invesco)",
    "IAU": "Gold Mini ETF (iShares)",
    "GDX": "Gold Miners ETF (VanEck)",
    "GDXJ": "Jr Gold Miners ETF (VanEck)",
    "XLE": "Energy Select Sector SPDR Fund",
    "XLF": "Financial Select Sector SPDR Fund",
    "XLV": "Health Care Select Sector SPDR Fund",
    "XLI": "Industrial Select Sector SPDR Fund",
    "XLY": "Consumer Discretionary Select Sector SPDR Fund",
    "XLP": "Consumer Staples Select Sector SPDR Fund",
    "XLC": "Communication Services Select Sector SPDR Fund",
    "XLK": "Technology Select Sector SPDR Fund",
    "XLU": "Utilities Select Sector SPDR Fund",
    "XLB": "Materials Select Sector SPDR Fund",
    "SPY": "S&P 500 ETF (SPDR)",
    "QQQ": "Nasdaq 100 ETF (Invesco)",
    "IWM": "Russell 2000 ETF (iShares)",
    "DIA": "Dow 30 ETF (SPDR)",
    "VOO": "Vanguard S&P 500 ETF",
    "VTI": "Vanguard Total Stock Market ETF",
    "VEA": "Vanguard FTSE Developed Markets ETF",
    "VWO": "Vanguard FTSE Emerging Markets ETF",
    "BND": "Vanguard Total Bond Market ETF",
    "LQD": "iShares iBoxx $ Investment Grade Corporate Bond ETF",
    "HYG": "iShares iBoxx $ High Yield Corporate Bond ETF",
    "TLT": "iShares 20+ Year Treasury Bond ETF"
    # Add more as needed, ensure all tickers from GLOBAL_TICKERS are covered
}

# Page Configuration
st.set_page_config(
    page_title="AlphaQuant Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional dark theme
st.markdown("""
<style>
    :root {
        --primary-color: #1f2937;
        --secondary-color: #4b5563;
        --accent-color: #10b981;
        --background-color: #111827;
        --surface-color: #1f2937;
        --text-color: #f9fafb;
        --text-secondary: #9ca3af;
        --positive-color: #10b981;
        --negative-color: #ef4444;
    }
    
    .main {
        background-color: var(--background-color);
        color: var(--text-color);
    }
    
    .stSelectbox div, .stTextInput div, .stDateInput div, .stNumberInput div {
        background-color: var(--surface-color) !important;
        border-radius: 8px !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.5) !important;
        color: var(--text-color) !important;
        border: 1px solid var(--secondary-color) !important;
    }
    
    .stButton>button {
        background-color: var(--accent-color) !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 0.5rem 1rem !important;
        font-weight: 500 !important;
        transition: all 0.2s !important;
        border: none !important;
    }
    
    .stButton>button:hover {
        background-color: #059669 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.5) !important;
    }
    
    .stMetric {
        background-color: var(--surface-color) !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.5) !important;
        color: var(--text-color) !important;
        border: 1px solid var(--secondary-color) !important;
    }
    
    .stDataFrame {
        border-radius: 12px !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.5) !important;
        color: var(--text-color) !important;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--accent-color) !important;
        color: white !important;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #1f2937 0%, #4b5563 100%) !important;
        color: white !important;
    }
    
    .sidebar .sidebar-content a {
        color: var(--accent-color) !important;
    }
    
    .sidebar .sidebar-content .stMarkdown h1, 
    .sidebar .sidebar-content .stMarkdown h2,
    .sidebar .sidebar-content .stMarkdown h3 {
        color: white !important;
    }
    
    .positive {
        color: var(--positive-color) !important;
    }
    
    .negative {
        color: var(--negative-color) !important;
    }
    
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: var(--surface-color);
        color: var(--text-color);
        text-align: center;
        padding: 10px;
        border-top: 1px solid var(--secondary-color);
    }
</style>
""", unsafe_allow_html=True)

# Title Section
st.title("📊 AlphaQuant Pro")
st.markdown("""
<div style="color: var(--text-secondary); margin-bottom: 2rem;">
    Advanced Stock Market Analysis & Forecasting Platform
</div>
""", unsafe_allow_html=True)

# Enhanced Ticker List (1000+ global companies)
GLOBAL_TICKERS = {
    "Technology": [
        "AAPL", "MSFT", "NVDA", "AVGO", "ASML", "TSM", "ADBE", "CSCO", "ACN", "CRM",
        "ORCL", "SAP", "INTU", "AMD", "INTC", "QCOM", "TXN", "AMAT", "LRCX", "KLAC",
        "SNOW", "PLTR", "U", "DDOG", "ZS", "CRWD", "NET", "MDB", "NOW", "TEAM", "SHOP",
        "SQ", "PYPL", "DOCU", "ZM", "OKTA", "TWLO", "FSLY", "PINS", "SPOT"
    ],
    "Finance": [
        "JPM", "BAC", "WFC", "C", "HSBC", "GS", "MS", "BLK", "SCHW", "AXP",
        "V", "MA", "PYPL", "SQ", "COIN", "MUFG", "RY", "TD", "BNPQY", "ING",
        "USB", "PNC", "TFC", "BK", "STT", "DFS", "ALLY", "CBOE", "MCO", "SPGI"
    ],
    "Consumer": [
        "AMZN", "WMT", "COST", "TGT", "HD", "LOW", "NKE", "MCD", "SBUX", "PEP",
        "KO", "PG", "UL", "NSRGY", "EL", "LVMUY", "KHC", "PM", "MO", "BUD",
        "DIS", "NFLX", "CMCSA", "T", "VZ", "CHTR", "TMUS", "DISH", "FOXA", "TWX"
    ],
    "Healthcare": [
        "JNJ", "PFE", "ABBV", "LLY", "MRK", "NVS", "AZN", "UNH", "DHR", "TMO",
        "ISRG", "SYK", "BDX", "BSX", "MDT", "ZTS", "VRTX", "REGN", "GILD", "BMY",
        "AMGN", "BIIB", "CVS", "WBA", "CI", "HUM", "ANTM", "DGX", "LH", "IQV"
    ],
    "Energy & Industrials": [
        "XOM", "CVX", "SHEL", "TTE", "BP", "ENB", "COP", "EOG", "BHP", "RIO",
        "CAT", "DE", "HON", "GE", "BA", "RTX", "LMT", "NOC", "GD", "MMM",
        "UPS", "FDX", "UNP", "CSX", "NSC", "EMR", "ITW", "ETN", "WM", "RSG"
    ],
    "Emerging Markets": [
        "BABA", "TCEHY", "JD", "PDD", "BIDU", "NTES", "TSM", "005930.KS", "000660.KS",
        "0688.HK", "3690.HK", "601318.SS", "600519.SS", "601288.SS", "RELIANCE.NS",
        "TATASTEEL.NS", "INFY", "HDB", "ICICIY", "ITUB", "VALE", "PBR", "BSBR", "BBD",
        "FMX", "AMX", "GRMN", "ASR", "GFNORTEO.MX", "BAP"
    ],
    "Crypto & Blockchain": [
        "COIN", "MARA", "RIOT", "MSTR", "HUT", "BITF", "CLSK", "BTBT", "MOGO", "SI",
        "CAN", "HIVE", "BTG", "ARBK", "CIFR", "DMGI", "GLXY", "BITI", "BITO", "BTCC"
    ],
    "EV & Clean Energy": [
        "TSLA", "NIO", "LI", "XPEV", "RIVN", "LCID", "FSR", "PLUG", "FCEL", "BE",
        "ENPH", "SEDG", "FSLR", "RUN", "SPWR", "NEE", "DQ", "JKS", "CSIQ", "SEDG",
        "NOVA", "MAXN", "ARRY", "SHLS", "STEM", "BLDP", "BEEM", "SPI", "NOVA", "ENVX"
    ],
    "Commodities & ETFs": [
        "GLD", "SLV", "USO", "UNG", "DBC", "IAU", "GDX", "GDXJ", "XLE", "XLF",
        "XLV", "XLI", "XLY", "XLP", "XLC", "XLK", "XLU", "XLB", "SPY", "QQQ",
        "IWM", "DIA", "VOO", "VTI", "VEA", "VWO", "BND", "LQD", "HYG", "TLT"
    ]
}

# --- Novita AI Integration ---
NOVITA_API_URL = "https://api.novita.ai/v3/openai/chat/completions"
NOVITA_API_KEY = "sk_uI9LhVcHFwMHwdSaO2uISiUdJkFYtMtulJfQTOFet34"

def generate_novita_insight(ticker, forecast_price, percent_change, direction, period="next week"):
    """
    Generate a human-readable financial insight using Novita AI.
    """
    prompt = (
        f"Write a professional financial analyst summary for {ticker}. "
        f"The predicted price is ${forecast_price:,.2f} ({direction}, {percent_change:+.2f}%) for {period}. "
        "Explain possible reasons for this movement in a concise, insightful way."
    )
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {NOVITA_API_KEY}"
    }
    data = {
        "model": "meta-llama/llama-3.2-1b-instruct",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "response_format": {"type": "text"}
    }
    try:
        response = requests.post(NOVITA_API_URL, headers=headers, json=data, timeout=20)
        response.raise_for_status()
        result = response.json()
        # Extract the generated text
        insight = result["choices"][0]["message"]["content"].strip()
        return insight
    except Exception as e:
        return f"Insight generation unavailable: {e}"

# Session State Management
if 'stock_data' not in st.session_state:
    st.session_state.stock_data = None
if 'current_ticker' not in st.session_state:
    st.session_state.current_ticker = None
if 'comparison_tickers' not in st.session_state:
    st.session_state.comparison_tickers = []
if 'forecast_data' not in st.session_state:
    st.session_state.forecast_data = None
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = None
if 'model_performance' not in st.session_state:
    st.session_state.model_performance = None

# Custom Session with Retries
def create_session():
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=[500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    })
    return session

# Enhanced Data Fetching Function with Technical Indicators
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_stock_data(ticker, start_date, end_date, interval='1d'):
    try:
        ticker_obj = yf.Ticker(ticker)
        data = ticker_obj.history(
            start=start_date,
            end=end_date,
            interval=interval,
            auto_adjust=False,
            actions=True
        )
        
        if data is None or data.empty:
            return None
            
        # Calculate technical indicators
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['SMA_200'] = data['Close'].rolling(window=200).mean()
        data['Daily_Return'] = data['Close'].pct_change()
        data['Cumulative_Return'] = (1 + data['Daily_Return']).cumprod() - 1
        
        # Add RSI
        rsi_indicator = RSIIndicator(close=data['Close'], window=14)
        data['RSI'] = rsi_indicator.rsi()
        
        # Add Bollinger Bands
        bb_indicator = BollingerBands(close=data['Close'], window=20, window_dev=2)
        data['BB_upper'] = bb_indicator.bollinger_hband()
        data['BB_middle'] = bb_indicator.bollinger_mavg()
        data['BB_lower'] = bb_indicator.bollinger_lband()
        
        # Add MACD
        macd_indicator = MACD(close=data['Close'], window_slow=26, window_fast=12, window_sign=9)
        data['MACD'] = macd_indicator.macd()
        data['MACD_signal'] = macd_indicator.macd_signal()
        data['MACD_diff'] = macd_indicator.macd_diff()
        
        # Format the data
        data = data.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Adj Close': 'adj_close',
            'Volume': 'volume'
        })
        
        data = data.reset_index().rename(columns={'Date': 'date'})
        return data[['date', 'open', 'high', 'low', 'close', 'adj_close', 'volume', 
                    'SMA_20', 'SMA_50', 'SMA_200', 'Daily_Return', 'Cumulative_Return',
                    'RSI', 'BB_upper', 'BB_middle', 'BB_lower', 'MACD', 'MACD_signal', 'MACD_diff']]
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None

# Forecasting Models
def sarima_forecast(data, column, p, d, q, seasonal_order, forecast_period):
    model = sm.tsa.statespace.SARIMAX(data[column], order=(p, d, q), seasonal_order=(p, d, q, seasonal_order))
    model = model.fit(disp=False)
    
    # Predict the future values
    predictions = model.get_prediction(start=len(data), end=len(data) + forecast_period)
    predictions = predictions.predicted_mean
    
    # Add index to the predictions
    predictions.index = pd.date_range(start=data['date'].iloc[-1] + pd.Timedelta(days=1), periods=len(predictions), freq='D')
    predictions = pd.DataFrame(predictions)
    predictions.insert(0, "date", predictions.index, True)
    predictions.reset_index(drop=True, inplace=True)
    predictions.columns = ['date', 'forecast']
    
    return model, predictions

def random_forest_forecast(data, column, forecast_period):
    # Feature engineering
    data['day'] = data['date'].dt.day
    data['month'] = data['date'].dt.month
    data['year'] = data['date'].dt.year
    data['day_of_week'] = data['date'].dt.dayofweek
    data['day_of_year'] = data['date'].dt.dayofyear
    
    # Prepare data
    X = data[['day', 'month', 'year', 'day_of_week', 'day_of_year']]
    y = data[column]
    
    # Train model
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)
    
    # Create future dates
    last_date = data['date'].iloc[-1]
    future_dates = [last_date + pd.Timedelta(days=x) for x in range(1, forecast_period+1)]
    
    # Prepare future features
    future_df = pd.DataFrame({'date': future_dates})
    future_df['day'] = future_df['date'].dt.day
    future_df['month'] = future_df['date'].dt.month
    future_df['year'] = future_df['date'].dt.year
    future_df['day_of_week'] = future_df['date'].dt.dayofweek
    future_df['day_of_year'] = future_df['date'].dt.dayofyear
    
    # Make predictions
    future_predictions = model.predict(future_df[['day', 'month', 'year', 'day_of_week', 'day_of_year']])
    future_df['forecast'] = future_predictions
    
    return model, future_df[['date', 'forecast']]

def lstm_forecast(data, column, forecast_period, seq_length=30, epochs=50, batch_size=32):
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[column].values.reshape(-1, 1))
    
    # Create sequences
    def create_sequences(dataset, seq_length):
        X, y = [], []
        for i in range(len(dataset) - seq_length):
            X.append(dataset[i:i + seq_length, 0])
            y.append(dataset[i + seq_length, 0])
        return np.array(X), np.array(y)
    
    X, y = create_sequences(scaled_data, seq_length)
    
    # Split into train/test (use all data for training in forecasting scenario)
    X_train, y_train = X, y
    
    # Reshape data for LSTM [samples, time steps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    
    # Build LSTM model
    model = Sequential()
    model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=100, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    
    # Prepare data for forecasting
    last_sequence = scaled_data[-seq_length:]
    future_predictions = []
    
    for _ in range(forecast_period):
        x_input = np.reshape(last_sequence, (1, seq_length, 1))
        pred = model.predict(x_input, verbose=0)
        future_predictions.append(pred[0,0])
        last_sequence = np.append(last_sequence[1:], pred)
    
    # Inverse transform predictions
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    
    # Create future dates
    last_date = data['date'].iloc[-1]
    future_dates = [last_date + pd.Timedelta(days=x) for x in range(1, forecast_period+1)]
    
    # Create result dataframe
    result_df = pd.DataFrame({
        'date': future_dates,
        'forecast': future_predictions.flatten()
    })
    
    return model, result_df

def prophet_forecast(data, column, forecast_period):
    # Prepare data for Prophet
    prophet_data = data[['date', column]].copy()
    prophet_data.columns = ['ds', 'y']
    
    # Create and fit model
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='multiplicative'
    )
    model.add_country_holidays(country_name='US')
    model.fit(prophet_data)
    
    # Make future dataframe
    future = model.make_future_dataframe(periods=forecast_period, freq='D')
    
    # Forecast
    forecast = model.predict(future)
    
    # Prepare result dataframe
    result_df = forecast[['ds', 'yhat']].tail(forecast_period)
    result_df.columns = ['date', 'forecast']
    
    return model, result_df

def get_display_ticker(ticker):
    name = TICKER_NAMES.get(ticker.upper(), ticker.upper())
    return f"{name} ({ticker.upper()})"

# Sidebar - Filters and Info
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="color: white;">AlphaQuant Pro</h1>
        <p style="color: rgba(255,255,255,0.8);">Advanced Market Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation
    page = st.radio(
        "Navigation",
        ["Market Analysis", "Forecasting"],
        index=0
    )
    
    # Sector Selection
    selected_sector = st.selectbox(
        "Select Sector",
        list(GLOBAL_TICKERS.keys()),
        index=0
    )
    
    # Ticker Selection
    ticker_options = GLOBAL_TICKERS[selected_sector]
    display_ticker_options = [get_display_ticker(t) for t in ticker_options]
    selected_display = st.selectbox(
        "Select Ticker",
        display_ticker_options,
        index=0
    )
    selected_ticker = ticker_options[display_ticker_options.index(selected_display)]

    # Date Range
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=dt.date(2020, 1, 1),
            min_value=dt.date(1980, 1, 1),
            max_value=dt.date.today()
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=dt.date.today(),
            min_value=dt.date(1980, 1, 1),
            max_value=dt.date.today()
        )
    
    # Comparison Tickers
    all_tickers = [t for sector in GLOBAL_TICKERS.values() for t in sector]
    display_all_tickers = [get_display_ticker(t) for t in all_tickers]
    selected_comparison_display = st.multiselect(
        "Compare With (Max 4)",
        display_all_tickers,
        default=[],
        max_selections=4
    )
    comparison_tickers = [all_tickers[display_all_tickers.index(d)] for d in selected_comparison_display]
    
    # Interval Selection
    interval = st.selectbox(
        "Data Interval",
        ["1d", "1wk", "1mo"],
        index=0
    )
    
    if page == "Forecasting":
        st.markdown("---")
        st.markdown("### Forecasting Parameters")
        
        # Model Selection
        st.session_state.selected_model = st.selectbox(
            "Select Forecasting Model",
            ["SARIMA", "Random Forest", "LSTM", "Prophet"],
            index=0
        )
        
        # Forecast Period
        forecast_period = st.slider(
            "Forecast Period (days)",
            1, 365, 30
        )
        
        # Model-specific parameters
        if st.session_state.selected_model == "SARIMA":
            col1, col2, col3 = st.columns(3)
            with col1:
                p = st.slider('p (AR)', 0, 5, 1)
            with col2:
                d = st.slider('d (I)', 0, 2, 1)
            with col3:
                q = st.slider('q (MA)', 0, 5, 1)
            
            seasonal_order = st.slider('Seasonal Period', 1, 365, 30)
            
        elif st.session_state.selected_model == "LSTM":
            seq_length = st.slider('Sequence Length', 1, 90, 30)
            epochs = st.slider('Epochs', 1, 200, 50)
            batch_size = st.slider('Batch Size', 1, 64, 32)
    
    st.markdown("---")
    
    # Social Links
    st.markdown("""
<h1 style="font-family: 'poppins'; font-weight: bold; color: Green;">👨‍💻Author: Muhammad Atif Latif</h1>

<a href="https://github.com/m-Atif-Latif" target="_blank">
    <img alt="GitHub" src="https://img.shields.io/badge/GitHub-Profile-blue?style=for-the-badge&logo=github">
</a>
<a href="https://www.kaggle.com/matiflatif" target="_blank">
    <img alt="Kaggle" src="https://img.shields.io/badge/Kaggle-Profile-blue?style=for-the-badge&logo=kaggle">
</a>
<a href="https://www.linkedin.com/in/muhammad-atif-latif-13a171318?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app" target="_blank">
    <img alt="LinkedIn" src="https://img.shields.io/badge/LinkedIn-Profile-blue?style=for-the-badge&logo=linkedin">
</a>
<a href="https://x.com/mianatif5867?s=09" target="_blank">
    <img alt="Twitter/X" src="https://img.shields.io/badge/Twitter-Profile-blue?style=for-the-badge&logo=twitter">
</a>
<a href="https://www.instagram.com/its_atif_ai/" target="_blank">
    <img alt="Instagram" src="https://img.shields.io/badge/Instagram-Profile-blue?style=for-the-badge&logo=instagram">
</a>
<a href="mailto:muhammadatiflatif67@gmail.com" target="_blank">
    <img alt="Email" src="https://img.shields.io/badge/Email-Contact Me-red?style=for-the-badge&logo=email">
</a>
""", unsafe_allow_html=True)

# Main Content
if page == "Market Analysis":
    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("📊 Fetch Market Data", use_container_width=True):
            with st.spinner(f"Loading data for {selected_ticker}..."):
                data = fetch_stock_data(selected_ticker, start_date, end_date, interval)
                if data is not None:
                    st.session_state.stock_data = data
                    st.session_state.current_ticker = selected_ticker
                    st.session_state.comparison_tickers = comparison_tickers
                    st.success("Data loaded successfully!")
    with col2:
        if st.button("🔄 Clear Data", use_container_width=True, type="secondary"):
            st.session_state.stock_data = None
            st.session_state.current_ticker = None
            st.session_state.comparison_tickers = []
            st.session_state.forecast_data = None
            st.rerun()

    # Display Data
    if st.session_state.stock_data is not None:
        df = st.session_state.stock_data
        ticker = st.session_state.current_ticker
        
        # Metrics Row
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            price_change = df.iloc[-1]['close'] - df.iloc[-2]['close']
            price_change_pct = (price_change / df.iloc[-2]['close']) * 100
            st.metric(
                "Current Price",
                f"${df.iloc[-1]['close']:,.2f}",
                f"{price_change:,.2f} ({price_change_pct:.2f}%)",
                delta_color="normal"
            )
        with col2:
            st.metric(
                "52 Week Range",
                f"${df['close'].min():,.2f} - ${df['close'].max():,.2f}"
            )
        with col3:
            daily_return = df.iloc[-1]['Daily_Return'] * 100
            st.metric(
                "Daily Return",
                f"{daily_return:.2f}%",
                delta_color="inverse" if daily_return < 0 else "normal"
            )
        with col4:
            vol = df['volume'].mean() / 1_000_000
            st.metric(
                "Avg Volume",
                f"{vol:,.1f}M"
            )
        
        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["Price Analysis", "Technical Indicators", "Performance Comparison", "Data Table"])
        
        with tab1:
            # Interactive Chart
            st.markdown(f"### {ticker} Price Analysis")
            
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                              vertical_spacing=0.05, row_heights=[0.7, 0.3])
            
            # Price and Moving Averages
            fig.add_trace(
                go.Candlestick(
                    x=df['date'],
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name="Price",
                    increasing_line_color='#2ecc71',
                    decreasing_line_color='#e74c3c'
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df['date'],
                    y=df['SMA_50'],
                    name="50-Day SMA",
                    line=dict(color='#3498db', width=2)
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df['date'],
                    y=df['SMA_200'],
                    name="200-Day SMA",
                    line=dict(color='#f39c12', width=2)
                ),
                row=1, col=1
            )
            
            # Volume
            fig.add_trace(
                go.Bar(
                    x=df['date'],
                    y=df['volume'],
                    name="Volume",
                    marker_color='#7f8c8d'
                ),
                row=2, col=1
            )
            
            fig.update_layout(
                height=800,
                showlegend=True,
                hovermode="x unified",
                template="plotly_dark",
                margin=dict(l=20, r=20, t=40, b=20),
                xaxis_rangeslider_visible=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # Technical Indicators
            st.markdown(f"### {ticker} Technical Indicators")
            
            # RSI Chart
            fig_rsi = go.Figure()
            fig_rsi.add_trace(
                go.Scatter(
                    x=df['date'],
                    y=df['RSI'],
                    name="RSI",
                    line=dict(color='#9b59b6', width=2)
                )
            )
            fig_rsi.add_hline(y=70, line_dash="dot", line_color="red", annotation_text="Overbought (70)")
            fig_rsi.add_hline(y=30, line_dash="dot", line_color="green", annotation_text="Oversold (30)")
            fig_rsi.update_layout(
                title="Relative Strength Index (RSI)",
                height=400,
                template="plotly_dark"
            )
            st.plotly_chart(fig_rsi, use_container_width=True)
            
            # Bollinger Bands
            fig_bb = go.Figure()
            fig_bb.add_trace(
                go.Scatter(
                    x=df['date'],
                    y=df['close'],
                    name="Price",
                    line=dict(color='#3498db', width=2)
                )
            )
            fig_bb.add_trace(
                go.Scatter(
                    x=df['date'],
                    y=df['BB_upper'],
                    name="Upper Band",
                    line=dict(color='#e74c3c', width=1)
                )
            )
            fig_bb.add_trace(
                go.Scatter(
                    x=df['date'],
                    y=df['BB_middle'],
                    name="Middle Band",
                    line=dict(color='#f39c12', width=1)
                )
            )
            fig_bb.add_trace(
                go.Scatter(
                    x=df['date'],
                    y=df['BB_lower'],
                    name="Lower Band",
                    line=dict(color='#2ecc71', width=1)
                )
            )
            fig_bb.update_layout(
                title="Bollinger Bands",
                height=400,
                template="plotly_dark"
            )
            st.plotly_chart(fig_bb, use_container_width=True)
            
            # MACD
            fig_macd = go.Figure()
            fig_macd.add_trace(
                go.Scatter(
                    x=df['date'],
                    y=df['MACD'],
                    name="MACD",
                    line=dict(color='#3498db', width=2)
                )
            )
            fig_macd.add_trace(
                go.Scatter(
                    x=df['date'],
                    y=df['MACD_signal'],
                    name="Signal Line",
                    line=dict(color='#f39c12', width=2)
                )
            )
            fig_macd.add_trace(
                go.Bar(
                    x=df['date'],
                    y=df['MACD_diff'],
                    name="Histogram",
                    marker_color=np.where(df['MACD_diff'] < 0, '#e74c3c', '#2ecc71')
                )
            )
            fig_macd.update_layout(
                title="MACD (Moving Average Convergence Divergence)",
                height=400,
                template="plotly_dark"
            )
            st.plotly_chart(fig_macd, use_container_width=True)
        
        with tab3:
            # Comparison Charts
            if st.session_state.comparison_tickers:
                st.markdown("### Performance Comparison")
                
                comparison_data = {}
                for comp_ticker in st.session_state.comparison_tickers:
                    comp_df = fetch_stock_data(comp_ticker, start_date, end_date, interval)
                    if comp_df is not None:
                        comparison_data[comp_ticker] = comp_df
                
                if comparison_data:
                    fig = go.Figure()
                    
                    # Normalize all prices to percentage change from start date
                    base_price = df.iloc[0]['close']
                    fig.add_trace(
                        go.Scatter(
                            x=df['date'],
                            y=(df['close'] / base_price - 1) * 100,
                            name=ticker,
                            line=dict(width=3)
                        )
                    )
                    
                    for comp_ticker, comp_df in comparison_data.items():
                        comp_base = comp_df.iloc[0]['close']
                        fig.add_trace(
                            go.Scatter(
                                x=comp_df['date'],
                                y=(comp_df['close'] / comp_base - 1) * 100,
                                name=comp_ticker
                            )
                        )
                    
                    fig.update_layout(
                        title="Normalized Performance Comparison",
                        yaxis_title="Percentage Change (%)",
                        hovermode="x unified",
                        height=500,
                        template="plotly_dark"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No comparison tickers selected. Please add some in the sidebar.")
        
        with tab4:
            # Data Table and Export
            st.markdown("### Market Data Table")
            
            # Show technical indicators in the table
            display_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 
                          'SMA_20', 'SMA_50', 'SMA_200', 'Daily_Return', 'RSI']
            
            st.dataframe(
                df[display_cols].rename(columns={
                    'date': 'Date',
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'close': 'Close',
                    'volume': 'Volume',
                    'SMA_20': '20-Day SMA',
                    'SMA_50': '50-Day SMA',
                    'SMA_200': '200-Day SMA',
                    'Daily_Return': 'Daily Return',
                    'RSI': 'RSI'
                }).style.format({
                    'Open': '{:,.2f}',
                    'High': '{:,.2f}',
                    'Low': '{:,.2f}',
                    'Close': '{:,.2f}',
                    'Volume': '{:,.0f}',
                    '20-Day SMA': '{:,.2f}',
                    '50-Day SMA': '{:,.2f}',
                    '200-Day SMA': '{:,.2f}',
                    'Daily Return': '{:.2%}',
                    'RSI': '{:.2f}'
                }).applymap(lambda x: 'color: #10b981' if isinstance(x, (int, float)) and x > 0 else 'color: #ef4444', 
                           subset=['Daily Return']),
                height=400,
                use_container_width=True
            )
            
            # Export Options
            st.markdown("---")
            st.markdown("### Export Data")
            col1, col2 = st.columns(2)
            with col1:
                csv = df.to_csv(index=False)
                st.download_button(
                    "📥 Download CSV",
                    csv,
                    file_name=f"{ticker}_market_data_{start_date}_to_{end_date}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            with col2:
                # Export Plotly chart as PNG
                try:
                    chart_png = fig.to_image(format="png")
                    st.download_button(
                        "📊 Download Chart as PNG",
                        chart_png,
                        file_name=f"{ticker}_chart.png",
                        mime="image/png",
                        use_container_width=True
                    )
                except Exception as e:
                    st.download_button(
                        "📊 Download Chart as PNG",
                        b"",
                        file_name=f"{ticker}_chart.png",
                        disabled=True,
                        help=f"PNG export failed: {str(e)}",
                        use_container_width=True
                    )

elif page == "Forecasting":
    if st.button("🔮 Run Forecast", use_container_width=True):
        # Always fetch fresh data and run forecast for the selected ticker
        with st.spinner(f"Loading and forecasting data for {selected_ticker}..."):
            data = fetch_stock_data(selected_ticker, start_date, end_date, interval)
            if data is None:
                st.error(f"No data found for {selected_ticker}.")
            else:
                # Update session state
                st.session_state.stock_data = data
                st.session_state.current_ticker = selected_ticker
                st.session_state.comparison_tickers = comparison_tickers
                # Run selected forecast model
                if st.session_state.selected_model == "SARIMA":
                    model, forecast = sarima_forecast(data, 'close', p, d, q, seasonal_order, forecast_period)
                elif st.session_state.selected_model == "Random Forest":
                    model, forecast = random_forest_forecast(data, 'close', forecast_period)
                elif st.session_state.selected_model == "LSTM":
                    model, forecast = lstm_forecast(data, 'close', forecast_period, seq_length, epochs, batch_size)
                elif st.session_state.selected_model == "Prophet":
                    model, forecast = prophet_forecast(data, 'close', forecast_period)
                # Save forecast data
                st.session_state.forecast_data = forecast
                st.success("Forecast completed successfully!")
    
    if st.session_state.forecast_data is not None and st.session_state.stock_data is not None:
        df = st.session_state.stock_data
        forecast = st.session_state.forecast_data
        ticker = st.session_state.current_ticker
        
        # Display forecast results
        st.markdown(f"### {ticker} {st.session_state.selected_model} Forecast")
        
        # Plot actual vs forecast
        fig = go.Figure()
        
        # Add actual data
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=df['close'],
                name="Actual",
                line=dict(color='#3498db', width=2)
            )
        )
        
        # Add forecast data
        fig.add_trace(
            go.Scatter(
                x=forecast['date'],
                y=forecast['forecast'],
                name="Forecast",
                line=dict(color='#f39c12', width=2, dash='dot')
            )
        )
        
        # Add confidence interval if available (Prophet)
        if st.session_state.selected_model == "Prophet":
            prophet_model = model
            future = prophet_model.make_future_dataframe(periods=forecast_period, freq='D')
            forecast_full = prophet_model.predict(future)
            
            fig.add_trace(
                go.Scatter(
                    x=forecast_full['ds'],
                    y=forecast_full['yhat_upper'],
                    fill=None,
                    mode='lines',
                    line_color='rgba(243,156,18,0.2)',
                    showlegend=False
                )
            )
            
            fig.add_trace(
                go.Scatter(
                    x=forecast_full['ds'],
                    y=forecast_full['yhat_lower'],
                    fill='tonexty',
                    mode='lines',
                    line_color='rgba(243,156,18,0.2)',
                    name="Confidence Interval"
                )
            )
        
        fig.update_layout(
            title=f"{ticker} Price Forecast ({st.session_state.selected_model})",
            xaxis_title="Date",
            yaxis_title="Price",
            height=600,
            template="plotly_dark"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show forecast table
        st.markdown("### Forecast Data")
        st.dataframe(
            forecast.style.format({
                'forecast': '{:,.2f}'
            }),
            height=400,
            use_container_width=True
        )
        # --- Novita AI Insight ---
        try:
            forecast_price = forecast['forecast'].iloc[-1]
            prev_price = df['close'].iloc[-1]
            percent_change = ((forecast_price - prev_price) / prev_price) * 100
            direction = "up" if percent_change > 0 else ("down" if percent_change < 0 else "flat")
            period = f"next {len(forecast)} days"
            ai_insight = generate_novita_insight(ticker, forecast_price, percent_change, direction, period)
            st.markdown(f"**AI Insight:** {ai_insight}")
        except Exception as e:
            st.warning(f"Could not generate AI insight: {e}")
        # Export forecast data
        st.markdown("---")
        st.markdown("### Export Forecast")
        col1, col2 = st.columns(2)
        with col1:
            csv = forecast.to_csv(index=False)
            st.download_button(
                "📥 Download Forecast CSV",
                csv,
                file_name=f"{ticker}_{st.session_state.selected_model}_forecast.csv",
                mime="text/csv",
                use_container_width=True
            )
        with col2:
            try:
                chart_png = fig.to_image(format="png")
                st.download_button(
                    "📊 Download Forecast Chart",
                    chart_png,
                    file_name=f"{ticker}_{st.session_state.selected_model}_forecast.png",
                    mime="image/png",
                    use_container_width=True
                )
            except Exception as e:
                st.download_button(
                    "📊 Download Forecast Chart",
                    b"",
                    file_name=f"{ticker}_{st.session_state.selected_model}_forecast.png",
                    disabled=True,
                    help=f"PNG export failed: {str(e)}",
                    use_container_width=True
                )

# Footer
st.markdown("""
<div class="footer">
    <p>AlphaQuant Pro • Advanced Stock Market Analysis & Forecasting</p>
    <p style="font-size: 0.8rem;">Data provided by Yahoo Finance • Updated at {}</p>
</div>
""".format(dt.datetime.now().strftime("%Y-%m-%d %H:%M")), unsafe_allow_html=True)
