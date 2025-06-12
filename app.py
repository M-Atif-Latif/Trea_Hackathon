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
import asyncio
from datetime import timedelta
import scipy.optimize as sco
from scipy import stats
import yfinance as yf

# --- Zilliz/Milvus Vector DB Integration ---
from pymilvus import MilvusClient
import hashlib
import json

CLUSTER_ENDPOINT = "https://in03-fea35679cd300bf.serverless.gcp-us-west1.cloud.zilliz.com"
TOKEN = "57305ae1a0886d848324bd569fad18c6adebc2b5bd1030e11f86a3de3d4098f118da70a8406f98f9d9116f4e95cb825dd010338a"

# Initialize Milvus client (singleton pattern)
@st.cache_resource(show_spinner=False)
def get_milvus_client():
    try:
        client = MilvusClient(uri=CLUSTER_ENDPOINT, token=TOKEN)
        # Try a simple operation to check connection
        _ = client.list_collections()
        return client
    except Exception as e:
        return None

milvus_client = get_milvus_client()

# Vector DB Helper Functions
def initialize_vector_collections():
    """Initialize vector database collections for different data types"""
    if milvus_client is None:
        return False
    
    try:
        # Collection for storing analysis reports
        analysis_schema = {
            "id": {"type": "int64", "primary": True, "auto_id": True},
            "ticker": {"type": "varchar", "max_length": 20},
            "analysis_type": {"type": "varchar", "max_length": 50},
            "timestamp": {"type": "varchar", "max_length": 50},
            "content": {"type": "varchar", "max_length": 65535},
            "embedding": {"type": "float_vector", "dim": 384}
        }
        
        # Collection for storing documents/reports
        document_schema = {
            "id": {"type": "int64", "primary": True, "auto_id": True},
            "filename": {"type": "varchar", "max_length": 255},
            "doc_type": {"type": "varchar", "max_length": 50},
            "content": {"type": "varchar", "max_length": 65535},
            "upload_date": {"type": "varchar", "max_length": 50},
            "embedding": {"type": "float_vector", "dim": 384}
        }
        
        # Collection for portfolio insights
        portfolio_schema = {
            "id": {"type": "int64", "primary": True, "auto_id": True},
            "portfolio_name": {"type": "varchar", "max_length": 100},
            "tickers": {"type": "varchar", "max_length": 1000},
            "insights": {"type": "varchar", "max_length": 65535},
            "performance_metrics": {"type": "varchar", "max_length": 5000},
            "created_date": {"type": "varchar", "max_length": 50},
            "embedding": {"type": "float_vector", "dim": 384}
        }
        
        collections = ["stock_analysis", "financial_documents", "portfolio_insights"]
        schemas = [analysis_schema, document_schema, portfolio_schema]
        
        for collection_name, schema in zip(collections, schemas):
            if not milvus_client.has_collection(collection_name):
                milvus_client.create_collection(
                    collection_name=collection_name,
                    dimension=384,
                    metric_type="COSINE"
                )
        
        return True
    except Exception as e:
        st.error(f"Error initializing vector collections: {e}")
        return False

def generate_embedding(text):
    """Generate embedding using a simple hash-based approach (placeholder for real embedding)"""
    # In production, use actual embedding models like sentence-transformers
    hash_obj = hashlib.sha256(text.encode())
    hash_hex = hash_obj.hexdigest()
    
    # Convert to 384-dim vector (simple approach for demo)
    vector = []
    for i in range(0, len(hash_hex), 2):
        val = int(hash_hex[i:i+2], 16) / 255.0
        vector.append(val)
    
    # Pad or truncate to 384 dimensions
    while len(vector) < 384:
        vector.extend(vector[:min(len(vector), 384-len(vector))])
    vector = vector[:384]
    
    return vector

def store_analysis_result(ticker, analysis_type, content):
    """Store analysis results in vector database"""
    if milvus_client is None:
        return False
    
    try:
        embedding = generate_embedding(content)
        data = {
            "ticker": ticker,
            "analysis_type": analysis_type,
            "timestamp": dt.datetime.now().isoformat(),
            "content": content,
            "embedding": embedding
        }
        
        result = milvus_client.insert(
            collection_name="stock_analysis",
            data=[data]
        )
        return True
    except Exception as e:
        st.error(f"Error storing analysis: {e}")
        return False

def store_document(filename, doc_type, content):
    """Store uploaded documents in vector database"""
    if milvus_client is None:
        return False
    
    try:
        embedding = generate_embedding(content)
        data = {
            "filename": filename,
            "doc_type": doc_type,
            "content": content,
            "upload_date": dt.datetime.now().isoformat(),
            "embedding": embedding
        }
        
        result = milvus_client.insert(
            collection_name="financial_documents",
            data=[data]
        )
        return True
    except Exception as e:
        st.error(f"Error storing document: {e}")
        return False

def search_similar_analysis(query, limit=5):
    """Search for similar analysis using vector similarity"""
    if milvus_client is None:
        return []
    
    try:
        query_embedding = generate_embedding(query)
        results = milvus_client.search(
            collection_name="stock_analysis",
            data=[query_embedding],
            limit=limit,
            output_fields=["ticker", "analysis_type", "timestamp", "content"]
        )
        return results[0] if results else []
    except Exception as e:
        st.error(f"Error searching analysis: {e}")
        return []

def get_analysis_history(ticker=None, limit=10):
    """Get analysis history from vector database"""
    if milvus_client is None:
        return []
    
    try:
        filter_expr = f'ticker == "{ticker}"' if ticker else None
        results = milvus_client.query(
            collection_name="stock_analysis",
            filter=filter_expr,
            output_fields=["ticker", "analysis_type", "timestamp", "content"],
            limit=limit
        )
        return results
    except Exception as e:
        st.error(f"Error retrieving analysis history: {e}")
        return []

def store_portfolio_insights(portfolio_name, tickers, insights, performance_metrics):
    """Store portfolio analysis in vector database"""
    if milvus_client is None:
        return False
    
    try:
        embedding = generate_embedding(insights)
        data = {
            "portfolio_name": portfolio_name,
            "tickers": ",".join(tickers),
            "insights": insights,
            "performance_metrics": json.dumps(performance_metrics),
            "created_date": dt.datetime.now().isoformat(),
            "embedding": embedding
        }
        
        result = milvus_client.insert(
            collection_name="portfolio_insights",
            data=[data]
        )
        return True
    except Exception as e:
        st.error(f"Error storing portfolio insights: {e}")
        return False

# Initialize collections on startup
if milvus_client:
    initialize_vector_collections()

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
    """Generate a human-readable financial insight using Novita AI."""
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
        insight = result["choices"][0]["message"]["content"].strip()
        return insight
    except Exception as e:
        return f"Insight generation unavailable: {e}"

def generate_analysis_summary(ticker, stock_data):
    """Generate comprehensive analysis summary using Novita AI."""
    if stock_data is None or stock_data.empty:
        return "No data available for analysis."
    
    latest_data = stock_data.iloc[-1]
    price_change = ((latest_data['close'] - stock_data.iloc[0]['close']) / stock_data.iloc[0]['close']) * 100
    volatility = stock_data['Daily_Return'].std() * np.sqrt(252) * 100  # Annualized volatility
    
    prompt = f"""
    Analyze {ticker} stock with the following data:
    - Current Price: ${latest_data['close']:.2f}
    - Price Change (Period): {price_change:.2f}%
    - RSI: {latest_data['RSI']:.2f}
    - Volume: {latest_data['volume']:,.0f}
    - Volatility: {volatility:.2f}%
    - 50-Day SMA: ${latest_data['SMA_50']:.2f}
    - 200-Day SMA: ${latest_data['SMA_200']:.2f}
    
    Provide a comprehensive analysis including:
    1. Technical outlook
    2. Risk assessment
    3. Trading recommendations
    4. Key support/resistance levels
    """
    
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
        response = requests.post(NOVITA_API_URL, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Analysis generation unavailable: {e}"

def ask_ai_question(question, context_data=None):
    """Answer user questions about market analysis using Novita AI."""
    if context_data:
        context = f"Context: {context_data}\n\nQuestion: {question}"
    else:
        context = f"Question about financial markets and stock analysis: {question}"
    
    prompt = f"""
    You are a professional financial analyst. Answer the following question with accurate, 
    helpful information. Be concise but comprehensive.
    
    {context}
    
    Provide a professional response that would help an investor make informed decisions.
    """
    
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
        response = requests.post(NOVITA_API_URL, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Unable to answer question: {e}"

def generate_market_news_summary(ticker):
    """Generate market news and sentiment summary."""
    prompt = f"""
    Provide a brief market sentiment and news summary for {ticker}. Include:
    1. Recent market trends affecting this stock
    2. Sector outlook
    3. Key factors investors should watch
    4. General market sentiment
    
    Keep it concise and focus on actionable insights.
    """
    
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
        response = requests.post(NOVITA_API_URL, headers=headers, json=data, timeout=25)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Market news unavailable: {e}"

# --- Advanced Financial Analytics Functions ---
def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """Calculate Sharpe ratio for risk-adjusted returns"""
    if returns.empty or returns.std() == 0:
        return 0
    excess_returns = returns - risk_free_rate/252
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

def calculate_var(returns, confidence_level=0.05):
    """Calculate Value at Risk"""
    if returns.empty:
        return 0
    return np.percentile(returns, confidence_level * 100)

def monte_carlo_portfolio_optimization(returns, num_simulations=10000):
    """Monte Carlo simulation for portfolio optimization"""
    if returns.empty or len(returns.columns) == 0:
        return np.zeros((4, 1))
    
    num_assets = len(returns.columns)
    results = np.zeros((4, num_simulations))
    
    for i in range(num_simulations):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        
        portfolio_return = np.sum(returns.mean() * weights) * 252
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
        sharpe_ratio = portfolio_return / portfolio_std if portfolio_std != 0 else 0
        
        results[0,i] = portfolio_return
        results[1,i] = portfolio_std
        results[2,i] = sharpe_ratio
        results[3,i] = weights[0] if num_assets > 0 else 0
    
    return results

def detect_chart_patterns(data):
    """Detect common chart patterns using technical analysis"""
    patterns = {}
    
    if data.empty or len(data) < 60:
        return patterns
    
    try:
        # Double bottom pattern
        lows = data['close'].rolling(window=20).min()
        if len(lows) > 40:
            recent_lows = lows.tail(40)
            if len(recent_lows.unique()) >= 2:
                patterns['double_bottom'] = abs(recent_lows.iloc[-1] - recent_lows.iloc[-20]) < 0.02 * recent_lows.iloc[-1]
        
        # Head and shoulders pattern detection
        highs = data['close'].rolling(window=20).max()
        if len(highs) > 60:
            recent_highs = highs.tail(60)
            left_shoulder = recent_highs.iloc[0:20].max()
            head = recent_highs.iloc[20:40].max()
            right_shoulder = recent_highs.iloc[40:60].max()
            
            patterns['head_and_shoulders'] = (head > left_shoulder and head > right_shoulder and 
                                            abs(left_shoulder - right_shoulder) < 0.03 * head)
        
        # Golden cross detection (50-day MA crossing above 200-day MA)
        if 'SMA_50' in data.columns and 'SMA_200' in data.columns:
            patterns['golden_cross'] = (data['SMA_50'].iloc[-1] > data['SMA_200'].iloc[-1] and
                                      data['SMA_50'].iloc[-5] <= data['SMA_200'].iloc[-5])
    except Exception as e:
        st.warning(f"Pattern detection error: {e}")
    
    return patterns

def calculate_fibonacci_levels(high, low):
    """Calculate Fibonacci retracement levels"""
    diff = high - low
    levels = {
        '23.6%': high - 0.236 * diff,
        '38.2%': high - 0.382 * diff,
        '50%': high - 0.5 * diff,
        '61.8%': high - 0.618 * diff,
        '78.6%': high - 0.786 * diff
    }
    return levels

def generate_trading_signals(data):
    """Generate comprehensive trading signals"""
    signals = {}
    
    if data.empty:
        return signals
    
    try:
        # RSI signals
        if 'RSI' in data.columns and not data['RSI'].isna().all():
            rsi = data['RSI'].iloc[-1]
            if rsi > 70:
                signals['RSI'] = 'SELL - Overbought'
            elif rsi < 30:
                signals['RSI'] = 'BUY - Oversold'
            else:
                signals['RSI'] = 'HOLD - Neutral'
        
        # MACD signals
        if 'MACD' in data.columns and 'MACD_signal' in data.columns:
            macd = data['MACD'].iloc[-1]
            macd_signal = data['MACD_signal'].iloc[-1]
            if macd > macd_signal:
                signals['MACD'] = 'BUY - Bullish crossover'
            else:
                signals['MACD'] = 'SELL - Bearish crossover'
        
        # Bollinger Bands signals
        if all(col in data.columns for col in ['close', 'BB_upper', 'BB_lower']):
            price = data['close'].iloc[-1]
            bb_upper = data['BB_upper'].iloc[-1]
            bb_lower = data['BB_lower'].iloc[-1]
            
            if price > bb_upper:
                signals['Bollinger'] = 'SELL - Above upper band'
            elif price < bb_lower:
                signals['Bollinger'] = 'BUY - Below lower band'
            else:
                signals['Bollinger'] = 'HOLD - Within bands'
        
        # Volume analysis
        if 'volume' in data.columns:
            avg_volume = data['volume'].rolling(20).mean().iloc[-1]
            current_volume = data['volume'].iloc[-1]
            
            if current_volume > 1.5 * avg_volume:
                signals['Volume'] = 'HIGH - Increased interest'
            elif current_volume < 0.5 * avg_volume:
                signals['Volume'] = 'LOW - Decreased interest'
            else:
                signals['Volume'] = 'NORMAL - Regular activity'
    except Exception as e:
        st.warning(f"Signal generation error: {e}")
    
    return signals

def create_crypto_correlation_matrix(crypto_symbols=['BTC-USD', 'ETH-USD', 'ADA-USD', 'DOT-USD']):
    """Create correlation matrix for crypto assets"""
    crypto_data = {}
    for symbol in crypto_symbols:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="3mo")
            if not hist.empty:
                crypto_data[symbol] = hist['Close'].pct_change().dropna()
        except:
            continue
    
    if crypto_data:
        crypto_df = pd.DataFrame(crypto_data)
        return crypto_df.corr()
    return None

def generate_sentiment_analysis(ticker):
    """Generate market sentiment analysis using AI"""
    prompt = f"""
    Analyze the current market sentiment for {ticker} based on:
    1. Recent news and market trends
    2. Social media sentiment
    3. Institutional investor activity
    4. Technical momentum indicators
    
    Provide a sentiment score (1-10) and key factors driving the sentiment.
    """
    
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
        response = requests.post(NOVITA_API_URL, headers=headers, json=data, timeout=25)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Sentiment analysis unavailable: {e}"

def generate_options_strategy(ticker, stock_price, volatility):
    """Generate options trading strategies"""
    prompt = f"""
    For {ticker} at ${stock_price:.2f} with {volatility:.1f}% volatility, suggest:
    1. Best options strategies for current market conditions
    2. Optimal strike prices and expiration dates
    3. Risk/reward analysis
    4. Entry and exit criteria
    
    Consider both bullish and bearish scenarios.
    """
    
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
        response = requests.post(NOVITA_API_URL, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Options strategy unavailable: {e}"

# Enhanced Session State Management
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
if 'show_analysis_history' not in st.session_state:
    st.session_state.show_analysis_history = False
if 'show_document_search' not in st.session_state:
    st.session_state.show_document_search = False
if 'show_similar_search' not in st.session_state:
    st.session_state.show_similar_search = False
if 'run_portfolio_analysis' not in st.session_state:
    st.session_state.run_portfolio_analysis = False
if 'show_saved_portfolios' not in st.session_state:
    st.session_state.show_saved_portfolios = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'show_ai_chat' not in st.session_state:
    st.session_state.show_ai_chat = False
if 'analysis_summary' not in st.session_state:
    st.session_state.analysis_summary = None
if 'price_alerts' not in st.session_state:
    st.session_state.price_alerts = []
if 'show_alerts' not in st.session_state:
    st.session_state.show_alerts = False
if 'crypto_data' not in st.session_state:
    st.session_state.crypto_data = None
if 'options_analysis' not in st.session_state:
    st.session_state.options_analysis = None
if 'sentiment_score' not in st.session_state:
    st.session_state.sentiment_score = None
if 'trading_signals' not in st.session_state:
    st.session_state.trading_signals = None

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
        ["Market Analysis", "Forecasting", "Portfolio Analysis"],
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
    
    # Vector Database Section (moved here)
    st.markdown("---")
    if milvus_client is not None:
        st.success("✅ Connected to Zilliz Vector DB")
        
        # Vector DB Management Section
        with st.expander("🗄️ Vector Database"):
            if st.button("📊 View Analysis History", use_container_width=True):
                st.session_state.show_analysis_history = True
            
            if st.button("📄 Document Search", use_container_width=True):
                st.session_state.show_document_search = True
            
            # Document Upload
            uploaded_file = st.file_uploader(
                "Upload Financial Document",
                type=['txt', 'pdf', 'csv'],
                help="Upload financial reports, analysis, or other documents"
            )
            
            if uploaded_file is not None:
                if st.button("📤 Upload to Vector DB"):
                    try:
                        if uploaded_file.type == 'text/plain':
                            content = str(uploaded_file.read(), "utf-8")
                        else:
                            content = str(uploaded_file.read())
                        
                        doc_type = uploaded_file.type
                        filename = uploaded_file.name
                        
                        if store_document(filename, doc_type, content):
                            st.success(f"Document '{filename}' uploaded successfully!")
                        else:
                            st.error("Failed to upload document")
                    except Exception as e:
                        st.error(f"Upload error: {e}")
    else:
        st.warning("⚠️ Could not connect to Zilliz Vector DB")
    
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
        
        # Enhanced Analysis Section
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🤖 AI Analysis", use_container_width=True):
                with st.spinner("Generating comprehensive analysis..."):
                    summary = generate_analysis_summary(ticker, df)
                    st.session_state.analysis_summary = summary
                    if store_analysis_result(ticker, "AI Analysis Summary", summary):
                        st.success("Analysis generated and saved to Vector DB!")
        
        with col2:
            if st.button("📊 Trading Signals", use_container_width=True):
                with st.spinner("Generating trading signals..."):
                    signals = generate_trading_signals(df)
                    st.session_state.trading_signals = signals
        
        with col3:
            if st.button("😊 Sentiment Analysis", use_container_width=True):
                with st.spinner("Analyzing market sentiment..."):
                    sentiment = generate_sentiment_analysis(ticker)
                    st.session_state.sentiment_score = sentiment
        
        with col4:
            if st.button("🎯 Options Strategy", use_container_width=True):
                with st.spinner("Generating options strategies..."):
                    volatility = df['Daily_Return'].std() * np.sqrt(252) * 100
                    options_strategy = generate_options_strategy(ticker, df.iloc[-1]['close'], volatility)
                    st.session_state.options_analysis = options_strategy
        
        # Display New Analysis Results
        if 'trading_signals' in st.session_state and st.session_state.trading_signals:
            st.markdown("### 🚦 Trading Signals")
            signal_cols = st.columns(len(st.session_state.trading_signals))
            for i, (indicator, signal) in enumerate(st.session_state.trading_signals.items()):
                with signal_cols[i]:
                    signal_type = signal.split(' - ')[0] if ' - ' in signal else signal
                    color = "🟢" if "BUY" in signal_type else "🔴" if "SELL" in signal_type else "🟡"
                    st.metric(f"{color} {indicator}", signal_type, signal.split(' - ')[1] if ' - ' in signal else "")
        
        if 'sentiment_score' in st.session_state and st.session_state.sentiment_score:
            st.markdown("### 😊 Market Sentiment")
            st.info(st.session_state.sentiment_score)
        
        if 'options_analysis' in st.session_state and st.session_state.options_analysis:
            st.markdown("### 🎯 Options Trading Strategies")
            st.warning(st.session_state.options_analysis)
        
        # Pattern Recognition
        st.markdown("### 🔍 Chart Pattern Recognition")
        patterns = detect_chart_patterns(df)
        if patterns:
            pattern_cols = st.columns(len(patterns))
            for i, (pattern_name, detected) in enumerate(patterns.items()):
                with pattern_cols[i]:
                    status = "✅ Detected" if detected else "❌ Not Found"
                    st.metric(pattern_name.replace('_', ' ').title(), status)
        
        # Fibonacci Levels
        if len(df) > 50:
            high_52w = df['close'].tail(252).max()
            low_52w = df['close'].tail(252).min()
            fib_levels = calculate_fibonacci_levels(high_52w, low_52w)
            
            st.markdown("### 📐 Fibonacci Retracement Levels")
            fib_cols = st.columns(len(fib_levels))
            for i, (level, price) in enumerate(fib_levels.items()):
                with fib_cols[i]:
                    st.metric(f"Fib {level}", f"${price:.2f}")
        
        # Enhanced Metrics with Risk Analytics
        st.markdown("---")
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
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
            # Sharpe Ratio
            sharpe = calculate_sharpe_ratio(df['Daily_Return'].dropna())
            st.metric("Sharpe Ratio", f"{sharpe:.2f}", "Risk-adjusted return")
        
        with col3:
            # Value at Risk
            var_5 = calculate_var(df['Daily_Return'].dropna())
            st.metric("VaR (5%)", f"{var_5:.2%}", "Daily risk")
        
        with col4:
            # Maximum Drawdown
            cumulative = (1 + df['Daily_Return']).cumprod()
            drawdown = (cumulative / cumulative.expanding().max() - 1).min()
            st.metric("Max Drawdown", f"{drawdown:.2%}", "Worst decline")
        
        with col5:
            # Beta calculation (vs SPY)
            try:
                spy_data = fetch_stock_data('SPY', df['date'].min(), df['date'].max())
                if spy_data is not None and len(spy_data) > 0:
                    # Align the data by dates
                    merged_data = pd.merge(df[['date', 'Daily_Return']], spy_data[['date', 'Daily_Return']], 
                                         on='date', suffixes=('_stock', '_spy'), how='inner')
                    if len(merged_data) > 1:
                        stock_returns = merged_data['Daily_Return_stock'].dropna()
                        spy_returns = merged_data['Daily_Return_spy'].dropna()
                        if len(stock_returns) > 1 and len(spy_returns) > 1 and spy_returns.var() != 0:
                            beta = stock_returns.cov(spy_returns) / spy_returns.var()
                            st.metric("Beta (vs SPY)", f"{beta:.2f}", "Market sensitivity")
                        else:
                            st.metric("Beta", "N/A", "Insufficient data")
                    else:
                        st.metric("Beta", "N/A", "No common dates")
                else:
                    st.metric("Beta", "N/A", "SPY data unavailable")
            except Exception as e:
                st.metric("Beta", "N/A", "Calculation error")
        
        with col6:
            # Information Ratio
            excess_return = df['Daily_Return'].mean() * 252 - 0.02  # vs 2% risk-free rate
            tracking_error = df['Daily_Return'].std() * np.sqrt(252)
            info_ratio = excess_return / tracking_error if tracking_error != 0 else 0
            st.metric("Info Ratio", f"{info_ratio:.2f}", "Alpha generation")
        
        # Enhanced Tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "Advanced Charts", 
            "Technical Analysis", 
            "AI Insights", 
            "Risk Analytics",
            "Crypto Comparison", 
            "Options Chain"
        ])
        
        with tab1:
            st.markdown(f"### {ticker} Advanced Price Analysis")
            
            # Enhanced chart with pattern overlays
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                              vertical_spacing=0.03, row_heights=[0.6, 0.2, 0.2])
            
            # Main price chart with patterns
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
            
            # Add Fibonacci levels
            if len(df) > 50:
                high_52w = df['close'].tail(252).max()
                low_52w = df['close'].tail(252).min()
                fib_levels = calculate_fibonacci_levels(high_52w, low_52w)
                for level, price in fib_levels.items():
                    fig.add_hline(y=price, line_dash="dot", line_color="gold", 
                                annotation_text=f"Fib {level}", row=1, col=1)
            
            # Volume with volume moving average
            fig.add_trace(
                go.Bar(x=df['date'], y=df['volume'], name="Volume", marker_color='#7f8c8d'),
                row=2, col=1
            )
            
            # RSI with overbought/oversold levels
            fig.add_trace(
                go.Scatter(x=df['date'], y=df['RSI'], name="RSI", line=dict(color='#9b59b6')),
                row=3, col=1
            )
            
            fig.update_layout(height=900, template="plotly_dark", title=f"{ticker} Advanced Technical Analysis")
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
            # AI Insights Tab
            st.markdown(f"### 🤖 AI-Powered Insights for {ticker}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Technical Analysis")
                if st.button("🔍 Analyze Technical Patterns"):
                    with st.spinner("Analyzing patterns..."):
                        tech_analysis = ask_ai_question(
                            f"Analyze the technical patterns for {ticker} with current RSI of {df.iloc[-1]['RSI']:.2f} and recent price action",
                            f"Stock data: Price ${df.iloc[-1]['close']:.2f}, 50-day SMA ${df.iloc[-1]['SMA_50']:.2f}, 200-day SMA ${df.iloc[-1]['SMA_200']:.2f}"
                        )
                        st.write(tech_analysis)
                
                st.markdown("#### Risk Assessment")
                if st.button("⚠️ Evaluate Risk Factors"):
                    with st.spinner("Assessing risks..."):
                        volatility = df['Daily_Return'].std() * np.sqrt(252) * 100
                        risk_analysis = ask_ai_question(
                            f"What are the key risk factors for investing in {ticker} right now?",
                            f"Current volatility: {volatility:.1f}%, Recent performance: {price_change_pct:.2f}%"
                        )
                        st.write(risk_analysis)
            
            with col2:
                st.markdown("#### Investment Recommendation")
                if st.button("💡 Get Investment Advice"):
                    with st.spinner("Generating recommendation..."):
                        investment_advice = ask_ai_question(
                            f"Should I buy, hold, or sell {ticker} based on current market conditions?",
                            f"Current analysis: RSI {df.iloc[-1]['RSI']:.2f}, Price trend: {price_change_pct:.2f}%, Volume: {df.iloc[-1]['volume']:,.0f}"
                        )
                        st.write(investment_advice)
                
                st.markdown("#### Sector Outlook")
                if st.button("🏭 Sector Analysis"):
                    with st.spinner("Analyzing sector..."):
                        sector_analysis = ask_ai_question(
                            f"What's the outlook for {ticker}'s sector and how might it affect the stock?",
                            f"Stock performance: {price_change_pct:.2f}% recent change"
                        )
                        st.write(sector_analysis)
            
            # Interactive Q&A Section
            st.markdown("---")
            st.markdown("#### 💬 Ask Custom Questions")
            custom_question = st.text_area(
                "Ask any question about this stock:",
                placeholder=f"What factors could drive {ticker} higher in the next quarter?"
            )
            
            if st.button("🚀 Get AI Answer") and custom_question:
                with st.spinner("Generating answer..."):
                    custom_answer = ask_ai_question(
                        custom_question,
                        f"Stock context: {ticker} at ${df.iloc[-1]['close']:.2f}, RSI {df.iloc[-1]['RSI']:.2f}"
                    )
                    st.success("**AI Response:**")
                    st.write(custom_answer)
                    
                    # Save Q&A to chat history
                    st.session_state.chat_history.append({
                        "question": custom_question,
                        "answer": custom_answer,
                        "timestamp": dt.datetime.now().strftime("%H:%M")
                    })
        
        with tab4:
            # Risk Analytics Tab
            st.markdown(f"### 📊 Risk Analytics for {ticker}")
            
            # Monte Carlo Simulation for portfolio if comparison tickers exist
            if st.session_state.comparison_tickers:
                st.markdown("#### Monte Carlo Portfolio Simulation")
                
                # Fetch comparison data
                portfolio_returns = pd.DataFrame()
                portfolio_returns[ticker] = df['Daily_Return']
                
                for comp_ticker in st.session_state.comparison_tickers[:3]:  # Limit to 3 for performance
                    comp_data = fetch_stock_data(comp_ticker, start_date, end_date, interval)
                    if comp_data is not None:
                        portfolio_returns[comp_ticker] = comp_data['Daily_Return']
                
                if len(portfolio_returns.columns) > 1:
                    # Run Monte Carlo simulation
                    mc_results = monte_carlo_portfolio_optimization(portfolio_returns.dropna())
                    
                    # Plot efficient frontier
                    fig_ef = go.Figure()
                    fig_ef.add_trace(go.Scatter(
                        x=mc_results[1], y=mc_results[0],
                        mode='markers',
                        marker=dict(
                            size=3,
                            color=mc_results[2],
                            colorscale='Viridis',
                            colorbar=dict(title="Sharpe Ratio")
                        ),
                        name='Portfolio Simulations'
                    ))
                    
                    fig_ef.update_layout(
                        title="Efficient Frontier - Monte Carlo Simulation",
                        xaxis_title="Volatility",
                        yaxis_title="Expected Return",
                        template="plotly_dark"
                    )
                    
                    st.plotly_chart(fig_ef, use_container_width=True)
                    
                    # Best portfolio metrics
                    max_sharpe_idx = np.argmax(mc_results[2])
                    st.markdown("#### Optimal Portfolio (Max Sharpe Ratio)")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Expected Return", f"{mc_results[0, max_sharpe_idx]:.2%}")
                    with col2:
                        st.metric("Volatility", f"{mc_results[1, max_sharpe_idx]:.2%}")
                    with col3:
                        st.metric("Sharpe Ratio", f"{mc_results[2, max_sharpe_idx]:.2f}")
            
            # Risk decomposition
            st.markdown("#### Risk Decomposition")
            returns = df['Daily_Return'].dropna()
            
            col1, col2 = st.columns(2)
            with col1:
                # Returns distribution
                fig_dist = go.Figure()
                fig_dist.add_trace(go.Histogram(x=returns, nbinsx=50, name="Returns Distribution"))
                fig_dist.update_layout(title="Daily Returns Distribution", template="plotly_dark")
                st.plotly_chart(fig_dist, use_container_width=True)
            
            with col2:
                # Rolling volatility
                rolling_vol = returns.rolling(30).std() * np.sqrt(252)
                fig_vol = go.Figure()
                fig_vol.add_trace(go.Scatter(x=df['date'].tail(len(rolling_vol)), y=rolling_vol, 
                                           name="30-Day Rolling Volatility"))
                fig_vol.update_layout(title="Rolling Volatility (30-Day)", template="plotly_dark")
                st.plotly_chart(fig_vol, use_container_width=True)
        
        with tab5:
            # Crypto Comparison Tab
            st.markdown(f"### ₿ Crypto Market Comparison")
            
            if 'crypto_symbol' in st.session_state:
                crypto_ticker = st.session_state.crypto_symbol
                crypto_data = fetch_stock_data(crypto_ticker, start_date, end_date, interval)
                
                if crypto_data is not None:
                    # Correlation analysis
                    stock_returns = df['Daily_Return'].dropna()
                    crypto_returns = crypto_data['Daily_Return'].dropna()
                    
                    # Align dates
                    common_dates = set(df['date']).intersection(set(crypto_data['date']))
                    if common_dates:
                        # Merge data on common dates
                        merged_data = pd.merge(df[['date', 'Daily_Return']], crypto_data[['date', 'Daily_Return']], 
                                             on='date', suffixes=('_stock', '_crypto'), how='inner')
                        if len(merged_data) > 1:
                            correlation = merged_data['Daily_Return_stock'].corr(merged_data['Daily_Return_crypto'])
                            
                            st.metric(f"Correlation with {crypto_ticker}", f"{correlation:.3f}")
                            
                            # Scatter plot
                            fig_scatter = go.Figure()
                            fig_scatter.add_trace(go.Scatter(
                                x=merged_data['Daily_Return_stock'], y=merged_data['Daily_Return_crypto'],
                                mode='markers',
                                name=f"{ticker} vs {crypto_ticker}"
                            ))
                            fig_scatter.update_layout(
                                title=f"{ticker} vs {crypto_ticker} Returns Correlation",
                                xaxis_title=f"{ticker} Daily Returns",
                                yaxis_title=f"{crypto_ticker} Daily Returns",
                                template="plotly_dark"
                            )
                            st.plotly_chart(fig_scatter, use_container_width=True)
                        else:
                            st.warning("No common dates found between stock and crypto data")
                    else:
                        st.warning("No overlapping dates found")
            
            # Crypto correlation matrix
            if st.session_state.crypto_data is not None:
                st.markdown("#### Crypto Correlation Matrix")
                fig_heatmap = px.imshow(st.session_state.crypto_data, 
                                      text_auto=True, aspect="auto",
                                      color_continuous_scale='RdBu_r')
                fig_heatmap.update_layout(title="Cryptocurrency Correlation Matrix")
                st.plotly_chart(fig_heatmap, use_container_width=True)
        
        with tab6:
            # Options Analysis Tab
            st.markdown(f"### 🎯 Options Analysis for {ticker}")
            
            current_price = df.iloc[-1]['close']
            volatility = df['Daily_Return'].std() * np.sqrt(252) * 100
            
            # Black-Scholes option pricing (simplified)
            st.markdown("#### Options Pricing Calculator")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                strike_price = st.number_input("Strike Price", value=float(current_price), step=1.0)
                option_type = st.selectbox("Option Type", ["Call", "Put"])
            
            with col2:
                days_to_expiry = st.number_input("Days to Expiry", value=30, min_value=1, max_value=365)
                risk_free_rate = st.number_input("Risk-free Rate (%)", value=5.0, step=0.1) / 100
            
            with col3:
                implied_vol = st.number_input("Implied Volatility (%)", value=float(volatility), step=1.0) / 100
            
            if st.button("Calculate Option Price"):
                try:
                    # Simplified Black-Scholes calculation
                    from scipy.stats import norm
                    import math
                    
                    T = days_to_expiry / 365
                    if T > 0 and implied_vol > 0 and current_price > 0 and strike_price > 0:
                        d1 = (math.log(current_price / strike_price) + (risk_free_rate + 0.5 * implied_vol**2) * T) / (implied_vol * math.sqrt(T))
                        d2 = d1 - implied_vol * math.sqrt(T)
                        
                        if option_type == "Call":
                            option_price = current_price * norm.cdf(d1) - strike_price * math.exp(-risk_free_rate * T) * norm.cdf(d2)
                        else:
                            option_price = strike_price * math.exp(-risk_free_rate * T) * norm.cdf(-d2) - current_price * norm.cdf(-d1)
                        
                        st.success(f"Theoretical {option_type} Option Price: ${option_price:.2f}")
                        
                        # Greeks calculation
                        delta = norm.cdf(d1) if option_type == "Call" else norm.cdf(d1) - 1
                        gamma = norm.pdf(d1) / (current_price * implied_vol * math.sqrt(T))
                        theta = -(current_price * norm.pdf(d1) * implied_vol) / (2 * math.sqrt(T)) - risk_free_rate * strike_price * math.exp(-risk_free_rate * T) * norm.cdf(d2)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Delta", f"{delta:.3f}")
                        with col2:
                            st.metric("Gamma", f"{gamma:.4f}")
                        with col3:
                            st.metric("Theta", f"{theta:.3f}")
                    else:
                        st.error("Invalid input parameters for option pricing")
                except Exception as e:
                    st.error(f"Error calculating option price: {e}")

# Real-time Alert System
if st.session_state.price_alerts and st.session_state.stock_data is not None:
    current_price = st.session_state.stock_data.iloc[-1]['close']
    current_ticker = st.session_state.current_ticker
    
    for alert in st.session_state.price_alerts:
        if alert['ticker'] == current_ticker:
            if (alert['type'] == 'Above' and current_price > alert['price']) or \
               (alert['type'] == 'Below' and current_price < alert['price']):
                st.error(f"🚨 PRICE ALERT: {current_ticker} is {alert['type'].lower()} ${alert['price']:.2f}! Current price: ${current_price:.2f}")

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

elif page == "Portfolio Analysis":
    st.markdown("## 📈 Portfolio Analysis & Vector Storage")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        portfolio_name = st.text_input("Portfolio Name", placeholder="My Tech Portfolio")
        
        # Multi-sector portfolio selection
        selected_sectors = st.multiselect(
            "Select Sectors for Portfolio",
            list(GLOBAL_TICKERS.keys()),
            default=["Technology", "Finance"]
        )
        
        # Get tickers from selected sectors
        available_tickers = []
        for sector in selected_sectors:
            available_tickers.extend(GLOBAL_TICKERS[sector])
        
        portfolio_tickers = st.multiselect(
            "Select Portfolio Stocks",
            available_tickers,
            default=["AAPL", "MSFT", "GOOGL"] if "GOOGL" in available_tickers else available_tickers[:3]
        )
    
    with col2:
        st.markdown("### Quick Actions")
        if st.button("📊 Analyze Portfolio", use_container_width=True) and portfolio_tickers:
            st.session_state.run_portfolio_analysis = True
        
        if st.button("📚 View Saved Portfolios", use_container_width=True):
            st.session_state.show_saved_portfolios = True
    
    if 'run_portfolio_analysis' in st.session_state and st.session_state.run_portfolio_analysis:
        st.markdown("---")
        
        with st.spinner("Analyzing portfolio..."):
            portfolio_data = {}
            total_return = 0
            portfolio_charts = {}
            
            # Fetch data for each ticker
            progress_bar = st.progress(0)
            for i, ticker in enumerate(portfolio_tickers):
                data = fetch_stock_data(ticker, start_date, end_date, interval)
                if data is not None:
                    returns = data['Daily_Return'].mean() * 252  # Annualized
                    volatility = data['Daily_Return'].std() * np.sqrt(252)
                    current_price = data['close'].iloc[-1]
                    price_change = ((current_price - data['close'].iloc[0]) / data['close'].iloc[0]) * 100
                    
                    portfolio_data[ticker] = {
                        'annual_return': returns,
                        'volatility': volatility,
                        'current_price': current_price,
                        'total_return': price_change
                    }
                    
                    total_return += returns
                    
                progress_bar.progress((i + 1) / len(portfolio_tickers))
            
            progress_bar.empty()
            
            if portfolio_data:
                # Generate insights
                avg_return = total_return / len(portfolio_tickers)
                best_performer = max(portfolio_data.keys(), key=lambda x: portfolio_data[x]['total_return'])
                worst_performer = min(portfolio_data.keys(), key=lambda x: portfolio_data[x]['total_return'])
                most_volatile = max(portfolio_data.keys(), key=lambda x: portfolio_data[x]['volatility'])
                
                insights = f"""
Portfolio '{portfolio_name}' Analysis Summary:
- Number of stocks: {len(portfolio_tickers)}
- Average annual return: {avg_return:.2%}
- Best performer: {best_performer} ({portfolio_data[best_performer]['total_return']:.2f}%)
- Worst performer: {worst_performer} ({portfolio_data[worst_performer]['total_return']:.2f}%)
- Most volatile: {most_volatile} ({portfolio_data[most_volatile]['volatility']:.2%})
- Analysis conducted on: {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}

Risk Assessment:
- Portfolio appears {'HIGH RISK' if avg_return > 0.15 else 'MODERATE RISK' if avg_return > 0.08 else 'LOW RISK'}
- Diversification: {'Well diversified' if len(portfolio_tickers) >= 5 else 'Consider more diversification'}
                """
                
                st.markdown("### 📊 Portfolio Insights")
                st.info(insights)
                
                # Display portfolio metrics
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Performance Metrics")
                    portfolio_df = pd.DataFrame(portfolio_data).T
                    st.dataframe(
                        portfolio_df.style.format({
                            'annual_return': '{:.2%}',
                            'volatility': '{:.2%}',
                            'current_price': '${:,.2f}',
                            'total_return': '{:.2f}%'
                        }),
                        use_container_width=True
                    )
                
                with col2:
                    st.markdown("#### Portfolio Composition")
                    # Pie chart of portfolio composition (equal weights for simplicity)
                    fig_pie = px.pie(
                        values=[1] * len(portfolio_tickers),
                        names=portfolio_tickers,
                        title="Portfolio Allocation (Equal Weights)"
                    )
                    fig_pie.update_layout(template="plotly_dark", height=300)
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                # Performance comparison chart
                st.markdown("#### Returns Comparison")
                returns_data = pd.DataFrame({
                    'Ticker': list(portfolio_data.keys()),
                    'Total Return (%)': [portfolio_data[t]['total_return'] for t in portfolio_data.keys()],
                    'Annual Return (%)': [portfolio_data[t]['annual_return'] * 100 for t in portfolio_data.keys()]
                })
                
                fig_bar = px.bar(
                    returns_data, 
                    x='Ticker', 
                    y='Total Return (%)',
                    title="Portfolio Returns Comparison",
                    color='Total Return (%)',
                    color_continuous_scale='RdYlGn'
                )
                fig_bar.update_layout(template="plotly_dark", height=400)
                st.plotly_chart(fig_bar, use_container_width=True)
                
                # Save to Vector DB
                if st.button("💾 Save Portfolio to Vector DB", use_container_width=True):
                    if store_portfolio_insights(portfolio_name, portfolio_tickers, insights, portfolio_data):
                        st.success("✅ Portfolio analysis saved to Vector Database!")
                    else:
                        st.error("❌ Failed to save portfolio analysis")
                
                st.session_state.run_portfolio_analysis = False

# Vector DB UI Sections
if 'show_analysis_history' in st.session_state and st.session_state.show_analysis_history:
    st.markdown("---")
    st.markdown("## 📊 Analysis History from Vector Database")
    
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    with col1:
        search_ticker = st.text_input("Search by Ticker (optional)", placeholder="AAPL")
    with col2:
        if st.button("🔍 Search"):
            history = get_analysis_history(search_ticker if search_ticker else None)
            st.session_state.search_results = history
            st.success("Search completed!")
    with col3:
        if st.button("🤖 AI Summary"):
            if 'search_results' in st.session_state and st.session_state.search_results:
                with st.spinner("Generating AI summary..."):
                    # Create summary of search results
                    summary_text = "\n".join([f"{item.get('ticker', 'N/A')}: {item.get('content', '')[:100]}" for item in st.session_state.search_results[:3]])
                    ai_summary = ask_ai_question(
                        f"Summarize these recent analyses: {summary_text}",
                        "Provide key insights and trends from these analyses"
                    )
                    st.session_state.history_ai_summary = ai_summary
    with col4:
        if st.button("❌ Close History"):
            st.session_state.show_analysis_history = False
            if 'search_results' in st.session_state:
                del st.session_state.search_results
            if 'history_ai_summary' in st.session_state:
                del st.session_state.history_ai_summary
            st.rerun()
    
    # Display AI Summary of History
    if 'history_ai_summary' in st.session_state:
        st.markdown("### 🤖 AI Summary of Analysis History")
        st.info(st.session_state.history_ai_summary)
    
    if 'search_results' in st.session_state and st.session_state.search_results:
        st.markdown("### Search Results")
        for item in st.session_state.search_results:
            with st.expander(f"📈 {item.get('ticker', 'N/A')} - {item.get('analysis_type', 'N/A')} ({item.get('timestamp', 'N/A')[:10]})"):
                st.write(item.get('content', 'No content available'))
    elif 'search_results' in st.session_state:
        st.info("No analysis history found for the specified criteria.")

if 'show_document_search' in st.session_state and st.session_state.show_document_search:
    st.markdown("---")
    st.markdown("## 🔍 Document Search")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        search_query = st.text_input("Search documents by content", placeholder="quarterly earnings report")
    with col2:
        if st.button("🔍 Semantic Search"):
            similar_docs = search_similar_analysis(search_query)
            st.session_state.doc_search_results = similar_docs
            st.success("Search completed!")
    with col3:
        if st.button("❌ Close Search"):
            st.session_state.show_document_search = False
            if 'doc_search_results' in st.session_state:
                del st.session_state.doc_search_results
            st.rerun()
    
    if 'doc_search_results' in st.session_state and st.session_state.doc_search_results:
        st.markdown("### Similar Documents")
        for doc in st.session_state.doc_search_results:
            score = getattr(doc, 'score', 0)
            entity = getattr(doc, 'entity', {})
            with st.expander(f"📄 Similarity: {score:.3f} - {entity.get('ticker', 'N/A')}"):
                st.write(entity.get('content', 'No content available'))
    elif 'doc_search_results' in st.session_state:
        st.info("No similar documents found.")

if 'show_similar_search' in st.session_state and st.session_state.show_similar_search:
    st.markdown("---")
    st.markdown("## 🔍 Find Similar Analysis")
    
    current_ticker = st.session_state.current_ticker
    query = f"analysis for {current_ticker} stock price technical indicators"
    
    similar_analyses = search_similar_analysis(query)
    
    if similar_analyses:
        st.markdown(f"### Similar Analyses for {current_ticker}")
        for analysis in similar_analyses:
            score = getattr(analysis, 'score', 0)
            entity = getattr(analysis, 'entity', {})
            with st.expander(f"📊 Similarity: {score:.3f} - {entity.get('analysis_type', 'N/A')} ({entity.get('timestamp', 'N/A')[:10]})"):
                st.write(entity.get('content', 'No content available'))
    else:
        st.info(f"No similar analyses found for {current_ticker}")
    
    if st.button("❌ Close Similar Search"):
        st.session_state.show_similar_search = False
        st.rerun()

# Enhanced Chat Interface (full screen when needed)
if st.session_state.show_ai_chat:
    st.markdown("---")
    st.markdown("## 💬 AI Financial Assistant")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("### Full Chat Interface")
        
        # Display full chat history
        if st.session_state.chat_history:
            for i, chat in enumerate(st.session_state.chat_history):
                with st.container():
                    st.markdown(f"**Q {i+1} ({chat['timestamp']}):** {chat['question']}")
                    st.markdown(f"**A:** {chat['answer']}")
                    st.markdown("---")
        
        # Large text area for detailed questions
        detailed_question = st.text_area(
            "Ask a detailed question about markets, your portfolio, or analysis:",
            height=100,
            placeholder="Ask anything about financial markets, stock analysis, trading strategies, risk management, etc."
        )
        
        if st.button("🚀 Get Detailed Answer", use_container_width=True):
            if detailed_question:
                with st.spinner("Generating comprehensive answer..."):
                    context = None
                    if st.session_state.stock_data is not None:
                        ticker = st.session_state.current_ticker
                        latest = st.session_state.stock_data.iloc[-1]
                        context = f"Current analysis context: {ticker} at ${latest['close']:.2f}, RSI {latest['RSI']:.2f}"
                    
                    detailed_answer = ask_ai_question(detailed_question, context)
                    st.session_state.chat_history.append({
                        "question": detailed_question,
                        "answer": detailed_answer,
                        "timestamp": dt.datetime.now().strftime("%H:%M:%S")
                    })
                    st.rerun()
    
    with col2:
        st.markdown("### Quick Actions")
        if st.button("📈 Market Outlook", use_container_width=True):
            with st.spinner("Getting market outlook..."):
                outlook = ask_ai_question("What's your outlook on the current market conditions and key trends investors should watch?")
                st.session_state.chat_history.append({
                    "question": "Market Outlook",
                    "answer": outlook,
                    "timestamp": dt.datetime.now().strftime("%H:%M")
                })
                st.rerun()
        
        if st.button("💡 Trading Tips", use_container_width=True):
            with st.spinner("Getting trading tips..."):
                tips = ask_ai_question("Give me 5 important trading tips for managing risk and maximizing returns in the current market environment.")
                st.session_state.chat_history.append({
                    "question": "Trading Tips",
                    "answer": tips,
                    "timestamp": dt.datetime.now().strftime("%H:%M")
                })
                st.rerun()
        
        if st.button("🏦 Economic Indicators", use_container_width=True):
            with st.spinner("Analyzing economic indicators..."):
                indicators = ask_ai_question("What are the most important economic indicators investors should monitor right now?")
                st.session_state.chat_history.append({
                    "question": "Economic Indicators",
                    "answer": indicators,
                    "timestamp": dt.datetime.now().strftime("%H:%M")
                })
                st.rerun()
        
        if st.button("❌ Close AI Chat", use_container_width=True):
            st.session_state.show_ai_chat = False
            st.rerun()

# ...existing Vector DB UI sections and Footer code...

# Enhanced Footer with Live Market Status
st.markdown("""
<div class="footer">
    <p>AlphaQuant Pro • Advanced Stock Market Analysis & Forecasting • Real-time Analytics & AI-Powered Insights</p>
    <p style="font-size: 0.8rem;">Data provided by Yahoo Finance • AI powered by Novita • Vector DB by Zilliz • Updated at {}</p>
</div>
""".format(dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)
