import streamlit as st
import pandas as pd
import plotly.express as px
import time
import warnings
from pytrends.request import TrendReq
from sklearn.linear_model import LinearRegression
import numpy as np

# Sembunyikan peringatan pembungkusan data
warnings.filterwarnings("ignore", category=FutureWarning)

st.set_page_config(
    page_title="Dynamic Market Trends Predictor",
    layout="wide"
)

# Inisialisasi koneksi Google Trends yang aman
@st.cache_resource
def get_pytrends():
    return TrendReq(hl='id-ID', tz=420)

pytrends = get_pytrends()

st.title("Dynamic Market Trends Predictor Engine")
st.subheader("Analisis Komparasi Tren Pasar Real-Time Berbasis Machine Learning")
st.markdown("---")

# PANEL KONTROL SIDEBAR
st.sidebar.header("Panel Kontrol Kustom")
st.sidebar.markdown("Masukkan kata kunci apa saja untuk memprediksi trennya di Indonesia.")

# Kolom Input Bebas untuk User
user_input = st.sidebar.text_input("Ketik Kata Kunci Baru:", "ai agent")
btn_cari = st.sidebar.button("Analisis dan Prediksi Tren")

# Fungsi untuk mengambil data langsung saat tombol diklik
def fetch_live_data(keyword):
    try:
        pytrends.build_payload([keyword], cat=0, timeframe='today 12-m', geo='ID', gprop='')
        df = pytrends.interest_over_time()
        if not df.empty:
            df = df.reset_index()
            df_clean = pd.DataFrame({
                'week_date': df['date'],
                'search_index': df[keyword]
            })
            return df_clean
        return None
    except Exception as e:
        st.error(f"Terjadi kendala pengambilan data. Eror: {e}")
        return None

# PROSES HITUNG DAN VISUALISASI
if user_input:
    with st.spinner(f"Sedang menarik data live {user_input} dari server Google..."):
        df_trends = fetch_live_data(user_input)
        
    if df_trends is not None and not df_trends.empty:
        # --- FEATURE ENGINEERING ---
        df_trends['moving_avg_4w'] = df_trends['search_index'].rolling(window=4, min_periods=1).mean()
        df_trends['row_num'] = np.arange(len(df_trends))
        
        # Latih Model Linear Regression Ringan
        X = df_trends[['row_num', 'moving_avg_4w']].values
        y = df_trends['search_index'].values
        
        model = LinearRegression()
