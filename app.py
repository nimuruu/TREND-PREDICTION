import streamlit as st
import pandas as pd
import plotly.express as px
import time
import warnings
from pytrends.request import TrendReq
from sklearn.linear_model import LinearRegression
import numpy as np

# Sembunyikan peringatan deprecation dari pandas/pytrends
warnings.filterwarnings("ignore", category=FutureWarning)

st.set_page_config(
    page_title="Dynamic Market Trends Predictor",
    page_icon="📈",
    layout="wide"
)

# Inisialisasi koneksi Google Trends
@st.cache_resource
def get_pytrends():
    return TrendReq(hl='id-ID', tz=420, retries=2, backoff_factor=1)

pytrends = get_pytrends()

st.title("📈 Dynamic Market Trends Predictor Engine")
st.subheader("Analisis Komparasi Tren Pasar Real-Time Berbasis Machine Learning")
st.markdown("---")

# 🎛️ PANEL KONTROL SIDEBAR
st.sidebar.header("🎛️ Panel Kontrol Kustom")
st.sidebar.markdown("Masukkan kata kunci apa saja untuk memprediksi trennya di Indonesia.")

# Kolom Input Bebas untuk User
user_input = st.sidebar.text_input("Ketik Kata Kunci Baru:", "ai agent")
btn_cari = st.sidebar.button("🚀 Analisis & Prediksi Tren")

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
        st.error(f"Terjadi batasan limit dari Google. Silakan coba beberapa saat lagi atau ganti kata kunci. Eror: {e}")
        return None

# PROSES HITUNG & VISUALISASI
if user_input:
    with st.spinner(f"⏳ Sedang menarik data live '{user_input}' dari server Google..."):
        df_trends = fetch_live_data(user_input)
        
    if df_trends is not None and not df_trends.empty:
        # --- FEATURE ENGINEERING (Meniru Logika PySpark) ---
        # 1. Menghitung Moving Average 4 Minggu
        df_trends['moving_avg_4w'] = df_trends['search_index'].rolling(window=4, min_periods=1).mean()
        
        # 2. Membuat Fitur Angka untuk Indeks Linear Regression
        df_trends['row_num'] = np.arange(len(df_trends))
        
        # Latih Model Linear Regression Ringan (Meniru MLlib)
        X = df_trends[['row_num', 'moving_avg_4w']].values
        y = df_trends['search_index'].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Prediksi 4 Minggu ke Depan
        df_trends['predicted_future_index'] = model.predict(X)
        # Batasi agar nilai prediksi tetap logis di skala 0-100
        df_trends['predicted_future_index'] = df_trends['predicted_future_index'].clip(0, 100)
        
        # Ambil baris data terakhir untuk ditampilkan pada widget info singkat
        latest_data = df_trends.iloc[-1]
        
        # 📊 MENAMPILKAN METRIK RINGKASAN
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                label=f"Indeks Pencarian Terakhir ({latest_data['week_date'].strftime('%d %b %Y')})", 
                value=f"{int(latest_data['search_index'])}/100"
            )
        with col2:
            st.metric(
                label="Moving Average (4 Minggu Lalu)", 
                value=f"{round(latest_data['moving_avg_4w'], 2)}"
            )
        with col3:
            delta_prediction = round(latest_data['predicted_future_index'] - latest_data['search_index'], 2)
            st.metric(
                label="Prediksi Indeks (4 Minggu Kedepan)", 
                value=f"{round(latest_data['predicted_future_index'], 2)}",
                delta=f"{delta_prediction} poin"
            )
            
        st.markdown(f"### 📊 Grafik Lini Masa Kata Kunci: **{user_input.upper()}**")
        
        # Transformasi Data untuk Grafik Multiline Plotly
        df_melted = df_trends.melt(
            id_vars=['week_date'], 
            value_vars=['search_index', 'predicted_future_index'],
            var_name='Tipe Indeks', 
            value_name='Skala Ketertarikan'
        )
        
        df_melted['Tipe Indeks'] = df_melted['Tipe Indeks'].map({
            'search_index': 'Tren Riil Saat Ini',
            'predicted_future_index': 'Prediksi Masa Depan (Regression Model)'
        })
        
        # Render Grafik Interaktif Plotly
        fig = px.line(
            df_melted, 
            x='week_date', 
            y='Skala Ketertarikan', 
            color='Tipe Indeks',
            labels={'week_date': 'Tanggal Mingguan', 'Skala Ketertarikan': 'Indeks Tren (0-100)'},
            color_discrete_sequence=["#1f77b4", "#ff7f0e"]
        )
        
        fig.update_layout(
            hovermode="x unified",
            legend=dict(orientation="h", y=1.05, x=0),
            margin=dict(l=20, r=20, t=20, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Menampilkan Tabel Data Mentah
        with st.expander("👁️ Lihat Tabel Data Analitik"):
            st.dataframe(
                df_trends[['week_date', 'search_index', 'moving_avg_4w', 'predicted_future_index']]
                .sort_values('week_date', ascending=False),
                use_container_width=True
            )
    else:
        st.warning("Data tidak ditemukan atau Google membatasi koneksi. Coba ketik kata kunci lain yang lebih umum.")
