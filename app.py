import streamlit as st
import pandas as pd
import plotly.express as px
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
user_input = st.sidebar.text_input("Ketik Kata Kunci Baru (Tekan Enter):", "ai")

# Fungsi untuk mengambil data langsung atau beralih ke simulator jika diblokir
def fetch_live_data(keyword):
    try:
        pytrends.build_payload([keyword], cat=0, timeframe='today 12-m', geo='ID', gprop='')
        df = pytrends.interest_over_time()
        if not df.empty:
            df = df.reset_index()
            df_clean = pd.DataFrame({
                'week_date': df['date'],
                'search_index': df[keyword],
                'status_data': 'Live Data (Google Trends)'
            })
            return df_clean, True
        return None, False
    except Exception as e:
        # Jika kena rate-limit Google, fungsi ini mengembalikan False agar sistem tahu harus memicu data fallback
        return None, False

# PROSES HITUNG DAN VISUALISASI DIRECT
if user_input.strip() != "":
    with st.spinner(f"Sedang memproses data untuk kata kunci: {user_input}..."):
        df_trends, is_live = fetch_live_data(user_input)
        
    # 🚨 JIKA GOOGLE MEMBLOKIR, BUAT DATA SIMULASI OTOMATIS AGAR WEB TIDAK BLANK
    if not is_live:
        st.sidebar.warning("Mode: Simulasi Tren Aktif (Server Google Limit)")
        # Membuat deret waktu 52 minggu ke belakang sampai hari ini
        dates = pd.date_range(end=pd.Timestamp.now(), periods=52, freq='W')
        
        # Membuat pola tren naik turun yang realistis berdasarkan panjang teks kata kunci
        seed_value = sum(ord(char) for char in user_input) % 20
        base_trend = np.linspace(20 + seed_value, 75 + seed_value, 52)
        noise = np.random.normal(0, 7, 52)
        simulated_index = np.clip(base_trend + noise, 0, 100).astype(int)
        
        df_trends = pd.DataFrame({
            'week_date': dates,
            'search_index': simulated_index,
            'status_data': 'Simulated Data (Fallback Mode)'
        })
    else:
        st.sidebar.success("Mode: Live Data Terhubung")

    # --- JALUR PROSES MACHINE LEARNING (Koneksi Live & Fallback melewati jalur yang sama) ---
    df_trends['moving_avg_4w'] = df_trends['search_index'].rolling(window=4, min_periods=1).mean()
    df_trends['row_num'] = np.arange(len(df_trends))
    
    # Latih Model Linear Regression Ringan
    X = df_trends[['row_num', 'moving_avg_4w']].values
    y = df_trends['search_index'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Prediksi Masa Depan
    df_trends['predicted_future_index'] = model.predict(X)
    df_trends['predicted_future_index'] = df_trends['predicted_future_index'].clip(0, 100)
    
    latest_data = df_trends.iloc[-1]
    
    # MENAMPILKAN METRIK RINGKASAN
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
        
    st.markdown(f"### Grafik Lini Masa Kata Kunci: {user_input.upper()}")
    st.caption(f"Sumber Status: {df_trends['status_data'].iloc[0]}")
    
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
    with st.expander("Lihat Tabel Data Analitik"):
        st.dataframe(
            df_trends[['week_date', 'search_index', 'moving_avg_4w', 'predicted_future_index']]
            .sort_values('week_date', ascending=False),
            use_container_width=True
        )
else:
    st.info("Silakan masukkan kata kunci pada kolom pencarian di sebelah kiri.")
