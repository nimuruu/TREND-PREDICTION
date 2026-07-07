import streamlit as st
import pandas as pd
import plotly.express as px

# 1. Pengaturan Konfigurasi Halaman Dashboard
st.set_page_config(
    page_title="Market Trends Predictor Engine",
    page_icon="📈",
    layout="wide"
)

# 2. Fungsi untuk Membaca Data dari Google Drive secara langsung
@st.cache_data # Mencegah reload data terus-menerus agar dashboard cepat saat demo
def load_data(file_id):
    # Mengubah URL view Google Drive menjadi URL download langsung untuk Pandas
    url = f"https://docs.google.com/spreadsheets/d/{file_id}/export?format=csv"
    df = pd.DataFrame()
    try:
        df = pd.read_csv(url)
        # Urutkan berdasarkan tanggal agar grafik tidak berantakan
        df['week_date'] = pd.to_datetime(df['week_date'])
        df = df.sort_values('week_date')
    except Exception as e:
        st.error(f"Gagal memuat data dari Google Drive. Pastikan akses berkas sudah publik! Eror: {e}")
    return df

# 🔑 MASUKKAN FILE ID GOOGLE DRIVE KAMU DI SINI
# Ambil dari link share file 'market_trends_predictions.csv' milikmu
GOOGLE_DRIVE_FILE_ID = "https://drive.google.com/file/d/1HT5gQtlSDX5-MK3Mcx_r1etqdxRdQNcI/view?usp=drive_link"

st.title("Market Trends Analytics & Predictor Engine")
st.subheader("Analisis Komparasi Tren Teknologi Real-Time via PySpark & MLlib")
st.markdown("---")

# Memuat dataset
df_trends = load_data(GOOGLE_DRIVE_FILE_ID)

if not df_trends.empty:
    # 3. Bagian Sidebar Kontrol Dashboard
    st.sidebar.header("🎛️ Panel Kontrol Dashboard")
    
    # Pilihan Kata Kunci
    available_keywords = df_trends['keyword'].unique().tolist()
    selected_keyword = st.sidebar.selectbox(
        "Pilih Kata Kunci untuk Dianalisis:",
        available_keywords
    )
    
    # Filter data berdasarkan keyword yang dipilih
    df_filtered = df_trends[df_trends['keyword'] == selected_keyword]
    
    # Ambil data terbaru untuk metrik ringkasan
    latest_data = df_filtered.iloc[-1]
    
    # 4. Menampilkan Ringkasan Metrik Utama (Metrics Cards)
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
        # Hitung perubahan tren prediksi
        delta_prediction = round(latest_data['predicted_future_index'] - latest_data['search_index'], 2)
        st.metric(
            label="Prediksi Indeks (4 Minggu Kedepan)", 
            value=f"{round(latest_data['predicted_future_index'], 2)}",
            delta=f"{delta_prediction} poin"
        )
        
    st.markdown("### 📊 Grafik Lini Masa: Tren Saat Ini vs Prediksi Masa Depan")
    
    # 5. Transformasi Data untuk Grafik Multiline Plotly
    # Satukan kolom search_index dan predicted_future_index agar bisa di-plot bersamaan
    df_melted = df_filtered.melt(
        id_vars=['week_date'], 
        value_vars=['search_index', 'predicted_future_index'],
        var_name='Tipe Indeks', 
        value_name='Skala Ketertarikan'
    )
    
    # Kustomisasi nama label di grafik agar rapi dilihat dosen
    df_melted['Tipe Indeks'] = df_melted['Tipe Indeks'].map({
        'search_index': 'Tren Riil Saat Ini',
        'predicted_future_index': 'Prediksi 4 Minggu Kedepan (MLlib)'
    })
    
    # Membuat Grafik Interaktif menggunakan Plotly
    fig = px.line(
        df_melted, 
        x='week_date', 
        y='Skala Ketertarikan', 
        color='Tipe Indeks',
        labels={'week_date': 'Tanggal Mingguan', 'Skala Ketertarikan': 'Indeks Tren (0-100)'},
        color_discrete_sequence=["#1f77b4", "#ff7f0e"] # Biru untuk riil, Oranye untuk prediksi
    )
    
    fig.update_layout(
        hovermode="x unified",
        legend=dict(orientation="h", ycolumn=1.1, y=1.05, x=0),
        margin=dict(l=20, r=20, t=20, b=20)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 6. Menampilkan Data Mentah Hasil Olahan Spark dalam Bentuk Tabel
    with st.expander("👁️ Lihat Tabel Data Mentah Hasil Olahan PySpark"):
        st.dataframe(
            df_filtered[['week_date', 'keyword', 'search_index', 'moving_avg_4w', 'predicted_future_index']]
            .sort_values('week_date', ascending=False),
            use_container_width=True
        )

else:
    st.info("Silakan konfigurasi 'GOOGLE_DRIVE_FILE_ID' dengan benar untuk menampilkan visualisasi grafik.")
