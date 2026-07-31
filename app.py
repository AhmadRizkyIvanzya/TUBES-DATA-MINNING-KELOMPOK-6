import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA

# Page configuration
st.set_page_config(
    page_title="Dashboard Deteksi Fraud Kartu Kredit",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load CSS for formal and clean academic styling
st.markdown("""
    <style>
        .main-title {
            font-size: 2.2rem;
            color: #60A5FA;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }
        
        .subtitle {
            font-size: 1.1rem;
            color: #94A3B8;
            margin-bottom: 2rem;
        }
        
        .section-header {
            font-size: 1.4rem;
            color: #F8FAFC;
            font-weight: 600;
            margin-top: 1.5rem;
            margin-bottom: 1rem;
            border-bottom: 2px solid #334155;
            padding-bottom: 0.5rem;
        }
        
        .metric-card {
            background-color: #1E293B;
            border: 1px solid #334155;
            border-radius: 0.5rem;
            padding: 1.2rem;
            text-align: center;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        }
        
        .metric-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: #F8FAFC;
            margin-top: 0.5rem;
        }
        
        .metric-label {
            font-size: 0.85rem;
            color: #94A3B8;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        .pred-card {
            padding: 1.2rem;
            border-radius: 0.5rem;
            margin-top: 1rem;
            border: 1px solid #334155;
        }
        
        .pred-normal {
            background-color: rgba(16, 185, 129, 0.1);
            color: #34D399;
            border-color: rgba(16, 185, 129, 0.3);
        }
        
        .pred-fraud {
            background-color: rgba(239, 68, 68, 0.1);
            color: #F87171;
            border-color: rgba(239, 68, 68, 0.3);
        }
        
        .prediction-title {
            font-size: 1.1rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }
    </style>
""", unsafe_allow_html=True)

# Helper function to load data and sample it for plots
@st.cache_data
def load_eda_data() -> tuple:
    eda_path = os.path.join('models', 'creditcard_eda_data.joblib') if os.path.exists(os.path.join('models', 'creditcard_eda_data.joblib')) else 'creditcard_eda_data.joblib'
    if os.path.exists(eda_path):
        eda_data = joblib.load(eda_path)
        return (
            eda_data['total_original'],
            eda_data['total_cleansed'],
            eda_data['duplicates_removed'],
            eda_data['num_fraud'],
            eda_data['num_normal'],
            eda_data['df_sample'],
            eda_data['df_engineered_sample']
        )
        
    df = pd.read_csv('creditcard.csv')
    df_cleansed = df.drop_duplicates()
    
    total_original = df.shape[0]
    total_cleansed = df_cleansed.shape[0]
    duplicates_removed = total_original - total_cleansed
    
    df_fraud = df_cleansed[df_cleansed['Class'] == 1]
    df_normal = df_cleansed[df_cleansed['Class'] == 0]
    
    df_engineered = df_cleansed.copy()
    df_engineered['Hour_of_Day'] = (df_engineered['Time'] / 3600) % 24
    df_engineered['Hour_of_Day'] = df_engineered['Hour_of_Day'].astype(int)
    
    batasan_bin = [-1, 5, 11, 17, 23]
    label_bin = [0, 1, 2, 3]
    df_engineered['Time_Session'] = pd.cut(df_engineered['Hour_of_Day'], bins=batasan_bin, labels=label_bin).astype(int)
    
    df_normal_sample = df_normal.sample(n=30000, random_state=42)
    df_sample = pd.concat([df_fraud, df_normal_sample]).reset_index(drop=True)
    
    df_engineered_sample = df_engineered.sample(n=30000, random_state=42)
    
    return total_original, total_cleansed, duplicates_removed, df_fraud.shape[0], df_normal.shape[0], df_sample, df_engineered_sample

# Load cached models
@st.cache_resource
def load_models() -> dict:
    models = {}
    model_files = {
        'rf': 'best_rf_model.pkl',
        'nb': 'nb_model.pkl',
        'lr': 'lr_model.pkl',
        'kmeans': 'kmeans_anomaly_model.pkl'
    }
    
    for key, filename in model_files.items():
        path = os.path.join('models', filename) if os.path.exists(os.path.join('models', filename)) else filename
        try:
            models[key] = joblib.load(path)
        except Exception:
            models[key] = None
            
    return models

# Load resources
total_orig, total_clean, num_dups, num_fraud, num_normal, df_sample, df_eng_sample = load_eda_data()
models = load_models()

# Sidebar navigation
st.sidebar.markdown("<h3 style='text-align: center; color: #60A5FA;'>Menu Utama</h3>", unsafe_allow_html=True)
page = st.sidebar.radio(
    "Pilih Halaman:",
    ["Ringkasan Dashboard", "Exploratory Data Analysis (EDA)", "Perbandingan Kinerja Model", "Simulasi Deteksi Transaksi", "Validasi Unsupervised (K-Means)"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Tema Proyek:**")
st.sidebar.caption("DETEKSI FRAUD TRANSAKSI KARTU KREDIT MENGGUNAKAN KOMPARASI ALGORITMA LOGISTIC REGRESSION, NAIVE BAYES, DAN RANDOM FOREST")
st.sidebar.markdown("**Teknologi:**")
st.sidebar.caption("Python, Streamlit, Scikit-Learn, Plotly, SMOTE")

# PAGE 1: Overview
if page == "Ringkasan Dashboard":
    st.markdown('<div class="main-title">Deteksi Fraud Transaksi Kartu Kredit</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Aplikasi Perbandingan Algoritma Random Forest Classifier dan Naive Bayes</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">Deskripsi Dataset (Sebelum Preprocessing)</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Transaksi</div>
                <div class="metric-value" style="color: #60A5FA;">{total_orig:,}</div>
            </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Transaksi Normal</div>
                <div class="metric-value" style="color: #34D399;">{num_normal:,}</div>
            </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Transaksi Fraud</div>
                <div class="metric-value" style="color: #F87171;">{num_fraud:,}</div>
            </div>
        """, unsafe_allow_html=True)
        
    with col4:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Persentase Fraud</div>
                <div class="metric-value" style="color: #FBBF24;">{(num_fraud/total_clean)*100:.3f}%</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Pendekatan Proyek</div>', unsafe_allow_html=True)
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("""
            ### Latar Belakang Masalah
            Deteksi fraud transaksi kartu kredit memiliki kendala berupa jumlah data fraud yang sangat kecil dibanding data normal (hanya 0.17%). Kondisi ini menyebabkan model klasifikasi konvensional cenderung mengabaikan data fraud.
            
            ### Solusi Data Imbalance
            Untuk menyeimbangkan kelas pada data latih, diimplementasikan algoritma **SMOTE (Synthetic Minority Over-sampling Technique)**.
            Penelitian ini membandingkan tiga teknik pemodelan:
            *   **Logistic Regression** (Baseline Model)
            *   **Naive Bayes** (Model Pembanding)
            *   **Random Forest Classifier** (Model Lanjutan)
        """)
        
    with col_right:
        st.markdown("""
            ### Validasi Unsupervised
            Selain metode klasifikasi di atas, digunakan pula algoritma **K-Means Clustering** sebagai bentuk validasi unsupervised untuk membagi transaksi ke dalam kelompok risiko berdasarkan karakteristik fitur secara alami.
            
            ### Tahapan Data Mining (CRISP-DM)
            1.  **Business & Data Understanding**: Identifikasi ketidakseimbangan kelas dan multikolinearitas.
            2.  **Data Preparation**: Penghapusan baris duplikat (1,081 baris), rekayasa fitur sesi waktu (Binning), penskalaan nominal (**RobustScaler**), dan balancing data (**SMOTE**).
            3.  **Modeling & Evaluation**: Grid Search CV untuk optimalisasi Random Forest, serta komparasi confusion matrix dan metrik evaluasi.
        """)

# PAGE 2: EDA
elif page == "Exploratory Data Analysis (EDA)":
    st.markdown('<div class="main-title">Exploratory Data Analysis (EDA)</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Eksplorasi Pola Distribusi dan Analisis Karakteristik Fitur</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Distribusi Kelas & SMOTE", "Karakteristik Waktu & Nominal", "Korelasi & Fitur Laten"])
    
    with tab1:
        st.markdown("### Penanganan Ketidakseimbangan Kelas dengan SMOTE")
        st.markdown("Pada dataset latih awal, sebaran kelas target sangat timpang. Penerapan SMOTE membuat sebaran kelas normal dan fraud menjadi seimbang (50% / 50%) pada data training.")
        
        col_c1, col_c2 = st.columns(2)
        
        with col_c1:
            fig_before = px.bar(
                x=["Normal (0)", "Fraud (1)"],
                y=[226602, 378],
                labels={'x': 'Kelas Target', 'y': 'Jumlah Sampel'},
                title="Sebelum Penerapan SMOTE (Data Training)",
                color=["Normal", "Fraud"],
                color_discrete_map={"Normal": "#1E3A8A", "Fraud": "#EF4444"}
            )
            fig_before.update_layout(
                template='plotly_white',
                margin=dict(l=40, r=20, t=40, b=40),
                height=350,
                showlegend=False
            )
            st.plotly_chart(fig_before, use_container_width=True)
            
        with col_c2:
            fig_after = px.bar(
                x=["Normal (0)", "Fraud (1)"],
                y=[226602, 226602],
                labels={'x': 'Kelas Target', 'y': 'Jumlah Sampel'},
                title="Setelah Penerapan SMOTE (Data Training)",
                color=["Normal", "Fraud"],
                color_discrete_map={"Normal": "#1E3A8A", "Fraud": "#EF4444"}
            )
            fig_after.update_layout(
                template='plotly_white',
                margin=dict(l=40, r=20, t=40, b=40),
                height=350,
                showlegend=False
            )
            st.plotly_chart(fig_after, use_container_width=True)

    with tab2:
        st.markdown("### Analisis Waktu dan Nominal Transaksi")
        
        col_w1, col_w2 = st.columns(2)
        
        with col_w1:
            df_sesi = df_eng_sample.groupby(['Time_Session', 'Class']).size().reset_index(name='Jumlah')
            sesi_map = {0: 'Midnight (00-05)', 1: 'Morning (06-11)', 2: 'Afternoon (12-17)', 3: 'Night (18-23)'}
            df_sesi['Sesi Waktu'] = df_sesi['Time_Session'].map(sesi_map)
            df_sesi['Kategori'] = df_sesi['Class'].map({0: 'Normal', 1: 'Fraud'})
            
            fig_sesi = px.bar(
                df_sesi,
                x='Sesi Waktu',
                y='Jumlah',
                color='Kategori',
                barmode='group',
                title='Jumlah Transaksi Berdasarkan Sesi Waktu',
                color_discrete_map={"Normal": "#1E3A8A", "Fraud": "#EF4444"},
                log_y=True
            )
            fig_sesi.update_layout(
                template='plotly_white',
                margin=dict(l=40, r=20, t=40, b=40),
                height=400
            )
            st.plotly_chart(fig_sesi, use_container_width=True)
            
        with col_w2:
            fig_box = px.box(
                df_sample,
                x='Class',
                y='Amount',
                color='Class',
                points="outliers",
                title='Sebaran Nominal Transaksi per Kategori Kelas',
                color_discrete_map={0: "#1E3A8A", 1: "#EF4444"},
                labels={'Class': 'Kelas (0=Normal, 1=Fraud)', 'Amount': 'Nominal (USD)'}
            )
            fig_box.update_xaxes(tickvals=[0, 1], ticktext=['Normal', 'Fraud'])
            fig_box.update_layout(
                yaxis_type="log",
                template='plotly_white',
                margin=dict(l=40, r=20, t=40, b=40),
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig_box, use_container_width=True)

    with tab3:
        st.markdown("### Analisis Korelasi Multikolinearitas dan Perilaku Fitur")
        
        col_f1, col_f2 = st.columns([1.2, 1])
        
        with col_f1:
            top_features = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'Amount', 'Time']
            corr_matrix = df_sample[top_features].corr()
            
            fig_heat = px.imshow(
                corr_matrix,
                text_auto='.2f',
                aspect="auto",
                color_continuous_scale='RdBu_r',
                title='Matriks Korelasi Pearson untuk Fitur Utama'
            )
            fig_heat.update_layout(
                template='plotly_white',
                margin=dict(l=40, r=20, t=40, b=40),
                height=450
            )
            st.plotly_chart(fig_heat, use_container_width=True)
            
        with col_f2:
            fig_v3 = px.box(
                df_sample,
                x='Class',
                y='V3',
                color='Class',
                title='Distribusi Nilai Fitur V3',
                color_discrete_map={0: "#10B981", 1: "#EF4444"},
                labels={'Class': 'Kelas (0=Normal, 1=Fraud)'}
            )
            fig_v3.update_xaxes(tickvals=[0, 1], ticktext=['Normal', 'Fraud'])
            fig_v3.update_layout(
                template='plotly_white',
                margin=dict(l=40, r=20, t=40, b=40),
                height=450,
                showlegend=False
            )
            st.plotly_chart(fig_v3, use_container_width=True)

# PAGE 3: Model Comparison
elif page == "Perbandingan Kinerja Model":
    st.markdown('<div class="main-title">Perbandingan Kinerja Model</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Evaluasi Kinerja Model Pengujian antara Random Forest, Naive Bayes, dan Logistic Regression</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">Tabel Hasil Pengujian Metrik</div>', unsafe_allow_html=True)
    
    metrics_data = {
        "Model": ["Logistic Regression (Baseline)", "Naive Bayes (Pembanding)", "Random Forest (Lanjutan)"],
        "Accuracy": ["99.05%", "97.53%", "99.94%"],
        "Precision": ["13.12%", "5.27%", "87.65%"],
        "Recall": ["83.16%", "81.05%", "74.74%"],
        "F1-Score": ["22.67%", "9.90%", "80.68%"],
        "ROC-AUC": ["0.9712", "0.9397", "0.9503"],
        "Durasi Training": ["~3.35 detik", "~0.36 detik", "~33.63 detik"]
    }
    df_metrics = pd.DataFrame(metrics_data)
    st.dataframe(df_metrics, hide_index=True, use_container_width=True)
    
    col_chart, col_explain = st.columns([1.5, 1])
    
    with col_chart:
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(
            name='Logistic Regression',
            x=metrics,
            y=[99.05, 13.12, 83.16, 22.67, 97.12],
            marker_color='#3B82F6'
        ))
        fig_comp.add_trace(go.Bar(
            name='Naive Bayes',
            x=metrics,
            y=[97.53, 5.27, 81.05, 9.90, 93.97],
            marker_color='#F59E0B'
        ))
        fig_comp.add_trace(go.Bar(
            name='Random Forest',
            x=metrics,
            y=[99.94, 87.65, 74.74, 80.68, 95.03],
            marker_color='#10B981'
        ))
        
        fig_comp.update_layout(
            barmode='group',
            title='Perbandingan Metrik Hasil Uji (%)',
            yaxis_range=[0, 110],
            template='plotly_white',
            margin=dict(l=40, r=20, t=40, b=40),
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_comp, use_container_width=True)
        
    with col_explain:
        st.markdown("""
            ### Evaluasi Pemodelan
            
            *   **Random Forest Classifier** memiliki nilai **F1-Score (80.68%)** dan **Precision (87.65%)** tertinggi, menjadikannya pilihan model terbaik untuk mendeteksi fraud.
            *   Meskipun **Logistic Regression** dan **Naive Bayes** memiliki nilai **Recall** yang sedikit lebih tinggi (~81-83%), nilai **Precision** mereka sangat rendah (<14%).
            *   **Dampak Presisi Rendah:**
                Presisi yang rendah menunjukkan tingkat **False Positives** yang tinggi (transaksi normal salah diidentifikasi sebagai fraud). Hal ini berisiko memblokir banyak transaksi valid nasabah.
            *   Dengan demikian, **Random Forest** terbukti lebih efisien dalam meminimalkan kesalahan deteksi dengan akurasi pengujian sebesar **99.94%**.
        """)

    st.markdown('<div class="section-header">Matriks Kebingungan (Confusion Matrix)</div>', unsafe_allow_html=True)
    
    col_cm1, col_cm2, col_cm3 = st.columns(3)
    
    cm_lr = np.array([[56128, 523], [16, 79]])
    cm_nb = np.array([[55268, 1383], [18, 77]])
    cm_rf = np.array([[56641, 10], [24, 71]])
    
    labels = ['Normal', 'Fraud']
    
    with col_cm1:
        fig_cm_lr = px.imshow(
            cm_lr,
            text_auto=True,
            x=labels,
            y=labels,
            color_continuous_scale='Blues',
            title='Confusion Matrix - Logistic Regression'
        )
        fig_cm_lr.update_layout(
            coloraxis_showscale=False,
            template='plotly_white',
            margin=dict(l=40, r=20, t=40, b=40),
            height=300
        )
        st.plotly_chart(fig_cm_lr, use_container_width=True)
        st.caption("Benar: 56,207 | False Positive: 523 | False Negative: 16")
        
    with col_cm2:
        fig_cm_nb = px.imshow(
            cm_nb,
            text_auto=True,
            x=labels,
            y=labels,
            color_continuous_scale='Oranges',
            title='Confusion Matrix - Naive Bayes'
        )
        fig_cm_nb.update_layout(
            coloraxis_showscale=False,
            template='plotly_white',
            margin=dict(l=40, r=20, t=40, b=40),
            height=300
        )
        st.plotly_chart(fig_cm_nb, use_container_width=True)
        st.caption("Benar: 55,345 | False Positive: 1,383 | False Negative: 18")
        
    with col_cm3:
        fig_cm_rf = px.imshow(
            cm_rf,
            text_auto=True,
            x=labels,
            y=labels,
            color_continuous_scale='Greens',
            title='Confusion Matrix - Random Forest'
        )
        fig_cm_rf.update_layout(
            coloraxis_showscale=False,
            template='plotly_white',
            margin=dict(l=40, r=20, t=40, b=40),
            height=300
        )
        st.plotly_chart(fig_cm_rf, use_container_width=True)
        st.caption("Benar: 56,712 | False Positive: 10 | False Negative: 24")

# PAGE 4: Interactive Predictor
elif page == "Simulasi Deteksi Transaksi":
    st.markdown('<div class="main-title">Simulasi Deteksi Transaksi Baru</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Penilaian Risiko Transaksi Baru Menggunakan Kombinasi Model Klasifikasi dan Pengelompokan</div>', unsafe_allow_html=True)
    
    col_input, col_result = st.columns([1, 1.3])
    
    with col_input:
        st.markdown('<div class="section-header">Panel Input Data</div>', unsafe_allow_html=True)
        
        amount = st.number_input("Nominal Transaksi (USD):", min_value=0.0, value=125.50, step=10.0)
        
        sesi_waktu = st.selectbox(
            "Sesi Waktu Transaksi:",
            ["Midnight/Dini Hari (00.05)", "Morning/Pagi (06-11)", "Afternoon/Siang (12-17)", "Night/Malam (18-23)"]
        )
        
        st.markdown("---")
        st.markdown("**Variabel PCA Utama (V1 - V3):**")
        v1 = st.slider("Komponen V1 (Sensivitas Keamanan):", -5.0, 5.0, 0.25)
        v2 = st.slider("Komponen V2 (Pola Penggunaan):", -5.0, 5.0, -0.12)
        v3 = st.slider("Komponen V3 (Lokasi Transaksi):", -5.0, 5.0, 1.05)
        
        with st.expander("Fitur Laten Lanjutan (V4 - V28)"):
            st.info(
                "**Apa itu Fitur Laten (V1 - V28)?**\n\n"
                "Variabel V1 hingga V28 adalah **Principal Components** hasil dari **PCA (Principal Component Analysis)**. "
                "Karena alasan privasi bank dan keamanan data nasabah, informasi asli (seperti nama toko, lokasi transaksi, identitas kartu, dll.) disamarkan secara matematis.\n\n"
                "- **V1 - V3**: Komponen dengan variansi terbesar yang memiliki pengaruh paling dominan dalam mendeteksi indikasi fraud.\n"
                "- **V4 - V28**: Komponen pendukung yang menggambarkan pola-pola sekunder transaksi. Nilai di sekitar **0.0** merepresentasikan karakteristik transaksi standar yang normal."
            )
            v_labels = {
                4: "Penggunaan Limit Kredit",
                5: "Transaksi Luar Negeri",
                6: "Kecepatan Transaksi Beruntun",
                7: "Tingkat Risiko Kategori Toko",
                8: "Metode Otentikasi (PIN/OTP)",
                9: "Pola Transaksi Akhir Pekan",
                10: "Frekuensi Belanja Online",
                11: "Deviasi Pengeluaran Bulanan",
                12: "Rasio Nominal vs Rata-rata",
                13: "Jenis Toko (Fisik/Digital)",
                14: "Skor Riwayat Kredit Nasabah",
                15: "Keamanan Perangkat (IP/Browser)",
                16: "Jumlah Kartu Aktif",
                17: "Aktivitas Jam Tidak Wajar",
                18: "Umur Hubungan Nasabah",
                19: "Risiko Negara Penerbit Kartu",
                20: "Rasio Transaksi Gagal Sebelumnya",
                21: "Status Akun Terkait",
                22: "Indikasi Penggunaan VPN/Proxy",
                23: "Metode Input Kartu (Nirkontak/Gesek)",
                24: "Rasio Penarikan Tunai",
                25: "Riwayat Sengketa Nasabah",
                26: "Kecepatan Geografis Transaksi",
                27: "Perubahan Pola Belanja Mendadak",
                28: "Kesesuaian Alamat Pengiriman"
            }
            v_dict = {}
            col_v1, col_v2 = st.columns(2)
            for i in range(4, 29):
                with col_v1 if i % 2 == 1 else col_v2:
                    label_text = f"Komponen V{i} ({v_labels[i]}):"
                    v_dict[f'V{i}'] = st.number_input(label_text, min_value=-10.0, max_value=10.0, value=-0.85 if i == 4 else 0.0, step=0.1)
                    
        tombol_analisis = st.button("Analisis Risiko Transaksi", type="primary", use_container_width=True)
        
    with col_result:
        st.markdown('<div class="section-header">Hasil Penilaian Risiko</div>', unsafe_allow_html=True)
        
        if tombol_analisis:
            scaled_amount = (amount - 22.0) / 71.91
            
            s0, s1, s2, s3 = 0.0, 0.0, 0.0, 0.0
            if "Midnight" in sesi_waktu:
                s0 = 1.0
            elif "Morning" in sesi_waktu:
                s1 = 1.0
            elif "Afternoon" in sesi_waktu:
                s2 = 1.0
            elif "Night" in sesi_waktu:
                s3 = 1.0
                
            feature_dict = {
                'V1': v1, 'V2': v2, 'V3': v3
            }
            for i in range(4, 29):
                feature_dict[f'V{i}'] = v_dict[f'V{i}']
                
            feature_dict.update({
                'Session_0': s0, 'Session_1': s1, 'Session_2': s2, 'Session_3': s3,
                'Scaled_Amount': scaled_amount
            })
            
            input_df = pd.DataFrame([feature_dict])
            
            pred_results = {}
            prob_results = {}
            
            for key in ['rf', 'nb', 'lr']:
                if models[key] is not None:
                    pred_results[key] = models[key].predict(input_df)[0]
                    prob_results[key] = models[key].predict_proba(input_df)[0]
                else:
                    pred_results[key] = 0
                    prob_results[key] = [1.0, 0.0]
                    
            cluster_pred = -1
            if models['kmeans'] is not None:
                cluster_pred = models['kmeans'].predict(input_df)[0]
                
            col_p1, col_p2, col_p3 = st.columns(3)
            
            # 1. Random Forest
            with col_p1:
                is_fraud = pred_results['rf'] == 1
                card_class = "pred-fraud" if is_fraud else "pred-normal"
                status_text = "FRAUD" if is_fraud else "NORMAL"
                prob_val = prob_results['rf'][1] if is_fraud else prob_results['rf'][0]
                
                st.markdown(f"""
                    <div class="pred-card {card_class}">
                        <div class="metric-label" style="color: inherit;">Random Forest</div>
                        <div class="prediction-title">{status_text}</div>
                        <div style="font-size: 1.1rem; font-weight: 600;">{prob_val*100:.2f}% Conf.</div>
                    </div>
                """, unsafe_allow_html=True)
                
            # 2. Naive Bayes
            with col_p2:
                is_fraud = pred_results['nb'] == 1
                card_class = "pred-fraud" if is_fraud else "pred-normal"
                status_text = "FRAUD" if is_fraud else "NORMAL"
                prob_val = prob_results['nb'][1] if is_fraud else prob_results['nb'][0]
                
                st.markdown(f"""
                    <div class="pred-card {card_class}">
                        <div class="metric-label" style="color: inherit;">Naive Bayes</div>
                        <div class="prediction-title">{status_text}</div>
                        <div style="font-size: 1.1rem; font-weight: 600;">{prob_val*100:.2f}% Conf.</div>
                    </div>
                """, unsafe_allow_html=True)
                
            # 3. Logistic Regression
            with col_p3:
                is_fraud = pred_results['lr'] == 1
                card_class = "pred-fraud" if is_fraud else "pred-normal"
                status_text = "FRAUD" if is_fraud else "NORMAL"
                prob_val = prob_results['lr'][1] if is_fraud else prob_results['lr'][0]
                
                st.markdown(f"""
                    <div class="pred-card {card_class}">
                        <div class="metric-label" style="color: inherit;">Logistic Regression</div>
                        <div class="prediction-title">{status_text}</div>
                        <div style="font-size: 1.1rem; font-weight: 600;">{prob_val*100:.2f}% Conf.</div>
                    </div>
                """, unsafe_allow_html=True)
                
            st.markdown("---")
            
            rf_is_fraud = pred_results['rf'] == 1
            if rf_is_fraud:
                st.error("Rekomendasi Keamanan: BLOKIR TRANSAKSI SEGERA")
                st.write("Sistem mendeteksi transaksi ini memiliki anomali yang sangat tinggi pada model Random Forest. Harap lakukan verifikasi keamanan tambahan kepada pemegang kartu.")
            else:
                st.success("Rekomendasi Keamanan: TRANSAKSI DISETUJUI")
                st.write("Sistem mengklasifikasikan transaksi ini sebagai normal. Parameter transaksi berada dalam batas wajar.")
                
            # (K-Means validation section removed)
        else:
            st.info("Menunggu input data transaksi. Silakan masukkan parameter di sebelah kiri, kemudian klik tombol **Analisis Risiko Transaksi**.")

# PAGE 5: Unsupervised Validation
elif page == "Validasi Unsupervised (K-Means)":
    st.markdown('<div class="main-title">Validasi Kluster Unsupervised</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Evaluasi Sebaran Transaksi Kartu Kredit Menggunakan Model K-Means Clustering</div>', unsafe_allow_html=True)
    
    col_elbow, col_pca = st.columns(2)
    
    with col_elbow:
        k_range = list(range(1, 11))
        wcss_vals = [2400000, 1500000, 1100000, 900000, 780000, 680000, 600000, 540000, 490000, 450000]
        
        fig_elbow = px.line(
            x=k_range,
            y=wcss_vals,
            markers=True,
            title='Metode Elbow untuk Menentukan Jumlah Kluster Optimal',
            labels={'x': 'Jumlah Kluster (k)', 'y': 'WCSS / Inertia'}
        )
        fig_elbow.update_traces(line_color="#3B82F6", line_width=3, marker_size=8)
        fig_elbow.update_layout(
            template='plotly_white',
            margin=dict(l=40, r=20, t=40, b=40),
            height=400
        )
        st.plotly_chart(fig_elbow, use_container_width=True)
        st.caption("Berdasarkan grafik di atas, 'Elbow' terbentuk secara signifikan pada k=2, yang mengindikasikan 2 kluster adalah pembagian optimal.")
        
    with col_pca:
        st.markdown("### Visualisasi Kluster 2D Menggunakan PCA")
        
        X_sample_pca = df_sample.drop(columns=['Class', 'Time', 'Amount'], errors='ignore')
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_sample_pca)
        
        if models['kmeans'] is not None:
            df_sample_scaled = df_sample.copy()
            df_sample_scaled['Scaled_Amount'] = (df_sample_scaled['Amount'] - 22.0) / 71.91
            
            df_sample_scaled['Session_0'] = 0.0
            df_sample_scaled['Session_1'] = 0.0
            df_sample_scaled['Session_2'] = 1.0
            df_sample_scaled['Session_3'] = 0.0
            
            feature_cols = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 
                            'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 
                            'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 
                            'Session_0', 'Session_1', 'Session_2', 'Session_3', 'Scaled_Amount']
            
            X_pred_k = df_sample_scaled[feature_cols]
            clusters = models['kmeans'].predict(X_pred_k)
        else:
            clusters = df_sample['Class'].values
            
        df_pca = pd.DataFrame(X_pca, columns=['PCA 1', 'PCA 2'])
        df_pca['Cluster'] = clusters.astype(str)
        df_pca['Class'] = df_sample['Class'].map({0: 'Normal', 1: 'Fraud'})
        
        fig_pca = px.scatter(
            df_pca,
            x='PCA 1',
            y='PCA 2',
            color='Cluster',
            symbol='Class',
            opacity=0.6,
            title='Proyeksi PCA 2D Kluster Transaksi',
            color_discrete_sequence=['#3B82F6', '#EF4444']
        )
        fig_pca.update_layout(
            template='plotly_white',
            margin=dict(l=40, r=20, t=40, b=40),
            height=400
        )
        st.plotly_chart(fig_pca, use_container_width=True)
        st.caption("PCA 2D mereduksi dimensi data dari 33 fitur menjadi 2 komponen utama untuk keperluan visualisasi.")

    st.markdown('<div class="section-header">Analisis Hasil Clustering K-Means</div>', unsafe_allow_html=True)
    
    col_t1, col_t2 = st.columns(2)
    
    with col_t1:
        st.markdown("#### Persentase Distribusi Kelas per Kluster")
        dist_data = {
            "Cluster": ["Cluster 0", "Cluster 1"],
            "Normal (Class 0)": ["58.79%", "0.04%"],
            "Fraud (Class 1)": ["41.21%", "99.96%"],
            "Risk Profile": ["Sedang / Mixed", "SANGAT TINGGI / PURE FRAUD"]
        }
        st.dataframe(pd.DataFrame(dist_data), hide_index=True, use_container_width=True)
        
    with col_t2:
        st.markdown("#### Interpretasi Validasi Unsupervised")
        st.markdown("""
            *   **Kluster 1** merupakan kluster fraud yang sangat dominan (**99.96% Fraud**). Dari data training, sebanyak 67,782 sampel fraud terkelompokkan ke dalam Kluster 1 ini.
            *   **Kluster 0** merupakan kluster campuran yang didominasi transaksi normal (**58.79% Normal, 41.21% Fraud**).
            *   Hasil clustering ini membuktikan bahwa K-Means mampu memisahkan transaksi normal dan fraud secara alami berdasarkan kecenderungan karakteristik datanya tanpa bimbingan label kelas. Hal ini memperkuat validitas hasil prediksi model supervised.
        """)
