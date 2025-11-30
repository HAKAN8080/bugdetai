import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="🤖 AI Bütçe Tahmin Sistemi", layout="wide")

# CSS
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .forecast-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Başlık
st.title("🤖 AI Destekli Bütçe Tahmin Sistemi")
st.markdown("**Machine Learning ile Otomatik Tahmin + Manuel Ayarlama**")

# Sidebar
with st.sidebar:
    st.header("⚙️ Ayarlar")
    
    forecast_mode = st.radio(
        "Tahmin Modu",
        ["🤖 ML Otomatik", "✋ Manuel Ayarlama", "🔀 Hibrit (ML + Manuel)"],
        help="ML: Geçmiş verilere göre otomatik tahmin\nManuel: Kendi parametrelerinizi girin\nHibrit: ML tahminini manuel ayarlayın"
    )
    
    st.divider()
    
    uploaded_file = st.file_uploader(
        "📊 Veri Dosyası Yükle",
        type=['xlsx', 'csv'],
        help="2024 ve 2025 satış verilerinizi yükleyin"
    )

# Ana içerik
if uploaded_file is None:
    st.info("👈 Lütfen sol panelden veri dosyanızı yükleyin")
    st.stop()

# Veriyi yükle
@st.cache_data
def load_data(file):
    try:
        df = pd.read_excel(file, sheet_name='Sayfa1', header=None)
        
        # Veriyi temizle
        data = df.iloc[2:].copy()
        data.columns = range(len(data.columns))
        
        data['Month'] = data[0]
        data['MainGroupDesc'] = data[1]
        data['Sales_2024'] = pd.to_numeric(data[4], errors='coerce')
        data['Sales_2025'] = pd.to_numeric(data[13], errors='coerce')
        
        clean_data = data[['Month', 'MainGroupDesc', 'Sales_2024', 'Sales_2025']].copy()
        clean_data = clean_data.dropna(subset=['Month', 'MainGroupDesc'])
        
        return clean_data
    except Exception as e:
        st.error(f"Veri yükleme hatası: {e}")
        return None

df = load_data(uploaded_file)

if df is None:
    st.stop()

# ML Forecaster sınıfı
class MLForecaster:
    def __init__(self):
        self.models = {}
        
    def prepare_prophet_data(self, df, category):
        cat_df = df[df['MainGroupDesc'] == category].copy()
        
        data_2024 = cat_df[cat_df['Sales_2024'].notna()][['Month', 'Sales_2024']].copy()
        data_2024['Year'] = 2024
        data_2024.rename(columns={'Sales_2024': 'Sales'}, inplace=True)
        
        data_2025 = cat_df[cat_df['Sales_2025'].notna()][['Month', 'Sales_2025']].copy()
        data_2025['Year'] = 2025
        data_2025.rename(columns={'Sales_2025': 'Sales'}, inplace=True)
        
        combined = pd.concat([data_2024, data_2025], ignore_index=True)
        combined['Month'] = combined['Month'].astype(int)
        combined['ds'] = pd.to_datetime(
            combined['Year'].astype(str) + '-' + 
            combined['Month'].astype(str).str.zfill(2) + '-01'
        )
        combined['y'] = combined['Sales']
        
        return combined[['ds', 'y']].sort_values('ds')
    
    def forecast_category(self, df, category):
        prophet_data = self.prepare_prophet_data(df, category)
        
        if len(prophet_data) < 2:
            return None
        
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode='multiplicative',
            changepoint_prior_scale=0.05,
        )
        
        model.fit(prophet_data)
        future = model.make_future_dataframe(periods=12, freq='MS')
        forecast = model.predict(future)
        
        forecast_2026 = forecast[forecast['ds'].dt.year == 2026].copy()
        forecast_2026['Month'] = forecast_2026['ds'].dt.month
        
        return forecast_2026[['Month', 'yhat', 'yhat_lower', 'yhat_upper']]

# ML tahminlerini hesapla
@st.cache_data
def get_ml_forecasts(_df):
    forecaster = MLForecaster()
    categories = _df['MainGroupDesc'].unique()
    
    all_forecasts = {}
    with st.spinner('🤖 ML modelleri eğitiliyor...'):
        progress_bar = st.progress(0)
        
        for i, category in enumerate(categories):
            forecast = forecaster.forecast_category(_df, category)
            if forecast is not None:
                all_forecasts[category] = forecast
            progress_bar.progress((i + 1) / len(categories))
        
        progress_bar.empty()
    
    return all_forecasts

# Mode'a göre işle
if forecast_mode == "🤖 ML Otomatik":
    st.header("🤖 Machine Learning Otomatik Tahmin")
    
    ml_forecasts = get_ml_forecasts(df)
    
    # Özet metrikler
    col1, col2, col3, col4 = st.columns(4)
    
    total_2024 = df['Sales_2024'].sum()
    total_2025 = df['Sales_2025'].sum() * (12/9)  # Tam yıl tahmini
    total_2026_ml = sum([f['yhat'].sum() for f in ml_forecasts.values()])
    
    with col1:
        st.metric("2024 Gerçek", f"{total_2024:.2f}")
    with col2:
        st.metric("2025 Tahmin", f"{total_2025:.2f}")
    with col3:
        st.metric("2026 ML Tahmin", f"{total_2026_ml:.2f}")
    with col4:
        growth = ((total_2026_ml - total_2025) / total_2025 * 100)
        st.metric("Büyüme %", f"{growth:.1f}%")
    
    st.divider()
    
    # Kategori seçimi
    selected_category = st.selectbox(
        "📂 Kategori Seçin",
        sorted(ml_forecasts.keys())
    )
    
    if selected_category:
        forecast_data = ml_forecasts[selected_category]
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader(f"📈 {selected_category} - 2026 Aylık Tahmin")
            
            # Grafik
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=forecast_data['Month'],
                y=forecast_data['yhat'],
                mode='lines+markers',
                name='ML Tahmin',
                line=dict(color='#667eea', width=3),
                marker=dict(size=8)
            ))
            
            fig.add_trace(go.Scatter(
                x=forecast_data['Month'],
                y=forecast_data['yhat_upper'],
                mode='lines',
                name='Üst Sınır',
                line=dict(color='rgba(102, 126, 234, 0.3)', dash='dash'),
                showlegend=True
            ))
            
            fig.add_trace(go.Scatter(
                x=forecast_data['Month'],
                y=forecast_data['yhat_lower'],
                mode='lines',
                name='Alt Sınır',
                line=dict(color='rgba(102, 126, 234, 0.3)', dash='dash'),
                fill='tonexty',
                showlegend=True
            ))
            
            fig.update_layout(
                height=400,
                xaxis_title="Ay",
                yaxis_title="Satış Tahmini",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📊 Detaylı Tahmin")
            
            forecast_display = forecast_data.copy()
            forecast_display.columns = ['Ay', 'Tahmin', 'Alt Limit', 'Üst Limit']
            forecast_display['Tahmin'] = forecast_display['Tahmin'].apply(lambda x: f"{x:.4f}")
            forecast_display['Alt Limit'] = forecast_display['Alt Limit'].apply(lambda x: f"{x:.4f}")
            forecast_display['Üst Limit'] = forecast_display['Üst Limit'].apply(lambda x: f"{x:.4f}")
            
            st.dataframe(forecast_display, use_container_width=True, height=400)
    
    # Tüm kategorileri karşılaştır
    st.divider()
    st.subheader("🔍 Kategori Karşılaştırma")
    
    comparison_data = []
    for category, forecast in ml_forecasts.items():
        cat_data = df[df['MainGroupDesc'] == category]
        sales_2024 = cat_data['Sales_2024'].sum()
        sales_2025 = cat_data['Sales_2025'].sum() * (12/9)
        forecast_2026 = forecast['yhat'].sum()
        
        comparison_data.append({
            'Kategori': category,
            '2024': sales_2024,
            '2025': sales_2025,
            '2026 ML': forecast_2026,
            'Büyüme %': ((forecast_2026 - sales_2025) / sales_2025 * 100) if sales_2025 > 0 else 0
        })
    
    comparison_df = pd.DataFrame(comparison_data).sort_values('2026 ML', ascending=False)
    
    # Grafikte göster
    fig = px.bar(
        comparison_df,
        x='Kategori',
        y=['2024', '2025', '2026 ML'],
        title="Kategori Bazında Yıllık Karşılaştırma",
        barmode='group',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Tablo
    st.dataframe(comparison_df, use_container_width=True)

elif forecast_mode == "✋ Manuel Ayarlama":
    st.header("✋ Manuel Bütçe Ayarlama")
    st.info("Bu modu geliştirmek ister misin? Mevcut manuel modülünü buraya entegre edebiliriz.")

else:  # Hibrit mod
    st.header("🔀 Hibrit Mod: ML + Manuel Ayarlama")
    
    ml_forecasts = get_ml_forecasts(df)
    
    st.info("💡 ML tahminini temel alıp, kendi parametrelerinizle ayarlayabilirsiniz")
    
    # Kategori seçimi
    selected_category = st.selectbox(
        "📂 Kategori Seçin",
        sorted(ml_forecasts.keys())
    )
    
    if selected_category:
        forecast_data = ml_forecasts[selected_category].copy()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader(f"📈 {selected_category} - ML Tahmin + Manuel Ayar")
            
            # Manuel ayar parametreleri
            st.markdown("**🎛️ Ayarlama Parametreleri:**")
            
            col_a, col_b = st.columns(2)
            with col_a:
                growth_rate = st.slider(
                    "Büyüme Oranı (%)",
                    min_value=-50,
                    max_value=100,
                    value=0,
                    step=5
                )
            
            with col_b:
                seasonality_boost = st.slider(
                    "Sezonsellik Çarpanı",
                    min_value=0.5,
                    max_value=2.0,
                    value=1.0,
                    step=0.1
                )
            
            # Ayarlanmış tahmin
            forecast_data['Adjusted'] = forecast_data['yhat'] * (1 + growth_rate/100) * seasonality_boost
            
            # Grafik
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=forecast_data['Month'],
                y=forecast_data['yhat'],
                mode='lines+markers',
                name='ML Tahmin (Orijinal)',
                line=dict(color='#667eea', width=2, dash='dot'),
                marker=dict(size=6)
            ))
            
            fig.add_trace(go.Scatter(
                x=forecast_data['Month'],
                y=forecast_data['Adjusted'],
                mode='lines+markers',
                name='Ayarlanmış Tahmin',
                line=dict(color='#f5576c', width=3),
                marker=dict(size=8)
            ))
            
            fig.update_layout(
                height=400,
                xaxis_title="Ay",
                yaxis_title="Satış Tahmini",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📊 Karşılaştırma")
            
            ml_total = forecast_data['yhat'].sum()
            adjusted_total = forecast_data['Adjusted'].sum()
            difference = adjusted_total - ml_total
            diff_pct = (difference / ml_total * 100) if ml_total > 0 else 0
            
            st.metric("ML Toplam", f"{ml_total:.4f}")
            st.metric("Ayarlanmış Toplam", f"{adjusted_total:.4f}", f"{diff_pct:+.1f}%")
            
            st.divider()
            
            # Detay tablo
            comparison_table = forecast_data[['Month', 'yhat', 'Adjusted']].copy()
            comparison_table.columns = ['Ay', 'ML', 'Ayarlanmış']
            comparison_table['Fark %'] = ((comparison_table['Ayarlanmış'] - comparison_table['ML']) / comparison_table['ML'] * 100).round(1)
            
            st.dataframe(comparison_table, use_container_width=True)

# Excel export
st.divider()
if st.button("💾 Sonuçları Excel'e Aktar"):
    # Export fonksiyonu eklenecek
    st.success("Excel export özelliği eklenecek!")
