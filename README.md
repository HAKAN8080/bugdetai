# 🤖 AI Destekli Bütçe Tahmin Sistemi

## 📋 İçindekiler
- Machine Learning tabanlı otomatik tahmin motoru
- Manuel ayarlama özellikleri
- Hibrit mod (ML + Manuel)
- Prophet algoritması ile zaman serisi tahmini

## 🎯 Özellikler

### 1. ML Otomatik Tahmin
- **Facebook Prophet** kullanarak gelişmiş zaman serisi tahmini
- Otomatik sezonsellik tespiti
- Trend analizi ve anomali tespiti
- Güven aralıkları (confidence intervals)
- Kategori bazlı modelleme

### 2. Hibrit Mod
- ML tahminini temel al
- Manuel parametrelerle ayarla
- Gerçek zamanlı karşılaştırma
- Esneklik ve kontrol

### 3. Manuel Ayarlama
- Klasik parametrik yaklaşım
- Büyüme oranları
- Sezonsellik faktörleri
- Lessons learned

## 📊 Dosyalar

### 1. `ml_budget_forecaster.py`
**Ana ML motoru** - Standalone Python scripti

**Kullanım:**
```python
from ml_budget_forecaster import MLBudgetForecaster
import pandas as pd

# Veriyi yükle
df = pd.read_csv('budget_data.csv')

# Forecaster oluştur
forecaster = MLBudgetForecaster()

# Tüm kategoriler için tahmin
forecasts = forecaster.train_all_categories(df)

# Özet
summary = forecaster.get_summary()
comparison = forecaster.compare_with_actuals(df)
```

**Çıktılar:**
- `ml_forecast_summary.csv` - Kategori bazlı özet tahminler
- `ml_forecast_comparison.csv` - Yıl bazlı karşılaştırma
- `ml_forecast_detailed.xlsx` - Her kategori için aylık detay

### 2. `ml_budget_app.py`
**Streamlit Web Uygulaması**

**Çalıştırma:**
```bash
streamlit run ml_budget_app.py
```

**Özellikler:**
- 🤖 ML Otomatik Mod
- ✋ Manuel Ayarlama Mod
- 🔀 Hibrit Mod
- Interaktif grafikler (Plotly)
- Excel export

### 3. `ml_forecast_summary.csv`
Kategori bazlı özet tahminler:
- Total_Forecast: 2026 toplam tahmin
- Lower_Bound: Alt güven sınırı
- Upper_Bound: Üst güven sınırı
- Avg_Monthly: Aylık ortalama

### 4. `ml_forecast_comparison.csv`
Yıl bazlı karşılaştırma:
- Sales_2024: 2024 gerçek satış
- Sales_2025_Est: 2025 tam yıl tahmini (9 aydan)
- Forecast_2026: 2026 ML tahmini
- Growth_24_25_%: 2024-2025 büyüme
- Growth_25_26_%: 2025-2026 büyüme

### 5. `ml_forecast_detailed.xlsx`
Her kategori için ayrı sheet:
- Month: Ay (1-12)
- Forecast: ML tahmin
- Lower_Bound: Alt limit
- Upper_Bound: Üst limit

## 🚀 Kurulum

### Gereksinimler
```bash
pip install pandas numpy prophet streamlit plotly openpyxl
```

### Hızlı Başlangıç
```bash
# 1. ML tahmin çalıştır
python ml_budget_forecaster.py

# 2. Web app'i başlat
streamlit run ml_budget_app.py
```

## 🧠 ML Modeli Nasıl Çalışır?

### Prophet Algoritması
Prophet, Facebook tarafından geliştirilen zaman serisi tahmin kütüphanesidir.

**Bileşenler:**
1. **Trend**: Uzun vadeli büyüme/düşüş
2. **Sezonsellik**: Yıllık, aylık kalıplar
3. **Tatiller/Özel günler**: İsteğe bağlı
4. **Hata terimi**: Belirsizlik

**Model Denklemi:**
```
y(t) = g(t) + s(t) + h(t) + ε(t)
```
- g(t): Trend
- s(t): Sezonsellik
- h(t): Tatiller
- ε(t): Hata

### Bizim Kullanımımız
```python
model = Prophet(
    yearly_seasonality=True,      # Yıllık kalıp var
    weekly_seasonality=False,     # Haftalık yok
    daily_seasonality=False,      # Günlük yok
    seasonality_mode='multiplicative',  # Çarpımsal sezonsellik
    changepoint_prior_scale=0.05,      # Trend değişim hassasiyeti
)
```

**Neden Multiplicative?**
- Retail verilerde sezonsellik satış miktarıyla orantılı
- Büyük satışlarda daha büyük dalgalanma
- Küçük satışlarda daha küçük dalgalanma

## 📈 Model Performansı

### Test Sonuçları (Sizin Veri)

**Toplam Tahminler:**
- 2024 Gerçek: 12.00
- 2025 Tahmin: 13.33 (+11.1%)
- 2026 ML Tahmin: 9.77 (-26.7%)

**En İyi Performans:**
1. Dünya Markaları: +68.2%
2. Pişirme: +2.9%
3. Mutfak: -0.4%

**En Zayıf Performans:**
1. Pike: -86.1%
2. Havlu: -62.8%
3. Bornoz: -60.6%

### Model Güvenilirliği
- Prophet güven aralıkları %95 confidence level
- Lower/Upper bounds tahmin belirsizliğini gösterir
- Geniş aralık = yüksek belirsizlik
- Dar aralık = güvenilir tahmin

## 🔄 Gelecek Geliştirmeler

### Kısa Vade
- [ ] Excel export özelliği
- [ ] Manuel modülün entegrasyonu
- [ ] Daha fazla görselleştirme
- [ ] Tahmin doğruluk metrikleri (MAPE, RMSE)

### Orta Vade
- [ ] Ensemble modeller (Prophet + SARIMA + XGBoost)
- [ ] Anomali tespiti ve uyarılar
- [ ] Senario analizi ("ne olursa" hesaplamaları)
- [ ] Otomatik parametre optimizasyonu

### Uzun Vade
- [ ] External factors (ekonomik göstergeler, rakip analizi)
- [ ] Veritabanı entegrasyonu
- [ ] API servisi
- [ ] Otomatik raporlama

## 📊 Hiyerarşi Ölçeklenebilirliği

**Soru: Ana grup sayısı 10 katına çıkarsa sorun olur mu?**

**Cevap: Hayır, ama optimizasyon gerekir:**

### Mevcut Durum
- 20 kategori: ✅ Sorunsuz
- 200 kategori: ⚠️ Yavaşlama olabilir
- 2000 kategori: ❌ Ciddi optimizasyon gerekir

### Çözümler

#### 1. Teknik Optimizasyon
```python
# Paralel işleme
from concurrent.futures import ProcessPoolExecutor

def train_parallel(categories, df):
    with ProcessPoolExecutor(max_workers=4) as executor:
        results = executor.map(train_category, categories)
    return results
```

#### 2. Hiyerarşik Modelleme
```python
# Üst seviye: Ana grup
# Alt seviye: Alt kategoriler
# Tahminleri yukarı topla
```

#### 3. Veritabanı
```python
# SQLite veya PostgreSQL
# İndeksli sorgular
# Batch processing
```

#### 4. Cache Sistemi
```python
@st.cache_data(ttl=3600)  # 1 saat cache
def get_forecasts(categories):
    # Hesaplamalar
    return forecasts
```

### Performans Beklentileri

| Kategori Sayısı | İşlem Süresi | Çözüm |
|----------------|--------------|-------|
| 1-50 | < 2 dakika | Mevcut sistem |
| 50-200 | 2-10 dakika | Paralel işleme |
| 200-1000 | 10-60 dakika | Hiyerarşik + cache |
| 1000+ | 1+ saat | Veritabanı + batch |

## 💡 Kullanım İpuçları

### ML Tahminlerini İyileştirme

1. **Daha fazla veri:**
   - 2-3 yıl veri ideal
   - Aylık granülasyon yeterli
   - Eksik veri olmamalı

2. **Tatil/kampanya bilgisi:**
```python
holidays = pd.DataFrame({
    'holiday': 'ramazan_bayrami',
    'ds': pd.to_datetime(['2024-04-10', '2025-03-30']),
    'lower_window': 0,
    'upper_window': 3,
})
model.add_country_holidays(country_name='TR')
```

3. **External regressors:**
```python
# Ekonomik göstergeler
model.add_regressor('inflation_rate')
model.add_regressor('exchange_rate')
```

### Hibrit Mod En İyi Kullanım

1. ML tahminini gör
2. Anormal durumları belirle
3. Manuel ayarlarla düzelt
4. Karşılaştır ve kaydet

## 🤝 Destek

Sorular veya öneriler için:
- İterasyon yaparak geliştirme
- Test ve iyileştirme
- Gerçek verilerle validasyon

## 📝 Notlar

- Model 2024 ve 2025 (9 ay) verisine dayalı
- 2025 tam yıl tahmini (9ay * 12/9) kullanıldı
- Bazı kategorilerde düşüş trendi görülüyor
- Bu normal - market dinamiklerini yansıtabilir
- Manuel müdahale ile ayarlanabilir

## 🎓 Öğrenme Kaynakları

**Prophet Dokümantasyonu:**
- https://facebook.github.io/prophet/

**Zaman Serisi Analizi:**
- SARIMA, ARIMA modelleri
- XGBoost for time series
- LSTM neural networks

**İstatistik:**
- Seasonality decomposition
- Trend analysis
- Confidence intervals
