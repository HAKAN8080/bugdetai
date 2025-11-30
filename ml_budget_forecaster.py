import pandas as pd
import numpy as np
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

class MLBudgetForecaster:
    """
    Machine Learning tabanlı bütçe tahmin motoru
    Prophet kullanarak kategori bazlı satış tahminleri yapar
    """
    
    def __init__(self):
        self.models = {}
        self.forecasts = {}
        
    def prepare_data_for_prophet(self, df, category=None):
        """
        Prophet için veri hazırlama
        Prophet formatı: ds (tarih), y (değer)
        """
        if category:
            df = df[df['MainGroupDesc'] == category].copy()
        
        # Uzun format'a çevir
        data_2024 = df[df['Sales_2024'].notna()][['Month', 'Sales_2024']].copy()
        data_2024['Year'] = 2024
        data_2024.rename(columns={'Sales_2024': 'Sales'}, inplace=True)
        
        data_2025 = df[df['Sales_2025'].notna()][['Month', 'Sales_2025']].copy()
        data_2025['Year'] = 2025
        data_2025.rename(columns={'Sales_2025': 'Sales'}, inplace=True)
        
        # Birleştir
        combined = pd.concat([data_2024, data_2025], ignore_index=True)
        
        # Tarih oluştur
        combined['Month'] = combined['Month'].astype(int)
        combined['ds'] = pd.to_datetime(
            combined['Year'].astype(str) + '-' + 
            combined['Month'].astype(str).str.zfill(2) + '-01'
        )
        combined['y'] = combined['Sales']
        
        # Sadece ds ve y kolonlarını al
        prophet_data = combined[['ds', 'y']].sort_values('ds')
        
        return prophet_data
    
    def train_model(self, category_data, category_name):
        """
        Belirli bir kategori için Prophet modeli eğit
        """
        # Model parametreleri
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode='multiplicative',  # Çarpımsal sezonsellik (retail için uygun)
            changepoint_prior_scale=0.05,  # Trend değişim hassasiyeti
        )
        
        # Modeli eğit
        model.fit(category_data)
        
        # Modeli kaydet
        self.models[category_name] = model
        
        return model
    
    def forecast_2026(self, model, periods=12):
        """
        2026 için tahmin yap
        """
        # Gelecek 12 ay için tarih oluştur
        future = model.make_future_dataframe(periods=periods, freq='MS')
        
        # Tahmin yap
        forecast = model.predict(future)
        
        return forecast
    
    def get_2026_forecast(self, forecast):
        """
        Sadece 2026 tahminlerini al
        """
        forecast_2026 = forecast[forecast['ds'].dt.year == 2026].copy()
        forecast_2026['Month'] = forecast_2026['ds'].dt.month
        forecast_2026 = forecast_2026[['Month', 'yhat', 'yhat_lower', 'yhat_upper']]
        forecast_2026.columns = ['Month', 'Forecast', 'Lower_Bound', 'Upper_Bound']
        
        return forecast_2026
    
    def train_all_categories(self, df):
        """
        Tüm kategoriler için model eğit ve tahmin yap
        """
        categories = df['MainGroupDesc'].unique()
        all_forecasts = {}
        
        print(f"🤖 {len(categories)} kategori için ML modeli eğitiliyor...\n")
        
        for i, category in enumerate(categories, 1):
            try:
                # Veriyi hazırla
                prophet_data = self.prepare_data_for_prophet(df, category)
                
                if len(prophet_data) < 2:
                    print(f"⚠️  {category}: Yetersiz veri, atlanıyor")
                    continue
                
                # Model eğit
                model = self.train_model(prophet_data, category)
                
                # Tahmin yap
                forecast = self.forecast_2026(model)
                forecast_2026 = self.get_2026_forecast(forecast)
                
                # Kaydet
                all_forecasts[category] = forecast_2026
                
                total_forecast = forecast_2026['Forecast'].sum()
                print(f"✅ {i:2d}. {category:20s} - 2026 Tahmini: {total_forecast:>12,.0f}")
                
            except Exception as e:
                print(f"❌ {category}: Hata - {str(e)}")
        
        self.forecasts = all_forecasts
        return all_forecasts
    
    def get_summary(self):
        """
        Tüm kategoriler için özet tahmin
        """
        summary = []
        
        for category, forecast in self.forecasts.items():
            summary.append({
                'Category': category,
                'Total_Forecast': forecast['Forecast'].sum(),
                'Lower_Bound': forecast['Lower_Bound'].sum(),
                'Upper_Bound': forecast['Upper_Bound'].sum(),
                'Avg_Monthly': forecast['Forecast'].mean()
            })
        
        summary_df = pd.DataFrame(summary)
        summary_df = summary_df.sort_values('Total_Forecast', ascending=False)
        
        return summary_df
    
    def compare_with_actuals(self, df):
        """
        2024 ve 2025 gerçek verilerle karşılaştırma
        """
        comparison = []
        
        for category in self.forecasts.keys():
            cat_data = df[df['MainGroupDesc'] == category]
            
            sales_2024 = cat_data['Sales_2024'].sum()
            sales_2025 = cat_data['Sales_2025'].sum()
            
            # 2025 tam yıl tahmini (9 ay veriden)
            if sales_2025 > 0:
                sales_2025_estimated = sales_2025 * (12/9)
            else:
                sales_2025_estimated = 0
            
            forecast_2026 = self.forecasts[category]['Forecast'].sum()
            
            # Büyüme oranları
            growth_24_25 = ((sales_2025_estimated - sales_2024) / sales_2024 * 100) if sales_2024 > 0 else 0
            growth_25_26 = ((forecast_2026 - sales_2025_estimated) / sales_2025_estimated * 100) if sales_2025_estimated > 0 else 0
            
            comparison.append({
                'Category': category,
                'Sales_2024': sales_2024,
                'Sales_2025_Est': sales_2025_estimated,
                'Forecast_2026': forecast_2026,
                'Growth_24_25_%': growth_24_25,
                'Growth_25_26_%': growth_25_26
            })
        
        comparison_df = pd.DataFrame(comparison)
        comparison_df = comparison_df.sort_values('Forecast_2026', ascending=False)
        
        return comparison_df


# Test edelim
if __name__ == "__main__":
    # Veriyi yükle
    df = pd.read_csv('/home/claude/budget_cleaned_data.csv')
    
    # ML Forecaster oluştur
    forecaster = MLBudgetForecaster()
    
    print("="*80)
    print("🚀 ML TABANLI BÜTÇE TAHMİN SİSTEMİ")
    print("="*80)
    print()
    
    # Tüm kategoriler için model eğit
    all_forecasts = forecaster.train_all_categories(df)
    
    print("\n" + "="*80)
    print("📊 ÖZET TAHMİNLER (2026)")
    print("="*80)
    
    # Özet
    summary = forecaster.get_summary()
    print(summary.to_string(index=False))
    
    print("\n" + "="*80)
    print("📈 BÜYÜME ANALİZİ")
    print("="*80)
    
    # Karşılaştırma
    comparison = forecaster.compare_with_actuals(df)
    print(comparison.to_string(index=False))
    
    print("\n" + "="*80)
    print("💾 Sonuçlar kaydediliyor...")
    
    # Sonuçları kaydet
    summary.to_csv('/home/claude/ml_forecast_summary.csv', index=False)
    comparison.to_csv('/home/claude/ml_forecast_comparison.csv', index=False)
    
    # Her kategori için detaylı tahmin
    with pd.ExcelWriter('/home/claude/ml_forecast_detailed.xlsx') as writer:
        for category, forecast in forecaster.forecasts.items():
            forecast.to_excel(writer, sheet_name=category[:30], index=False)
    
    print("✅ Tamamlandı!")
