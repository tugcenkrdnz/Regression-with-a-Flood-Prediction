import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="Sel Riski Tahmin Sistemi", layout="wide")

# Modeli Yükle
@st.cache_resource
def load_model():
    return joblib.load('lightgbm_flood_model.pkl')

model = load_model()

st.title("🌊 Sel Olasılığı Tahmin Uygulaması")
st.write("Lütfen aşağıdaki 20 parametreyi doldurun. Model, istatistiksel özellikleri (sum, mean vb.) otomatik hesaplayacaktır.")

# 20 Orijinal Sütun İsmi (Eğitimdeki sırayla aynı olmalı!)
original_features = [
    'MonsoonIntensity', 'TopographyDrainage', 'RiverManagement',
    'Deforestation', 'Urbanization', 'ClimateChange', 'DamsQuality',
    'Siltation', 'AgriculturalPractices', 'Encroachments',
    'IneffectiveDisasterPreparedness', 'DrainageSystems',
    'CoastalVulnerability', 'Landslides', 'Watersheds',
    'DeterioratingInfrastructure', 'PopulationScore', 'WetlandLoss',
    'InadequatePlanning', 'PoliticalFactors'
]

# Kullanıcıdan 20 özelliği alalım (4 sütunlu bir yapıyla şık dursun)
input_values = []
cols = st.columns(4)

for i, feature in enumerate(original_features):
    with cols[i % 4]:
        val = st.number_input(f"{feature}", min_value=0, max_value=20, value=5)
        input_values.append(val)

# TAHMİN BUTONU
if st.button("Risk Analizi Yap", use_container_width=True):
    # 1. Kullanıcı verilerini DataFrame yapalım
    df_input = pd.DataFrame([input_values], columns=original_features)
    
    # 2. MODELİN BEKLEDİĞİ 5 EK ÖZELLİĞİ HESAPLA (Eğitimdekiyle birebir aynı)
    df_input['sum'] = df_input[original_features].sum(axis=1)
    df_input['std'] = df_input[original_features].std(axis=1)
    df_input['mean'] = df_input[original_features].mean(axis=1)
    df_input['max'] = df_input[original_features].max(axis=1)
    df_input['min'] = df_input[original_features].min(axis=1)
    
    # 3. Tahmin yap
    prediction = model.predict(df_input)
    
    # 4. Sonuçları Göster
    st.divider()
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        st.metric("Hesaplanan Sel Olasılığı", f"{prediction[0]:.4f}")
    
    with res_col2:
        if prediction[0] > 0.55:
            st.error("🚨 KRİTİK: Çok Yüksek Sel Riski!")
        elif prediction[0] > 0.45:
            st.warning("⚠️ UYARI: Orta Derece Risk.")
        else:
            st.success("✅ GÜVENLİ: Düşük Risk Seviyesi.")

    # Detaylı Bilgi
    st.info(f"Girilen değerlerin toplam puanı: {df_input['sum'].values[0]}")