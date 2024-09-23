import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


# 1. Adım: Veri setini yükleme
df = pd.read_csv('datasets/ObesityDataSet_raw_and_data_sinthetic.csv')



# 2. Adım: SQLite veritabanına veri setini kaydetme
conn = sqlite3.connect('obesity_data.db')
df.to_sql('obesity', conn, if_exists='replace', index=False)
conn.close()

# 3. Adım: Veri Analizi ve Korelasyon Matrisi Görselleştirme
corr_matrix = df.corr()
plt.figure(figsize=(12,8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Korelasyon Matrisi')
plt.show()

# 4. Adım: Veri setini eğitim ve test verilerine ayırma
X = df.drop(columns='NObeyesdad')  # Bağımsız değişkenler
y = df['NObeyesdad']  # Hedef değişken

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Adım: Veriyi ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Adım: Logistic Regression Modeli
logreg = LogisticRegression()
logreg.fit(X_train_scaled, y_train)

# Tahmin yapma ve değerlendirme
y_pred_logreg = logreg.predict(X_test_scaled)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_logreg))
print(classification_report(y_test, y_pred_logreg))

# 7. Adım: XGBoost Modeli
xgb_model = XGBClassifier()
xgb_model.fit(X_train_scaled, y_train)

y_pred_xgb = xgb_model.predict(X_test_scaled)
print("XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))
print(classification_report(y_test, y_pred_xgb))

# 8. Adım: Streamlit Web Uygulaması
st.title('Obezite Tahmin Modeli')

# Kullanıcı girdisi
age = st.slider('Yaş', min_value=10, max_value=80, value=25)
height = st.number_input('Boy (cm)', value=170)
weight = st.number_input('Kilo (kg)', value=70)

# Model tahmini yapma
if st.button('Tahmin Et'):
    input_data = scaler.transform([[age, height, weight]])  # Girdileri ölçeklendir
    prediction = xgb_model.predict(input_data)
    st.write('Tahmin edilen obezite durumu:', prediction[0])

# Streamlit uygulamasını çalıştırmak için terminalde şu komutu kullanın:
# streamlit run app.py
from sklearn.preprocessing import LabelEncoder

# Label encoding for categorical variables
label_encoder = LabelEncoder()
df['Cinsiyet'] = label_encoder.fit_transform(df['Cinsiyet'])  # 'Cinsiyet' sütununu uygun şekilde değiştirin
