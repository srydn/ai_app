import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgboost
from matplotlib import pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import joblib

df = pd.read_csv(r'C:\Users\srydn\PycharmProjects\AITechniques\data\StudentsPerformance.csv')
df = df.drop(["race/ethnicity"], axis=1)
df.columns = ["cinsiyet", "ebeveynEgitim", "ogleYemegi", "testHazirlik", "matSkoru", "okumaSkoru", "yazmaSkoru"]

duplicate_rows = df[df.duplicated()]
print('Tekrar Eden Satirlar: ',duplicate_rows)
########################################
# EDA: Kesifsel Veri Analizi, veriyi inceleme
# print(df.info()) # veriye genel bakis
# print(df.describe()) # degerleri gorme
# print(df.isnull().sum()) # eksik deger yok

## KATEGORIK VERI ANALIZI
# print(df['cinsiyet'].value_counts())
# sns.countplot(x='cinsiyet', data=df)
# plt.xlabel("Cinsiyet")
# plt.ylabel("Adet")
# plt.show()
# print(df['ebeveynEgitim'].value_counts())
# sns.countplot(x='ebeveynEgitim', data=df)
# plt.xlabel("Ebeveyn Eğitim Seviyesi")
# plt.ylabel("Adet")
# plt.show()
# print(df['ogleYemegi'].value_counts())
# sns.countplot(x='ogleYemegi', data=df)
# plt.xlabel("Öğle Yemeği Ücreti")
# plt.ylabel("Adet")
# plt.show()
# print(df['testHazirlik'].value_counts())
# sns.countplot(x='testHazirlik', data=df)
# plt.xlabel("Hazırlık Testi")
# plt.ylabel("Adet")
# plt.show()
## SAYISAL VERI ANALIZI
# sns.histplot(df['matSkoru'], kde=True)
# plt.show()
# sns.histplot(df['okumaSkoru'], kde=True)
# plt.show()
# sns.histplot(df['yazmaSkoru'], kde=True)
# plt.show()
# x = df.groupby("testHazirlik")["matSkoru"].mean()
# print(x)
# x = df.groupby("ebeveynEgitim")["matSkoru"].mean()
# print(x)
# x = df.groupby("ogleYemegi")["matSkoru"].mean()
# print(x)
# sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm') # heatmap ile korelasyon analizi
# plt.show()
#######################################


# preprocessing
y = df["matSkoru"]
# print(df.columns)
df = pd.get_dummies(df, columns=["cinsiyet", "ebeveynEgitim", "ogleYemegi", "testHazirlik"], drop_first=True, dtype=int)
x = df.drop(["matSkoru"], axis=1)

# print(df)
print(df.columns)

# # 80:20 split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=28)

models = {
    "Lineer Regresyon": LinearRegression(),
    "Ridge Regresyon": Ridge(alpha=0.01),
    "Lasso Regresyon": Lasso(alpha=0.0001),
    "ElasticNet": ElasticNet(alpha=0.0001, l1_ratio=0.5),
    "Decision Tree": DecisionTreeRegressor(random_state=28),
    "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=5, random_state=28),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200),
    "XGBoost": xgboost.XGBRegressor(objective='reg:squarederror', random_state=28),
    "KNeighbors": KNeighborsRegressor(n_neighbors=5),
    "Support Vector Regressor": SVR(kernel='linear')
}

results = []
trained_models = {}
for n, model in models.items():
    model.fit(x_train, y_train)
    if (n == "Lineer Regresyon"):
        trained_models[n] = model
    y_pred = model.predict(x_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    results.append({
        "Model": n,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    })

results_df = pd.DataFrame(results).sort_values(by="MAE")
print(results_df)
pd.DataFrame(results).set_index('Model')[["MAE", "RMSE", "R2"]].plot(kind="bar", figsize=(10, 6),
                                                                     title="Regresyon Modellerinin Karşılaştırılması")
plt.ylabel("Skor")
plt.tight_layout()
# plt.show()

print(x_train.columns)
# Feature importance
feature_names = x_train.columns
# print(feature_names)
imp = pd.Series(trained_models["Lineer Regresyon"].coef_[0], index=feature_names)
imp = imp.sort_values(key=abs, ascending=False)

plt.figure(figsize=(12, 6))
imp.plot(kind='bar')
plt.title("Lineer Regresyon Öznitelik Önem Dereceleri")
plt.ylabel("Katsayı değeri")
plt.tight_layout()
# plt.show()

# feature importance yaptıktan sonra
x_dropped_train = df.drop(["matSkoru", "okumaSkoru"],
                          axis=1)  # sadece okumaSkoru elendiğinde model biraz daha iyi skorlar veriyor.
# print(x_dropped_train.columns)
# # 80:20 split
x_train_2, x_test_2, y_train, y_test = train_test_split(x_dropped_train, y, train_size=0.8, random_state=28)

lr = LinearRegression()
model_featured = lr.fit(x_train_2, y_train)
y_pred = model_featured.predict(x_test_2)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(mae, rmse, r2)
# joblib.dump(trained_models["Lineer Regresyon"],"student_performance_model.pkl")
