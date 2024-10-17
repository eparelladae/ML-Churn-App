import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from scipy.stats import randint, uniform

# Cargar el dataset procesado sin la columna 'TotalCharges'
@st.cache_data
def load_data():
    data = pd.read_csv('Telco-Customer-Churn-Final.csv').drop(columns=['TotalCharges'])
    # Feature Engineering: crear nuevas variables
    data['MonthlyCharge_Tenure'] = data['MonthlyCharges'] * data['tenure']
    data['HighCharges'] = (data['MonthlyCharges'] > 70).astype(int)
    return data

data = load_data()

# Dividir datos en variables predictoras y variable objetivo
X = data.drop(['Churn'], axis=1)
y = data['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Selección del modelo
st.title("Predicción de Churn de Clientes con Optimización Avanzada")
model_type = st.selectbox("Selecciona el tipo de modelo", ["Random Forest", "Logistic Regression", "Decision Tree", "XGBoost"])

# Configuración de hiperparámetros para RandomizedSearchCV
param_distributions = {
    "Random Forest": {
        "model__n_estimators": randint(50, 150),
        "model__max_depth": randint(10, 40),
        "model__min_samples_split": randint(2, 10)
    },
    "Logistic Regression": {
        "model__C": uniform(0.1, 5),
        "model__solver": ['lbfgs', 'liblinear']
    },
    "Decision Tree": {
        "model__max_depth": randint(5, 30),
        "model__min_samples_split": randint(2, 8),
        "model__min_samples_leaf": randint(1, 5),
        "model__class_weight": ['balanced', None]
    },
    "XGBoost": {
        "model__n_estimators": randint(50, 100),
        "model__max_depth": randint(3, 8),
        "model__learning_rate": uniform(0.01, 0.2)
    }
}

# Instanciar el modelo y configurar RandomizedSearchCV
if model_type == "Random Forest":
    model = RandomForestClassifier(random_state=42)
elif model_type == "Logistic Regression":
    model = LogisticRegression(max_iter=500, class_weight='balanced')
elif model_type == "Decision Tree":
    model = DecisionTreeClassifier(random_state=42)
else:
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

# Configurar pipeline con escalado y búsqueda aleatoria de hiperparámetros
scaler = StandardScaler()
pipe = Pipeline([
    ('scaler', scaler),
    ('model', model)
])
random_search = RandomizedSearchCV(pipe, param_distributions[model_type], n_iter=20, cv=3, n_jobs=-1, scoring='accuracy', random_state=42)

# Entrenar el modelo con la mejor combinación de hiperparámetros
random_search.fit(X_train, y_train)
best_model = random_search.best_estimator_

# Validación cruzada en el conjunto de prueba para evaluar el modelo
cv_scores = cross_val_score(best_model, X_test, y_test, cv=3, scoring='accuracy')
cv_mean_score = cv_scores.mean()

# Hacer predicciones en el conjunto de prueba y calcular métricas
preds = best_model.predict(X_test)
accuracy = accuracy_score(y_test, preds)
error = mean_squared_error(y_test, preds)

# Mostrar resultados de precisión y errores del modelo optimizado
st.write(f"Precisión del modelo {model_type} optimizado en test: {accuracy:.2%}")
st.write(f"Error cuadrático medio del modelo en test: {error:.2f}")
st.write(f"Precisión promedio de validación cruzada en test: {cv_mean_score:.2%}")

# Entradas de usuario para predicción de nuevo cliente
st.sidebar.title("Introduce datos de un nuevo cliente:")
tenure = st.sidebar.slider("Antigüedad (en meses)", min_value=int(X['tenure'].min()), max_value=int(X['tenure'].max()), step=1)
monthly_charges = st.sidebar.number_input("Monthly Charges", min_value=0.0, step=0.1)

# Entradas para variables categóricas
gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
payment_method = st.sidebar.selectbox("Payment Method", ["Bank transfer (automatic)", "Credit card (automatic)",
                                                         "Electronic check", "Mailed check"])

# Conversión a variables dummy
input_data = pd.DataFrame(columns=X.columns)
input_data.loc[0] = 0  # Inicia todas las variables en 0

# Asignar valores a las variables numéricas
input_data['tenure'] = tenure
input_data['MonthlyCharges'] = monthly_charges
input_data['MonthlyCharge_Tenure'] = monthly_charges * tenure
input_data['HighCharges'] = int(monthly_charges > 70)

# Convertir género a dummies
if gender == "Male":
    input_data['gender_Male'] = 1

# Convertir contract a dummies
if contract == "One year":
    input_data['Contract_One year'] = 1
elif contract == "Two year":
    input_data['Contract_Two year'] = 1

# Convertir payment_method a dummies
if payment_method == "Electronic check":
    input_data['PaymentMethod_Electronic check'] = 1
elif payment_method == "Mailed check":
    input_data['PaymentMethod_Mailed check'] = 1
elif payment_method == "Credit card (automatic)":
    input_data['PaymentMethod_Credit card (automatic)'] = 1

# Escalar las variables numéricas del input de usuario usando el escalador entrenado
input_data_scaled = scaler.transform(input_data)

# Predecir y mostrar resultado
churn_proba = best_model.predict_proba(input_data_scaled)[0][1] * 100  # probabilidad de churn
st.write(f"Probabilidad de Churn: {churn_proba:.2f}%")

if churn_proba >= 50:
    st.write("El modelo predice: **Churn (abandono)**")
else:
    st.write("El modelo predice: **No Churn (retención)**")
