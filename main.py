import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

# Cargar el dataset procesado sin la columna 'TotalCharges'
@st.cache_data
def load_data():
    data = pd.read_csv('Telco-Customer-Churn-Final.csv').drop(columns=['TotalCharges'])
    return data

data = load_data()

# Dividir datos en variables predictoras y variable objetivo
X = data.drop(['Churn'], axis=1)
y = data['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Selección del modelo
st.title("Predicción de Churn de Clientes con Optimización de Hiperparámetros")
model_type = st.selectbox("Selecciona el tipo de modelo", ["Random Forest", "Logistic Regression", "Decision Tree", "XGBoost"])

# Configuración de hiperparámetros para GridSearchCV
param_grids = {
    "Random Forest": {
        "model__n_estimators": [50, 100, 150],
        "model__max_depth": [10, 20, 30],
        "model__min_samples_split": [2, 5, 10]
    },
    "Logistic Regression": {
        "model__C": [0.1, 1.0, 10.0],
        "model__solver": ['lbfgs', 'liblinear']
    },
    "Decision Tree": {
        "model__max_depth": [10, 20, 30],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4]
    },
    "XGBoost": {
        "model__n_estimators": [50, 100, 150],
        "model__max_depth": [3, 6, 10],
        "model__learning_rate": [0.01, 0.1, 0.2]
    }
}

# Instanciar el modelo y configurar GridSearchCV
if model_type == "Random Forest":
    model = RandomForestClassifier(random_state=42)
elif model_type == "Logistic Regression":
    model = LogisticRegression(max_iter=1000)
elif model_type == "Decision Tree":
    model = DecisionTreeClassifier(random_state=42)
else:
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

# Configurar pipeline con escalado y grid search
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', model)
])
grid_search = GridSearchCV(pipe, param_grids[model_type], cv=5, n_jobs=-1, scoring='accuracy')

# Entrenar el modelo con la mejor combinación de hiperparámetros
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
preds = best_model.predict(X_test)
accuracy = accuracy_score(y_test, preds)
error = mean_squared_error(y_test, preds)

# Mostrar los resultados del modelo optimizado

st.write(f"Precisión del modelo {model_type} optimizado: {accuracy:.2%}")
st.write(f"Error cuadrático medio del modelo: {error:.2f}")

# Entradas de usuario para predicción de nuevo cliente
st.sidebar.title("Introduce datos de un nuevo cliente:")
tenure = st.sidebar.slider("Tenure", min_value=int(X['tenure'].min()), max_value=int(X['tenure'].max()), step=1)
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

# Predecir y mostrar resultado
churn_proba = best_model.predict_proba(input_data)[0][1] * 100  # probabilidad de churn
st.write(f"Probabilidad de Churn: {churn_proba:.2f}%")

if churn_proba >= 50:
    st.write("El modelo predice: **Churn (abandono)**")
else:
    st.write("El modelo predice: **No Churn (retención)**")
