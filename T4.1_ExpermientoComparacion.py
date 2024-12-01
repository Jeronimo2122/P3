import mlflow
import mlflow.sklearn
from tensorflow.keras.models import load_model # type: ignore
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Importar el dataset
df = pd.read_csv("DF_FINAL.csv")

# Definir variables independientes (X) y dependiente (y)
X = df.drop(columns=['punt_global'])  # Excluir la columna de puntaje global
y = df['punt_global']  # Variable objetivo

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Configurar el URI del artefacto para MLflow
mlflow.set_tracking_uri("file:///c:/Users/Jeronimo Vargas/OneDrive/Documentos/8 SEMESTRE/ANALITICA/P3/mlruns")

mlflow.set_experiment("Experimento Proyecto 3")


# Comenzar el seguimiento de experimentos en MLflow
with mlflow.start_run():

    # 1. Cargar el modelo 1 (Red Neuronal Simple)
    with mlflow.start_run(nested=True):
        model1 = load_model("model1.h5")  # Cargar el modelo 1
        y_pred_model1 = model1.predict(X_test)
        mse_model1 = mean_squared_error(y_test, y_pred_model1)
        r2_model1 = r2_score(y_test, y_pred_model1)

        # Loggear el modelo 1 y sus métricas en MLflow
        mlflow.log_param("model_type", "Neural Network Simple (M1)")
        mlflow.log_metric("mse", mse_model1)
        mlflow.log_metric("r2", r2_model1)
        mlflow.keras.log_model(model1, "model")

        print(f"Model 1 (Neural Network Simple) - MSE: {mse_model1}")
        print(f"Model 1 (Neural Network Simple) - R^2: {r2_model1}")
    
    # 2. Cargar el modelo 2 (Red Neuronal Profunda)
    with mlflow.start_run(nested=True):
        model2 = load_model("model2.h5")  # Cargar el modelo 2
        y_pred_model2 = model2.predict(X_test)
        mse_model2 = mean_squared_error(y_test, y_pred_model2)
        r2_model2 = r2_score(y_test, y_pred_model2)

        # Loggear el modelo 2 y sus métricas en MLflow
        mlflow.log_param("model_type", "Neural Network Profunda (M2)")
        mlflow.log_metric("mse", mse_model2)
        mlflow.log_metric("r2", r2_model2)
        mlflow.keras.log_model(model2, "model")

        print(f"Model 2 (Neural Network Profunda) - MSE: {mse_model2}")
        print(f"Model 2 (Neural Network Profunda) - R^2: {r2_model2}")
    
    # 3. Cargar el modelo 3 (Red Neuronal con Dropout y Regularización L2)
    with mlflow.start_run(nested=True):
        model3 = load_model("model3.h5")  # Cargar el modelo 3
        y_pred_model3 = model3.predict(X_test)
        mse_model3 = mean_squared_error(y_test, y_pred_model3)
        r2_model3 = r2_score(y_test, y_pred_model3)

        # Loggear el modelo 3 y sus métricas en MLflow
        mlflow.log_param("model_type", "Neural Network con Dropout y Regularización L2 (M3)")
        mlflow.log_metric("mse", mse_model3)
        mlflow.log_metric("r2", r2_model3)
        mlflow.keras.log_model(model3, "model")

        print(f"Model 3 (Neural Network con Dropout y Regularización L2) - MSE: {mse_model3}")
        print(f"Model 3 (Neural Network con Dropout y Regularización L2) - R^2: {r2_model3}")
