import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. Selección del Dataset
# Usamos el dataset de Wisconsin Breast Cancer (Diagnóstico)
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

print("1. Dataset cargado:")
print(f"Dimensiones: {df.shape}")
print(df.head())

# 2. Preprocesamiento y Normalización
# Separamos características (X) y etiqueta (y)
X = df.drop('target', axis=1)
y = df['target']

# División en entrenamiento (70%) y prueba (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalización (Estandarización)
# La regresión logística requiere que las variables estén en la misma escala
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) # Ajustar y transformar train
X_test_scaled = scaler.transform(X_test)       # Solo transformar test

print("\n2. Preprocesamiento completado.")
print("Datos estandarizados (media=0, var=1).")

# 3. Implementación del Modelo
# Modelo: Regresión Logística
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# 4. Evaluación del Modelo
y_pred = model.predict(X_test_scaled)

# Métricas
acc = accuracy_score(y_test, y_pred)
print(f"\n4. Exactitud (Accuracy): {acc:.4f}")
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred, target_names=data.target_names))

# 5. Visualización
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=data.target_names, 
            yticklabels=data.target_names)
plt.title('Matriz de Confusión - Regresión Logística')
plt.ylabel('Verdadero')
plt.xlabel('Predicho')
plt.tight_layout()

plt.savefig('matriz_confusion.png')
plt.show()