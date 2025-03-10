import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
import os

# Diretório do modelo salvo
MODEL_PATH = "models/alexNet/saved_model/alexnet_cifar10.h5"

# Verificar se o modelo existe
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Modelo não encontrado em {MODEL_PATH}. Treine o modelo primeiro.")

# Carregar o modelo treinado
model = load_model(MODEL_PATH)
print("Modelo carregado com sucesso!")

# Carregar o conjunto de testes CIFAR-10
(_, _), (x_test, y_test) = cifar10.load_data()

# Normalizar os dados
test_images = x_test / 255.0

y_test_one_hot = to_categorical(y_test, 10)

# Avaliar o modelo no conjunto de testes
loss, accuracy = model.evaluate(test_images, y_test_one_hot, verbose=1)
print(f"Acurácia no conjunto de teste: {accuracy * 100:.2f}%")

# Fazer previsões
predictions = model.predict(test_images)
predicted_classes = np.argmax(predictions, axis=1)

# Comparar com os rótulos reais
y_true = y_test.flatten()

# Calcular métricas
from sklearn.metrics import classification_report, confusion_matrix

print("Relatório de Classificação:")
print(classification_report(y_true, predicted_classes, target_names=[str(i) for i in range(10)]))

print("Matriz de Confusão:")
print(confusion_matrix(y_true, predicted_classes))
