from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, cohen_kappa_score, confusion_matrix, roc_auc_score, f1_score
from sklearn.neural_network import MLPClassifier
import numpy as np
import pandas as pd
from sklearn.svm import SVC
import database_credentials as dc
from pymongo import MongoClient

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, cohen_kappa_score, roc_auc_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def expand_fingerprint(mfp_data, size=1024):
    bits = mfp_data.get('bits', [])  # Acceder al diccionario bits en mfp
    fingerprint = np.zeros(size)
    fingerprint[bits] = 1 # Colocamos un 1 donde nos indique en la mfp
    return fingerprint

import os
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, cohen_kappa_score, roc_auc_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def random_forest(X_train, y_train, X_test, y_test, n_estimators=100, max_depth=10):
    model = RandomForestClassifier(random_state=42, n_estimators=n_estimators, max_depth=max_depth)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # Evaluación del modelo
    accuracy = accuracy_score(y_test, y_pred)
    f1_sco = f1_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    # AURoc (necesita probabilidades predichas, no etiquetas)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    # Imprimir métricas
    print(f'Accuracy: {accuracy:.2f}')
    print(f'Kappa: {kappa:.2f}')
    print(f'ROC AUC: {roc_auc:.2f}')
    print(f'F1 Score: {f1_sco:.2f}')

    # Random Forest
    cm_rf = confusion_matrix(y_test, y_pred)
    print('Confusion Matrix Random Forest: \n', cm_rf)
    
    # Visualizar la matriz de confusión
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_rf)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix - Random Forest")
    
    # Guardar la imagen en el directorio donde está el script
    plt.savefig('confusion_matrix_random_forest.png')
    plt.close()

def svm(X_train, y_train, X_test, y_test):
    model = SVC()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # Evaluación del modelo
    accuracy = accuracy_score(y_test, y_pred)
    f1_sco = f1_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    # AURoc (necesita probabilidades predichas, no etiquetas)
    y_pred_proba = model.decision_function(X_test)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    # Imprimir métricas
    print(f'Accuracy: {accuracy:.2f}')
    print(f'Kappa: {kappa:.2f}')
    print(f'ROC AUC: {roc_auc:.2f}')
    print(f'F1 Score: {f1_sco:.2f}')

    # Support Vector Machine
    cm_rf = confusion_matrix(y_test, y_pred)
    print('Confusion Matrix SVM: \n', cm_rf)
    
    # Visualizar la matriz de confusión
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_rf)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix - SVM")
    
    # Guardar la imagen en el directorio donde está el script
    plt.savefig('confusion_matrix_svm.png')
    plt.close()

def mlp(X_train, y_train, X_test, y_test):
    model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10, 10), random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # Evaluación del modelo
    accuracy = accuracy_score(y_test, y_pred)
    f1_sco = f1_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    # AURoc (necesita probabilidades predichas, no etiquetas)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    # Imprimir métricas
    print(f'Accuracy: {accuracy:.2f}')
    print(f'Kappa: {kappa:.2f}')
    print(f'ROC AUC: {roc_auc:.2f}')
    print(f'F1 Score: {f1_sco:.2f}')

    # Multi-layer Perceptron
    cm_rf = confusion_matrix(y_test, y_pred)
    print('Confusion Matrix MLP: \n', cm_rf)

    # Visualizar la matriz de confusión
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_rf)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix - MLP")
    
    # Guardar la imagen en el directorio donde está el script
    plt.savefig('confusion_matrix_mlp.png')
    plt.close()


if __name__ == "__main__":
  
  # Preprocesamiento de datos:
    client, db, collection_CDS16  = dc.connect("CDS16", "molecules")
    client, db, collection_CDS29  = dc.connect("CDS29", "molecules")
    data_CDS16 = pd.DataFrame(list(collection_CDS16.find()))  
    data_CDS29 = pd.DataFrame(list(collection_CDS29.find()))  
    
    # Variables predictoras: mfp (1024)
    # Aplicar la función a cada fila
    data_CDS29['mfp'] = data_CDS29['mfp'].apply(lambda mfp_data: expand_fingerprint(mfp_data))
    data_CDS16['mfp'] = data_CDS16['mfp'].apply(lambda mfp_data: expand_fingerprint(mfp_data))
    
    
    # Extraemos las variables predictoras:
    # Variables predictoras y variable respuesta
    X_CDS29 = pd.DataFrame(data_CDS29['mfp'].tolist())
    X_CDS16 = pd.DataFrame(data_CDS16['mfp'].tolist())
    
    y_CDS29 = pd.DataFrame(data_CDS29['class'].map({'Active':True, 'Inactive':False})) # Convertimos a binario
    y_CDS16 = pd.DataFrame(data_CDS16['class'].map({'Active':True, 'Inactive':False}))
    
    # Dataset Balanceado
    # print(y_CDS16.value_counts()[0]) # 41
    # print(y_CDS16.value_counts()[1]) # 41
    
    
    # Dataset no Balanceado
    # print(y_CDS29.value_counts()[0]) # 5641
    # print(y_CDS29.value_counts()[1]) # 1374
    
    # Dividir en conjunto de entrenamiento y prueba con estratificación solo en el dataset desbalanceado
    X_train_CDS29, X_test_CDS29, y_train_CDS29, y_test_CDS29 = train_test_split(X_CDS29, y_CDS29, test_size=0.2, random_state=42,stratify=y_CDS29)
    X_train_CDS16, X_test_CDS16, y_train_CDS16, y_test_CDS16 = train_test_split(X_CDS16, y_CDS16, test_size=0.2, random_state=42)
    
    ## Clasificacion
    X_train = X_train_CDS16
    y_train = y_train_CDS16
    
    X_test = X_test_CDS16
    y_test = y_test_CDS16
    # 1. RandomForest
    random_forest(X_train, y_train, X_test, y_test, n_estimators=100, max_depth=10)
    print("=============================")
    # 2. SVM
    svm(X_train, y_train, X_test, y_test)
    print("=============================")
    # 3. MLP (Multilayer Perceptron)
    mlp(X_train, y_train, X_test, y_test)


    