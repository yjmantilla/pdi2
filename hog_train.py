
#--------------------------------------------------------------------------
#------- Entrenamiento de Modelos con base en Caracteristicas HOG ---------
#------- Para Reconocimiento de Placas-------------------------------------
#------- Por: Yorguin José Mantilla Ramos    yorguinj.mantilla@udea.edu.co -
#-------      Estudiante de Ingeniería Electrónica -----------------
#-------      CC 1127617499 , Tel +13053591904,  Wpp +573115154452 ---------
#------- Curso Básico de Procesamiento de Imágenes y Visión Artificial-----
#------- Mayo de 2023--------------------------------------------------
#--------------------------------------------------------------------------


#--------------------------------------------------------------------------
#--Importacion de Librerias  -----------------------------------------------
#--------------------------------------------------------------------------

import os
import sys
sys.path.insert(0, os.path.abspath('..'))
import glob
import numpy as np
from hog_features import get_letters
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from supervised import AutoML
from sklearn.model_selection import train_test_split
from supervised.automl import AutoML
import psutil

# Directorios de Entrada y Salida
single_char_path = "Y:/code/pdi2/placas/single-char-dataset"
output_path = "Y:/code/pdi2/placas/single-char-explore_train"
os.makedirs(output_path,exist_ok=True)

# Lectura de la base de datos
hog_files = glob.glob(os.path.join(single_char_path,'*_hog.npy'))
char_files = [x.replace('_hog','_char') for x in hog_files]
feature_files = [x.replace('_hog','_feature') for x in hog_files]
labels = np.array([os.path.basename(x).split('-')[1] for x in hog_files])
hogs = np.array([np.load(x,allow_pickle=True) for x in hog_files])
chars = np.array([np.load(x,allow_pickle=True) for x in char_files])
features = np.array([np.load(x,allow_pickle=True) for x in feature_files])

#--------------------------------------------------------------------------
# Caracterización de los carácteres únicos en los datos (y ploteo)
#--------------------------------------------------------------------------

unique_chars,counts = np.unique(labels,return_counts=True)
for uchar,ucount in zip(unique_chars,counts):
    print(uchar)
    pp = os.path.join(output_path,f'{uchar}.png')
    if not os.path.isfile(pp):
        idxs=np.where(labels==uchar)
        mean_hog=hogs[idxs].mean(axis=0)
        mean_char = chars[idxs].mean(axis=0)
        mean_feats = features[idxs].mean(axis=0)

        fig,axes = plt.subplots(1,3)
        fig.set_size_inches(16,9)
        axes[1].imshow(mean_hog)
        axes[1].set_xlabel('MEAN HOG')
        axes[0].imshow(mean_char)
        axes[0].set_xlabel('MEAN CHAR')
        axes[2].plot(mean_feats,np.arange(len(mean_feats)))
        axes[2].xaxis.tick_top()
        axes[2].set_xlabel('FEATURE VECTOR')
        fig.suptitle(f'{uchar}:{ucount} counts')
        fig.savefig(pp)
        plt.close('all')

label_map = {c:i for i,c in enumerate(unique_chars)}
int_labels = [label_map[c] for c in labels]
njobs=1 #len(psutil.Process().cpu_affinity())

# Creacion de variables para entrenar los modelos
X = features
y = np.array(labels) #np.array(int_labels)

# Definicion de Estrategia de validación cruzada 
cv={
    "validation_type": "kfold",
    "k_folds": 5,
    "shuffle": True,
    "stratify": True,
    "random_seed": 123
}

# Entrenamiento de los modelos
for mode in ['Explain']:
    folder = f'AutoML-HOG-char-{mode}-cv'
    os.makedirs(folder,exist_ok=True)
    automl = AutoML(results_path=folder,ml_task='multiclass_classification',eval_metric='accuracy',validation_strategy=cv,explain_level=1)#,n_jobs=njobs)
    automl.fit(X, y)

    predictions = automl.predict(X)

    print(f"Accuracy: {accuracy_score(y, predictions)*100.0:.2f}%" )

    pickle.dump( automl, open( f"automl-hog-char-{mode}.pickle", "wb" ) )