
#--------------------------------------------------------------------------
#------- Prediccion de placa con base en Caracteristicas HOG ---------
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
import pickle
from hog_features import pipeline,get_letters,locations
import numpy as np

# Entrada
placa = 'placas/dataset/ACU185.jpg'
placa = 'placas/dataset/OKQ117.jpg'
answer = os.path.basename(placa).split('.')[0]

# Preprocesamiento de la placa
chars,letters,features,hogs,bounds,figs = get_letters(placa,True,locations)

# Carga del modelo ML
def load_pickle(f):
    with open(f, 'rb') as fp:
        a = pickle.load(fp)
    return a
model = load_pickle('automl-hog-char-Explain.pickle')

# Prediccion
prediction = ''.join(model.predict(np.array(features)).tolist())

print(prediction,'==',answer,':',answer==prediction)


