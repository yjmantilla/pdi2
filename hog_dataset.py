import os
import sys
sys.path.insert(0, os.path.abspath('..'))
import glob
import numpy as np
from hog_features import get_letters
import matplotlib.pyplot as plt
single_char_path = "Y:/code/pdi2/placas/single-char-dataset"
output_path = "Y:/code/pdi2/placas/single-char-explore"
os.makedirs(output_path,exist_ok=True)

hog_files = glob.glob(os.path.join(single_char_path,'*_hog.npy'))
char_files = [x.replace('_hog','_char') for x in hog_files]
feature_files = [x.replace('_hog','_feature') for x in hog_files]
labels = np.array([os.path.basename(x).split('-')[1] for x in hog_files])

hogs = np.array([np.load(x,allow_pickle=True) for x in hog_files])
chars = np.array([np.load(x,allow_pickle=True) for x in char_files])
features = np.array([np.load(x,allow_pickle=True) for x in feature_files])

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

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from supervised import AutoML
from sklearn.model_selection import train_test_split
from supervised.automl import AutoML
import psutil
njobs=len(psutil.Process().cpu_affinity())
X = features
y = np.array(int_labels)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25
)
os.makedirs('AutoML-HOG',exist_ok=True)
automl = AutoML(results_path="AutoML-HOG",ml_task='multiclass_classification',n_jobs=njobs)
automl.fit(X_train, y_train)

predictions = automl.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, predictions)*100.0:.2f}%" )