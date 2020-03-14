# kaggle.py module

import csv
import numpy as np

### Graciously provided by your fellow classmate, Renos.
def kaggleize(predictions,file):

    if(len(predictions.shape)==1):
        predictions.shape = [predictions.shape[0],1]

    ids = np.arange(predictions.shape[0], dtype = float)[None].T

    kaggle_predictions = np.hstack((ids,predictions))

    np.savetxt(file, kaggle_predictions, header="Id,Predicted", delimiter=",")
