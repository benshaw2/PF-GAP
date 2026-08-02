#!/usr/bin/env python
# coding: utf-8

# In[1]:


import PF_wrapper as PF
import numpy as np
import pandas as pd
import random

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_blobs

from aeon.datasets import load_japanese_vowels


# In[2]:


def inject_missing_values(data, missing_rate=0.1, seed=None):
    """
    Inject missing values into a dataset.

    Parameters:
    - data: numpy array (2D or 3D) or list of 2D numpy arrays
    - missing_rate: proportion of values to set as missing
    - seed: random seed for reproducibility

    Returns:
    - data with missing values injected (same type as input)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    def inject_into_array(arr):
        arr = np.copy(arr)
        total_elements = arr.size
        num_missing = int(total_elements * missing_rate)
        indices = [(i, j) for i in range(arr.shape[0]) for j in range(arr.shape[1])]
        missing_indices = random.sample(indices, num_missing)
        for i, j in missing_indices:
            arr[i][j] = np.nan
        return arr

    if isinstance(data, list):
        return [inject_into_array(arr) for arr in data]
    elif isinstance(data, np.ndarray):
        if data.ndim == 2:
            return inject_into_array(data)
        elif data.ndim == 3:
            return np.array([inject_into_array(data[i]) for i in range(data.shape[0])])
        else:
            raise ValueError("Unsupported array dimensionality: expected 2D or 3D numpy array")
    else:
        raise TypeError("Unsupported data type: expected numpy array or list of numpy arrays")


# ## Test NaN-Euclidean

# In[3]:


X, y = make_blobs(n_samples=int(1e4), n_features=int(1e2),random_state=0)


# In[4]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)


# In[5]:


y_train_df = pd.DataFrame(y_train)
y_train_df.to_csv("Data/blob_labels_train.csv", index=False, header=None)

y_test_df = pd.DataFrame(y_test)
y_test_df.to_csv("Data/blob_labels_test.csv", index=False, header=None)

X_train_df = pd.DataFrame(inject_missing_values(X_train))
X_train_df.to_csv("Data/blob_train.csv", index=False, header=None)

X_test_df = pd.DataFrame(inject_missing_values(X_test))
X_test_df.to_csv("Data/blob_test.csv", index=False, header=None)


# In[7]:


dir1 = "training_output"
dir2 = "training_predictions"


# In[14]:


PF.train("Data/blob_train.csv", test_file="Data/blob_test.csv", distances=['nan_euclidean'], 
                train_labels="Data/blob_labels_train.csv", test_labels="Data/blob_labels_test.csv",
                return_proximities=False, output_directory=dir1, #array_separator=":", entry_separator=",", 
                model_name="Waldo", data_dimension=1, return_training_outlier_scores=False,
                num_trees=11, parallel_trees=False, r=5, parallel_prox=False, impute_training_data=False,
                impute_testing_data=False, impute_iterations=0, return_imputed_training=False, return_imputed_testing=False,
                #initial_imputer="linear", DTWImpute=True, 
         return_predictions=True, numeric_data=False)


# ## Test DTW-AROW

# In[15]:


PF.train("Data/blob_train.csv", test_file="Data/blob_test.csv", distances=['dtwarow'], 
                train_labels="Data/blob_labels_train.csv", test_labels="Data/blob_labels_test.csv",
                return_proximities=False, output_directory=dir1, #array_separator=":", entry_separator=",", 
                model_name="Waldo", data_dimension=1, return_training_outlier_scores=False,
                num_trees=11, parallel_trees=False, r=5, parallel_prox=False, impute_training_data=False,
                impute_testing_data=False, impute_iterations=0, return_imputed_training=False, return_imputed_testing=False,
                #initial_imputer="linear", DTWImpute=True, 
         return_predictions=True, numeric_data=False)


# ## Test NaN-Euclidean_i

# In[ ]:





# ## Test DTW-AROW_i and DTW-AROW_d

# In[16]:


X3, y3 = load_japanese_vowels(split="TRAIN")
X3t, y3t = load_japanese_vowels(split="TEST")


# In[17]:


def aeonToFile(X3, filename, entry_separator = ',', array_separator = ':'):

    with open(filename, 'w') as f:
        # Loop over the first dimension (the "planes")
        for plane in X3:
            # Loop over the second dimension (the "rows")
            for i, row in enumerate(plane):
                # Convert the row to a string, joining with the item separator
                row_str = entry_separator.join(str(item) for item in row)

                # Write the row string to the file
                f.write(row_str)

                # Add the sub-array separator if not the last row
                if i < len(plane) - 1:
                    f.write(array_separator)

            # Add a newline after each plane
            f.write('\n')


# In[18]:


dir1 = "training_output"
dir2 = "training_predictions"

y3df = pd.DataFrame(y3)
y3df.to_csv("Data/ThreeDlabels.csv", index=False, header=None)

y3tdf = pd.DataFrame(y3t)
y3tdf.to_csv("Data/ThreeDlabels_test.csv", index=False, header=None)

aeonToFile(inject_missing_values(X3), 'Data/ThreeDdata.txt')
aeonToFile(inject_missing_values(X3t), 'Data/ThreeDdata_test.txt')


# In[23]:


PF.train("Data/ThreeDdata.txt", test_file="Data/ThreeDdata_test.txt", distances=['dtwarow_i','dtwarow_d'], 
                train_labels="Data/ThreeDlabels.csv", test_labels="Data/ThreeDlabels_test.csv",
                return_proximities=False, output_directory=dir1, array_separator=":", entry_separator=",", 
                model_name="Waldo", data_dimension=2, return_training_outlier_scores=False,
                num_trees=11, parallel_trees=False, r=5, parallel_prox=False, impute_training_data=False,
                impute_testing_data=False, impute_iterations=0, return_imputed_training=False, return_imputed_testing=False,
                #initial_imputer="linear", DTWImpute=True, 
         return_predictions=True, numeric_data=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




