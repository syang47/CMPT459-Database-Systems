## CMPT 459 Final Milestone 

"code/project.ipynb" is the project's code file.


The data files are stored at "code/data/folder" that we will use are: 

*   "cleaned_train_df.csv"
*   "cleaned_test_df.csv"

It is recommended to import these to files to run the notebook, as running the preprocessing code cell takes a very long time. 


Original Datafiles are converted to .csv files and stored in the data folder.


Required Python Version: 
*   Python 3.7.13

Required Python Libraries/Packages:
*   pandas, numpy, seaborn, matplotlib
*   datetime, geopy, certifi, ssl, copy, csv
*   urllib, imblearn, sklearn, collections

How we Imported Libraries/packages:
*   import pandas as pd
*   import numpy as np
*   import seaborn as sns
*   import matplotlib.pyplot as plt
*   import datetime
*   import geopy
*   import certifi
*   import ssl
*   import copy
*   import csv
*   from geopy.geocoders import Nominatim
*   from urllib.parse import urlparse
*   from collections import Counter
*   from imblearn.pipeline import make_pipeline, Pipeline
*   from sklearn.preprocessing import StandardScaler
*   from sklearn.model_selection import StratifiedKFold
*   from sklearn.svm import SVC
*   from sklearn.metrics import accuracy_score, f1_score, make_scorer
*   from sklearn.model_selection import GridSearchCV
*   from sklearn.ensemble import RandomForestClassifier
*   from sklearn.neighbors import KNeighborsClassifier
*   from imblearn.over_sampling import SMOTE
