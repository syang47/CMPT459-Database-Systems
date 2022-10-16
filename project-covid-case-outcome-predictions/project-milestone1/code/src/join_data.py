import numpy as np
import pandas as pd

def join_train_data (location,cases_train):    
    location['Country_Region'].replace({"US": "United States"}, inplace=True)
    location['Country_Region'].replace({"Korea, South": "South Korea"}, inplace=True)
    initial_Merge = pd.merge(cases_train, location, left_on=['province', 'country'], right_on=['Province_State', 'Country_Region'], how = 'left')
    initial_Merge = initial_Merge.drop(columns=['Province_State', 'Country_Region'])
    return initial_Merge

def join_test_data (location,cases_test):
    location['Country_Region'].replace({"US": "United States"}, inplace=True)
    location['Country_Region'].replace({"Korea, South": "South Korea"}, inplace=True)
    initial_Merge = pd.merge(cases_test, location, left_on=['province', 'country'], right_on=['Province_State', 'Country_Region'], how = 'left')
    initial_Merge = initial_Merge.drop(columns=['Province_State', 'Country_Region'])
    return initial_Merge
