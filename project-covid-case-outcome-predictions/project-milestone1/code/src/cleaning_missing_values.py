import pandas as pd
import numpy as np
import geopy
from geopy.geocoders import Nominatim
import certifi
import ssl
import copy
from datetime import datetime
import cleaning_outcome_labels
from urllib.parse import urlparse

ctx = ssl.create_default_context(cafile=certifi.where())
geopy.geocoders.options.default_ssl_context = ctx
geopy.geocoders.options.default_user_agent = "cmpt459"

geolocator = Nominatim(timeout= None)
cases_train = pd.read_csv("../data/cases_2021_train.csv")

def process_age(df):
    temp_copy = copy.deepcopy(df)
    
    age_col = temp_copy['age']
    for index, row in age_col.iteritems():
        if '-' in row:
            temp = row.split('-')
            if temp[0].isdigit() and temp[1].isdigit():
                age_col[index]=int((int(temp[0])+int(temp[1]))/2)
            else:
                age_col[index]=int(temp[0])
        else:
            age_col[index]=int(round(float(row))) 
    temp_copy['age'] = age_col
    return temp_copy

def process_gender(df):
    df['sex'] = df['sex'].fillna('unknown')
    return df
def process_province_and_country(df, s):
    cleaned_df = copy.deepcopy(df)
    
    for index, row in cleaned_df.iterrows():
        if s == 0:
            lat = str(row.latitude)
            lon = str(row.longitude)
            if 'nan' not in (lat or lon):
                coordinate = lat + "," + lon
                # check for nan countries
                if str(row.country) == "nan":               
                    cleaned_df.country[index] = "China"

                # check for nan provinces
                if str(row.province) == "nan":
                    location = geolocator.reverse(coordinate, language="en")

                    # modify province attribute when address is not none
                    if location is not None:
                        add_lst = location.address.split(', ')
                        if len(add_lst) < 2:
                            cleaned_df.province[index] = add_lst[0]

                        elif len(add_lst) > 2:
                            cleaned_df.province[index] = add_lst[-3]
        elif s == 1: 
            lat = str(row.Lat)
            lon = str(row.Long_)
            coordinate = lat + "," + lon
            if lat != 'nan' and lon != 'nan':               
                if "nan" in str(row.Country_Region):  
            
                    location = geolocator.reverse(coordinate, language="en")
                    country = location.address.split()[-1]
                    cleaned_df['Country_Region'][index] = country

                # check for nan provinces
                if str(row.Province_State) == "nan":
                    location = geolocator.reverse(coordinate, language="en")

                    # modify province attribute when address is not none
                    if location is not None:
                        add_lst = location.address.split(', ')
                        if len(add_lst) < 2:                
                            cleaned_df.Province_State[index] = add_lst[0]

                        elif len(add_lst) > 2:
                            cleaned_df.Province_State[index] = add_lst[-3]
    return cleaned_df

def cleaning_date_format(df):
    if '-' in df:
        df = df.split('-')
        return df[0].strip()
    else:
        return df
        
def impute_missing_vals():
    # read data files
    print("reading files\n")
    cases_train = pd.read_csv("../data/cases_2021_train.csv")
    cases_test = pd.read_csv("../data/cases_2021_test.csv")
    location = pd.read_csv("../data/location_2021.csv")

    # cleaning training data set
    
    print("cleaning train dataset...\n")
    # 1.1 clean outcome labels for training dataset
    cleaned_train_df = cleaning_outcome_labels.clean_outcome_labels(cases_train)

    ## 1.2 ##
    # drop missing age rows
    cleaned_train_df = cleaned_train_df.dropna(subset=['age']).reset_index(drop=True)

    # filling na vals with empty strings
    cleaned_train_df.additional_information = cleaned_train_df.additional_information.fillna(' ')
    cleaned_train_df.source = cleaned_train_df.source.fillna('')

    # parsing urls from source
    cleaned_train_df.source = cleaned_train_df.source.apply(lambda x: urlparse(x).netloc)
    cleaned_train_df.source = cleaned_train_df.source.fillna(' ')
    


    # cleaning age, gender, and province/country columns
    cleaned_train_df = process_age(cleaned_train_df)
    cleaned_train_df = process_gender(cleaned_train_df)
    cleaned_train_df = process_province_and_country(cleaned_train_df,0)

    # processing date column
    cleaned_train_df.date_confirmation = cleaned_train_df.date_confirmation.fillna(cleaned_train_df.date_confirmation.mode().iloc[0])
    cleaned_train_df.date_confirmation = cleaned_train_df.date_confirmation.apply(cleaning_date_format)

    cleaned_train_df = cleaned_train_df.drop_duplicates()    

    ## Cleaning Test Data
    print("cleaning test dataset...\n")

    # drop missing age rows
    cleaned_test_df = cases_test.dropna(subset=['age']).reset_index(drop=True)

    # filling na vals with empty strings
    cleaned_test_df.additional_information = cleaned_test_df.additional_information.fillna(' ')
    cleaned_test_df.outcome_group = cleaned_test_df.outcome_group.fillna(' ')
    cleaned_test_df.source = cleaned_test_df.source.fillna('')


    # parsing urls from source
    cleaned_test_df.source = cleaned_test_df.source.apply(lambda x: urlparse(x).netloc)
    cleaned_test_df.source = cleaned_test_df.source.fillna(' ')


    # cleaning age, gender, and province/country columns
    cleaned_test_df = process_age(cleaned_test_df)
    cleaned_test_df = process_gender(cleaned_test_df)
    cleaned_test_df = process_province_and_country(cleaned_test_df, 0)

    # processing date column
    cleaned_test_df.date_confirmation = cleaned_test_df.date_confirmation.fillna(cleaned_test_df.date_confirmation.mode().iloc[0])
    cleaned_test_df.date_confirmation = cleaned_test_df.date_confirmation.apply(cleaning_date_format)
    cleaned_test_df = cleaned_test_df.drop_duplicates()

    # Cleaning Location Data
    
    print("cleaning location dataset...\n")
    cleaned_location_df = process_province_and_country(location, 1)


    cleaned_location_df.Recovered = cleaned_location_df.Recovered.fillna(cleaned_location_df.Recovered.mean())
    cleaned_location_df.Active = cleaned_location_df.Active.fillna(cleaned_location_df.Active.mean())
    cleaned_location_df.Case_Fatality_Ratio = cleaned_location_df.Case_Fatality_Ratio.fillna(cleaned_location_df.Case_Fatality_Ratio.mean())
    cleaned_location_df.Incident_Rate = cleaned_location_df.Incident_Rate.fillna(cleaned_location_df.Incident_Rate.mean())

    cleaned_location_df = cleaned_location_df.dropna(subset=['Lat']).reset_index(drop=True)
    
    cleaned_location_df = cleaned_location_df.dropna(subset=['Long_']).reset_index(drop=True)
    # cleaned_location_df = cleaned_location_df.dropna(subset=['Active']).reset_index(drop=True)
    # cleaned_location_df = cleaned_location_df.dropna(subset=['Case_Fatality_Ratio']).reset_index(drop=True)
    # cleaned_location_df = cleaned_location_df.dropna(subset=['Incident_Rate']).reset_index(drop=True)


    cleaned_location_df = cleaned_location_df.drop_duplicates()

    # save cleaned datasets to csv
    
    print("saving csv files...\n")
    cleaned_train_df.to_csv("../results/cases_2021_train_processed.csv", index=False)
    cleaned_test_df.to_csv("../results/cases_2021_test_processed.csv", index=False)
    cleaned_location_df.to_csv("../results/location_2021_processed.csv", index=False)

    return cleaned_train_df, cleaned_test_df, cleaned_location_df