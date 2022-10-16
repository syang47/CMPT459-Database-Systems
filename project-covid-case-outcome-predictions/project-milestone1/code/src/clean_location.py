import numpy as np
import pandas as pd

def clean_location_file(cleaned_location_df):
    
    na_fatality_df = cleaned_location_df.loc[cleaned_location_df.Case_Fatality_Ratio.isna()]
    na_incident_df = cleaned_location_df.loc[cleaned_location_df.Incident_Rate.isna()]
    na_active_df = cleaned_location_df.loc[cleaned_location_df.Active.isna()]
    na_recovered_df = cleaned_location_df.loc[cleaned_location_df.Recovered.isna()]

    file_without_na_fatality = cleaned_location_df.loc[cleaned_location_df.Case_Fatality_Ratio.isna() == False]
    file_without_na_incident = cleaned_location_df.loc[cleaned_location_df.Incident_Rate.isna() == False]
    file_without_na_active = cleaned_location_df.loc[cleaned_location_df.Active.isna() == False]
    file_without_na_recovered = cleaned_location_df.loc[cleaned_location_df.Recovered.isna() == False]


    for i, row in na_fatality_df.Country_Region.iteritems():
        cleaned_location_df.Case_Fatality_Ratio.iloc[i] = file_without_na_fatality.Case_Fatality_Ratio.loc[file_without_na_fatality.Country_Region == row].mean()

    for i, row in na_incident_df.Country_Region.iteritems():
        cleaned_location_df.Incident_Rate.iloc[i] = file_without_na_incident.Incident_Rate.loc[file_without_na_incident.Country_Region == row].mean()
    for i, row in na_active_df.Active.iteritems():
            cleaned_location_df.Active.iloc[i] = file_without_na_active.Active.loc[file_without_na_active.Country_Region == row].mean()
    for i, row in na_recovered_df.Recovered.iteritems():
            cleaned_location_df.Recovered.iloc[i] = file_without_na_recovered.Recovered.loc[file_without_na_recovered.Country_Region == row].mean()

    return cleaned_location_df
