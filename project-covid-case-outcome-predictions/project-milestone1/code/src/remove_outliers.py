import pandas as pd
import numpy as np
from datetime import date,timedelta,datetime

def drop_outliers(df):
    #find outliers with out of bounds coordinates
    print("rows with longitude out of bounds:",len(df.loc[df["longitude"]>180])+len(df.loc[df["longitude"]<-180]))
    print("rows with latitude out of bounds:",len(df.loc[df["latitude"]>90])+len(df.loc[df["latitude"]<-90]))

    #find outliers with date range
    covid19_start = date(2019,12,31)
    report_date = date(2021,3,31)
    for idx,row in df.iterrows():
        this_date = datetime.strptime(row["date_confirmation"],"%d.%m.%Y")
        if this_date.date() <= covid19_start or this_date.date() >= report_date:
            print(row)

    #find duplicate rows with same additional information
    df_added_info = df[df["additional_information"]!='']
    duplicates = df_added_info[df_added_info.duplicated()]

    #drop duplicate rows with same additional information
    df = df.drop(duplicates.index)

    #find rows with age 0 or less
    age_zeros = df[df["age"]<=0]

    #drop rows with age 0 or less
    df = df.drop(age_zeros.index)
    return df