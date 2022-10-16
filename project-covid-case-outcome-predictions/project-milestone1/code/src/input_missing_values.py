import pandas as pd
import numpy as np
from datetime import date,timedelta,datetime

def fill_missing_values(df):
    #drop all rows missing age
    df = df[df["age"].notnull()]

    #imute empty addition_information with ""
    df["additional_information"] = df["additional_information"].replace(np.nan,"")

    #replace emptu source with ""
    df["source"] = df["source"].replace(np.nan,"")

    #replace age values that are a range with its middle value
    nonnumeric_age_rows = pd.to_numeric(df["age"],errors="coerce").isnull()
    for idx,row in df[nonnumeric_age_rows].iterrows():
        bounds = row["age"].split('-')
        lower = int(bounds[0])
        upper = bounds[1]!='' and int(bounds[1]) or lower
        df.at[idx,"age"] = (upper-lower)/2
    df["age"] = df["age"].astype(float)

    #calculate the mean date
    dates_nonempty = df["date_confirmation"].notnull()
    today = date(2020,2,3)
    mean = timedelta(0)
    count = 0
    for idx,row in df[dates_nonempty].iterrows():
        date_pieces = row["date_confirmation"].split('.')
        if len(date_pieces) != 3:
            date_pieces = row["date_confirmation"][0:10].split('.')
            df.at[idx,"date_confirmation"] = row["date_confirmation"][0:10]
        day = int(date_pieces[0])
        month = int(date_pieces[1])
        year = int(date_pieces[2])
        this_date = date(year,month,day)
        mean = mean + this_date-today
        count += 1
    mean = mean/count
    formatted = (today+mean).strftime("%d.%m.%Y")
    #set all missing date_confirmation to the mean date
    df["date_confirmation"] = df["date_confirmation"].replace(np.nan,formatted)
    return df