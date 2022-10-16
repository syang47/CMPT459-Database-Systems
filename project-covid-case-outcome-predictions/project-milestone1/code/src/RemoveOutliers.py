import pandas as pd
import numpy as np
import seaborn
import matplotlib.pyplot as plt

def remove_outliers(test_data):
    numeric_col = ['Confirmed','Deaths','Active','Recovered']
    numeric_col1 = ['Case_Fatality_Ratio','Incident_Rate']
    fig1 = plt.figure()
    plot1 = test_data.boxplot(numeric_col)
    fig2 = plt.figure()
    plot2 = test_data.boxplot(numeric_col1)
    for x in ['Incident_Rate']:
        q75,q25 = np.percentile(test_data.loc[:,x],[75,25])
        intr_qr = q75-q25
 
        max = q75+(1.5*intr_qr)
        min = q25-(1.5*intr_qr)
 
        test_data.loc[test_data[x] < min,x] = np.nan
        test_data.loc[test_data[x] > max,x] = np.nan
    removed_data = test_data
    fig1.savefig("../plots/boxplot1.png")
    fig2.savefig("../plots/boxplot2.png")
    return removed_data