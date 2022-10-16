import numpy as np

def clean_outcome_labels(df):
    conditions = [
        (df['outcome'].str.contains("discharge|hospitalized|critical condition", case=False)),
        (df['outcome'].str.contains("alive|treatment|hom|released|stable", case=False)),
        (df['outcome'].str.contains("dead|death|deceased|died", case=False)),
        (df['outcome'].str.contains("recovered", case=False))
    ]

    values = ['hospitalized','nonhospitalized','deceased','recovered']
    df.drop("outcome", inplace=True, axis=1)
    df['outcome_group'] = np.select(conditions,values)
    return df