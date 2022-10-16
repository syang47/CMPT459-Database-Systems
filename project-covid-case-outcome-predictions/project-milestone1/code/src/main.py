import numpy as np
import pandas as pd

import cleaning_outcome_labels
import cleaning_missing_values
import join_data
import remove_outliers
import clean_location
import RemoveOutliers
def main():
    # 1.1 cleaning messy outcome labels
    # 1.4 cleaning and imputing missing values   

    cleaned_train_df,cleaned_test_df,cleaned_location_df = cleaning_missing_values.impute_missing_vals()
    cleaned_location_df = clean_location.clean_location_file(cleaned_location_df)

    # 1.5 removing outliers
    # cleaned_train_df = remove_outliers.drop_outliers(cleaned_train_df)
    # cleaned_test_df = remove_outliers.drop_outliers(cleaned_test_df)
    cleaned_location_df = RemoveOutliers.remove_outliers(cleaned_location_df)

    # 1.6 joining dataframes
    merge_loc_train_df = join_data.join_train_data(cleaned_location_df, cleaned_train_df)
    merge_loc_test_df = join_data.join_test_data(cleaned_location_df, cleaned_test_df)

    # 1.7 selecting features
    merge_loc_train_df = merge_loc_train_df.drop(columns=['additional_information', "Last_Update", "Combined_Key", "chronic_disease_binary"])
    merge_loc_test_df = merge_loc_test_df.drop(columns=['additional_information', "Last_Update", "Combined_Key", "chronic_disease_binary"])
    merge_loc_train_df.to_csv("../results/trained_processed_features.csv", index=False)
    merge_loc_test_df.to_csv("../results/test_processed_features.csv", index=False)

    
if __name__ == '__main__':
    main()