import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt

def feature_selection(dataset):
    X = dataset.iloc[:,12:17]
    y = dataset.iloc[:,10]
    model = ExtraTreesClassifier()
    model.fit(X,y)
    print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
    #plot graph of feature importances for better visualization
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    pic = feat_importances.plot(kind='barh')
    pic.savefig("../results/feature_selection.png")


def main():
    cleaned_train_df = pd.read_csv("../results/cases_2021_train_processed.csv")
    cleaned_test_df = pd.read_csv("../results/cases_2021_test_processed.csv")
    test_feature = feature_selection(cleaned_test_df)
    train_feature = feature_selection(cleaned_train_df)

if __name__ == '__main__':
    main()