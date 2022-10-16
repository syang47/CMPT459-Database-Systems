import pandas as pd
import argparse
from random_forest import RandomForest

def parse_args():
    parser = argparse.ArgumentParser(description='Run random forrest with specified input arguments')
    parser.add_argument('--n-classifiers', type=int,
                        help='number of features to use in a tree',
                        default=1)
    parser.add_argument('--train-data', type=str, default='data.csv',
                        help='train data path')
    parser.add_argument('--test-data', type=str, default='data.csv',
                        help='test data path')
    parser.add_argument('--criterion', type=str, default='entropy',
                        help='criterion to use to split nodes. Should be either gini or entropy.')
    parser.add_argument('--maxdepth', type=int, help='maximum depth of the tree',
                        default=5)
    parser.add_argument('--min-sample-split', type=int, help='The minimum number of samples required to be at a leaf node',
                        default=20)
    parser.add_argument('--max-features', type=int,
                        help='number of features to use in a tree',
                        default=12)
    a = parser.parse_args()
    return(a.n_classifiers, a.train_data, a.test_data, a.criterion, a.maxdepth, a.min_sample_split, a.max_features)


def read_data(path):
    data = pd.read_csv(path)
    return data

def main():
    n_classifiers, train_data_path, test_data_path, criterion, max_depth, min_sample_split, max_features = parse_args()
    train_data = read_data(train_data_path)
    test_data = read_data(test_data_path)
    # YOU NEED TO HANDLE MISSING VALUES HERE
    # ...
    train_data = train.drop(columns="fnlwgt")
    test_data = train.drop(columns="fnlwgt")
    test_data['income'] = test_data['income'].str.replace(".", "")
    random_forest = RandomForest(n_classifiers=n_classifiers,
                  criterion = criterion,
                  max_depth=  max_depth,
                  min_samples_split = min_sample_split ,
                  max_features = max_features )

    print("train_data: ", random_forest.fit(train_data, 'income'))
    print("train_data: ", random_forest.evaluate(train_data, 'income'))
    print("test_data: ", random_forest.evaluate(test_data, 'income'))

    # train = pd.read_csv("./data/train.csv")
    # test = pd.read_csv("./data/test.csv")
    # test['income'] = test['income'].str.replace(".", "")
    # q1 = RandomForest(n_classifiers=10,criterion='gini',max_depth=10,min_samples_split=20,max_features=11)
    # q1_acc = q1.fit(train,'income')
    
    # q1_test = q1.evaluate(test,'income')
    # q2 = RandomForest(n_classifiers=10,criterion='entropy',max_depth=10,min_samples_split=20,max_features=11)
    # q2_acc = q2.fit(train,'income')
    # q2_test = q2.evaluate(test,'income')
    # q4_acc = []
    # q4_test = []
    # for i in range(1,11):
    #     q4 = RandomForest(n_classifiers=10,criterion='gini',max_depth=i,min_samples_split=10,max_features=10)
    #     q4_acc.append(q4.fit(train, 'income'))
    #     q4_test.append(q4.evaluate(test,'income'))

    # print("q1_acc: ", q1_acc)
    # print("q2_acc: ", q2_acc)
    
    # print("q1_t_acc: ", q1_test)
    # print("q2_t_acc: ", q2_test)

    # plotting graph
    # plt.plot(range(1,11),q4_test)
    # # naming the x axis
    # plt.xlabel('Depth')
    # # naming the y axis
    # plt.ylabel('Test Accuracy')
    
    # # giving a title to my graph
    # plt.title('Test Accuracy vs. Max Depth')
if __name__ == '__main__':
    main()

