from turtle import right
from typing import Optional, Sequence, Mapping
import numpy as np
import pandas as pd
import random
import math
from collections import Counter
import matplotlib.pyplot as plt

class Node(object):
    def __init__(self, node_size: int, node_class: str, depth: int, single_class:bool = False):
        # Every node is a leaf unless you set its 'children'
        self.is_leaf = True
        # Each 'decision node' has a name. It should be the feature name
        self.name = None
        # All children of a 'decision node'. Note that only decision nodes have children
        self.children = {}
        # Whether corresponding feature of this node is numerical or not. Only for decision nodes.
        self.is_numerical = None
        # Threshold value for numerical decision nodes. If the value of a specific data is greater than this threshold,
        # it falls under the 'ge' child. Other than that it goes under 'l'. Please check the implementation of
        # get_child_node for a better understanding.
        self.threshold = None
        # The class of a node. It determines the class of the data in this node. In this assignment it should be set as
        # the mode of the classes of data in this node.
        self.node_class = node_class
        # Number of data samples in this node
        self.size = node_size
        # Depth of a node
        self.depth = depth
        # Boolean variable indicating if all the data of this node belongs to only one class. This is condition that you
        # want to be aware of so you stop expanding the tree.
        self.single_class = single_class
        self.mode_val = None

    def set_children(self, children):
        self.is_leaf = False
        self.children = children

    def get_child_node(self, feature_value)-> 'Node':
        if not self.is_numerical:
            return self.children[feature_value]
        else:          
            if feature_value >= self.threshold:
                return self.children['ge'] # ge stands for greater equal
            else:
                
                return self.children['l'] # l stands for less than


class RandomForest(object):
    def __init__(self, n_classifiers: int,
                 criterion: Optional['str'] = 'gini',
                 max_depth: Optional[int] = None,
                 min_samples_split: Optional[int] = None,
                 max_features: Optional[int] = None):
        """
        :param n_classifiers:
            number of trees to generated in the forrest
        :param criterion:
            The function to measure the quality of a split. Supported criteria are “gini” for the Gini
            impurity and “entropy” for the information gain.
        :param max_depth:
            The maximum depth of the trees.
        :param min_samples_split:
            The minimum number of samples required to be at a leaf node
        :param max_features:
            The number of features to consider for each tree.
        """
        self.n_classifiers = n_classifiers
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []
        self.criterion_func = self.entropy if criterion == 'entropy' else self.gini
        self.split_value = None
        self.attributes = []
        self.numerical = []
        self.categorical = []
        self.feature_count = 0

    def fit(self, X: pd.DataFrame, y_col: str)->float:
        """
        :param X: data
        :param y_col: label column in X
        :return: accuracy of training dataset
        """

        # process features from dataset
        features = self.process_features(X, y_col)
        
        for i in features:
          self.attributes.append(i['name'])
          if i['dtype'] == 'int64':
            self.numerical.append(i['name'])
          else:
            self.categorical.append(i['name'])        

        if len(self.trees) > 0:
          self.trees = []
        
        # create n_classifiers trees
        for i in range(self.n_classifiers):
          
          tree = self.generate_tree(X, y_col, features)

          self.trees.append(tree)
        return self.evaluate(X, y_col)

    def calc_predict_val(self, X, tree:Node):
      if (tree.is_leaf == True): 
        return tree.node_class
      else:        
        value = X[tree.name]

        if tree.is_numerical == False:
            if value in tree.children:
                return self.calc_predict_val(X, tree.get_child_node(value))
            else:
                return tree.node_class
        else:
          return self.calc_predict_val(X, tree.get_child_node(value))

    def predict(self, X: pd.DataFrame)->np.ndarray:
        """
        :param X: data
        :return: aggregated predictions of all trees on X. Use voting mechanism for aggregation.
        """
        predictions = []
        # Your code
        
        for _, x in X.iterrows():
            predict_list = []
            for tree in self.trees:
                predict_list.append(self.calc_predict_val(x,tree))
            temp = Counter(predict_list)
            predictions.append(temp.most_common(1)[0][0])          
        return predictions

        
    def evaluate(self, X: pd.DataFrame, y_col: str)-> int:
        """
        :param X: data
        :param y_col: label column in X
        :return: accuracy of predictions on X
        """
        
        preds = self.predict(X)
        
        acc = sum(preds == X[y_col]) / len(preds)
        print("accuracy with ", self.criterion, ": ",acc)
        return acc

    def generate_tree(self, X: pd.DataFrame, y_col: str,   features: Sequence[Mapping])->Node:
        """
        Method to generate a decision tree. This method uses self.split_tree() method to split a node.
        :param X:
        :param y_col:
        :param features:
        :return: root of the tree
        """
        root = Node(X.shape[0], X[y_col].mode()[0], 0)
        # Your code
        
        sampled_data = X.sample(n=len(X), replace=True)

        self.split_node(root, sampled_data, y_col, features)
        
        return root

    def split_node(self, node: Node, X: pd.DataFrame, y_col:str, features: Sequence[Mapping]) -> None:
        """
        This is probably the most important function you will implement. This function takes a node, uses criterion to
        find the best feature to slit it, and splits it into child nodes. I recommend to use revursive programming to
        implement this function but you are of course free to take any programming approach you want to implement it.
        :param node:
        :param X:
        :param y_col:
        :param features:
        :return:
        """   

        # If current dataset contains instances of only one class then return
        if (
            (node.depth >= self.max_depth) or 
            (len(X) <= self.min_samples_split) or 
            (node.single_class == True) or 
            len(X[y_col].unique()) <= 1
            ):
            node.is_leaf = True
            node.single_class = True
            return
        else:
            node.is_leaf = False
        # randomly select x% of the possible splitting features in X
        randomly_selected_features = random.choices(features, k=random.randint(2, self.max_features))

        # Select the feature F with the highest Information gain/gini index
        if self.criterion == "entropy":
            maxGain = -1
        else:
            maxGain = 1
        maxGain_attribute = None
        
        # split items into attributes list, numerical list, and categorical list
        attributes = [] 
        for i in randomly_selected_features:
            attributes.append(i['name'])
        
        # calc
        for col in attributes:
            best_gain = self.criterion_func(X, X[col], y_col)

            if self.criterion == "entropy":
                if maxGain <= best_gain:
                    maxGain = best_gain
                    maxGain_attribute = col
            else:
                if best_gain < maxGain:
                    maxGain = best_gain
                    maxGain_attribute = col

        node.name = maxGain_attribute
      
        ## if the best gain's feature is categorical value
        if maxGain_attribute not in self.numerical:
          node.is_numerical = False
          best_feature_all_class = X[maxGain_attribute].unique()
          
          for cur_class in best_feature_all_class:
            cur_dataset = X[X[maxGain_attribute] == cur_class]
            n_current = len(cur_dataset[y_col])

            # create new node
            new_node = Node(n_current, cur_dataset[y_col].mode()[0], node.depth+1)
            new_node.name = maxGain_attribute
            new_node.is_numerical = False
            node.single_class = False
            
            node.children[cur_class] = new_node    
        
            self.split_node(new_node, cur_dataset, y_col, features)

        # numerical
        else:
          
            node.is_numerical = True
            node.threshold = self.split_value
            
            cur_dataset_L = X[X[maxGain_attribute] < node.threshold]
            cur_dataset_U = X[X[maxGain_attribute] >= node.threshold]
            if len(cur_dataset_L) < 1 or len(cur_dataset_U) < 1:
                node.is_leaf=True
                node.single_class = True
                return
            new_node_l = Node(len(cur_dataset_L), cur_dataset_L[y_col].mode()[0], node.depth+1)
            new_node_l.name = maxGain_attribute
            new_node_l.is_numerical = False
            
            node.children['l'] = new_node_l   
            node.single_class=False
            self.split_node(new_node_l, cur_dataset_L, y_col, features)
            
            new_node_ge = Node(len(cur_dataset_U), cur_dataset_U[y_col].mode()[0], node.depth+1)
            new_node_ge.name = maxGain_attribute
            new_node_ge.is_numerical = False

            node.children['ge'] = new_node_ge   
            node.single_class=False
            self.split_node(new_node_ge, cur_dataset_U, y_col, features)
            return 
        return

    def gini_calc(self, lower, upper):
        split_gain = 0
        for targets in [lower, upper]:
            gini = 1
            for i in range(len(targets.unique())):
                prob = targets.value_counts()[i] * 1.0/len(targets)
                gini -= prob ** 2
            split_gain += len(targets)*1.0/(len(lower)+len(upper))*gini
        return split_gain

    def gini(self, X: pd.DataFrame, feature: Mapping, y_col: str) -> float:
        """
        Returns gini index of the give feature
        :param X: data
        :param feature: the feature you want to use to get compute gini score
        :param y_col: name of the label column in X
        :return:
        """
        unique_f = None
        if feature.name in self.numerical:
          
            cur_dataset = X[[feature.name,y_col]].sort_values(feature.name)
            unique_f = list(cur_dataset[feature.name].unique())
            best_gain = 1
            unique_f = np.percentile(unique_f, [85,75,50,25,15])
            for i in unique_f:
                lower = cur_dataset[cur_dataset[feature.name] < i][y_col]
                
                upper = cur_dataset[cur_dataset[feature.name] >= i][y_col]
                best_gain = self.gini_calc(lower, upper)
                self.split_value = i 
        else:
            cur_dataset = X[[feature.name,y_col]].sort_values(feature.name)
            unique_f = cur_dataset[feature.name].value_counts(sort=True)
            total_size = unique_f.sum()
            best_gain = 1
            prob = 0
            for i in unique_f:
                prob += (i/total_size)
                best_gain -= prob ** 2        
        return best_gain
      
    def entropy_calc(self, target_feature):
        # convert to integers to avoid runtime errors
        counts = target_feature.value_counts()
        
        # probabilities of each class label 
        percentages = counts/len(target_feature)

        # calculate parent entropy
        entropy = 0
        for pct in percentages:
            if pct > 0:
                entropy += pct * np.log2(pct)
        return -1*entropy
    
    def entropy(self, X: pd.DataFrame, feature: Mapping, y_col: str) ->float:
        """
        Returns gini index of the give feature
        :param X: data
        :param feature: the feature you want to use to get compute gini score
        :param y_col: name of the label column in X
        :return:
        """

        parent_entropy = self.entropy_calc(X[y_col])
        n_all = len(X)
        
        # categorical
        if feature.name not in self.numerical:
          all_class = X[feature.name].unique()
          infoGain_val = 0
          for cur_class in all_class:
                cur_dataset = X[X[feature.name] == cur_class][y_col]
                n_current = len(cur_dataset)
                infoGain_val = infoGain_val + (n_current/n_all) * self.entropy_calc(cur_dataset)
          infoGain_val = parent_entropy - infoGain_val
          return infoGain_val
        
        # numerical
        else:
            cur_dataset = X[[feature.name,y_col]].sort_values(feature.name)
            unique = list(cur_dataset[feature.name].unique())
            unique = np.percentile(unique, [85,75,50,25,15])
            infoGain_val = 1
            for i in unique:
                lower = cur_dataset[cur_dataset[feature.name] < i][y_col]
                upper = cur_dataset[cur_dataset[feature.name] >= i][y_col]
                lower_entropy = self.entropy_calc(lower)
                upper_entropy = self.entropy_calc(upper)
                tmp_infoGain_val = (len(lower)/len(cur_dataset))*lower_entropy + (len(upper)/len(cur_dataset))*upper_entropy
                if(tmp_infoGain_val <= infoGain_val):
                    infoGain_val = tmp_infoGain_val
                    self.split_value = i
            
            infoGain_val = parent_entropy - infoGain_val
            return infoGain_val
            
    def process_features(self, X: pd.DataFrame, y_col: str)->Sequence[Mapping]:
        """
        :param X: data
        :param y_col: name of the label column in X
        :return:
        """
        features = []
        for n,t in X.dtypes.items():
            if n == y_col:
                continue
            f = {'name': n, 'dtype': t}
            features.append(f)
        return features



