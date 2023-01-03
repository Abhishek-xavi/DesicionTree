#Loading libraries

import pandas as pd
import numpy as np
from collections import Counter

# Defining Node class

class Node:

    # Defining class variable, to store the unique output labels.
    UNIQUE_CLASS = 0

    # Defining class constructor
    def __init__(
        self,
        Y: list,
        X: pd.DataFrame,
        min_samples_split = 20,
        max_depth = 5,
        depth = 0,
        node_type = 'root',
        rule = ''
    ):

        #Saving data to node
        self.X = X
        self.Y = Y
        
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        
        #Depth of node
        self.depth = depth
        #Creating a list of features
        self.features = list(self.X.columns)
        #Type of node
        self.node_type = node_type

        #Rule for splitting
        self.rule = rule

        #Calculating count of Y in node
        self.counts = Counter(Y)

        if self.node_type == 'root':
            Node.UNIQUE_CLASS = len(self.counts)
        self.UNIQUE_CLASS = Node.UNIQUE_CLASS

        #Getting gini distribution based on distribution of Y
        self.gini_impurity = self.get_GINI()

        #Sorting counts and saving final prediction of the node
        counts_sorted = list(sorted(self.counts.items(), key=lambda item: item[1]))

        #Getting last item
        yhat = None

        if len(counts_sorted) > 0:
            yhat = counts_sorted[-1][0]
        
        #Node predicts the class with the most frequent class
        self.yhat = yhat

        #Saving number of observations in node
        self.n = len(Y)

        #Initialising left and right nodes
        self.left = None
        self.right = None

        #Default split values
        self.best_feature = None
        self.best_value = None


    #Defining a static method for the class, to find the gini impurity given the number of observations
    #Arg class_counts contains number of observations for each class
    @staticmethod
    def GINI_impurity(class_counts) -> float:

        n = sum(class_counts)

        #for n=0, return the lowest gini impurity
        if n == 0:
            return 0.0

        # Probability of each class
        # p1 = y1_count / n
        # p2 = y2_count / n

        probab_class = [ count/n for count in class_counts ]
        gini_value = sum(i**2 for i in probab_class)
        
        # Calculate Gini
        # gini = 1 - (p1**2 + p2**2)
        gini = 1 - gini_value

        return gini

    #Calculate gini impurity of a node
    def get_GINI(self):

        #Getting count of all output classes
        # y1_count, y2_count = self.counts.get(0), self.counts.get(1)

        class_counts = [ self.counts.get(i) if self.counts.get(i) is not None else 0 for i in range(self.UNIQUE_CLASS) ]
        

        # for i in  range(len(self.counts)):
        #     class_counts.append(self.counts.get(i))

        #get GINI impurity
        return self.GINI_impurity(class_counts)

    #Define method to calculate moving average
    @staticmethod
    def ma(x:np.array , window : int) -> np.array:
        return np.convolve(x, np.ones(window), 'valid') / window

    #Finding best split for the node
    def best_split(self) -> tuple:

        df = self.X.copy()
        df['Y'] = self.Y

        #Getting gini impurity for the base input
        GINI_base = self.get_GINI()

        #Finding best split with maximum gini gain
        max_gain = 0

        #Best feature and score
        best_feature = None
        best_value = None


        for feature in self.features:
            #Drop missing values
            Xdf = df.dropna().sort_values(feature)

            #Sort values and get rolling average
            xmeans = self.ma(Xdf[feature].unique(), 2)

            for value in xmeans:
                #Split the dataset
                left_counts = Counter(Xdf[Xdf[feature]<value]['Y'])
                right_counts = Counter(Xdf[Xdf[feature]>=value]['Y'])

                # Getting Y distribution
                # y0_left, y1_left, y0_right, y1_right = left_counts.get(0), left_counts.get(1), \
                #                                         right_counts.get(0), right_counts.get(1)

                y_left_class_count = [ left_counts.get(i) if left_counts.get(i) is not None else 0 for i in  range(self.UNIQUE_CLASS) ]
                y_right_class_count = [ right_counts.get(i) if right_counts.get(i) is not None else 0 for i in  range(self.UNIQUE_CLASS) ]
                
                #getting left and right gini impurity
                gini_left = self.GINI_impurity(y_left_class_count)
                gini_right = self.GINI_impurity(y_right_class_count)

                #Getting obs count from the left and right nodes
                n_left = sum(y_left_class_count)
                n_right = sum(y_right_class_count)


                #Calculating weights of each node
                w_left = n_left / (n_left + n_right)
                w_right = n_right / (n_left + n_right)

                #Calculate weighted gini impurity
                wGINI = w_left * gini_left + w_right * gini_right

                #Calculating gini gain
                GINIgain = GINI_base - wGINI

                #Is this the best split
                if GINIgain > max_gain:
                    best_feature = feature
                    best_value = value
                    max_gain = GINIgain
            
        return (best_feature, best_value)

    #Creating desicion tree
    def grow_tree(self):

        df = self.X.copy()
        df['Y'] = self.Y

        #Splitting the tree further based on hyper parameter constraint
        if (self.depth < self.max_depth) and (self.n >= self.min_samples_split):

            # Getting the best split 
            best_feature, best_value = self.best_split()

            if best_feature is not None:

                self.best_feature = best_feature
                self.best_value = best_value

                # Getting the left and right nodes
                left_df, right_df = df[df[best_feature]<=best_value].copy(), df[df[best_feature]>best_value].copy()

                # Creating the left and right nodes
                left = Node(
                    left_df['Y'].values.tolist(), 
                    left_df[self.features], 
                    depth=self.depth + 1, 
                    max_depth=self.max_depth, 
                    min_samples_split=self.min_samples_split, 
                    node_type='left_node',
                    rule=f"{best_feature} <= {round(best_value, 3)}"
                    )

                self.left = left 
                self.left.grow_tree()

                right = Node(
                    right_df['Y'].values.tolist(), 
                    right_df[self.features], 
                    depth=self.depth + 1, 
                    max_depth=self.max_depth, 
                    min_samples_split=self.min_samples_split,
                    node_type='right_node',
                    rule=f"{best_feature} > {round(best_value, 3)}"
                    )

                self.right = right
                self.right.grow_tree()

    # Printing information about the tree
    def print_info(self, width=4):

        # Defining the number of spaces 
        const = int(self.depth * width ** 1.5)
        spaces = "-" * const
        
        if self.node_type == 'root':
            print("Root")
        else:
            print(f"|{spaces} Split rule: {self.rule}")
        print(f"{' ' * const}   | GINI impurity of the node: {round(self.gini_impurity, 2)}")
        print(f"{' ' * const}   | Class distribution in the node: {dict(self.counts)}")
        print(f"{' ' * const}   | Predicted class: {self.yhat}") 

    #Printing the whole tree
    def print_tree(self):

        self.print_info() 
        
        if self.left is not None: 
            self.left.print_tree()
        
        if self.right is not None:
            self.right.print_tree()

    #Predicting the dataset
    def predict(self, X:pd.DataFrame):
    
        predictions = []

        for _, x in X.iterrows():
            values = {}
            for feature in self.features:
                values.update({feature: x[feature]})
        
            predictions.append(self.predict_obs(values))
        
        return predictions

    #Predicting the class
    def predict_obs(self, values: dict) -> int:
        
        cur_node = self
        while cur_node.depth < cur_node.max_depth:
            # Traversing the nodes all the way to the bottom
            best_feature = cur_node.best_feature
            best_value = cur_node.best_value

            if cur_node.n < cur_node.min_samples_split:
                break 

            if (values.get(best_feature) < best_value):
                if self.left is not None:
                    cur_node = cur_node.left
            else:
                if self.right is not None:
                    cur_node = cur_node.right
            
        return cur_node.yhat