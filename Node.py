###########################
#### LOADING LIBRARIES ####
###########################

import pandas as pd
import numpy as np
from collections import Counter
import math

##############################################
#### DEFINING NODE CLASS AS DECISION TREE ####
##############################################

class Node:

    """Details about my desicion tree classifier

    Parameters
    
    ----------------------------------------------------------------

    criterion : {"gini", "entropy"}, default value = "gini"
        This helps define on what statistical concenpt the node is split.

    splitter : {"best", "random"}, default="best"
        The strategy used to choose the split at each node. Supported
        strategies are "best" to choose the best split and "random" to choose
        the best random split.

    max_depth : int, default=None
        Define the maximum number of levels our tree must contain.
        For default value of None, the tree keeps splitting until pure leaf node or when we hit min_sample_leaf value

    min_samples_split : int, default=2
        This is the minimum number of samples present in a node to split it further

    min_samples_leaf : int, default=1
        The minimum number of samples required to make a node.

    max_features : int, {"auto", "sqrt", "log2"}, default=None
        The number of features need to be considered while splitting a node.
        For auto and sqrt we take max features to be sqrt(features).
        For log2 we consider max features to be log2(features).

    random_state : int, default=None
        Controls the randomness of our classifier.

    min_impurity_decrease : float, default=0.0
        A threshold that decides a minimum information gain for every node split.

        """

    # Class variable storing for storing distinct output classes
    CLASS_COUNT = 0

    # Defining class constructor

    def __init__(
        self,
        Y: list,
        X: pd.DataFrame,
        criterion = 'gini',
        max_depth = 5,
        min_samples_split = 2,
        min_samples_leaf = 1,
        max_features = None,
        random_state = None,
        min_impurity_decrease = 0.0,
        depth = 0,
        node_type = 'root',
        rule = ''
    ):

        #Saving data to node
        self.X = X
        self.Y = Y
        self.criterion = criterion
        self.max_depth = int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = max_features
        self.random_state = random_state
        self.min_impurity_decrease = min_impurity_decrease
        

        ####################################################################
        #### DEFINING VARIABLES TO MAINTAIN INTERPRETABILITY OF OUR TREE ###
        ####################################################################

        # self.depth is counter to keep a track of the tree's depth
        self.depth = depth
        # This variable holds the number of observations in each 
        self.class_counts = Counter(Y)
        # Storing list of features available for the node.
        self.features = list(self.X.columns)
        # This holds the information if it's a right node or left node of the split
        self.node_type = node_type
        # This variable contains the details regarding which feature and corresponding value was the node split upon.
        self.rule = rule
        
        # Setting the count of each output label in the class variable Unique Class.
        # This count is stored in every Node object.
        if self.node_type == 'root':
            Node.CLASS_COUNT = len(self.class_counts)
        self.CLASS_COUNT = Node.CLASS_COUNT

        
        self.node_criterion_value = self.get_node_criterion_value()

        # Sorting the class_count variable and fetching the label with highest variable
        # This would be the prediction value of the node.
        counts_sorted = list(sorted(self.class_counts.items(), key=lambda item: item[1]))

        #Getting last item
        # yhat = None

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

    ########################
    #### CALCULATE GINI ####
    ########################

    def get_node_criterion_value(self):

        #SToring count of each class in a list
        class_counts = [ self.class_counts.get(i) if self.class_counts.get(i) is not None else 0 for i in range(self.CLASS_COUNT) ]

        # Calculating criterion based on argument
        if self.criterion == 'entropy':
            criterion_value = self.get_entropy(class_counts)
        else:
            criterion_value =  self.get_gini_impurity(class_counts)

        #get GINI impurity
        return criterion_value
        

    #################################
    #### CALCULATE GINI IMPURITY ####
    #################################

    # Defining a static method for the class, to find the gini impurity given the number of observations
    # Arg class_counts contains number of observations for each class
    @staticmethod
    def get_gini_impurity(class_counts) -> float:

        n = sum(class_counts)

        #for n=0, return the lowest gini impurity
        if n == 0:
            return 0.0

        # Calculating probability of each class
        probab_class = [ count/n for count in class_counts ]
        #FInding gini value
        gini_value = sum(i**2 for i in probab_class)
        #Calculating gini impurity
        gini = 1 - gini_value

        return gini

    ###########################
    #### CALCULATE ENTROPY ####
    ###########################

    # Defining a static method for the class, to find the entropy given the number of observations
    # Arg class_counts contains number of observations for each class
    @staticmethod
    def get_entropy(class_counts) -> float:

        n = sum(class_counts)

        #for n=0, return the lowest gini impurity
        if n == 0:
            return 0.0

        # Calculating probability of each class
        probab_class = [ -(count/n) for count in class_counts ]
        #multiplying probab with log probab
        entropy = sum( [( i * math.log(i,2) ) for i in probab_class] )

        return entropy


    ##############################################
    #### CALCULATE MOVING AVERAGE OF FEATURES ####
    ##############################################

    @staticmethod
    def get_moving_average(x:np.array , window : int) -> np.array:
        return np.convolve(x, np.ones(window), 'valid') / window

    ##########################################
    #### CALCULATE BEST SPLIT OF FEATURES ####
    ##########################################

    def best_split(self) -> tuple:

        df = self.X.copy()
        df['Y'] = self.Y

        # Setting node criterion value to calculate criterion gain later on
        base_criterion_value = self.node_criterion_value

        # Setting a threshold for the information gain at every split
        max_gain = self.min_impurity_decrease

        #Best feature and score
        best_feature = None
        best_value = None


        for feature in self.features:
            #Drop missing values
            Xdf = df.dropna().sort_values(feature)

            #Sort values and get rolling average
            xmeans = self.get_moving_average(Xdf[feature].unique(), 2)

            for value in xmeans:
                #Split the dataset
                left_counts = Counter(Xdf[Xdf[feature]<value]['Y'])
                right_counts = Counter(Xdf[Xdf[feature]>=value]['Y'])

                #Calculate various class counts in both the nodes.
                y_left_class_count = [ left_counts.get(i) if left_counts.get(i) is not None else 0 for i in  range(self.CLASS_COUNT) ]
                y_right_class_count = [ right_counts.get(i) if right_counts.get(i) is not None else 0 for i in  range(self.CLASS_COUNT) ]
                
                #getting left and right gini impurity
                gini_left = self.get_gini_impurity(y_left_class_count)
                gini_right = self.get_gini_impurity(y_right_class_count)

                #Getting obs count from the left and right nodes
                n_left = sum(y_left_class_count)
                n_right = sum(y_right_class_count)


                #Calculating weights of each node
                w_left = n_left / (n_left + n_right)
                w_right = n_right / (n_left + n_right)

                #Calculate weighted gini impurity
                wGINI = w_left * gini_left + w_right * gini_right

                #Calculating gini gain
                GINIgain = base_criterion_value - wGINI

                #Is this the best split
                if GINIgain > max_gain:
                    best_feature = feature
                    best_value = value
                    max_gain = GINIgain
            
        return (best_feature, best_value)

    ##############################
    #### CREATE DECISION TREE ####
    ##############################

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

    ################################
    #### PRINT TREE INFORMATION ####
    ################################

    def print_info(self, width=4):

        # Defining the number of spaces 
        const = int(self.depth * width ** 1.5)
        spaces = "-" * const
        
        if self.node_type == 'root':
            print("Root")
        else:
            print(f"|{spaces} Split rule: {self.rule}")
        print(f"{' ' * const}   | GINI impurity of the node: {round(self.get_gini_impurity, 2)}")
        print(f"{' ' * const}   | Class distribution in the node: {dict(self.class_counts)}")
        print(f"{' ' * const}   | Predicted class: {self.yhat}") 

    ################################
    #### PRINT TREE INFORMATION ####
    ################################

    def print_tree(self):

        self.print_info() 
        
        if self.left is not None: 
            self.left.print_tree()
        
        if self.right is not None:
            self.right.print_tree()

    ##########################
    #### PREDICT FUNCTION ####
    ##########################

    def predict(self, X:pd.DataFrame):
    
        predictions = []

        for _, x in X.iterrows():
            values = {}
            for feature in self.features:
                values.update({feature: x[feature]})
        
            predictions.append(self.predict_obs(values))
        
        return predictions

    ##########################
    #### PREDICT FUNCTION ####
    ##########################

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