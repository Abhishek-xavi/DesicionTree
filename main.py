###########################
#### LOADING LIBRARIES ####
###########################

from Node import Node
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics


#######################
#### MAIN FUNCTION ####
#######################

def main():
        
    ###########################
    #### LOADING DATAFRAME ####
    ###########################

    data = pd.read_table("/Users/AbhishekGupta/Documents/Lancaster/Programming/dataSet/iris/iris.data",sep=",",names=["SepalLength", "SepalWidth", "PetalLength", "PetalWidth", "Class"], header = None)
    
    #################################
    #### HANDLING MISSING VALUES ####
    #################################

    # Dropping missing values
    dtree = data[["SepalLength", "SepalWidth", "PetalLength", "PetalWidth", "Class"]].dropna().copy()
    
    if not isinstance(dtree["Class"], (int, float)):
        dtree.Class, output_label_dict = pd.Series(dtree.Class).factorize()

    # Defining the X and Y matrices
    Y = dtree["Class"].values
    X = dtree[["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]]

    ##################################
    #### SPLITTING TEST TRAIN DATA####
    ##################################

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state = 1)
    
    # # Saving the feature list 
    # features = list(X.columns)

    #############################################
    #### CALLING SKLEARN DESICION TREE CLASS ####
    #############################################

    clf = DecisionTreeClassifier()
    clf = clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)
    print('Accuracy of sklearn tree : ', metrics.accuracy_score(Y_test, Y_pred))


    #########################################
    #### CALLING OUR DESICION TREE CLASS ####
    #########################################

    hp = {
     'max_depth': 3,
     'min_samples_split': 50
    }

    root = Node(Y_train, X_train, **hp)
    root.grow_tree()
    # root.print_tree()
    Y_pred = root.predict(X_test)
    print('Accuracy of my Tree : ', metrics.accuracy_score(Y_test, Y_pred))

    
if __name__ == '__main__':
    main()

