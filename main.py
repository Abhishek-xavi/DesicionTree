from Node import Node
import pandas as pd


def main():
    # Loading data
    d = pd.read_csv("/Users/AbhishekGupta/Documents/Lancaster/Programming/dataSet/titanic/train.csv")
    # Dropping missing values
    dtree = d[['Survived', 'Age', 'Fare']].dropna().copy()
    # Defining the X and Y matrices
    Y = dtree['Survived'].values
    X = dtree[['Age', 'Fare']]
    # Saving the feature list 
    features = list(X.columns)

    hp = {
     'max_depth': 3,
     'min_samples_split': 50
    }

    root = Node(Y, X, **hp)

    root.grow_tree()

    root.print_tree()

    print('Tree is constructed')
    
if __name__ == '__main__':
    main()

