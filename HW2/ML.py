import pandas as pd
import networkx as nx
from sklearn.ensemble import RandomForestClassifier


def main():

    ########################### your code goes here #########################
    # input: undirected graph readed from edges.txt
    # output: DataFrame including one column named 'node' 
    # for node numbers , and other column(s) including node
    # feature(s)



    ###########################################################################

    nodes_label = pd.read_csv('./soc-nodes.txt')
    # adding data labels to data
    data = pd.merge(data,nodes_label, on = ['node'])

    # splitting data to training and test sets
    train = data[data['partition']=='train']
    test = data[data['partition']=='test']

    # prepare data for model
    X_train = train.drop(['node', 'class', 'partition'], axis = 1)
    X_test = test.drop(['node', 'class', 'partition'], axis = 1)
    y_train = train['class']
    y_test = test['class']

    # training the model
    model = RandomForestClassifier(max_depth=2, random_state=0)
    model.fit(X_train, y_train)

    # make prediction
    test_prediction = model.predict(X_test)

    # calculate the accuracy
    true_predicted = 0
    for i in range(len(test_prediction)):
        if test_prediction[i] == list(y_test)[i]:
            true_predicted +=1

    numberOfTestNodes = len(y_test)

    print('Accuracy: ', true_predicted/numberOfTestNodes)

if __name__ == '__main__':
    
    main()