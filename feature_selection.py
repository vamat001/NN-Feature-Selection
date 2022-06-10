from dis import dis
import pandas as pd
import numpy as np
import sys

# df = pd.read_fwf("CS205_SP_2022_SMALLtestdata__28.txt",sep=" ",header=None)
# cols = [4,1,2]
# data = df.loc[:,cols]
# data.columns = range(data.columns.size)
# print(data)

def main():
    # print("Welcome to Vivek Amatya's Feature Selection Algorithm.")
    # file = input("Type in the name of the file to test: ")
    # algorithm = input("\nType the number of the algorithm you want to run.\n\n\t1) Forward Selection\n\t2) Backward Elimination\n")
    feature_search()
    # df = pd.read_fwf("CS205_SP_2022_SMALLtestdata__28.txt",sep=" ",header=None)
    # print(accuracy(df,0,0))

# leave one out cross validation
def accuracy(df,current_set,feature_to_add):

    number_correctly_classfied = 0
    features = current_set
    features.append(feature_to_add)
    data = df.loc[:,features] # get only the features we care about
    data.columns = range(data.columns.size)

    # begin outer loop
    for i in range(len(data)):

        object_to_classify = data.iloc[i][1:]
        label = data.iloc[i][0]

        # initialize nearest neighbor variables
        nn_dist = sys.maxsize
        nn_loc = sys.maxsize

        #begin inner loop
        for k in range(len(data)):

            if k != i: # don't compare to self
                # 1-nearest neighbor using euclidean distance
                dist = np.sqrt(np.sum(np.square(object_to_classify-data.iloc[k][1:])))
                if dist < nn_dist:
                    nn_dist = dist
                    nn_loc = k
                    nn_label = data.iloc[nn_loc][0]
        # end inner loop
        if label == nn_label:
            number_correctly_classfied += 1
    # end outer loop
    return number_correctly_classfied/len(data)

# feature search
def feature_search():

    df = pd.read_fwf("CS205_SP_2022_SMALLtestdata__28.txt",sep=" ",header=None) # read text file into pandas dataframe
    num_features = len(df.iloc[0])-1 # number of features is just the length of the second dimension of dataframe
    current_set_of_features = [] # initialize to empty set

    # outer feature set loop
    for i in range(num_features):
        print("On the " + str(i+1) + "th level of the search tree")
        feature_to_add = [] # feature to add at this level
        best_so_far_accuracy = 0 # keep track of highest accuracy

        # inner feature set loop
        for k in range(num_features):
            # if k in current_set_of_features:
            #     continue
            print("--Considering adding the " + str(k+1) + " feature")
            acc = accuracy(df,current_set_of_features,k+1) # check accuracy using leave one out cross validation
            print(acc)

            if acc > best_so_far_accuracy:
                best_so_far_accuracy = acc
                feature_to_add = k

        current_set_of_features.append(feature_to_add)
        print("On level " + str(i+1) + " i added feature " + str(feature_to_add+1) + " to current set")

        # end inner loop

    # end outer loop
    return

if __name__ == "__main__":
    main()
