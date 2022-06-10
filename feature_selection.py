import pandas as pd
import numpy as np
import sys

def main():
    # print("Welcome to Vivek Amatya's Feature Selection Algorithm.")
    # file = input("Type in the name of the file to test: ")
    # algorithm = input("\nType the number of the algorithm you want to run.\n\n\t1) Forward Selection\n\t2) Backward Elimination\n")
    # feature_search()
    df = pd.read_fwf("CS205_SP_2022_SMALLtestdata__28.txt",sep=" ",header=None)
    accuracy(df)

# leave one out cross validation
def accuracy(df):
    for i in range(len(df)):
        object_to_classify = df.iloc[i][1:]
        label = df.iloc[i][0]

        # initialize nearest neighbor variables
        nn_dist = sys.maxsize
        nn_loc = sys.maxsize

        for k in range(len(df)):

            if k != i: # don't compare to self
                print('Ask if ' + str(i+1) + " is nearest neighbor with " + str(k+1))
                # 1-nearest neighbor using euclidean distance
                dist = np.sqrt(np.sum(np.square(object_to_classify-df.iloc[k][1:])))

# feature search
def feature_search():

    df = pd.read_fwf("CS205_SP_2022_SMALLtestdata__28.txt",sep=" ",header=None) # read text file into pandas dataframe
    num_features = len(df.iloc[0])-1 # number of features is just the length of the second dimension of dataframe
    current_set_of_features = [] # initialize to empty set

    # outer feature set loop
    for i in range(1,num_features+1):
        print("On the " + str(i) + "th level of the search tree")
        feature_to_add = None # feature to add at this level
        best_so_far_accuracy = 0 # keep track of highest accuracy

        # inner feature set loop
        for k in range(1,num_features+1):
            if k in current_set_of_features:
                continue
            print("--Considering adding the " + str(k) + " feature")
            acc = accuracy(df) # check accuracy using leave one out cross validation

            if acc > best_so_far_accuracy:
                best_so_far_accuracy = acc
                feature_to_add = k

        current_set_of_features.append(feature_to_add)
        print("On level " + str(i) + " i added feature " + str(feature_to_add) + " to current set")

        # end inner loop

    # end outer loop
    return

# if __name__ == "__main__":
#     main()
