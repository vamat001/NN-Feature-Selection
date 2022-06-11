from dis import dis
from operator import indexOf
import pandas as pd
import numpy as np
import sys
import copy
import multiprocessing as mp
from multiprocessing import Pool
from itertools import product
from functools import partial
import time

def main():
    # print("Welcome to Vivek Amatya's Feature Selection Algorithm.")
    # file = input("Type in the name of the file to test: ")
    # algorithm = input("\nType the number of the algorithm you want to run.\n\n\t1) Forward Selection\n\t2) Backward Elimination\n")
    df = pd.read_fwf("CS205_SP_2022_SMALLtestdata__28.txt",sep=" ",header=None)
    forward_selection(df)

# leave one out cross validation
def accuracy(df,current_set,feature_to_add):

    number_correctly_classfied = 0
    features = copy.deepcopy(current_set)
    features.append(feature_to_add)
    features.insert(0,0) # we always need the 0th column for classes
    data = df.loc[:,features] # get only the features we care about
    data.columns = range(data.columns.size)
    # print(data)

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
def forward_selection():

    df = pd.read_fwf("CS205_SP_2022_Largetestdata__27.txt",sep=" ",header=None) # read text file into pandas dataframe
    num_features = len(df.iloc[0])-1 # number of features is just the length of the second dimension of dataframe
    current_set_of_features = [] # initialize to empty set
    best_set_of_features = [[0],0] # the set of features that gives highest accuracy

    # outer feature set loop
    for i in range(1,num_features+1):
        print("On the " + str(i) + "th level of the search tree")
        feature_to_add = None # feature to add at this level
        best_so_far_accuracy = 0 # keep track of highest accuracy

        args = []
        for x in range(1,num_features+1):
            if x not in current_set_of_features:
                args.append(x)
        with Pool(processes=mp.cpu_count()) as p:
            func = partial(accuracy, df, current_set_of_features)
            results = p.map(func,args)
        best_so_far_accuracy = max(results)
        feature_to_add = args[results.index(best_so_far_accuracy)]
        # inner feature set loop
        # for k in range(1,num_features+1):
        #     if k in current_set_of_features:
        #         continue
        #     print("--Considering adding the " + str(k) + " feature")
        #     pool = mp.Pool(processes=10)
            
        #     acc = accuracy(df,current_set_of_features,k) # check accuracy using leave one out cross validation
        #     # print(acc)

        #     if acc > best_so_far_accuracy:
        #         best_so_far_accuracy = acc
        #         feature_to_add = k
        # # end inner loop

        current_set_of_features.append(feature_to_add)
        if best_so_far_accuracy > best_set_of_features[1]:
            best_set_of_features[0] = copy.deepcopy(current_set_of_features)
            best_set_of_features[1] = best_so_far_accuracy
        print("On level " + str(i) + " i added feature " + str(feature_to_add) + " to current set with accuracy " + str(best_so_far_accuracy))
        print("Best set of features so far: ", best_set_of_features[0])
        print("With accuracy: ",best_set_of_features[1])

    # end outer loop

    print("Search finished! Best set of features: ", best_set_of_features[0])
    print("With accuracy: ",best_set_of_features[1])

    return

def backward_elimination():

    df = pd.read_fwf("CS205_SP_2022_Largetestdata__27.txt",sep=" ",header=None) # read text file into pandas dataframe
    num_features = len(df.iloc[0])-1 # number of features is just the length of the second dimension of dataframe
    current_set_of_features = [] # initialize to empty set
    best_set_of_features = [[0],0] # the set of features that gives highest accuracy

    # outer feature set loop
    for i in range(1,num_features+1):
        print("On the " + str(i) + "th level of the search tree")
        feature_to_add = None # feature to add at this level
        best_so_far_accuracy = 0 # keep track of highest accuracy

        args = []
        for x in range(1,num_features+1):
            if x not in current_set_of_features:
                args.append(x)
        with Pool(processes=mp.cpu_count()) as p:
            func = partial(accuracy, df, current_set_of_features)
            results = p.map(func,args)
        best_so_far_accuracy = max(results)
        feature_to_add = args[results.index(best_so_far_accuracy)]
        # inner feature set loop
        # for k in range(1,num_features+1):
        #     if k in current_set_of_features:
        #         continue
        #     print("--Considering adding the " + str(k) + " feature")
        #     pool = mp.Pool(processes=10)
            
        #     acc = accuracy(df,current_set_of_features,k) # check accuracy using leave one out cross validation
        #     # print(acc)

        #     if acc > best_so_far_accuracy:
        #         best_so_far_accuracy = acc
        #         feature_to_add = k
        # # end inner loop

        current_set_of_features.append(feature_to_add)
        if best_so_far_accuracy > best_set_of_features[1]:
            best_set_of_features[0] = copy.deepcopy(current_set_of_features)
            best_set_of_features[1] = best_so_far_accuracy
        print("On level " + str(i) + " i added feature " + str(feature_to_add) + " to current set with accuracy " + str(best_so_far_accuracy))
        print("Best set of features so far: ", best_set_of_features[0])
        print("With accuracy: ",best_set_of_features[1])

    # end outer loop

    print("Search finished! Best set of features: ", best_set_of_features[0])
    print("With accuracy: ",best_set_of_features[1])

    return

if __name__ == "__main__":
    start_time = time.time()
    mp.set_start_method('fork')
    main()
    runtime = time.time()-start_time
    if runtime < 500:
        print("--- %s seconds ---" % runtime)
    elif runtime < 1200:
        runtime /= 60
        print("--- %s minutes ---" % runtime)
    else:
        runtime /= 360
        print("--- %s hours ---" % runtime)
