import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import math  
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestRegressor

#stops scientific notation
np.set_printoptions(suppress=True)
#reads the data from the CSV
data= pd.read_csv('MLdata.csv')
#shuffles the data
data = shuffle(data)

#seperates into input and output for model
data_input = data.iloc[:, 1:13]
data_output = data.select_dtypes(include=[object])

#changes output into ones and zeros for the models ease
preproc = preprocessing.LabelEncoder()
data_output = data_output.apply(preproc.fit_transform)

#further splits the data into training and test data
trainx, testx, trainy, testy = train_test_split(data_input, data_output, test_size = 0.10)

#sets up array for models leaf/ node count, and array for output of the models
modelnodes = [50, 100, 150, 200, 300, 400, 500, 600, 800, 1000]
modeloutputs = np.zeros((2, 10))


for values in modelnodes:

    #NN model


    #create and train model
    NNmodel = MLPClassifier(hidden_layer_sizes=(values, values), max_iter=500)
    NNmodel.fit(trainx, trainy.values.ravel())

    #get predictions
    NNpredictions = NNmodel.predict(testx)

    #print report/save value
    #print(classification_report(testy,NNpredictions))
    modeloutputs[0, modelnodes.index(values)] = accuracy_score(testy, NNpredictions)


    #random forests model


    #create and train model
    RFmodel = RandomForestRegressor(n_estimators = values, min_samples_leaf = 5)
    RFmodel.fit(trainx, trainy.values.ravel())

    #get predictions
    RFpredictions = RFmodel.predict(testx)

    #print model accuracy/save value
    #print(f'Model Accuracy: {RFmodel.score(trainx, trainy.values.ravel())}')
    modeloutputs[1, modelnodes.index(values)] = RFmodel.score(trainx, trainy.values.ravel())

    print('itteration ' , (modelnodes.index(values) + 1), ' of 10 complete')

print(modeloutputs)

plt.clf()
plt.plot(modelnodes, modeloutputs[0, :], '-r', label = 'NN model')
plt.plot(modelnodes, modeloutputs[1, :], '-b', label = 'RF model')
plt.legend()
plt.xlabel('iterrations')
plt.ylabel('accuracy')
plt.show()