import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import random
import math  

#function to initialise the centroids
def initialise_centroids():
    #0 is x, 1 is y, 2 is count of connected points, 3 is non permanant x total, 4 is non permanant y total, 5 is non permanant euclidian distance
    centroids = np.zeros((Ccount, 6))
    for loop in range(0,Ccount):
        rand = random.randint(0, (np.prod(x.shape) - 1))
        centroids[loop][0] = x[rand]
        centroids[loop][1] = y[rand]
    return(centroids)

#computes the distance between 2 points
def compute_distance(x1, x2, y1, y2):
    response = math.sqrt(((x1 - x2) **2) + ((y1 - y2) **2))
    return(response)

#reads the data from a csv
data= pd.read_csv('task2.csv').values
x = data[:,0]
y = data[:,1]
Ccount = 5
itterations = 10
objective_function = np.zeros((Ccount, itterations))

centroids = initialise_centroids()


#iterations
for loop in range(itterations):
    #goes through all data
    for val in range(np.prod(x.shape) ):

        #calculates euclidian distance
        for val2 in range(0, Ccount):
            centroids[val2][5] = compute_distance(centroids[val2][0], x[val], centroids[val2][1], y[val])
        
        lowest_position = np.argmin(centroids[:,5])

        centroids[lowest_position][2] += 1
        centroids[lowest_position][3] += x[val]
        centroids[lowest_position][4] += y[val]

    for val2 in range(0, Ccount):
        objective_function[val2][loop] = compute_distance(centroids[val2][0], (centroids[val2][3] / centroids[val2][2]), centroids[val2][1], (centroids[val2][4] / centroids[val2][2]))
        centroids[val2][0] = (centroids[val2][3] / centroids[val2][2])
        centroids[val2][1] = (centroids[val2][4] / centroids[val2][2])
        centroids[val2][2] = 0
        centroids[val2][3] = 0
        centroids[val2][4] = 0
        centroids[val2][5] = 0

for val in range(np.prod(x.shape) ):

    for val2 in range(0, Ccount):
        centroids[val2][5] = math.sqrt(((centroids[val2][0] - x[val]) **2) + ((centroids[val2][1] - y[val]) **2))

    lowest_position = np.argmin(centroids[:,5])
    #plots data points
    if (lowest_position == 0):
        plt.scatter(x[val], y[val], color='b', marker='o', )
    if (lowest_position == 1):
        plt.scatter(x[val], y[val], color='g', marker='o', )
    if (lowest_position == 2):
        plt.scatter(x[val], y[val], color='m', marker='o', )
    if (lowest_position == 3):
        plt.scatter(x[val], y[val], color='c', marker='o', )
    if (lowest_position == 4):
        plt.scatter(x[val], y[val], color='y', marker='o', )

#plots centroids
for val in range(0, Ccount):
    if (val == 0):
        colors, label = 'cornflowerblue', 'cluster 1'
    if (val == 1):
        colors, label = 'lime', 'cluster 2'
    if (val == 2):
        colors, label = 'violet', 'cluster 3'
    if (val == 3):
        colors, label = 'aqua', 'cluster 4'
    if (val == 4):
        colors, label = 'orange', 'cluster 5'
    plt.scatter(centroids[val][0],centroids[val][1], color=colors, marker='x', label = label)


plt.legend()
plt.xlabel('height')
plt.ylabel('leg lenth')
plt.show()



plt.clf()
for val in range(0, Ccount):
    if (val == 0):
        colors, label = 'cornflowerblue', 'cluster 1'
    if (val == 1):
        colors, label = 'lime', 'cluster 2'
    if (val == 2):
        colors, label = 'violet', 'cluster 3'
    if (val == 3):
        colors, label = 'aqua', 'cluster 4'
    if (val == 4):
        colors, label = 'orange', 'cluster 5'
    plt.plot(range(1, itterations +1), objective_function[val], '-r', color=colors, label = label)

plt.legend()
plt.xlabel('iterration number')
plt.ylabel('objective function')
plt.show()