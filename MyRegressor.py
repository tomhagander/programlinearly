from sklearn.metrics import mean_absolute_error
from scipy.optimize import linprog
import numpy as np
import random
import time
import matplotlib.pyplot as plt

class MyRegressor:
    def __init__(self, alpha):
        self.weight = None
        self.bias = None
        self.training_cost = 0   # N * M
        self.alpha = alpha

    def select_features_random(self, trainX, trainY, k):
        N, M = trainX.shape
        if k==M:
            return np.linspace(0,M-1,num=M, dtype=np.int64)
        return np.random.choice(np.linspace(0,M-1,num=M, dtype=np.int64), k)

    def select_sample_random(self, trainX, trainY, k, timeout):
        N, M = trainX.shape
        indices = np.random.choice(np.linspace(0,N-1, num=N, dtype=np.int64), k)
        return trainX[indices, :], trainY[indices], False

        
    def select_features(self, trainX, trainY, k):
    
        N, M = trainX.shape
        if (k == M):
            return np.linspace(0,M-1,num=M, dtype=np.int64) # all samples

        print(N)
        print(M)
        angles = np.zeros(M)
        for i in range(M):
            a_i = trainX[:,i]
            angles[i] = (abs(np.dot(trainY, a_i)/(np.linalg.norm(trainY)*np.linalg.norm(a_i))))
        
        # taking the k highest values
        indices = np.argpartition(angles, -k)[-k:]

        return indices # selected_feat # The index List of selected features
        
        
    def select_sample(self, trainX, trainY, k, timeout):
        t0 = time.time()
        terminated_on_timeout = False

        N, M = trainX.shape

        if (k == N):
            return trainX, trainY, False

        class Cluster:
            def __init__(self, point):
                self.centroid = point
                self.points = []

            def distance_to(self, point):
                # l2 norm
                sum = 0
                for idx in range(self.centroid.size):
                    sum += (self.centroid[idx] - point[idx])**2
                return sum**0.5
            
            def update_centroid(self):
                # check if empty
                if len(self.points) == 0:
                    raise Exception('No points in cluster, should never happen')
                sum = 0
                for point in self.points:
                    sum += point
                self.previous_centroid = self.centroid
                self.centroid = sum/len(self.points)

            def clear_points(self):
                self.points = []

            def check_if_changed(self):
                # should be called right after assignment step
                return not np.array_equal(self.centroid, self.previous_centroid)

            def find_closest(self, points):
                closest = None
                min_dist = float('inf')
                for point in points:
                    dist = self.distance_to(point)
                    if dist < min_dist:
                        closest = point
                        min_dist = dist
                return closest

        clusters = []
        points = []

        # initialization
        randos = np.random.choice(N-1, k)
        for i in range(k):
            idx = randos[i]
            pnt = np.append(trainX[idx], trainY[idx])
            points.append(pnt)
            clusters.append(Cluster(pnt)) # create cluster with pnt as centroid

        terminal = False
        while not terminal:
            # Assignment
            print('assigning')
            for point in points:
                min_dist = float('inf')
                closest_cluster = None
                for cluster in clusters:
                    dist_to_centroid = cluster.distance_to(point)
                    if dist_to_centroid < min_dist:
                        closest_cluster = cluster
                        min_dist = dist_to_centroid
                closest_cluster.points.append(point)

            # Update
            print('updating')
            for cluster in clusters:
                if len(cluster.points) == 0: # empty cluster, needs a new data point so that it can define a centroid
                    # adding point
                    found_cluster_to_steal_from = False
                    cluster_to_steal_from = None
                    while not found_cluster_to_steal_from:
                        other_cluster = random.choice(clusters)
                        if len(other_cluster.points) > 1: #dont want that cluster to become empty
                            found_cluster_to_steal_from = True
                            cluster_to_steal_from = other_cluster
                    cluster.points.append(cluster_to_steal_from.points.pop())
                cluster.update_centroid()

            # clear old points
            for cluster in clusters:
                cluster.clear_points()

            # check timeout
            if time.time() - t0 >= timeout:
                terminal = True
                terminated_on_timeout = True

            # check terminal
            found_change = False
            for cluster in clusters:
                if cluster.check_if_changed():
                    found_change = True
                    break
            if not found_change:
                terminal = True
            
        # comes here when terminal
        print('terminal')
        selected_trainX = np.zeros((k, M))
        selected_trainY = np.zeros((k, ))
        for idx, cluster in enumerate(clusters):
            closest_point = cluster.find_closest(points)
            selected_trainX[idx] = closest_point[:-1]
            selected_trainY[idx] = closest_point[-1]
        
        return selected_trainX, selected_trainY, terminated_on_timeout    # A subset of trainX and trainY


    def select_data(self, trainX, trainY, N_perc, M_perc): #sample percentage determined from the other two.
        
        N,M = trainX.shape
        if (N_perc*M_perc == 1):
            return trainX, trainY, np.linspace(0,M-1,num=M, dtype=np.int64) # all samples

        N_use = round(N*N_perc)
        M_use = round(M*M_perc)
        print('M use N use')
        print(M_use)
        print(N_use)

        indices = self.select_features(trainX, trainY, M_use)
        selectX1 = trainX[:, indices]

        selectX, selectY, timeout = self.select_sample(selectX1, trainY, N_use, timeout=100)
        
        return selectX, selectY, indices
    
    
    def train(self, trainX, trainY): 

        # create matrices, derivation on paper
        N, M = trainX.shape
        trainYm = trainY.reshape(-1, 1)
        A = np.block([[-np.eye(N), np.zeros((N, M)), -trainX, -np.ones((N,1))],
            [-np.eye(N), np.zeros((N, M)), trainX, np.ones((N,1))],
            [np.zeros((M, N)), -np.eye(M), np.eye(M), np.zeros((M,1))],
            [np.zeros((M, N)), -np.eye(M), -np.eye(M), np.zeros((M,1))]])

        c = np.block([np.ones((1,N))/N, self.alpha*np.ones((1, M)), np.zeros((1,M+1))])

        b = np.block([[-trainYm],
        [trainYm],
        [np.zeros((M,1))],
        [np.zeros((M,1))]])

        # solve LP
        res = linprog(c, A_ub=A, b_ub=b, method='highs')
        # extract relevant solutions
        self.weight = res.x[-(M+1): -1]
        self.bias = res.x[-1]

        # return predY and error
        return self.evaluate(trainX, trainY)
    
    def train_online(self, trainX, trainY, cutoff):
        N, M = trainX.shape
        # initial training (set all weights and biases to zero)
        self.weight = np.zeros(M)
        self.bias = 0
        previous_samples = []
        central_node_X = None
        central_node_Y = None
        self.training_cost = 0

        # we simulate the online setting by handling training data samples one by one
        for index, x in enumerate(trainX):
            y = trainY[index]

            # always take first datapoint
            if index == 0:
                central_node_X = np.array([x])
                central_node_Y = np.array([y])
                continue
            
            # calculate distance to each to the contral node previously known sample
            diffs = []
            for sample in previous_samples:
                diffs.append(np.linalg.norm(x - sample))

            # find lowest distance
            min_distance = float('inf')
            for diff in diffs:
                if diff < min_distance:
                    min_distance = diff
            if min_distance >= cutoff:
                # add sample
                previous_samples.append(x)
                self.training_cost += M
                central_node_X = np.concatenate((central_node_X, np.array([x])))
                central_node_Y = np.concatenate((central_node_Y, np.array([y])))

        predY, error = self.train(central_node_X, central_node_Y)
            
        return self.training_cost, error
        # thoughts
        # might just keep all features despite analysis being really good
        # Reasoning: In the beginning it makes no sense to remove a feature since there 
        # really isnt enough information. In the end, we need to throw away previously recieved data points.
        # Rebuttals: Sunk cost fallacy, and since we can send old samples,
        # We can remove in the beginning and add later. 
        # idea: (keep all samples) Remove features like crazy, see what happens. At each iteration, add features
        # in proportion to error. Kinda sucks because there is no testing error and training error is going to be great
        # Tree search no bueno.
        # Keep all features, add full sample if 'good' (unique?)
        # Keep it simple lets do that

    
    def evaluate(self, X, Y):
        predY = X @ self.weight + self.bias
        error = mean_absolute_error(Y, predY)
        
        return predY, error
    
    
    def get_params(self):
        return self.weight, self.bias