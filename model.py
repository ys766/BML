import numpy as np
import csv

class model():
    def __init__(self,
                 training_set_index,
                 hyp = [1,1,1,1],
                 directory = 'D:/DATA/'):
        """Load data into user feature, movie feature, training set and test set.
           training_set_index: choose from 1-5. Indicating which training set to load.
           user/movie feature dimension: (data size) X (feature number)
           indexed training/test set dimension: (data size) X [user_id, movie_id, rating]
        """
        self.user_feature = self.__load_data__(directory + 'user_feature.csv')[1:]
        self.movie_feature = self.__load_data__(directory + 'movie_feature.csv')[1:]
        self.indexed_training_set = np.array(self.__load_data__(directory + 'train_'+ str(training_set_index) + '.csv'),dtype = np.int32)
        self.indexed_testing_set = np.array(self.__load_data__(directory + 'test_' + str(training_set_index) + '.csv'),dtype = np.int32)
        a1,b1,a2,b2 = hyp
        self.hyp = hyp
        self.Kuser = self.__kernel__(a1,b1,self.user_feature)
        self.Kmovie = self.__kernel__(a2,b2,self.movie_feature)
        self.__distance_measure__()
    @staticmethod
    def __load_data__(path):
        data = []
        with open(path) as f:
            csv_file = csv.reader(f)
            for row in csv_file:
                data.append(row)
            data = np.array(data,dtype = np.float32)
        return data
    @staticmethod
    def __kernel__(a,b,x):
        # x:m x d
        # a,b hyperparameter
        m = x.shape[0]
        return np.matmul(x,np.transpose(x))*a*a + np.ones([m,m])*b*b
    
    def __distance_measure__(self):
        
        user_feature_shape = self.user_feature.shape
        movie_feature_shape = self.movie_feature.shape
        user_feature1 = np.repeat(np.expand_dims(self.user_feature,axis = 2),user_feature_shape[0],axis = 2)
        movie_feature1 = np.repeat(np.expand_dims(self.movie_feature,axis = 2),movie_feature_shape[0],axis = 2)
        user_feature2 = np.repeat(np.expand_dims(np.transpose(self.user_feature),axis = 0),user_feature_shape[0],axis = 0)
        movie_feature2 = np.repeat(np.expand_dims(np.transpose(self.movie_feature),axis = 0),movie_feature_shape[0],axis = 0)
        self.user_distance = np.squeeze(np.sum(0.5*np.abs(user_feature1 - user_feature2),axis = 1))
        self.movie_distance = np.squeeze(np.sum(0.5*np.abs(movie_feature1 - movie_feature2),axis = 1))

    def choose_inducing_point(self,ratio,pattern = 'random',source = 'training set'):
        # function used to choose inducing point from self.training_set
        # ratio: inducing_point/data_size
        # Extra choosing method could be changed by pattern
        if pattern == 'random' and source == 'training set':
            n = self.indexed_training_set.shape[0]
            m = np.ceil(n*ratio).astype(int)
            self.indexed_inducing_set = self.indexed_training_set[np.random.choice(n,m,replace = False)]

    def build_Kuu(self,ratio,pattern = 'random'):
        # hyp = [a_user,b_user,a_movie,b_movie]
        m = self.indexed_inducing_set.shape[0] # size of the inducing point set
        self.Kuu = np.matrix(np.ndarray([m,m],dtype = np.float32))
        for i in range(m):
            for j in range(m):
                user_i,movie_i,*rest = self.indexed_inducing_set[i]
                user_j,movie_j,*rest = self.indexed_inducing_set[j]
                self.Kuu[i,j] = self.Kuser[user_i,user_j]*self.Kmovie[movie_i,movie_j]
    
            
    def build_sparse_W(self,pattern = 'train'):
        #Create sparse matrix self.W based on:
        #1.self.indexed_training_set or self.indexed_testing_set
        #2.self.indexed_inducing_set
        #call choose_inducing_point before calling this method
        if pattern == 'train':
            X = self.indexed_training_set
        elif pattern == 'test':
            X = self.indexed_testing_set
        U = self.indexed_inducing_set
        m = len(X)
        n = len(U)
        W = np.matrix(np.zeros([m,n]))
        for i in range(m):
            low_1,low_2,index_1,index_2 = [47,47,-1,-1]
            for j in range(n):
                ui,uj = X[i][0],U[j][0]
                vi,vj = X[i][1],U[j][1]
                buffer = self.user_distance[ui-1,uj-1] + self.movie_distance[vi-1,vj-1]
                if buffer <= low_1:
                    low_2 = low_1
                    low_1 = buffer
                    index_2 = index_1
                    index_1 = j
                    if low_1 == 0 and low_2 == 0:
                        break
            if low_1 == low_2:
                W[i,[index_1,index_2]] = 0.5
            else:
                W[i,[index_1,index_2]] = np.array([low_2,low_1])/(low_2 + low_1)
        if pattern == 'train':
            self.training_W = W
        elif pattern == 'test':
            self.testing_W = W

    def Linear_Conjugate_Gradient(self):
        #Given W,Kuu out put the inverse of WKuuWT
        self.inverse_WKuuWT
        pass

    def Approximate_log_determinant(self):
        #Approximate the log|WKuuWT|
        pass

    def predict(self):
        self.mean = (self.training_W*self.Kuu.T*self.testing_W.T) * self.inverse_WKuuWT * np.matrix(self.indexed_training_set[:,2]) # K(x,x*) * inv(W*Kuu*WT) * y
        self.variance = (self.testing_W*self.Kuu.T*self.testing_W.T) + (self.training_W*self.Kuu.T*self.testing_W.T) * self.inverse_WKuuWT * (self.testing_W*self.Kuu.T*self.training_W.T)
        return [self.mean,self.variance]

    def optimize_hyp(self):
        pass
        
        
                
                    
                    
            
        
        
        
        

    """
    def build_grid_strict(self,max_guessing_point_per_user = 0):
        # Not finished
        self.dict = {}
        for i in np.arange(1,945):
            self.dict[str(i)] = set(self.training_set[self.training_set[:,0] == i,1])
        self.maximum_user_grid = []
        buffer = []
        def expoit(user_sub):
            if len(user_sub) == 0:
                shared_movie_set = set()
            elif len(user_sub) == 1:
                shared_movie_set = self.dict[str(user_sub[0])]
            else:
                shared_movie_set = set.intersection(*[self.dict[str(i)] for i in user_sub])
            length = 0
            for i in set(self.dict.keys())-set(user_sub):
                if len(self.dict[i] & shared_movie_set) >= length:
                    length = len(self.dict[i] & shared_movie_set)
                    user_id = i
            return [user_sub.append(str(i)),length]
        user_sub = []
        length = 0
        while(1):
            K = expoit(user_sub)
            if K[0]*K[1] >= len(user_sub)*length:
    """
