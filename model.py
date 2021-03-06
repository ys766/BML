import numpy as np
import csv
import scipy

class model():
    def __init__(self,
                 training_set_index,
                 hyp = [1,1,1,1,1],
                 directory = ''):
        """Load data into user feature, movie feature, training set and test set.
           training_set_index: choose from 1-5. Indicating which training set to load.
           user/movie feature dimension: (data size) X (feature number)
           indexed training/test set dimension: (data size) X [user_id, movie_id, rating]
        """
        self.user_feature = self.__load_data__(directory + 'user_feature.csv')[1:]
        self.movie_feature = self.__load_data__(directory + 'movie_feature.csv')[1:]
        self.indexed_training_set = np.array(self.__load_data__(directory + 'train_'+ str(training_set_index) + '.csv'),dtype = np.int32)
        self.indexed_testing_set = np.array(self.__load_data__(directory + 'test_' + str(training_set_index) + '.csv'),dtype = np.int32)
        a1,b1,a2,b2,*rest = hyp
        self.hyp = hyp
        self.Kuser = self.__kernel__(a1,b1,self.user_feature)
        self.Kmovie = self.__kernel__(a2,b2,self.movie_feature)
        self.__distance_measure__()
        self.__build_R__()
        self.choose_inducing_point(0.1)
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
        return  np.eye(m)*a*a + np.matmul(x,np.transpose(x))*b*b
    
    def __distance_measure__(self):
        
        user_feature_shape = self.user_feature.shape
        movie_feature_shape = self.movie_feature.shape
        user_feature1 = np.repeat(np.expand_dims(self.user_feature,axis = 2),user_feature_shape[0],axis = 2)
        movie_feature1 = np.repeat(np.expand_dims(self.movie_feature,axis = 2),movie_feature_shape[0],axis = 2)
        user_feature2 = np.repeat(np.expand_dims(np.transpose(self.user_feature),axis = 0),user_feature_shape[0],axis = 0)
        movie_feature2 = np.repeat(np.expand_dims(np.transpose(self.movie_feature),axis = 0),movie_feature_shape[0],axis = 0)
        self.user_distance = np.squeeze(np.sum(0.5*np.abs(user_feature1 - user_feature2),axis = 1))
        self.movie_distance = np.squeeze(np.sum(0.5*np.abs(movie_feature1 - movie_feature2),axis = 1))

    def __build_R__(self):
        self.R = -1*np.ones([len(self.user_feature),len(self.movie_feature)])
        for i in self.indexed_training_set:
            self.R[i[0]-1,i[1]-1] = i[2]
        
        

    def choose_inducing_point(self,ratio,pattern = 'random',source = 'training set'):
        # function used to choose inducing point from self.training_set
        # ratio: inducing_point/data_size
        # Extra choosing method could be changed by pattern
        if pattern == 'random' and source == 'training set':
            n = self.indexed_training_set.shape[0]
            m = np.ceil(n*ratio).astype(int)
            self.indexed_inducing_set = self.indexed_training_set[np.random.choice(n,m,replace = False)]

    def refresh_um_kernel(self):
        a1,b1,a2,b2,*rest = self.hyp
        self.Kuser = self.__kernel__(a1,b1,self.user_feature)
        self.Kmovie = self.__kernel__(a2,b2,self.movie_feature)

    def build_Kuu(self):
        # hyp = [a_user,b_user,a_movie,b_movie]
        m = self.indexed_inducing_set.shape[0] # size of the inducing point set
        self.Ku = np.matrix(np.ndarray([m,m],dtype = np.float32))
        self.Km = np.matrix(np.ndarray([m,m],dtype = np.float32))
        for i in range(m):
            for j in range(m):
                user_i,movie_i,*rest = self.indexed_inducing_set[i]
                user_j,movie_j,*rest = self.indexed_inducing_set[j]
                self.Ku[i,j] = self.Kuser[user_i-1,user_j-1]
                self.Km[i,j] = self.Kmovie[movie_i-1,movie_j-1]
        self.Kuu = np.multiply(self.Ku,self.Km)
                
    
            
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
        n = len(X)
        m = len(U)
        W = np.matrix(np.zeros([n,m]))
        for i in range(n):
            low_1,low_2,index_1,index_2 = [47,47,-1,-1]
            for j in range(m):
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

    def build_alpha(self):
        #Given W and Kuu, return inv(WKuuWT + sigma**2*I)*y
        sigma = self.hyp[4]
        y = self.indexed_training_set[:,2]
        self.y_mean = np.mean(y)
        y = y - self.y_mean
        y = y.reshape([-1,1])
        sigma_square = sigma**2
        self.alpha = y/sigma_square - (self.training_W/sigma_square) * (np.linalg.inv(np.linalg.inv(self.Kuu) + self.training_W.T*self.training_W/sigma_square) * (self.training_W.T*y/sigma_square))


    def predict(self):
        self.predict_mean = (self.testing_W*self.Kuu)*(self.training_W.T*self.alpha) + self.y_mean
        m = np.array(self.predict_mean - self.indexed_testing_set[:,2])
        return np.sqrt(np.mean(m**2))
        

    def hyp_g(self):
        hyp = self.hyp
        m = self.Kuu.shape[0]
        Kuu_inv = np.matrix(np.linalg.inv(self.Kuu))
        y = np.matrix(self.indexed_inducing_set[:,2].reshape([-1,1]))
        dkda1 = np.matrix(self.Km*2*hyp[0])
        dkdb1 = np.matrix(np.multiply(self.Km,self.Ku-hyp[0]**2*np.eye(m))*2/hyp[1])
        dkda2 = np.matrix(self.Km*2*hyp[2])
        dkdb2 = np.matrix(np.multiply(self.Ku,self.Km-hyp[2]**2*np.eye(m))*2/hyp[3])
        dkdsigma = np.matrix(np.eye(m))*2*hyp[4]
        def dl(x):
            return np.matrix.trace(Kuu_inv*x) - y.T*Kuu_inv*x*Kuu_inv*y
        self.hyp_grad = np.squeeze(np.array([dl(dkda1),dl(dkdb1),dl(dkda2),dl(dkdb2),dl(dkdsigma)]))

    def optimize_hyp(self,step = 0.0001,n = 1000):
        for i in range(n):
            self.hyp_g()
            self.hyp -= step*self.hyp_grad
            self.refresh_um_kernel()
            self.build_Kuu()

                
                    
                    
            
        
        
        
        

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
