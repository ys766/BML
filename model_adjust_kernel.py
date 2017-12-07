import numpy as np
import csv
import scipy

class model():
    def __init__(self,
                 training_set_index,
                 filter_ratio = 1,
                 hyp = [1,1,1,1,0.1,1,0.1,1],
                 directory = 'D:/DATA/'
                 ):
        """Load data into user feature, movie feature, training set and test set.
           training_set_index: choose from 1-5. Indicating which training set to load.
           user/movie feature dimension: (data size) X (feature number)
           indexed training/test set dimension: (data size) X [user_id, movie_id, rating]
        """
        self.user_feature = self.__load_data__(directory + 'user_feature.csv')[1:]
        self.movie_feature = self.__load_data__(directory + 'movie_feature.csv')[1:]
        self.indexed_training_set = np.array(self.__load_data__(directory + 'train_'+ str(training_set_index) + '.csv'),dtype = np.int32)
        self.indexed_testing_set = np.array(self.__load_data__(directory + 'test_' + str(training_set_index) + '.csv'),dtype = np.int32)
        self.shrink_training_set(filter_ratio)
        self.__build_R__()
        self.hyp = hyp
        self.build_um_kernel()
        
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
    def __l2norm__(x):
        # x is a n x m matrix
        # we want to measure the distance between n points and return n x n distance measure
        shape = x.shape
        x1 = np.repeat(np.expand_dims(x,axis = 2),shape[0],axis = 2)
        x2 = np.transpose(x1,axes = (2,1,0))
        return np.squeeze(np.sum((x1-x2)**2,axis = 1))

    @staticmethod
    def __rbf_kernel__(x,l):
        shape = x.shape
        x1 = np.repeat(np.expand_dims(x,axis = 2),shape[0],axis = 2)
        x2 = np.transpose(x1,axes = (2,1,0))
        return np.squeeze(np.exp(np.sum((x1-x2)**2,axis = 1)/(-2*l**2)))
    
    '''
    @staticmethod
    def __rating_kernel__(R,l):
        R_L = R != 1e-8
        R -= np.mean(R,axis = 1).reshape([-1,1])
        R[~R_L] = 1e-8
        buf = np.matmul(R,R.T)/np.matmul(R_L,R_L.T)
        buf[np.isinf(buf)] = 0
        return buf
    '''
        
        

    @staticmethod
    def __movie_feature_kernel__(x,l):
        buf = np.matmul(x,x.T)
        for i in range(buf.shape[0]):
            n = np.sqrt(buf[i,i])
            buf[i,:] = buf[i,:]/n
            buf[:,i] = buf[:,i]/n
        return buf**(1/l**2)

        
    
    def __user_kernel__(self):
        a,l1,l2,l3,sigma_u = self.hyp[0:5]
        age = self.user_feature[:,:5]
        occupation = self.user_feature[:,5:26]
        gender = self.user_feature[:,26:]
        R = self.R
        k1 = self.__rbf_kernel__(age,l1)
        k2 = self.__rbf_kernel__(occupation,l2)
        k3 = self.__rbf_kernel__(gender,l3)
        k4 = np.eye(len(self.user_feature))*sigma_u
        self.Kuser = a*k1*k2*k3+k4


    def __movie_kernel__(self):
        l5,sigma_m = self.hyp[5:7]
        k1 = self.__movie_feature_kernel__(self.movie_feature,l5)
        k2 = np.eye(len(self.movie_feature))*sigma_m
        self.Kmovie = k1+k2
    
    def __distance_measure__(self):
        user_feature = self.user_feature.astype(np.float64)
        movie_feature = self.movie_feature.astype(np.float64)
        user_feature[:,:5] = (user_feature[:,:5]/self.hyp[1])**2
        user_feature[:,5:26] = (user_feature[:,5:26]/self.hyp[2])**2
        user_feature[:,26:] = (user_feature[:,26:]/self.hyp[3])**2
        movie_feature = (movie_feature/self.hyp[6])**2
        user_feature_shape = user_feature.shape
        movie_feature_shape = movie_feature.shape
        user_feature1 = np.repeat(np.expand_dims(user_feature,axis = 2),user_feature_shape[0],axis = 2)
        movie_feature1 = np.repeat(np.expand_dims(movie_feature,axis = 2),movie_feature_shape[0],axis = 2)
        user_feature2 = np.repeat(np.expand_dims(np.transpose(user_feature),axis = 0),user_feature_shape[0],axis = 0)
        movie_feature2 = np.repeat(np.expand_dims(np.transpose(movie_feature),axis = 0),movie_feature_shape[0],axis = 0)
        self.user_distance = np.squeeze(np.sum(0.5*np.abs(user_feature1 - user_feature2),axis = 1))
        self.movie_distance = np.squeeze(np.sum(0.5*np.abs(movie_feature1 - movie_feature2),axis = 1))

    def __build_R__(self):
        self.R = 1e-8*np.ones([len(self.user_feature),len(self.movie_feature)])
        for i in self.indexed_training_set:
            self.R[i[0]-1,i[1]-1] = i[2]

    def build_um_kernel(self):
        self.__user_kernel__()
        self.__movie_kernel__()

    def shrink_training_set(self,ratio,pattern = 'random',source = 'training set'):
        # function used to choose inducing point from self.training_set
        # ratio: inducing_point/data_size
        # Extra choosing method could be changed by pattern
        if pattern == 'random' and source == 'training set':
            n = self.indexed_training_set.shape[0]
            m = np.ceil(n*ratio).astype(int)
            self.indexed_training_set = self.indexed_training_set[np.random.choice(n,m,replace = False)]


        
    def choose_inducing_point(self,ratio,pattern = 'random',source = 'training set'):
        # function used to choose inducing point from self.training_set
        # ratio: inducing_point/data_size
        # Extra choosing method could be changed by pattern
        if pattern == 'random' and source == 'training set':
            n = self.indexed_training_set.shape[0]
            m = np.ceil(n*ratio).astype(int)
            self.indexed_inducing_set = self.indexed_training_set[np.random.choice(n,m,replace = False)]

    def build_Kuu(self,ratio = 0.01,label = 0):
        self.choose_inducing_point(ratio)
        # hyp = [a_user,b_user,a_movie,b_movie]
        m = self.indexed_inducing_set.shape[0] # size of the inducing point set
        if label == 1:
            self.Kuu = np.eye(m)
            return 
        self.Kuu = np.matrix(np.ndarray([m,m],dtype = np.float32))
        for i in range(m):
            for j in range(m):
                user_i,movie_i,*rest = self.indexed_inducing_set[i]
                user_j,movie_j,*rest = self.indexed_inducing_set[j]
                user = self.Kuser[user_i-1,user_j-1]
                movie = self.Kmovie[movie_i-1,movie_j-1]
                self.Kuu[i,j] = user*movie
    
            
    def build_W(self,pattern = 'train'):
        #Create sparse matrix self.W based on:
        #1.self.indexed_training_set or self.indexed_testing_set
        #2.self.indexed_inducing_set
        #call choose_inducing_point before calling this method
        self.__distance_measure__()
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
        sigma = self.hyp[7]
        y = self.indexed_training_set[:,2]
        self.y_mean = np.mean(y)
        y = y - self.y_mean
        y = y.reshape([-1,1])
        sigma_square = sigma**2
        self.alpha = y/sigma_square - (self.training_W/sigma_square) * (np.linalg.inv(np.linalg.inv(self.Kuu) + self.training_W.T*self.training_W/sigma_square) * (self.training_W.T*y/sigma_square))

    def Approximate_log_determinant(self):
        #Approximate the log|WKuuWT|
        pass

    def predict(self):
        self.predict_mean = (self.testing_W*self.Kuu)*(self.training_W.T*self.alpha) + self.y_mean

    def rmse(self):
        m = np.array(self.predict_mean - self.indexed_testing_set[:,2])
        return np.sqrt(np.mean(m**2))

    def full_inference(self,inducing_ratio = 0.01):
        self.build_um_kernel()
        self.choose_inducing_point(inducing_ratio)
        self.build_Kuu()
        self.build_W()
        self.build_W('test')
        self.build_alpha()
        self.predict()
        print(self.rmse())

    def optimize_hyp(self):
        inv_Kuu = np.linalg.inv(self.Kuu)
        y = self.indexed_inducing_set[:,2].reshape([-1,1])
        dKuuda = self.Kuu/self.hyp[0]
        tao_l1 = self.__l2norm__(self.user_feature[:,:5])
        tao_l2 = self.__l2norm__(self.user_feature[:,5:26])
        tao_l3 = self.__l2norm__(self.user_feature[:,26:])
        tao_l5 = self.__l2norm__(self.movie_feature)
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
