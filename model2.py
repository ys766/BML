import numpy as np
import csv
import scipy

class model():
    def __init__(self,
                 training_set_index,
                 hyp = [1,1,1,1,1,1],
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
        a1,b1,c1,a2,b2,c2 = hyp
        self.hyp = np.array(hyp,dtype = np.float64)
        self.__build_R__()
        self.Kuser = self.__kernel__(a1,b1,c1,label = 'user')
        self.Kmovie = self.__kernel__(a2,b2,c2,label = 'movie')
        self.Kuser_inv = np.linalg.inv(self.Kuser)
        self.Kmovie_inv = np.linalg.inv(self.Kmovie)
        self.__compute_mean__()
        self.tR = self.R+self.user_mean+self.movie_mean
        self.tR = self.tR/2
        self.refresh()
    @staticmethod
    def __load_data__(path):
        data = []
        with open(path) as f:
            csv_file = csv.reader(f)
            for row in csv_file:
                data.append(row)
            data = np.array(data,dtype = np.float32)
        return data
    def __kernel__(self,a,b,c,label,use_tR = False):
        # x:m x d
        # a,b hyperparameter
        if use_tR == False:
            R = self.R
        else:
            R = self.tR
        if label == 'user':
            feature = self.user_feature
            rating = R
        elif label == 'movie':
            feature = self.movie_feature
            rating = R.T
        else:
            print('wrong label')
        
        return np.matmul(feature,np.transpose(feature))*a*a + np.matmul(rating,np.transpose(rating))*b*b + np.eye(len(feature))*c*c
    
    def refresh_kernel(self,new_hyp = [],tR = False):
        if len(new_hyp)>0:
            hyp = new_hyp
            self.hyp = hyp
        else:
            hyp = self.hyp
        self.Kuser = self.__kernel__(hyp[0],hyp[1],hyp[2],'user',tR)
        self.Kmovie = self.__kernel__(hyp[3],hyp[4],hyp[5],'movie',tR)
        self.Kuser_inv = np.linalg.inv(self.Kuser)
        self.Kmovie_inv = np.linalg.inv(self.Kmovie)
        

    def __compute_mean__(self):
        self.user_mean = (np.sum(self.R,axis = 1)/np.sum(self.R != 1e-8,axis = 1)).reshape([len(self.user_feature),1])
        self.movie_mean = (np.sum(self.R,axis = 0)/np.sum(self.R != 1e-8,axis = 0)).reshape([1,len(self.movie_feature)])
        self.all_mean = np.sum(self.R)/len(self.indexed_training_set)
        self.user_mean[np.isinf(self.user_mean)] = self.all_mean
        self.movie_mean[np.isinf(self.movie_mean)] = self.all_mean

    def refresh(self):
        for i in self.indexed_training_set:
            self.tR[i[0]-1,i[1]-1] = i[2]

    def iteration(self,n):
        for i in range(n):
            self.tR -= 1*np.matmul(self.Kuser_inv,self.tR-self.movie_mean)
            self.tR -= 1*np.matmul(self.tR-self.user_mean,self.Kmovie_inv)
            if i % 10 ==0:
                self.refresh()
            if i % 100 == 0:
                self.refresh_kernel(self.hyp,True)
                print(self.rmse())
    def optimize_hyp(self,use_tR = False,alpha = 1e-4):
        a = self.R.T*self.Kuser_inv
        b = self.Kuser_inv*self.R
        c = self.R*self.Kmovie_inv
        d = self.Kmovie_inv*self.R.T
        if use_tR:
            R = self.tR
        else:
            R = self.R
        hyp_grad = [np.trace(-1*a*np.matmul(self.user_feature,np.transpose(self.user_feature))*b)]
        hyp_grad += [np.trace(-1*a*np.matmul(R,R.T)*b)]
        hyp_grad += [np.trace(-1*a*np.eye(len(self.user_feature))*b)]
        hyp_grad += [np.trace(-1*c*np.matmul(self.movie_feature,np.transpose(self.movie_feature))*d)]
        hyp_grad += [np.trace(-1*c*np.matmul(R.T,R)*d)]
        hyp_grad += [np.trace(-1*c*np.eye(len(self.movie_feature))*d)]
        hyp_grad = np.array(hyp_grad)
        self.hyp = self.hyp - hyp_grad*alpha*self.hyp

    def train(self,n,alpha):
        for i in range(n):
            self.optimize_hyp(alpha)
            self.refresh_kernel()

    def rmse(self):
        buf = 0
        for i in self.indexed_testing_set:
            buf += (self.tR[i[0]-1,i[1]-1]-i[2])**2
        buf = np.sqrt(buf/len(self.indexed_testing_set))
        return buf

    def __build_R__(self):
        self.R = np.matrix(1e-8*np.ones([len(self.user_feature),len(self.movie_feature)]))
        for i in self.indexed_training_set:
            self.R[i[0]-1,i[1]-1] = i[2]
        
        

    
