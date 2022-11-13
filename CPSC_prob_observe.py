from scipy.io import loadmat
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
from sklearn.model_selection import train_test_split


class CPSC():
    '''
    The CPSC Class
    '''

    def __init__(self, delta, epsilon, t):
        self.delta = delta  # hyperparameter :threshold for shrinking contrast
        self.epsilon = epsilon  # hyperparameter : threshold for label prediction with p_value in CP
        self.t = t  # data distillation - to make the softmax more soft

    pass

    def fit(self, x_proper_train, y_proper_train, x_cal_train, y_cal_train):
        '''
        This function fit the CPSC model with the data and labels given by users
        :param X: the feature matrix of train set with shape (N, D), np.array
        :param Y: the label of the data of train label with shape (N,), np.array
        '''

        self.num_of_classes = None
        self.class_count = []

        self.dk = None
        self.dk_hat = None
        self.mean_x_k = []
        self.mean_x_overall = []
        self.mean_x_k_hat = []
        self.mean_squared_deviation = []
        self.squared_deviation_overall = []
        self.A_train = []
        self.A_test = []


        y_proper_train = y_proper_train.reshape(-1, )
        y_cal_train = y_cal_train.reshape(-1, )

        classes, self.class_count = np.unique(y_proper_train, return_counts=True)
        self.num_of_classes = len(classes)
        self.mean_x_k, self.mean_x_overall = self.Calculate_raw_centroids(x_proper_train, y_proper_train)
        self.mean_x_k_hat, self.mean_squared_deviation = self.Calculate_nomalized_contrast(x_proper_train,
                                                                                           y_proper_train,
                                                                                           self.mean_x_k,
                                                                                           self.mean_x_overall,
                                                                                           self.delta)
        self.A_train = self.Calculate_A_train_1(x_cal_train, y_cal_train, self.mean_x_k_hat,
                                                self.mean_squared_deviation, self.t)

    def transform(self,x_new):
        return x_new[:,self.remained_feature_index]#Return x_new without the features whose d to centroids are all zeros (<1e-4) after SC

    def fit_transform(self, x_proper_train, y_proper_train):
            '''
            This function fit the CPSC model with the data and labels given by users, and transform the training data.
            :param X: the feature matrix of train set with shape (N, D), np.array
            :param Y: the label of the data of train label with shape (N,), np.array
            '''

            self.num_of_classes = None
            self.class_count = []

            self.dk = None
            self.dk_hat = None
            self.mean_x_k = []
            self.mean_x_overall = []
            self.mean_x_k_hat = []
            self.mean_squared_deviation = []
            self.squared_deviation_overall = []
            self.A_train = []
            self.A_test = []

            y_proper_train = y_proper_train.reshape(-1, )

            classes, self.class_count = np.unique(y_proper_train, return_counts=True)
            self.num_of_classes = len(classes)
            self.mean_x_k, self.mean_x_overall = self.Calculate_raw_centroids(x_proper_train, y_proper_train)
            self.mean_x_k_hat, self.mean_squared_deviation = self.Calculate_nomalized_contrast(x_proper_train,
                                                                                               y_proper_train,
                                                                                               self.mean_x_k,
                                                                                               self.mean_x_overall,
                                                                                               self.delta)
            return x_proper_train[:,self.remained_feature_index]#Return x_new without the features whose d to centroids are all zeros (<1e-4) after SC


    def predict_CPSC(self, x_new):
        '''
        This function predict the label of new samples given by users
        :param x_new: the feature matrix of train set with shape (N, D), np.array
        '''

        self.A_test, self.c_k, self.sigma_k_matrix = self.Calculate_A_test_1(x_new, self.mean_x_k_hat,
                                                                             self.mean_squared_deviation,self.t)
        self.p_value = []
        self.forced_prediction = []  # forced prediction(single output based on max p_value)
        self.credibility_prediction=[]
        self.credibility_index=[]
        self.credibility = []  # forced_prediction credibility
        self.confidence = []  # forced_prediction confidence
        self.second_max_pvalue=[]

        self.region_prediction_output = []
        self.region_prediction_pvalue = []  # multi-output p_value

        self.p_value = self.Calculate_pvalue(self.A_train, self.A_test)
        self.region_prediction_pvalue = self.p_value * (self.p_value > self.epsilon)

        region_prediction_list = np.zeros((len(x_new), self.num_of_classes)).astype(int)
        # inform leo
        self.region_prediction_output = np.full((len(x_new), self.num_of_classes), np.nan)

        for i in range(len(x_new)):
            region_prediction_list[i, :] = np.arange(self.num_of_classes).astype(int)
            p_value_array = self.p_value[i, :]
            credibility = np.max(p_value_array)  # the max p-value
            max_index = np.argmax(p_value_array)  # the index of max p-value (its predicted label)
            second_p_value_array = np.delete(p_value_array, max_index, axis=0)
            p_value_sec_max = np.max(second_p_value_array)
            confidence = 1 - p_value_sec_max

            self.forced_prediction.append(max_index)
            self.credibility.append(credibility)
            self.confidence.append(confidence)
            self.second_max_pvalue.append(p_value_sec_max)

            # region prediction  # inform leo
            self.region_prediction_output[i, np.where(p_value_array >= self.epsilon)] = \
                region_prediction_list[i, np.where(p_value_array >= self.epsilon)]

            # if len(region_prediction)>0:
            #     self.region_prediction_output.append(region_prediction)
            # else:
            #     self.region_prediction_output.append(np.nan)

        #         self.region_prediction=self.region_prediction*(self.p_value>self.epsilon)
        self.confidence = np.array(self.confidence)
        self.credibility = np.array(self.credibility)

        self.forced_prediction = np.array(self.forced_prediction)
        self.credibility_prediction=np.array([self.forced_prediction[m] for m in range(len(self.forced_prediction)) if self.credibility[m]>=self.epsilon])
        self.credibility_index=np.array([m for m in range(len(self.forced_prediction)) if self.credibility[m]>=self.epsilon  and self.credibility[m]>3*self.second_max_pvalue[m]])#The forced predictions satisfiying epsilon
        self.region_prediction_output = np.array(self.region_prediction_output)

        return self.credibility_index,self.forced_prediction,self.credibility,self.confidence,self.p_value,

    def predict_SC(self, x_new):

        self.SC_prediction = []  # SC prediction
        self.SC_prob = []  # the probability of SC output
        self.A_test, self.c_k, self.sigma_k_matrix = self.Calculate_A_test_1(x_new, self.mean_x_k_hat,
                                                                             self.mean_squared_deviation, self.t)
        self.SC_prediction = np.argmax(self.sigma_k_matrix, axis=1)
        assert self.c_k.shape[1] == 2
        self.SC_prob = self.c_k

        return self.SC_prediction,self.SC_prob

    def Calculate_raw_centroids(self, X, Y):
        '''
        This function calculate the original centroids of the training data and labels given by users
        :param X: the feature matrix of train set with shape (N, D), np.array
        :param Y: the label of the data of train label with shape (N,), np.array
        '''

        self.mean_x_k = np.zeros((self.num_of_classes, X.shape[1]))  # x_k_bar matrix(K*D)
        for k in range(self.num_of_classes):
            self.mean_x_k[k] = np.mean(X[Y == k, :], axis=0)  # (1*D)

        self.mean_x_overall = np.mean(X, axis=0)  # mean of features of all samples(1*D)

        return self.mean_x_k, self.mean_x_overall

    def Calculate_nomalized_contrast(self, X, Y, mean_x_k, mean_x_overall, delta):
        '''
        This function calculate the nomalized contrast of the training data and labels given by users
        :param X: the feature matrix of train set with shape (N, D), np.array
        :param Y: the label of the data of train label with shape (N,), np.array
        :param mean_x_k: the mean of 'class k' data, with shape (C,D), np.array
        :param mean_x_overall: the mean of all samples, with shape(1,D),np.array
        '''


        self.squared_deviation_overall = np.zeros(X.shape[1])  # s^2 (1*D)

        for k in range(self.num_of_classes):
            squared_deviation_k = np.sum((X[Y == k, :] - mean_x_k[k]) ** 2, axis=0)
            self.squared_deviation_overall += squared_deviation_k

        self.mean_squared_deviation = (1 / (
                X.shape[0] - self.num_of_classes)) * self.squared_deviation_overall  # Sj^2 (1*D)
        self.standard_deviation = np.sqrt(self.mean_squared_deviation)  # Sj (1*D)

        self.dk = (mean_x_k - mean_x_overall) / self.standard_deviation  # Normalized contrast of class k
        self.dk = np.nan_to_num(self.dk)

        # Shrunk class centroids

        self.dk_hat = np.sign(self.dk) * np.maximum(np.abs(self.dk) - self.delta, 0)  # delta is a hyperparameter
        self.mean_x_k_hat = mean_x_overall + self.standard_deviation * self.dk_hat  # shrunken contrast
        self.remained_feature_index=[i for i in range(self.dk_hat.shape[1]) if all(np.abs(self.dk_hat[:, i]) > 1e-4)]#the features whose d to centroids are all above zeros (<1e-4) after SC


        return self.mean_x_k_hat, self.mean_squared_deviation

    def Calculate_A_train_1(self, X, Y, mean_x_k_hat, mean_squared_deviation, t):
        '''
        This function calculate the nonconformity measurement with training data and labels given by users
        :param X: the feature matrix of train set with shape (N, D), np.array
        :param Y: the label of the data of train label with shape (N,), np.array
        :param mean_x_k_hat: the shrunk mean of 'class k' data, with shape (C,D), np.array
        :param mean_squared_deviation: the mean of squared_deviation among k centroids and overall centroid, with shape(1,D),np.array
        '''
        A_label = np.zeros(len(X))
        for i in range(len(X)):
            label = int(Y[i])
            c_k = np.zeros(self.num_of_classes)
            sigma_k = self.Calculate_sigma(X[i], mean_x_k_hat, mean_squared_deviation)
            c_k = np.exp(sigma_k / t) / np.sum(np.exp(sigma_k /t))  # softmax (1*K)
            if np.isnan(c_k).any():
                c_k= np.zeros(self.num_of_classes)
                c_k[np.argmax(sigma_k)]=1
            A_label[i] = 0.5 - 0.5 * (c_k[label] - np.max(np.delete(c_k, label)))
        return A_label

    def Calculate_sigma(self, x_new, mean_x_k_hat, mean_squared_deviation):  # Input only a SINGLE sample once
        '''
        This function calculate the discriminant score of new samples given by users
        :param x_new: the feature matrix of test set with shape (N, D), np.array
        :param mean_x_k_hat: the shrunk mean of 'class k' data, with shape (C,D), np.array
        :param mean_squared_deviation: the mean of squared_deviation among k centroids and overall centroid, with shape(1,D),np.array
        '''
        sigma_k = np.zeros(self.num_of_classes)
        pi_k = self.class_count / np.sum(self.class_count)  # prior
        sigma_k = np.log(pi_k) - np.sum(0.5 * ((x_new - mean_x_k_hat) ** 2) / mean_squared_deviation, axis=1)
        return sigma_k  # (K,)

    def Calculate_A_test_1(self, x_new, mean_x_k_hat, mean_squared_deviation,t):  # Method 1_
        '''
        This function calculate the nonconformal measurement of new samples given by users
        :param x_new: the feature matrix of test set with shape (N, D), np.array
        :param mean_x_k_hat: the shrunk mean of 'class k' data, with shape (C,D), np.array
        :param mean_squared_deviation: the mean of squared_deviation among k centroids and overall centroid, with shape(1,D),np.array
        '''

        c_k = np.zeros((len(x_new), self.num_of_classes))  # possibility of k
        A_k = np.zeros((len(x_new), self.num_of_classes))  # Noncomformity measurement
        self.sigma_k_matrix = np.empty((x_new.shape[0], self.num_of_classes))  # (N_test,K)

        for i in range(len(x_new)):

            sigma_k = self.Calculate_sigma(x_new[i], mean_x_k_hat, mean_squared_deviation)
            self.sigma_k_matrix[i] = sigma_k

            c_k[i] = np.exp(sigma_k/t) / np.sum(np.exp(sigma_k/t))  # (1*K)
            if np.isnan(c_k[i]).any():
                c_k[i]= np.zeros(self.num_of_classes)
                c_k[i][np.argmax(sigma_k)]=1

            for k in range(self.num_of_classes):
                A_k[i][k] = 0.5 - 0.5 * (c_k[i][k] - np.max(np.delete(c_k[i], k)))

        return A_k, c_k, self.sigma_k_matrix

    def Calculate_pvalue(self, A_train, A_test):
        '''
        This function calculate the p_value of new samples given by users
        :param A_train: the nonconformity measurement matrix of training set, with shape of (N, 1), np.array
        :param A_test: the nonconformity measurement matrix of test data, with shape (N,C), np.array
        '''

        self.p_value = np.zeros((len(A_test), self.num_of_classes))
        for m in range(len(A_test)):
            for n in range(self.num_of_classes):
                cnt = len(A_train[A_train >= A_test[m, n]])
                self.p_value[m, n] = (cnt + 1) / (len(A_train) + 1)
        return self.p_value

if __name__ == '__main__':

        dataset_from_mat = loadmat("full_trainingset_categories.mat")
        all_dataset_raw = dataset_from_mat['full_trainingset']



        # set label
        all_label = []
        for i in range(12):
            for j in range(50):
                all_label.append(i)
        all_label=np.array(all_label)

        # data preprocessing
        scaler.fit(all_dataset_raw)
        processed_dataset = scaler.transform(all_dataset_raw)

        #dataset partition
        x_train,x_other,y_train,y_other=train_test_split(processed_dataset,all_label,test_size=360, random_state=10)
        x_pro_train,x_cal_train,y_pro_train,y_cal_train=train_test_split(x_train,y_train,test_size=120, random_state=10)
        x_active,x_cal_test,y_active,y_cal_test=train_test_split(x_other,y_other,test_size=120, random_state=10)
        x_val,x_test,y_val,y_test=train_test_split(x_cal_test,y_cal_test,test_size=60, random_state=10)


        cpsc=CPSC(0.01,0.5,1000)#delta,epsilon,temperature
        cpsc.fit(x_pro_train,y_pro_train,x_cal_train,y_cal_train)
        y_predict=cpsc.predict_CPSC(x_test)
        print(accuracy_score(y_test,y_predict))
