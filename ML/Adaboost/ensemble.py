import pickle
import numpy as np
from sklearn.tree import DecisionTreeClassifier

class AdaBoostClassifier:
    
    '''A simple AdaBoost Classifier.'''
    alpha=[]
    weak_classifiers=[]

        
    def __init__(self, weak_classifier, n_weakers_limit):
        '''
        Initialize AdaBoostClassifier
        Args:
            weak_classifier: The class of weak classifier, which is recommend to be sklearn.tree.DecisionTreeClassifier.
            n_weakers_limit: The maximum number of weak classifier the model can use.
        '''
        
        self.weak_classifier=weak_classifier
        self.n_weakers_limit=n_weakers_limit
        
    def is_good_enough(self):
        '''Optional'''
        

    def fit(self,X,y):
        '''
        Build a boosted classifier from the training set (X, y).
        Returns:
            X: An ndarray indicating the samples to be trained, which shape should be (n_samples,n_features).
            y: An ndarray indicating the ground-truth labels correspond to X, which shape should be (n_samples,1).
        '''
        m,n=X.shape
        D=np.zeros([m])
        for i in range(m):
            D[i]=1/m
        #print("D:",D)
        for i in range(self.n_weakers_limit):
                      
            self.weak_classifier.fit(X, y, sample_weight=D)          
            y_predict=self.weak_classifier.predict(X)
            ebacro=0
            for j in range(m):
                if(y_predict[j]!=y[j]):
                    ebacro+=D[i]
            self.alpha.append(1/2*np.log((1-ebacro)/ebacro))
            
            result=np.exp(np.multiply(-self.alpha[i]*y[0].T,y_predict))
            Z=np.sum(np.multiply(D,result))
            D=np.multiply(D/Z,result)
            self.weak_classifiers.append(self.weak_classifier)
            
        print("Compiled train...")

    def predict_scores(self,X):
        '''
        Calculate the weighted sum score of the whole base classifiers for given samples.
        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).
        Returns:
            An one-dimension ndarray indicating the scores of differnt samples, which shape should be (n_samples,1).
        '''
        

    def predict(self,X,threshold=0):
        #print("self.alpha:",self.alpha)
        m,n=X.shape
        G=np.zeros([m],dtype=np.float64)
        for i in range(self.n_weakers_limit):
            G+=self.alpha[i]*self.weak_classifiers[i].predict(X)
        
        print("G:",G)
        G=np.where(G>threshold,1,-1)
        return G.reshape([m,1])
        '''
        Predict the catagories for geven samples.
        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).
            threshold: The demarcation number of deviding the samples into two parts.
        Returns:
            An ndarray consists of predicted labels, which shape should be (n_samples,1).
        '''
        

    @staticmethod
    def save(model, filename):
        with open(filename, "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
