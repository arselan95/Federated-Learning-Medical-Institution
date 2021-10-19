import warnings
warnings.filterwarnings("ignore")
from sklearn.datasets import load_boston
from sklearn import preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from sklearn.linear_model import SGDRegressor
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from numpy import random
from sklearn.model_selection import train_test_split
import pickle
from matplotlib.pyplot import figure
import math
#ddic=pd.DataFrame(columns=['nodenumber','event'])
#ddic.to_csv('demodatabase.csv')
'''
dfile=pd.read_csv("demodatabase.csv")
print(len(dfile.index))
if dfile.empty is False:
    print(dfile['event'].to_string(index=False))
else:
    print("No new events")
'''
class model():
    loss=0
    losshelper=[]
    weights=[]
    bias=[]
    def getdata(mydata):
        return mydata;

    def modelparameters(s,learning_rate,n_iter,k,divideby):
        params=[]
        params.append(learning_rate)
        params.append(n_iter)
        params.append(k)
        params.append(divideby)
        return params
    def SGDModel(data,X, learning_rate,n_iter,k,divideby,tlosshelper,a,c):    
        # Initially we will keep our W and B as 0 as per the Training Data
        model().losshelper=tlosshelper
        w=np.zeros(shape=(1,X.shape[1]-1))
        b=0
        cur_iter=1
        while(cur_iter<=n_iter): 

            # We will create a small training data set of size K
            temp=X.sample(k)
            
            # We create our X and Y from the above temp dataset
            y=np.array(temp['beds'])
            x=np.array(temp.drop('beds',axis=1))
            
            # We keep our initial gradients as 0
            w_gradient=np.zeros(shape=(1,X.shape[1]-1))
            b_gradient=0
            
            for i in range(k): # Calculating gradients for point in our K sized dataset
                prediction=np.dot(w,x[i])+b
                w_gradient=w_gradient+(a)*x[i]*(y[i]-(prediction))
                b_gradient=b_gradient+(c)*(y[i]-(prediction))
            
            #Updating the weights(W) and Bias(b) with the above calculated Gradients
            w=w-learning_rate*(w_gradient/k)
            b=b-learning_rate*(b_gradient/k)
            
            # Incrementing the iteration value
            cur_iter=cur_iter+1
            
            #Dividing the learning rate by the specified value
            learning_rate=learning_rate/divideby

            
        return w,b #Returning the weights and Bias

    def predict(a,x,w,b):
        y_pred=[]
        for i in range(len(x)):
            y=np.asscalar(np.dot(w,x[i])+b)
            y_pred.append(y)
        return np.array(y_pred)

    def getloss(someloss):
        model().loss=np.asarray(model().loss)
        return model().loss

    def getweight(m,wb):
        model().weights=np.asarray(model().weights)
        model().weights=wb
        return wb
    def getbias(k,bi):
        model().bias=np.asarray(model().bias)
        model().bias=bi
        return bi


def split_and_shuffle_labels(y_data, seed, amount):
    y_data=pd.DataFrame(y_data,columns=["labels"])
    y_data["i"]=np.arange(len(y_data))
    label_dict = dict()
    for i in range(3):
        var_name="label" + str(i)
        label_info=y_data
        #print("labels")
        #print(label_info)
        np.random.seed(i)
        label_info=np.random.permutation(label_info)
        label_info=label_info[0:amount]
        label_info=pd.DataFrame(label_info, columns=["labels","i"])
        #print(label_info)
        label_dict.update({var_name: label_info})
        print("updating label dic")
        #print(label_dict)
    return label_dict


def get_iid_subsamples_indices(label_dict, number_of_samples, amount):
    sample_dict= dict()
    batch_size=int(math.floor(amount/number_of_samples))
    for i in range(number_of_samples):
        sample_name="sample"+str(i)
        dumb=pd.DataFrame()
        for j in range(3):
            label_name=str("label")+str(j)
            a=label_dict[label_name][i*batch_size:(i+1)*batch_size]
            #print(a)
            dumb=pd.concat([dumb,a], axis=0)
        dumb.reset_index(drop=True, inplace=True)    
        sample_dict.update({sample_name: dumb})
        print("updating sample dict")
        #print(sample_dict) 
    return sample_dict


def create_iid_subsamples(sample_dict, x_data, y_data, x_name, y_name):
    x_data_dict= dict()
    y_data_dict= dict()
    
    for i in range(len(sample_dict)):  ### len(sample_dict)= number of samples
        xname= x_name+str(i)
        yname= y_name+str(i)
        sample_name="sample"+str(i)
        
        indices=np.sort(np.array(sample_dict[sample_name]["i"]))
        
        x_info= x_data[indices,:]
        x_data_dict.update({xname : x_info})
        
        y_info= y_data[indices]
        y_data_dict.update({yname : y_info})
        
    return x_data_dict, y_data_dict


#train_amount=150
train_amount=180
number_of_samples=3
test_amount=25
print_amount=2

j=[]
def create_model_optimizer_criterion_dict(number_of_samples):
    model_dict = dict()
    optimizer_dict= dict()
    criterion_dict = dict()
    
    for i in range(number_of_samples):
        model_name="model"+str(i)
        model_info=model()
        model_dict.update({model_name : model_info })
        
        optimizer_name="optimizer"+str(i)
        optimizer_info = model_info.modelparameters(0.0001,i*400,40,1)
        optimizer_dict.update({optimizer_name : optimizer_info })
        
        criterion_name = "criterion"+str(i)
        criterion_info = model_info.getloss()
        criterion_dict.update({criterion_name : criterion_info})
        
    return model_dict, optimizer_dict, criterion_dict 


model_dict, optimizer_dict, criterion_dict=create_model_optimizer_criterion_dict(3)

'''
print("model")
print(model_dict)
print("opti")
print(optimizer_dict)
print("crite")
print(criterion_dict)

name_of_models=list(model_dict.keys())
name_of_optimizers=list(optimizer_dict.keys())
name_of_criterions=list(criterion_dict.keys())
def send_main_model_to_nodes_and_update_model_dict(main_model, model_dict, number_of_samples):
    for i in range(number_of_samples):

        model_dict[name_of_models[i]].weights=main_model.getweight([])
        model_dict[name_of_models[i]].bias =main_model.getbias([])
    
    return model_dict

main_model=model()
model_dict=send_main_model_to_nodes_and_update_model_dict(main_model, model_dict, number_of_samples)
print(model_dict)
'''
