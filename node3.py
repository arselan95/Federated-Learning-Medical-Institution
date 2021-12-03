import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from numpy import random
import pickle
from matplotlib.pyplot import figure
import math
from globalmodel import model
import csv
import pymysql
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import time
import json
import uuid


#check node3 database to get type of predictions entered by user
conn=None
conn=pymysql.connect(
    host='localhost',
    user='root',
    password='',
    db='node3')

cursor=None
cursor=conn.cursor()
sql1='select predictiontype from node3info where node3id=2;'
cursor.execute(sql1)
row=cursor.fetchone()
predictiontype=row[0]
cursor.close()
conn.close()

print("testing dataabse")
print(predictiontype)

finalrecord=[]
finaltrecord=[]
default=200


'''
with open('calidata.csv', 'r') as infile, open('updatecalidata.csv', 'w',newline='') as outfile:
    csv_reader = csv.reader(infile)
    csv_writer = csv.writer(outfile)
    for line in csv_reader:
        if all(line):
            csv_writer.writerow(line)

dfile=pd.read_csv("updatecalidata.csv")
dfile.head()
pickfile=open("updatedcalidata.pickle","wb")
pickle.dump(dfile,pickfile)
'''
with open('updatedbayareanode3.pickle','rb') as f1:
    x = pickle.load(f1)
print(type(x))
print(len(x.columns))
print(x.count)

x = x[x['hospitalized_covid_confirmed_patients'] > 0 ]

x = x[x['hospitalized_suspected_covid_patients'] > 0]

x = x[x['hospitalized_covid_patients'] > 0 ]

x = x[x['all_hospital_beds'] > 0]

x = x[x['icu_covid_confirmed_patients'] > 0 ]

x = x[x['icu_suspected_covid_patients'] > 0 ]

x = x[x['icu_available_beds'] > 0  ]


x=x.reset_index()
print(x.count)

i=0

if predictiontype=='beds':
    for i in range(3000):
        default=800
        zrecord=[]
        zrecord.append(x['hospitalized_covid_confirmed_patients'][i])
        zrecord.append(x['hospitalized_suspected_covid_patients'][i])
        zrecord.append(x['hospitalized_covid_patients'][i])
        zrecord.append(x['all_hospital_beds'][i])
        zrecord.append(x['icu_covid_confirmed_patients'][i])
        zrecord.append(x['icu_suspected_covid_patients'][i])
        zrecord.append(x['icu_available_beds'][i])
        zrecord = np.asarray(zrecord)
        finalrecord.append(zrecord)
        finaltrecord.append(x['all_hospital_beds'][i])

if predictiontype=='icubeds':
    for i in range(3000):
        default=500
        zrecord=[]
        zrecord.append(x['hospitalized_covid_confirmed_patients'][i])
        zrecord.append(x['hospitalized_suspected_covid_patients'][i])
        zrecord.append(x['hospitalized_covid_patients'][i])
        zrecord.append(x['all_hospital_beds'][i])
        zrecord.append(x['icu_covid_confirmed_patients'][i])
        zrecord.append(x['icu_suspected_covid_patients'][i])
        zrecord.append(x['icu_available_beds'][i])
        zrecord = np.asarray(zrecord)
        finalrecord.append(zrecord)
        finaltrecord.append(x['icu_available_beds'][i])

if predictiontype=='covidpatients':
    for i in range(3000):
        default=600
        zrecord=[]
        zrecord.append(x['hospitalized_covid_confirmed_patients'][i])
        zrecord.append(x['hospitalized_suspected_covid_patients'][i])
        zrecord.append(x['hospitalized_covid_patients'][i])
        zrecord.append(x['all_hospital_beds'][i])
        zrecord.append(x['icu_covid_confirmed_patients'][i])
        zrecord.append(x['icu_suspected_covid_patients'][i])
        zrecord.append(x['icu_available_beds'][i])
        zrecord = np.asarray(zrecord)
        finalrecord.append(zrecord)
        finaltrecord.append(x['hospitalized_covid_patients'][i])

if predictiontype=='icupatients':
    for i in range(3000):
        default=900
        zrecord=[]
        zrecord.append(x['hospitalized_covid_confirmed_patients'][i])
        zrecord.append(x['hospitalized_suspected_covid_patients'][i])
        zrecord.append(x['hospitalized_covid_patients'][i])
        zrecord.append(x['all_hospital_beds'][i])
        zrecord.append(x['icu_covid_confirmed_patients'][i])
        zrecord.append(x['icu_suspected_covid_patients'][i])
        zrecord.append(x['icu_available_beds'][i])
        zrecord = np.asarray(zrecord)
        finalrecord.append(zrecord)
        finaltrecord.append(x['icu_covid_confirmed_patients'][i])

if predictiontype=='suspectedcovid':
    for i in range(3000):
        default=400
        zrecord=[]
        zrecord.append(x['hospitalized_covid_confirmed_patients'][i])
        zrecord.append(x['hospitalized_suspected_covid_patients'][i])
        zrecord.append(x['hospitalized_covid_patients'][i])
        zrecord.append(x['all_hospital_beds'][i])
        zrecord.append(x['icu_covid_confirmed_patients'][i])
        zrecord.append(x['icu_suspected_covid_patients'][i])
        zrecord.append(x['icu_available_beds'][i])
        zrecord = np.asarray(zrecord)
        finalrecord.append(zrecord)
        finaltrecord.append(x['hospitalized_suspected_covid_patients'][i])


finaltrecord=np.asarray(finaltrecord)
finalrecord = np.asarray(finalrecord)

x_train,x_test,y_train,y_test=train_test_split(finalrecord,finaltrecord,test_size=0.5)
# standardizing data
scaler = preprocessing.StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test=scaler.transform(x_test)

## Adding the BED Column in the data
train_data=pd.DataFrame(x_train)
train_data['predictions']=y_train
train_data.head(3)

x_test=np.array(x_test)
y_test=np.array(y_test)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

#global model database for admin
conn=None
conn=pymysql.connect(
    host='localhost',
    user='root',
    password='',
    db='globalnode')
cursor=None
wca=[]
bca=[]

#FEDERATED 
print("FEDERATED")
print(train_data.shape)
#default=800
print("default")
print(default)

#timer
node3starttime=time.time()
#jobid
jobid=uuid.uuid4()

#MysqlDatabase
cursor=conn.cursor()
sql='Insert into managenodes(nodename, nodeid, starttime, jobstatus,jobid) values (%s,2,now(),%s,%s);'
sqlinsert=("node3","running",jobid.hex)
cursor.execute(sql,sqlinsert)
conn.commit()
cursor.close()

errors=[]
#original epoch is 25
for itrca in range(25):
    itrca=itrca+1
    localmodel=model()
    
    
    main_model_bias=localmodel.getbias(bca)
    main_model_weights=localmodel.getweight(wca)
    
    if(len(wca)!=0):
        print("training locally")

        print("old weight")
        print(main_model_weights)

        print("old bias")
        print(main_model_bias)
    
    
        print("sending parameters to global model for aggregation")
        print("taking average of the nodes and sending nodes to main model")

    #local data training
    wca,bca=localmodel.SGDModel(train_data, 0.006,itrca*default,1500,1,x_test,-2,-2)
    y1ca_pred_train=localmodel.predict(x_test,wca,bca)
    
    
    #federated
    federatedmodel=model()
    newweight=federatedmodel.getweight(wca)
    newbias=federatedmodel.getbias(bca)

    avgweight=[]
    avgbias=[]

    for wei in range(len(main_model_weights)):
        main_model_weights[wei] += newweight[wei]
        main_model_weights[wei] = main_model_weights[wei] /2
        avgweight.append(main_model_weights[wei])

    for bei in range(len(main_model_bias)):
        main_model_bias[bei] += newbias[bei]
        main_model_bias[bei] = main_model_bias[bei] /2
        avgbias.append(main_model_bias[bei])
    
    avgweight=np.array(avgweight)
    avgbias=np.array(avgbias)

    print(type(avgweight))
    print("updated avg weights")
    print(avgweight)

   
    print("updated avg bias")
    print(main_model_bias)

    if(itrca<=1):
        print("data loss captured by updated weights sent by Global model")
        print((mean_squared_error(y_test,y1ca_pred_train))/10)
        loss=mean_squared_error(y_test,y1ca_pred_train)/10
        errors.append(loss)


    if(len(avgweight)!=0):
        #Predictions from Federated Model
        y2ca_pred_train=federatedmodel.predict(x_test,avgweight,avgbias)
        print("sending updated model back to the node")
        print("data loss captured by updated weights sent by Global model")
        print((mean_squared_error(y_test,y2ca_pred_train))/10)
        loss=mean_squared_error(y_test,y2ca_pred_train)/10
        errors.append(loss)

print()
print()
print("predict")
xdict={'xvalues':list(y_test)}
ydict={'yvalues':list(y2ca_pred_train)}
errlist={'loss':list(errors)}
print(y_test)
print(len(y_test))
print(y2ca_pred_train)
print(len(y2ca_pred_train))

plt.figure(figsize=(25,6))
plt.plot(y_test, label='Actual')
plt.plot(y2ca_pred_train, label='Predicted')
plt.legend(prop={'size': 16})
plt.savefig('node3predictions.png')
print("################")

totaltime=time.time()-node3starttime
totaltime=str(totaltime)

#mysql update global model for admin
cursor=conn.cursor()
sql2='update managenodes set totaltime=%s, jobstatus=%s where jobid=%s;'
sql2where=(totaltime,"completed",jobid.hex)
cursor.execute(sql2,sql2where)
conn.commit()
cursor.close()
conn.close()

#mysql update node 3 with current latest predictions
conn=None
conn=pymysql.connect(
    host='localhost',
    user='root',
    password='',
    db='node3')
cursor=None
cursor=conn.cursor()
sql3='update node3info set xpredvalues=%s, ypredvalues=%s, dataloss=%s, jobstatus=%s where node3id=2'
sql3where=(json.dumps(xdict), json.dumps(ydict),json.dumps(errlist), "completed")
cursor.execute(sql3,sql3where)
conn.commit()
cursor.close()


#mysql update node 3 prediction history
cursor=None
cursor=conn.cursor()
sql4='Insert into node3predictions(jobid,xpredvalues,ypredvalues,predictiontype,dataloss) values (%s,%s,%s,%s,%s);'
sql4insert=(jobid.hex,json.dumps(xdict), json.dumps(ydict),predictiontype,json.dumps(errlist))
cursor.execute(sql4,sql4insert)
conn.commit()
cursor.close()
conn.close()


