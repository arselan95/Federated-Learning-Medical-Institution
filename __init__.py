from flask import Flask,request,redirect,url_for,render_template,session
import subprocess
import pymysql
import requests
import subprocess
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt,mpld3
from matplotlib.pyplot import figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.widgets import Cursor
import numpy as np
import csv
import pickle

app = Flask(__name__)
conn=None
app.secret_key = 'super secret key'
#conn==pymysql.connect()

@app.route("/login", methods = ['POST', 'GET'])
def login():
	if request.method=='GET':
		session.clear()
		return render_template('loginpage.html', data="")
	if request.method=="POST":
		user=request.form['username']
		password=request.form['password']
		user=user.replace(" ","")
		password=password.replace(" ","")
		user=str(user)
		password=str(password)

		if(user=="node2institution" and password=="node2123"):
			print("here")
			session['node2session']="1"
			if 'node2session' in session:
				print(session['node2session'])
				conn=None
				conn=pymysql.connect(
				    host='localhost',
				    user='root',
				    password='',
				    db='node2')
				cursor=None
				cursor=conn.cursor()
				sql0='select jobstatus from node2info where node2id=1;'
				cursor.execute(sql0)
				row=cursor.fetchone()
				jobupdate=row[0]
				cursor.close()
				conn.close()

				counties = []
				beds = []
				covidpatients=[]
				icubeds=[]
				icucovid=[]
				suspectedcovid=[]
				  
				with open('updatedbayareanode2.pickle','rb') as f1:
					tempframe = pickle.load(f1)
				for i in range(2000):
					counties.append(tempframe['county'][i])
					beds.append(tempframe['all_hospital_beds'][i])
					icubeds.append(tempframe['icu_available_beds'][i])
				beds=np.array(beds)
				counties=np.array(counties)
				icubeds=np.array(icubeds)

				fignode2data,axnode2data=plt.subplots(figsize=(25,5))
				plt.scatter(counties, icubeds)
				plt.xlabel('Counties')
				plt.ylabel('ICU Beds')
				plt.legend(prop={'size': 30})
				#mpld3.show()
				node2piepngname="static"+"/"+"node2datasetscatter"+".png"
				plt.savefig(node2piepngname)

				fignode2data2,axnode2data2=plt.subplots(figsize=(25,5))
				plt.bar(counties, beds, color = 'lightblue', width = 0.72, label = "beds vs counties")
				plt.xlabel('Counties')
				plt.ylabel('Beds')
				plt.legend(prop={'size': 16})
				node2databarpngname="static"+"/"+"node2datasetbar"+".png"
				plt.savefig(node2databarpngname)



				if(jobupdate=="started"):
					return render_template('dashboard.html',data="node2", predictionstatus="running")
				else:
					return render_template('dashboard.html',data="node2", predictionstatus="completed")


		if(user=="node3institution" and password=="node3123"):
			session['node3session']="2"
			if 'node3session' in session:
				print(session['node3session'])
				conn=None
				conn=pymysql.connect(
				    host='localhost',
				    user='root',
				    password='',
				    db='node3')
				cursor=None
				cursor=conn.cursor()
				sqlnode30='select jobstatus from node3info where node3id=2;'
				cursor.execute(sqlnode30)
				row=cursor.fetchone()
				jobupdate=row[0]
				cursor.close()
				conn.close()

				counties = []
				beds = []
				covidpatients=[]
				icubeds=[]
				icucovid=[]
				suspectedcovid=[]
				  
				with open('updatedbayareanode3.pickle','rb') as f1:
					tempframe = pickle.load(f1)
				for i in range(2000):
					counties.append(tempframe['county'][i])
					beds.append(tempframe['all_hospital_beds'][i])
					icubeds.append(tempframe['icu_available_beds'][i])
				beds=np.array(beds)
				counties=np.array(counties)
				icubeds=np.array(icubeds)

				fignode2data,axnode2data=plt.subplots(figsize=(25,5))
				plt.scatter(counties, icubeds)
				plt.xlabel('Counties')
				plt.ylabel('ICU Beds')
				plt.legend(prop={'size': 30})
				#mpld3.show()
				node3piepngname="static"+"/"+"node3datasetscatter"+".png"
				plt.savefig(node3piepngname)

				fignode3data3,axnode3data3=plt.subplots(figsize=(25,5))
				plt.bar(counties, beds, color = 'lightblue', width = 0.72, label = "beds vs counties")
				plt.xlabel('Counties')
				plt.ylabel('Beds')
				plt.legend(prop={'size': 16})
				node3databarpngname="static"+"/"+"node3datasetbar"+".png"
				plt.savefig(node3databarpngname)

				if(jobupdate=="started"):
					return render_template('dashboard.html',data="node3", predictionstatus="running")
				else:
					return render_template('dashboard.html',data="node3", predictionstatus="completed")

		if(user=="admin" and password=="admin123"):
			print("here")
			session['adminsession']="3"
			if 'adminsession' in session:
				print(session['adminsession'])
				runningjobs=[]
				completedjobs=[]
				conn=None
				conn=pymysql.connect(
				    host='localhost',
				    user='root',
				    password='',
				    db='globalnode')
				cursor=None
				cursor=conn.cursor()
				sqladmin0='select * from managenodes;'
				cursor.execute(sqladmin0)
				records=cursor.fetchall()
				for row in range(len(records)):
					index=row+1
					nodename=records[row][0]
					print(nodename)
					starttime=records[row][2]
					jobstatus=records[row][4]
					if (jobstatus=="running"):
						temprunningjob={"index":index,"node":nodename, "starttime":starttime,"status":jobstatus}
						runningjobs.append(temprunningjob)
					else:
						totaltime=records[row][3]
						tempcompletedjob={"index":index,"node":nodename, "starttime":starttime,"status":jobstatus,"totaltime":totaltime}
						completedjobs.append(tempcompletedjob)
				cursor.close()
				conn.close()
				print(runningjobs)
				print("---")
				print(completedjobs)

				return render_template('admin.html',runjobs=runningjobs,donejobs=completedjobs)

		else:
			return render_template('loginpage.html', data="wrong credentials")

@app.route("/home", methods = ['POST', 'GET'])
def home():
	if request.method=="GET":
		if 'node2session' in session:
				conn=None
				conn=pymysql.connect(
				    host='localhost',
				    user='root',
				    password='',
				    db='node2')
				cursor=None
				cursor=conn.cursor()
				sqlhomenode2='select jobstatus from node2info where node2id=1;'
				cursor.execute(sqlhomenode2)
				row=cursor.fetchone()
				jobupdate=row[0]
				cursor.close()
				conn.close()
				if(jobupdate=="started"):
					return render_template('dashboard.html',data="node2", predictionstatus="running")
				else:
					return render_template('dashboard.html',data="node2", predictionstatus="completed")

		if 'node3session' in session:
				conn=None
				conn=pymysql.connect(
				    host='localhost',
				    user='root',
				    password='',
				    db='node3')
				cursor=None
				cursor=conn.cursor()
				sqlhomenode3='select jobstatus from node3info where node3id=2;'
				cursor.execute(sqlhomenode3)
				row=cursor.fetchone()
				jobupdate=row[0]
				cursor.close()
				conn.close()
				if(jobupdate=="started"):
					return render_template('dashboard.html',data="node3", predictionstatus="running")
				else:
					return render_template('dashboard.html',data="node3", predictionstatus="completed")

		if 'adminsession' in session:
			print("her2")
			print(session['adminsession'])
			runningjobs=[]
			completedjobs=[]
			conn=None
			conn=pymysql.connect(
			    host='localhost',
			    user='root',
			    password='',
			    db='globalnode')
			cursor=None
			cursor=conn.cursor()
			sqladmin1='select * from managenodes;'
			cursor.execute(sqladmin1)
			records=cursor.fetchall()
			for row in range(len(records)):
				index=row+1
				nodename=records[row][0]
				print(nodename)
				starttime=records[row][2]
				jobstatus=records[row][4]
				if (jobstatus=="running"):
					temprunningjob={"index":index,"node":nodename, "starttime":starttime,"status":jobstatus}
					runningjobs.append(temprunningjob)
				else:
					totaltime=records[row][3]
					tempcompletedjob={"index":index,"node":nodename, "starttime":starttime,"status":jobstatus,"totaltime":totaltime}
					completedjobs.append(tempcompletedjob)
			cursor.close()
			conn.close()
			print(runningjobs)
			print("---")
			print(completedjobs)

			return render_template('admin.html',runjobs=runningjobs,donejobs=completedjobs)



@app.route("/submitPredictionJob", methods = ['POST', 'GET'])
def submitPredictionJob():
	if request.method=="POST":
		if 'node2session' in session:
			predictiontypeinput=request.form['pred']
			print(predictiontypeinput)
			conn=None
			conn=pymysql.connect(
			    host='localhost',
			    user='root',
			    password='',
			    db='node2')
			cursor=None
			cursor=conn.cursor()
			sql1='update node2info set predictiontype=%s, xpredvalues=null, ypredvalues=null, dataloss=null,jobstatus=%s where node2id=1'
			sql1where=(predictiontypeinput, "started")
			cursor.execute(sql1,sql1where)
			conn.commit()
			cursor.close()
			conn.close()
			subprocess.Popen(["python3","node2.py"])
			return render_template ('dashboard.html', data="node2", predictionstatus="running")

		if 'node3session' in session:
			predictiontypeinput=request.form['pred']
			print(predictiontypeinput)
			conn=None
			conn=pymysql.connect(
			    host='localhost',
			    user='root',
			    password='',
			    db='node3')
			cursor=None
			cursor=conn.cursor()
			sql2='update node3info set predictiontype=%s, xpredvalues=null, ypredvalues=null, dataloss=null,jobstatus=%s where node3id=2'
			sql2where=(predictiontypeinput, "started")
			cursor.execute(sql2,sql2where)
			conn.commit()
			cursor.close()
			conn.close()
			subprocess.Popen(["python3","node3.py"])
			return render_template ('dashboard.html', data="node3", predictionstatus="running")

@app.route("/viewJobList", methods = ['POST', 'GET'])
def viewJobList():
	if request.method=="GET":
		if 'node2session' in session:
			finaljoblist=[]
			conn=None
			conn=pymysql.connect(
			    host='localhost',
			    user='root',
			    password='',
			    db='globalnode')
			cursor=None
			cursor=conn.cursor()
			sqljobnode2='select * from managenodes where nodename="node2" order by starttime DESC;'
			cursor.execute(sqljobnode2)
			row=cursor.fetchall()
			for jobs in range(len(row)):
				index=jobs+1
				print(index)
				timestart=row[jobs][2]
				status=row[jobs][4]
				print(row[jobs][5])
				joblink="/viewJobList/"+row[jobs][5]
				if status=="running":
					totaltime= "-"
				else:
					totaltime=row[jobs][3]
					totaltime=float(totaltime)/50
				tempjoblist={"No.": index, "timestarted": timestart, "status": status, "totaltime": totaltime, "joblink" : joblink}
				finaljoblist.append(tempjoblist)
			cursor.close()
			conn.close()
			return render_template("joblist.html",jobslist=finaljoblist, data="node2")
		
		if 'node3session' in session:
			finaljoblist=[]
			conn=None
			conn=pymysql.connect(
			    host='localhost',
			    user='root',
			    password='',
			    db='globalnode')
			cursor=None
			cursor=conn.cursor()
			sqljobnode3='select * from managenodes where nodename="node3" order by managenodes.starttime DESC;'
			cursor.execute(sqljobnode3)
			row=cursor.fetchall()
			for jobs in range(len(row)):
				index=jobs+1
				print(index)
				timestart=row[jobs][2]
				status=row[jobs][4]
				print(row[jobs][5])
				joblink="/viewJobList/"+row[jobs][5]
				if status=="running":
					totaltime= "-"
				else:
					totaltime=row[jobs][3]
					totaltime=float(totaltime)/50
				tempjoblist={"No.": index, "timestarted": timestart, "status": status, "totaltime": totaltime, "joblink" : joblink}
				finaljoblist.append(tempjoblist)
			cursor.close()
			conn.close()
			return render_template("joblist.html",jobslist=finaljoblist, data="node3")

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]			

@app.route("/viewJobList/<jobid>", methods = ['POST', 'GET'])
def viewJob(jobid):
	if request.method=="GET":
		if 'node2session' in session:
			conn=None
			conn=pymysql.connect(
			    host='localhost',
			    user='root',
			    password='',
			    db='node2')
			cursor=None
			cursor=conn.cursor()
			sqlnode2job="Select * from node2predictions where jobid=%s"
			sqlnode2jobwhere=(jobid)
			cursor.execute(sqlnode2job,sqlnode2jobwhere)
			row=cursor.fetchone()
			
			predictiontype=row[3]
			
			xvalues=row[1]
			xvalues=json.loads(xvalues)
			xvalues=xvalues['xvalues']
			
			yvalues=row[2]
			yvalues=json.loads(yvalues)
			yvalues=yvalues['yvalues']
			
			dataloss=row[4]
			dataloss=json.loads(dataloss)
			dataloss=dataloss['loss']
			print(dataloss)

			senddataloss=[]
			for los in range(len(dataloss)):
				if predictiontype=="beds":
					templos={"index":los,"error":dataloss[los]}
				else:
					templos={"index":los,"error":dataloss[los]*100}
				senddataloss.append(templos)

			xvalues=np.array(xvalues)
			yvalues=np.array(yvalues)
			

			fig,ax=plt.subplots(figsize=(25,5))
			plt.plot(yvalues, label="Predicted")
			plt.plot(xvalues, label='Original')
			plt.legend(prop={'size': 30})
			pltpngname="static"+"/"+"node2plot"+jobid+".png"
			plt.savefig(pltpngname)

			fig2,ax2=plt.subplots(figsize=(25,5))
			plt.scatter(xvalues,yvalues)
			plt.xlabel('Original')
			plt.ylabel('Predictions')
			plt.legend(prop={'size': 30})
			sctpngname="static"+"/"+"node2scatter"+jobid+".png"
			plt.savefig(sctpngname)
			
			fig3,ax3=plt.subplots(figsize=(25,5))
			plt.bar(xvalues, yvalues, color = 'g', width = 0.72, label = "Predictions")
			plt.xlabel('Original')
			plt.ylabel('Predictions')
			plt.legend(prop={'size': 30})
			barpngname="static"+"/"+"node2bar"+jobid+".png"
			plt.savefig(barpngname)

			
			fig4,ax4=plt.subplots(figsize=(25,5))
			plt.plot(xvalues, yvalues, color = 'g', linestyle = 'dashed',marker = 'o',label = "Predictions")
			plt.xlabel('Original')
			plt.ylabel('Predicted')
			plt.grid()
			plt.legend(prop={'size': 30})
			gridpngname="static"+"/"+"node2grid"+jobid+".png"
			plt.savefig(gridpngname)

			fig5,ax5=plt.subplots(1, figsize=(25,5))
			plt.plot(xvalues,yvalues,color='blue')
			plt.xlabel('Original')
			plt.ylabel('Predicted')
			#mpld3.show()

			slopechartpngname="static"+"/"+"node2slopechart"+jobid+".png"
			plt.savefig(gridpngname)
			

			csvheader=['Predicted Beds']
			csvdata=[]
			for y in yvalues:
				temp=[]
				temp.append(y)
				csvdata.append(temp)

			with open('updatedbayareanode2.pickle','rb') as f1:
				tempframe = pickle.load(f1)
			tempframe=tempframe.head(len(yvalues))


			
			#plt.plot(dates, label='Original')
			#plt.legend(prop={'size': 30})
			#pltpredpngname="static"+"/"+"node2plotpred"+jobid+".png"
			#plt.savefig(pltpredpngname)
			
			

			tempframe[predictiontype]=xvalues
			tempframe['Predictions']=yvalues
			tempframe2=tempframe['Predictions'].unique()

			tempframe3=tempframe['icu_covid_confirmed_patients'].unique()
			tempframe4=tempframe['hospitalized_covid_confirmed_patients'].unique()
			tempframe5=tempframe['all_hospital_beds'].unique()
			tempframe6=tempframe['icu_available_beds'].unique()
			tempframe7=tempframe[predictiontype].unique()

			tempframe3=np.sort(tempframe3)
			tempframe4=np.sort(tempframe4)
			tempframe5=np.sort(tempframe5)
			tempframe6=np.sort(tempframe6)
			tempframe7=np.sort(tempframe7)
			tempframe2=np.sort(tempframe2)
			

			if(predictiontype=='icubeds'):
				tempframe2=tempframe2[::-1]
				dates=[]
				beds=[]
				for j in range(len(tempframe3)):
					dates.append(tempframe3[j])
					beds.append(tempframe2[j])

				dates=np.array(dates)
				beds=np.array(beds)


				fig6,ax6=plt.subplots(1,figsize=(25,5))
				plt.plot(dates,beds)
				plt.xlabel('ICU COVID Patients')
				plt.ylabel('Predicted ICU Beds')
				print(len(tempframe2))
				node2icubedpngname="static"+"/"+"node2icubed"+jobid+".png"
				plt.savefig(node2icubedpngname)
				#mpld3.show()

			if(predictiontype=='beds'):
				tempframe2=tempframe2[::-1]
				alldates=[]
				allbeds=[]
				for j in range(len(tempframe4)):
					alldates.append(tempframe4[j])
					allbeds.append(tempframe2[j])

				alldates=np.array(alldates)
				allbeds=np.array(allbeds)


				fig7,ax7=plt.subplots(1,figsize=(25,5))
				plt.plot(allbeds,alldates)
				plt.xlabel('COVID Patients')
				plt.ylabel('Predicted Beds')
				print(len(tempframe2))
				node2bedpngname="static"+"/"+"node2bed"+jobid+".png"
				plt.savefig(node2bedpngname)
				#mpld3.show()

			csvfilename="static"+"/"+"node2csvpredictions"+jobid+".csv"
			tempframe.to_csv(csvfilename)
			
			pltlink="/viewplt/"+jobid
			sctlink="/viewsct/"+jobid
			barlink="/viewbar/"+jobid
			gridlink="/viewgrid/"+jobid


			return render_template("viewjob.html", data="node2", jobid=jobid, predictiontype=predictiontype, meanloss=senddataloss,
				pltlink=pltlink, sctlink=sctlink, barlink=barlink, gridlink=gridlink)

		if 'node3session' in session:
			conn=None
			conn=pymysql.connect(
			    host='localhost',
			    user='root',
			    password='',
			    db='node3')
			cursor=None
			cursor=conn.cursor()
			sqlnode3job="Select * from node3predictions where jobid=%s"
			sqlnode3jobwhere=(jobid)
			cursor.execute(sqlnode3job,sqlnode3jobwhere)
			row=cursor.fetchone()
			
			predictiontype=row[3]
			
			x3values=row[1]
			x3values=json.loads(x3values)
			x3values=x3values['xvalues']
			
			y3values=row[2]
			y3values=json.loads(y3values)
			y3values=y3values['yvalues']
			
			data3loss=row[4]
			data3loss=json.loads(data3loss)
			data3loss=data3loss['loss']
			print(data3loss)

			senddata3loss=[]
			for los in range(len(data3loss)):
				if predictiontype=="beds":
					templos={"index":los,"error":data3loss[los]}
				else:
					templos={"index":los,"error":data3loss[los]*10}
				senddata3loss.append(templos)

			x3values=np.array(x3values)
			y3values=np.array(y3values)
			

			fig3,ax3=plt.subplots(figsize=(25,5))
			plt.plot(y3values, label="Predicted")
			plt.plot(x3values, label='Original')
			plt.legend(prop={'size': 30})
			plt3pngname="static"+"/"+"node3plot"+jobid+".png"
			plt.savefig(plt3pngname)

			fig23,ax23=plt.subplots(figsize=(25,5))
			plt.scatter(x3values,y3values)
			plt.xlabel('Original')
			plt.ylabel('Predictions')
			plt.legend(prop={'size': 30})
			sct3pngname="static"+"/"+"node3scatter"+jobid+".png"
			plt.savefig(sct3pngname)
			
			fig33,ax33=plt.subplots(figsize=(25,5))
			plt.bar(x3values, y3values, color = 'g', width = 0.72, label = "Predictions")
			plt.xlabel('Original')
			plt.ylabel('Predictions')
			plt.legend(prop={'size': 30})
			bar3pngname="static"+"/"+"node3bar"+jobid+".png"
			plt.savefig(bar3pngname)

			
			fig43,ax43=plt.subplots(figsize=(25,5))
			plt.plot(x3values, y3values, color = 'g', linestyle = 'dashed',marker = 'o',label = "Predictions")
			plt.xlabel('Original')
			plt.ylabel('Predicted')
			plt.grid()
			plt.legend(prop={'size': 30})
			grid3pngname="static"+"/"+"node3grid"+jobid+".png"
			plt.savefig(grid3pngname)
			

			csv3header=['Predicted Beds']
			csv3data=[]
			for y in y3values:
				temp=[]
				temp.append(y)
				csv3data.append(temp)

			with open('updatedbayareanode3.pickle','rb') as f1:
				tempframe = pickle.load(f1)
			tempframe=tempframe.head(len(y3values))
			print(tempframe)
			
			tempframe[predictiontype]=x3values
			tempframe['Predictions']=y3values
			tempframe2=tempframe['Predictions'].unique()

			tempframe3=tempframe['icu_covid_confirmed_patients'].unique()
			tempframe4=tempframe['hospitalized_covid_confirmed_patients'].unique()
			tempframe5=tempframe['all_hospital_beds'].unique()
			tempframe6=tempframe['icu_available_beds'].unique()

			tempframe3=np.sort(tempframe3)
			tempframe4=np.sort(tempframe4)
			tempframe5=np.sort(tempframe5)
			tempframe6=np.sort(tempframe6)
			tempframe2=np.sort(tempframe2)
			

			if(predictiontype=='icubeds'):
				tempframe2=tempframe2[::-1]
				dates=[]
				beds=[]
				for j in range(len(tempframe3)):
					dates.append(tempframe3[j])
					beds.append(tempframe2[j])

				dates=np.array(dates)
				beds=np.array(beds)


				fig36,ax36=plt.subplots(1,figsize=(25,5))
				plt.plot(dates,beds)
				plt.xlabel('ICU COVID Patients')
				plt.ylabel('Predicted ICU Beds')
				print(len(tempframe2))
				node3icubedpngname="static"+"/"+"node3icubed"+jobid+".png"
				plt.savefig(node3icubedpngname)
				#mpld3.show()

			if(predictiontype=='beds'):
				tempframe2=tempframe2[::-1]
				alldates=[]
				allbeds=[]
				for j in range(len(tempframe4)):
					alldates.append(tempframe4[j])
					allbeds.append(tempframe2[j])

				alldates=np.array(alldates)
				allbeds=np.array(allbeds)


				fig37,ax37=plt.subplots(1,figsize=(25,5))
				plt.plot(alldates,allbeds)
				plt.xlabel('COVID Patients')
				plt.ylabel('Predicted Beds')
				print(len(tempframe2))
				node3bedpngname="static"+"/"+"node3bed"+jobid+".png"
				plt.savefig(node3bedpngname)


			tempframe[predictiontype]=x3values
			tempframe['Predictions']=y3values
			csv3filename="static"+"/"+"node3csvpredictions"+jobid+".csv"
			tempframe.to_csv(csv3filename)
			
			plt3link="/viewplt/"+jobid
			sct3link="/viewsct/"+jobid
			bar3link="/viewbar/"+jobid
			grid3link="/viewgrid/"+jobid


			return render_template("viewjob.html", data="node3", jobid=jobid, predictiontype=predictiontype, meanloss=senddata3loss,
				pltlink=plt3link, sctlink=sct3link, barlink=bar3link, gridlink=grid3link)


@app.route("/viewplt/<jobid>", methods = ['POST', 'GET'])
def viewplt(jobid):
	if request.method=="GET":
		if 'node2session' in session:
			conn=None
			conn=pymysql.connect(
			    host='localhost',
			    user='root',
			    password='',
			    db='node2')
			cursor=None
			cursor=conn.cursor()
			sqlnode2plot="Select * from node2predictions where jobid=%s"
			sqlnode2plotwhere=(jobid)
			cursor.execute(sqlnode2plot,sqlnode2plotwhere)
			row=cursor.fetchone()
						
			xvalues=row[1]
			xvalues=json.loads(xvalues)
			xvalues=xvalues['xvalues']
			
			yvalues=row[2]
			yvalues=json.loads(yvalues)
			yvalues=yvalues['yvalues']

			xvalues=np.array(xvalues)
			yvalues=np.array(yvalues)

			figplot,axplot=plt.subplots(figsize=(25,5))
			plt.plot(yvalues, label="Predicted")
			plt.plot(xvalues, label='Original')
			plt.legend(prop={'size': 30})
			mpld3.show()

		if 'node3session' in session:
			conn=None
			conn=pymysql.connect(
			    host='localhost',
			    user='root',
			    password='',
			    db='node3')
			cursor=None
			cursor=conn.cursor()
			sqlnode3plot="Select * from node3predictions where jobid=%s"
			sqlnode3plotwhere=(jobid)
			cursor.execute(sqlnode3plot,sqlnode3plotwhere)
			row=cursor.fetchone()
						
			xvalues=row[1]
			xvalues=json.loads(xvalues)
			xvalues=xvalues['xvalues']
			
			yvalues=row[2]
			yvalues=json.loads(yvalues)
			yvalues=yvalues['yvalues']

			xvalues=np.array(xvalues)
			yvalues=np.array(yvalues)

			fig3plot,ax3plot=plt.subplots(figsize=(25,5))
			plt.plot(yvalues, label="Predicted")
			plt.plot(xvalues, label='Original')
			plt.legend(prop={'size': 30})
			mpld3.show()

@app.route("/viewsct/<jobid>", methods = ['POST', 'GET'])
def viewsct(jobid):
	if request.method=="GET":
		if 'node2session' in session:
			conn=None
			conn=pymysql.connect(
			    host='localhost',
			    user='root',
			    password='',
			    db='node2')
			cursor=None
			cursor=conn.cursor()
			sqlnode2sct="Select * from node2predictions where jobid=%s"
			sqlnode2sctwhere=(jobid)
			cursor.execute(sqlnode2sct,sqlnode2sctwhere)
			row=cursor.fetchone()
						
			xvalues=row[1]
			xvalues=json.loads(xvalues)
			xvalues=xvalues['xvalues']
			
			yvalues=row[2]
			yvalues=json.loads(yvalues)
			yvalues=yvalues['yvalues']

			xvalues=np.array(xvalues)
			yvalues=np.array(yvalues)

			figsct,axsct=plt.subplots(figsize=(25,5))
			plt.scatter(xvalues,yvalues)
			plt.xlabel('Original')
			plt.ylabel('Predictions')
			plt.legend(prop={'size': 30})
			mpld3.show()

		if 'node3session' in session:
			conn=None
			conn=pymysql.connect(
			    host='localhost',
			    user='root',
			    password='',
			    db='node3')
			cursor=None
			cursor=conn.cursor()
			sqlnode3sct="Select * from node3predictions where jobid=%s"
			sqlnode3sctwhere=(jobid)
			cursor.execute(sqlnode3sct,sqlnode3sctwhere)
			row=cursor.fetchone()
						
			xvalues=row[1]
			xvalues=json.loads(xvalues)
			xvalues=xvalues['xvalues']
			
			yvalues=row[2]
			yvalues=json.loads(yvalues)
			yvalues=yvalues['yvalues']

			xvalues=np.array(xvalues)
			yvalues=np.array(yvalues)

			fig3sct,ax3sct=plt.subplots(figsize=(25,5))
			plt.scatter(xvalues,yvalues)
			plt.xlabel('Original')
			plt.ylabel('Predictions')
			plt.legend(prop={'size': 30})
			mpld3.show()

@app.route("/viewbar/<jobid>", methods = ['POST', 'GET'])
def viewbar(jobid):
	if request.method=="GET":
		if 'node2session' in session:
			conn=None
			conn=pymysql.connect(
			    host='localhost',
			    user='root',
			    password='',
			    db='node2')
			cursor=None
			cursor=conn.cursor()
			sqlnode2bar="Select * from node2predictions where jobid=%s"
			sqlnode2barwhere=(jobid)
			cursor.execute(sqlnode2bar,sqlnode2barwhere)
			row=cursor.fetchone()
						
			xvalues=row[1]
			xvalues=json.loads(xvalues)
			xvalues=xvalues['xvalues']
			
			yvalues=row[2]
			yvalues=json.loads(yvalues)
			yvalues=yvalues['yvalues']

			xvalues=np.array(xvalues)
			yvalues=np.array(yvalues)

			figbar,axbar=plt.subplots(figsize=(25,5))
			plt.bar(xvalues, yvalues, color = 'g', width = 0.72, label = "Predictions")
			plt.xlabel('Original')
			plt.ylabel('Predictions')
			plt.legend(prop={'size': 30})
			mpld3.show()

		if 'node3session' in session:
			conn=None
			conn=pymysql.connect(
			    host='localhost',
			    user='root',
			    password='',
			    db='node3')
			cursor=None
			cursor=conn.cursor()
			sqlnode3bar="Select * from node3predictions where jobid=%s"
			sqlnode3barwhere=(jobid)
			cursor.execute(sqlnode3bar,sqlnode3barwhere)
			row=cursor.fetchone()
						
			xvalues=row[1]
			xvalues=json.loads(xvalues)
			xvalues=xvalues['xvalues']
			
			yvalues=row[2]
			yvalues=json.loads(yvalues)
			yvalues=yvalues['yvalues']

			xvalues=np.array(xvalues)
			yvalues=np.array(yvalues)

			fig3bar,ax3bar=plt.subplots(figsize=(25,5))
			plt.bar(xvalues, yvalues, color = 'g', width = 0.72, label = "Predictions")
			plt.xlabel('Original')
			plt.ylabel('Predictions')
			plt.legend(prop={'size': 30})
			mpld3.show()

@app.route("/viewgrid/<jobid>", methods = ['POST', 'GET'])
def viewgrid(jobid):
	if request.method=="GET":
		if 'node2session' in session:
			conn=None
			conn=pymysql.connect(
			    host='localhost',
			    user='root',
			    password='',
			    db='node2')
			cursor=None
			cursor=conn.cursor()
			sqlnode2grid="Select * from node2predictions where jobid=%s"
			sqlnode2gridwhere=(jobid)
			cursor.execute(sqlnode2grid,sqlnode2gridwhere)
			row=cursor.fetchone()
						
			xvalues=row[1]
			xvalues=json.loads(xvalues)
			xvalues=xvalues['xvalues']
			
			yvalues=row[2]
			yvalues=json.loads(yvalues)
			yvalues=yvalues['yvalues']

			xvalues=np.array(xvalues)
			yvalues=np.array(yvalues)

			figgrid,axgrid=plt.subplots(figsize=(25,5))
			plt.plot(xvalues, yvalues, color = 'g', linestyle = 'dashed',marker = 'o',label = "Predictions")
			plt.xlabel('Original')
			plt.ylabel('Predicted')
			plt.grid()
			plt.legend(prop={'size': 30})
			mpld3.show()

		if 'node3session' in session:
			conn=None
			conn=pymysql.connect(
			    host='localhost',
			    user='root',
			    password='',
			    db='node3')
			cursor=None
			cursor=conn.cursor()
			sqlnode3grid="Select * from node3predictions where jobid=%s"
			sqlnode3gridwhere=(jobid)
			cursor.execute(sqlnode3grid,sqlnode3gridwhere)
			row=cursor.fetchone()
						
			xvalues=row[1]
			xvalues=json.loads(xvalues)
			xvalues=xvalues['xvalues']
			
			yvalues=row[2]
			yvalues=json.loads(yvalues)
			yvalues=yvalues['yvalues']

			xvalues=np.array(xvalues)
			yvalues=np.array(yvalues)

			fig3grid,ax3grid=plt.subplots(figsize=(25,5))
			plt.plot(xvalues, yvalues, color = 'g', linestyle = 'dashed',marker = 'o',label = "Predictions")
			plt.xlabel('Original')
			plt.ylabel('Predicted')
			plt.grid()
			plt.legend(prop={'size': 30})
			mpld3.show()



@app.route("/logout", methods = ['POST', 'GET'])
def logout():
	if request.method=="POST":
		if 'node2session' in session:
			session.pop('node2session')
			return render_template ('loginpage.html', data="")
		if 'node3session' in session:
			session.pop('node3session')
			return render_template ('loginpage.html', data="")
		if 'adminsession' in session:
			session.pop('adminsession')
			return render_template ('loginpage.html', data="")



			
if __name__ == "__main__":
    app.run()