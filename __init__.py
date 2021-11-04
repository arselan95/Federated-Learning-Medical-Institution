from flask import Flask,request,redirect,url_for,render_template,session
import subprocess
import pymysql
import requests

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

			return render_template('sessiontest.html')

		if(user=="node3institution" and password=="node3123"):
			session['node3session']="2"
			if 'node3session' in session:
				print(session['node3session'])
			return render_template('sessiontest.html')


		else:
			return render_template('loginpage.html', data="wrong credentials")

'''
@app.route("/testsession", methods = ['POST', 'GET'])
def testsession():
	if request.method=="GET":
		print(session)
		return "session"
	if request.method=="POST":

		print(session)

		if 'node3session' in session:
			print("session3")
			return "node3"
		
		if 'node2session' in session:
			print("session2")
			return "node2"
'''
			
if __name__ == "__main__":
    app.run()