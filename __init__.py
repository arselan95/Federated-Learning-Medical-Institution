from flask import Flask,request,redirect,url_for,render_template
import subprocess
import pymysql
import requests

app = Flask(__name__)
conn=None
#conn==pymysql.connect()

@app.route("/login", methods = ['POST', 'GET'])
def login():
	if request.method=='GET':
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
			return "correct credentials - login node 2"

		if(user=="node3institution" and password=="node3123"):
			return "correct credentials - login node 3"


		return render_template('loginpage.html', data="wrong credentials")
		
if __name__ == "__main__":
    app.run()