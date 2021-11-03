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
		return render_template('loginpage.html')

if __name__ == "__main__":
    app.run()