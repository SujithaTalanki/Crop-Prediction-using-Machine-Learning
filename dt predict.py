import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from flask import Flask, request, render_template
app=Flask(__name__)
@app.route('/')
@app.route('/home')
def home():
     return render_template("predict.html")
@app.route('/pred', methods =["POST", "GET"])
def pred():
    n_ip=request.form.get("N")
    p_ip=request.form.get("P")
    k_ip=request.form.get("K")
    temp_ip=request.form.get("temp")
    hum_ip=request.form.get("hum")
    rain_ip=request.form.get("rain")
    ph_ip=request.form.get("ph")
    df=pd.read_csv("Crop_recommendation.csv")
    features = df[['N','P','K','temperature','humidity','ph','rainfall']]
    target = df['label']
    X_train,X_test,y_train,y_test = train_test_split(features,target,test_size=1,random_state = 2)
    DecisionTree=DecisionTreeClassifier(criterion='entropy',max_depth=5,random_state=2)
    DecisionTree.fit(X_train,y_train)
    n_max=df['N'].max()
    n_min=df['N'].min()
    p_max=df['P'].max()
    p_min=df['P'].min()
    k_max=df['K'].max()
    k_min=df['K'].min()
    temp_max=df['temperature'].max()
    temp_min=df['temperature'].min()
    hum_max=df['humidity'].max()
    hum_min=df['humidity'].min()
    ph_max=df['ph'].max()
    ph_min=df['ph'].min()
    rain_max=df['rainfall'].max()
    rain_min=df['rainfall'].min()
    ip=[]
    ip.append(n_ip)
    ip.append(p_ip)
    ip.append(k_ip)
    ip.append(temp_ip)
    ip.append(hum_ip)
    ip.append(ph_ip)
    ip.append(rain_ip)
    crop=DecisionTree.predict([ip])
    return render_template('predict.html',crop=crop)
        

if __name__=='__main__':
    app.run(debug=True,port=5002) 
'''n_ip=int(input("Enter Nitrogen value (should be between "+str(n_min)+" and "+str(n_max)+") :"))
p_ip=int(input("Enter Phosophorous value (should be between "+str(p_min)+" and "+str(p_max)+"):"))
k_ip=int(input("Enter Pottasium value (should be between "+str(k_min)+" and "+str(k_max)+") :"))
temp_ip=float(input("Enter Temperature value (should be between "+str(temp_min)+" and "+str(temp_max)+"):"))
hum_ip=float(input("Enter Humidity value (should be between "+str(hum_min)+" and "+str(hum_max)+") :"))
ph_ip=float(input("Enter Ph value (should be between "+str(ph_min)+" and "+str(ph_max)+") :"))
rain_ip=float(input("Enter Rainfall value (should be between "+str(rain_min)+" and "+str(rain_max)+") :"))'''
'''#KNN
knn_model=KNeighborsClassifier(n_neighbors = 3)
knn_model.fit(X_train,y_train)
knn_ip=pd.core.frame.DataFrame(ip)
print(knn_model.predict([ip]))'''
