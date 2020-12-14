from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from django.http import HttpResponse, JsonResponse
import json
import os


def survie(request):
    module_dir = os.path.dirname(__file__)  
    file_path = os.path.join(module_dir, 'titanic3.xls')  
    data = pd.read_excel(file_path)
    data = data.drop(['name','fare','body','sibsp','parch','ticket','cabin','embarked','boat','body','home.dest'],axis=1)
    data.dropna(axis = 0 , inplace=True)
    data['sex'].replace(['male','female'],[0,1],inplace=True)
    #model = KNeighborsClassifier()
    model = KNeighborsClassifier()
    y = data['survived']
    X = data.drop('survived',axis=1)
    model.fit(X,y)
    model.score(X,y)

    pclass = request.GET.get('class',None)
    sex = request.GET.get('sexe',None)
    age = request.GET.get('aage',None)
    print(pclass,sex,age)
    x = np.array([pclass,sex,age]).reshape(1,3)

    predict = model.predict(x)
    prob = model.predict_proba(x)
    probabili = prob[0]

    prediction = predict[0]
    if prediction == 1:
        probabilite = probabili[1]
    else:
        probabilite = probabili[0]

    # print(model.predict(x))
    # print("la probabilite est de ",model.predict_proba(x))
    json_data = json.dumps({'prediction':int(prediction),'probabilite':float(probabilite)})
    # json_data = json.dumps({'prediction': 2, 'probabilite': 5})

    return HttpResponse(json_data)
