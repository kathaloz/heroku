import numpy as np
import pickle
from flask import Flask, jsonify, request, json
from sklearn import preprocessing

def abrirModelo(ruta):
    modelo = pickle.load(open(ruta,'rb' ))
    return modelo

def transformarPCA(pca,X):
    return pca.transform(X)

def predecirSVM(modelo,X):
    return modelo.predict(X)
    
def predecirRED(modelo,X):
    return modelo.predict_classes(X)

def predecirKNN(modelo,X):
    y = modelo.predict(X)
    return pd.DataFrame(y).idxmax(axis=1)

app = Flask(__name__)

@app.route('/modeloRedsinPCA',methods=['POST'])
def modeloRedsinPCA():
    content = request.get_json()
    aux = [[content['var1'],content['var2'],content['var3'],content['var4'],content['var5'],content['var6'], content['var7'],content['var8'],content['var9'],content['var10'],content['var11'],content['var12'],content['var13'],content['var14'],content['var15'],content['var16'],content['var17']]]         
    X1 = np.array(aux)
    X1= preprocessing.scale(X1)
    result= predecirRED(abrirModelo('modeloRN.h5'),X1)
    d={'prediccion':str(result[0])}
    response = app.response_class(
        response=json.dumps(d),
        status = 200,
        mimetype='application/json'
    )
    return response



if __name__ == "__main__":
    app.run()
