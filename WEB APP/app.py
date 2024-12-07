from flask import Flask, request, render_template
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib  # Para guardar y cargar modelos

# Crear la aplicación Flask
app = Flask(__name__)

# Cargar el modelo entrenado
modelo_lr = joblib.load("modelo_regresion_lineal.pkl")  # Guarda tu modelo previamente

# Ruta para la página principal
@app.route("/")
def index():
    return render_template("index.html")

# Ruta para manejar el formulario y realizar predicciones
@app.route("/predecir", methods=["POST"])
def predecir():
    # Obtener los valores del formulario
    valores = {
        'FacCCPC_T12': float(request.form['FacCCPC_T12']),
        'FacCI_T12': float(request.form['FacCI_T12']),
        'FacCN_T12': float(request.form['FacCN_T12']),
        'PagoNac_T12': float(request.form['PagoNac_T12']),
        'UsoLI_T12': float(request.form['UsoLI_T12']),
        'Txs_T12': float(request.form['Txs_T12']),
        'FlgAct_T12': float(request.form['FlgAct_T12']),
    }

    # Convertir los valores a un DataFrame
    df_valores = pd.DataFrame([valores])

    # Realizar la predicción
    prediccion = modelo_lr.predict(df_valores)

    # Retornar la predicción al usuario
    return render_template("index.html", resultado=prediccion[0])

if __name__ == "__main__":
    app.run(debug=True)
