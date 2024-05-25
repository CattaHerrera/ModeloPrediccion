from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import sklearn.preprocessing as pp
import pickle
from azure.storage.blob import BlobServiceClient

app = Flask(__name__)

# Cargar los datos para obtener los rangos de escala
datos = pd.read_csv("C:/Users/kosmo/Downloads/CierreAgricola2019.csv", encoding='latin-1')
datos = datos.drop(["Municipio", "Generico", "Modalidad", "Siniestrada", "Rendimiento",
                    "CicloProductivo", "Unidad", "Precio", "Unnamed: 14", "Año", "FechaPubDOF",
                    "NombreCiudad", "Division", "Grupo", "Subclase", "Clase", "Especificacion",
                    "Cantidad", "Unidad.1"], axis=1)
categorias = ["Estado", "Cultivo"]
datos = pd.get_dummies(datos, columns=categorias)

# Crear y ajustar el escalador
escalador = pp.MinMaxScaler()
esc_entrenado = escalador.fit_transform(datos)
df = pd.DataFrame(esc_entrenado, columns=datos.columns)

# Definir el objetivo y las características
Y = np.array(df['PrecioPromedio'])
datos3 = df
datos2 = df.drop('PrecioPromedio', axis=1)
X = np.array(datos2)

def load_model_from_blob():
    connect_str = 'DefaultEndpointsProtocol=https;AccountName=modeloprediccion;AccountKey=0CaZVbaEwovf42S1fb9O5ucEFTB6TGyBF+oB6F/yX0e5+YL4FppbD80/p5x+i5/y9jVOyp5MA7VY+AStpnHx5Q==;EndpointSuffix=core.windows.net'
    container_name = 'modelopi'
    blob_name = 'ModeloFinal.pkl'

    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    blob_client = blob_service_client.get_container_client(container_name).get_blob_client(blob_name)
    
    blob_data = blob_client.download_blob().readall()
    model = pickle.loads(blob_data)
    return model

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        try:
            # Recupera los datos del formulario
            anio = float(request.form['Anio'])
            estado = request.form['Estado']
            cultivo = request.form['Cultivo']

            # Crea un DataFrame con las columnas seleccionadas durante el entrenamiento
            selected_columns = ['Anio', 'Estado_Michoacán', 'Estado_Puebla', 'Estado_Sinaloa', 'Estado_Veracruz', 'Cultivo_Aguacate', 'Cultivo_Durazno']
            input_data = pd.DataFrame({
                'Anio': [anio],
                'Estado_Michoacán': [1 if estado == 'Michoacán' else 0],
                'Estado_Puebla': [1 if estado == 'Puebla' else 0],
                'Estado_Sinaloa': [1 if estado == 'Sinaloa' else 0],
                'Estado_Veracruz': [1 if estado == 'Veracruz' else 0],
                'Cultivo_Aguacate': [1 if cultivo == 'Aguacate' else 0],
                'Cultivo_Durazno': [1 if cultivo == 'Durazno' else 0]
            })[selected_columns]

            # Escalar el vector de entrada
            escalador2 = pp.MinMaxScaler()
            escalador2.fit(datos[selected_columns])  # Ajustar el escalador con los datos originales
            X = escalador2.transform(input_data)

            # Cargar el modelo desde Blob Storage
            model = load_model_from_blob()

            # Realizar la predicción
            pred = model.predict(X)

            # Desescalar la predicción
            precioEsc = datos['PrecioPromedio'].values.reshape(-1, 1)
            escalador3 = pp.MinMaxScaler()
            escalador3.fit(precioEsc)  # Ajustar el escalador con los precios originales

            # Asegúrate de que la predicción está en el rango adecuado
            pred_descaled = escalador3.inverse_transform(pred.reshape(-1, 1))

            # Devuelve el resultado de la predicción
            return render_template('formulariogod.html', prediction_text=f'Precio: {pred_descaled[0][0]:.2f}')
        except Exception as e:
            return f"Error: {str(e)}"
    return render_template('formulariogod.html')

if __name__ == "__main__":
    app.run()
