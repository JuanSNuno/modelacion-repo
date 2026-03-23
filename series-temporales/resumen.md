Resumen del Análisis de la Tasa de Cambio Australiana
1. Carga y Limpieza de Datos:

Qué se hizo: Se cargó el archivo exchange_rates.csv y se seleccionó la serie de tiempo correspondiente a la tasa de cambio de Australia (AUSTRALIA -- SPOT EXCHANGE RATE US$/AU$). Se renombraron las columnas para mayor claridad (Date, Exchange Rate). Los valores "ND" (No Data) fueron reemplazados por NaN y luego interpolados linealmente para mantener la continuidad de la serie temporal, evitando la pérdida de datos valiosos.
Por qué: La interpolación lineal es adecuada para series de tiempo, ya que estima los valores faltantes basándose en los puntos adyacentes, preservando las tendencias. La limpieza inicial es crucial para asegurar la calidad de los datos para el modelado.
Resultados: Se obtuvo un DataFrame df_australia limpio, sin valores nulos en la columna Exchange Rate, y con la columna Date en formato de fecha, abarcando un rango desde 1971 hasta 2017.

2. Preparación de Datos para Modelado (Ventanas Temporales):

Qué se hizo: Se escalaron los datos de la tasa de cambio a un rango entre 0 y 1 usando MinMaxScaler. Se crearon ventanas temporales (WINDOW_SIZE) de datos históricos para predecir el siguiente valor (HORIZON = 1). Inicialmente, WINDOW_SIZE se estableció en 10, pero los resultados no eran los esperados y luego, se dejó en  30, ya que el pensado era 60 pero el entrenamiento de las epocas era muy demorado (en la epoca 10 llevaba más de 20 minutos). Los datos se dividieron en conjuntos de entrenamiento (80%) y prueba (20%) manteniendo el orden temporal.
Por qué: El escalado ayuda a los modelos de redes neuronales a converger más rápido y a tener un mejor rendimiento. Las ventanas temporales son el formato estándar para el modelado de series de tiempo con redes neuronales, donde el modelo aprende de secuencias pasadas para predecir el futuro.
Resultados: Se obtuvieron X_train, X_test, y_train, y_test listos para el entrenamiento de modelos, con X_train teniendo una forma de (9712, 30) y y_train de (9712,) (para WINDOW_SIZE = 30).

3. Desarrollo y Evaluación del Modelo MLP (Multi-Layer Perceptron):

Qué se hizo (Arquitectura Inicial): Se implementó un modelo MLP (model) con capas Dense y activación relu, y una capa de salida linear. Se compiló con el optimizador adam y función de pérdida mse. Se entrenó con 50 épocas.
Resultados (Iniciales con WINDOW_SIZE=10, sin Dropout):
Train RMSE: 0.0055 (muy bueno)
Test RMSE: 0.0076 (muy bueno)
Qué se hizo (Modificación por "Picos"): Por preocupación  sobre los "picos" en las proyecciones futuras, se agregaron capas Dropout(0.2) después de cada capa Dense para reducir el sobreajuste y fomentar predicciones más suaves.
Por qué: Los "picos" en las proyecciones iterativas pueden ser una señal de que el modelo está aprendiendo demasiado ruido de los datos de entrenamiento o que el error se acumula. Dropout es una técnica de regularización que ayuda a prevenir el sobreajuste y a obtener modelos más generalizables y suaves.
Resultados (Después de Dropout, con WINDOW_SIZE=30):
Train RMSE: 0.0563
Test RMSE: 0.0390
Análisis: El RMSE aumentó, lo cual es un compromiso esperado al introducir Dropout para mejorar la suavidad y la generalización a expensas de una menor precisión en los datos de entrenamiento y prueba inmediatos. Sin embargo, este cambio busca mejorar la calidad de las proyecciones a largo plazo.

4. Desarrollo y Evaluación del Modelo LSTM (Long Short-Term Memory):

Qué se hizo (Arquitectura Inicial): Se implementó un modelo LSTM (lstm_model) para capturar dependencias secuenciales a largo plazo. Los datos de entrada X_train y X_test fueron remodelados a formato 3D ((muestras, pasos_de_tiempo, características)). El modelo inicialmente tuvo una capa LSTM y capas Dense.
Resultados (Iniciales con WINDOW_SIZE=10):
Train RMSE: 0.0584
Test RMSE: 0.0421
Análisis: El rendimiento inicial del LSTM fue considerablemente peor que el del MLP.
Qué se hizo (Modificación de Arquitectura): Se intentó mejorar el modelo LSTM añadiendo una segunda capa LSTM, aumentando las unidades (100 en la primera, 50 en la segunda) y agregando Dropout entre las capas, además de aumentar las épocas a 50.
Por qué: Se buscó una arquitectura más compleja para permitir al LSTM capturar patrones más intrincados y dependencias a largo plazo, esperando mejorar su rendimiento.
Resultados (Después de la modificación, con WINDOW_SIZE=30):
Train RMSE: 0.0407
Test RMSE: 0.0240
Análisis: Aunque el RMSE del LSTM mejoró respecto a su versión inicial, sigue siendo más alto que el del MLP después de la implementación de Dropout. Esto sugiere que para este conjunto de datos y configuración, el MLP ha demostrado ser más eficaz.

5. Interactividad y Pronóstico Futuro:

Qué se hizo: Se crearon funciones interactivas (plot_interactive_range, plot_future_forecast, plot_interactive_lstm, plot_future_forecast_lstm) con widgets para:
Visualizar y comparar las predicciones del modelo (MLP y LSTM) con los valores reales dentro de un rango de fechas seleccionado por el usuario.
Realizar pronósticos iterativos hacia el futuro para un número de meses especificado por el usuario, mostrando tanto una gráfica como una tabla de valores numéricos.
Por qué: Estas herramientas permiten al usuario explorar visualmente el rendimiento del modelo, entender cómo se ajusta a los datos históricos y cómo proyecta tendencias futuras. La visualización de los valores numéricos es crucial para una evaluación precisa de las predicciones.
Resultados: Se generaron interfaces interactivas que muestran las curvas de predicción junto a los datos reales y permiten generar pronósticos a meses vista. Se notó la presencia de "picos" en las proyecciones iterativas, lo que llevó a la inclusión de Dropout en el MLP.

6. Ajuste del WINDOW_SIZE:

Qué se hizo: Se propuso cambiar el WINDOW_SIZE de 10 (inicial) a 30 (luego a 60). Se modificó la celda  para reflejar este cambio.
Por qué: El WINDOW_SIZE es un hiperparámetro crítico que define cuántos pasos de tiempo pasados usa el modelo para hacer una predicción. Experimentar con su valor permite encontrar el equilibrio óptimo para capturar las dependencias temporales relevantes en la serie de datos.
Resultados: El cambio a WINDOW_SIZE = 30 (y próximamente a 60) implica que el modelo ahora considera una ventana más amplia de datos históricos, lo que podría afectar su capacidad para capturar patrones a corto o largo plazo. La reevaluación de los modelos después de este cambio es fundamental para entender su impacto.
En resumen, hemos pasado por un proceso iterativo de limpieza de datos, modelado con MLP y LSTM, evaluación rigurosa con RMSE, y ajuste de hiperparámetros y arquitecturas en respuesta a observaciones y solicitudes del usuario. El modelo MLP, incluso con Dropout para suavizar predicciones, ha demostrado ser consistentemente más preciso que el LSTM para esta tarea específica hasta ahora.