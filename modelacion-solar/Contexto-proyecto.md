🌞 SolarMotion Tracker & Modeler

Contexto General para el Agente de Codificación (AI Developer Context)

🎯 Objetivo del Proyecto

Desarrollar una aplicación web Full-Stack (Frontend, Backend, Base de Datos) que permita modelar matemáticamente el movimiento aparente del sol y la intensidad de la luz natural a lo largo del día.

El sistema procesará datos empíricos recolectados por usuarios mediante dos experimentos físicos distintos, utilizará análisis numérico (Mínimos Cuadrados y Eliminación de Gauss) para encontrar la curva que mejor se ajuste, y aplicará cálculo diferencial para hallar la velocidad de los cambios observados.

🔬 Experimentos y Modelos Físicos a Soportar

La aplicación debe manejar dos modos de experimentación. El usuario podrá alternar entre ellos en la interfaz:

Modo 1: Intensidad Lumínica (Luxómetro)

Variable Independiente (X): Tiempo (Horas del día, ej. 7.0, 8.5, 12.0).

Variable Dependiente (Y): Intensidad de luz (Lux).

Comportamiento Esperado: Una curva en forma de campana (parábola invertida), con su pico máximo cerca del mediodía solar.

Modo 2: Rastreo de Sombras y Velocidad (Gnomon)

Contexto: Se utiliza un "Gnomon" (un palo recto clavado perpendicularmente al suelo). A medida que el sol se mueve, la longitud de la sombra cambia.

Variable Independiente (X): Tiempo (Horas del día).

Variable Dependiente (Y): Longitud de la sombra (cm o metros).

Comportamiento Esperado: Una curva en forma de "U" (parábola normal), donde las sombras son más largas al amanecer/atardecer y más cortas al mediodía.

Cálculo Adicional (Velocidad): Para modelar la "velocidad" del movimiento de la sombra (que es un reflejo de la velocidad angular del sol), el sistema debe derivar matemáticamente el polinomio resultante de la longitud de la sombra ($Y'$).

🛠️ Stack Tecnológico Requerido

El agente de codificación debe apegarse a las siguientes tecnologías:

Frontend: React.js, Tailwind CSS (para diseño rápido y responsivo), Recharts (para las gráficas interactivas).

Backend: Node.js con Express.js (API RESTful).

Base de Datos: PostgreSQL (Relacional) o MongoDB (NoSQL) - El agente debe elegir la más óptima según su configuración de entorno, preferiblemente PostgreSQL.

Lógica Matemática: JavaScript/TypeScript puro en el Backend (sin librerías matemáticas externas pesadas para las regresiones, implementar los algoritmos desde cero para control total).

📋 Features Descriptivos para el Agente (Implementation Guide)

A continuación, se describen las funcionalidades (features) con el nivel de detalle técnico necesario para que el agente inicie la codificación:

Feature 1: Interfaz de Entrada de Datos Dual (UI)

Toggle de Experimento: Un switch o pestañas (Tabs) en el frontend para seleccionar entre "Experimento de Luz" y "Experimento de Sombra".

Tabla de Datos Dinámica: Un formulario en formato de tabla donde el usuario pueda agregar $N$ filas.

Columnas para Modo 1: Hora (X), Intensidad Lux (Y).

Columnas para Modo 2: Hora (X), Longitud de Sombra (Y).

Selector de Grado: Un slider numérico (rango 1 a 6) para seleccionar el grado ($n$) del polinomio de regresión.

Feature 2: Motor Matemático (Backend API)

Endpoint de Procesamiento: Un endpoint POST /api/model/calculate que reciba un array de puntos [{x, y}], el grado n y el tipo de experimento.

Algoritmo de Regresión Polinomial (Mínimos Cuadrados): Construir la matriz de sumatorias de las potencias de $X$ y el vector de sumatorias de $X^k \cdot Y$.

Resolución de Sistemas de Ecuaciones: Implementar el algoritmo de Eliminación de Gauss (con pivoteo parcial para evitar división por cero) que resuelva la matriz anterior y retorne el array de coeficientes $[a_0, a_1, a_2, ..., a_n]$.

Ecuación Generada: $Y(x) = a_0 + a_1x + a_2x^2 + ... + a_nx^n$.

Cálculo de Errores: Implementar funciones para calcular el $MSE$ (Error Cuadrático Medio) y el $R^2$ (Coeficiente de Determinación).

Feature 3: Módulo de Cálculo de Velocidad (Derivadas)

Lógica de Derivación: Si el tipo de experimento es "Sombra", el motor debe tomar el array de coeficientes de posición y aplicar la regla de la potencia para obtener la ecuación de la velocidad (derivada primera).

Fórmula: Si $Y(x) = a_0 + a_1x + a_2x^2 + a_3x^3$, entonces la velocidad $V(x) = Y'(x) = a_1 + 2a_2x + 3a_3x^2$.

Retorno: El endpoint debe devolver los coeficientes de la ecuación de velocidad y un array de puntos evaluados para poder graficarlos.

Feature 4: Visualización Gráfica Avanzada (Recharts)

Gráfica Principal (Posición/Luz vs Tiempo): * Eje X: Tiempo. Eje Y: Lux o Longitud.

Mostrar los puntos experimentales introducidos por el usuario como puntos dispersos (Scatter).

Trazar una línea continua suave que represente el polinomio generado evaluado en intervalos pequeños (ej. cada 0.1 horas).

Gráfica Secundaria (Velocidad vs Tiempo - Solo Modo Sombra):

Aparece únicamente cuando se procesan datos de sombras.

Eje X: Tiempo. Eje Y: Velocidad (Tasa de cambio de la sombra, ej. cm/hora).

Trazar la línea del polinomio derivado.

Feature 5: Panel de Resultados y Generación de PDF

Display de Ecuaciones: Renderizar en formato amigable (ej. usando estilos de texto monoespaciado o LaTeX/MathJax) el modelo obtenido: y = 1.2x^2 - 0.5x + 3. Si es sombra, mostrar también v = 2.4x - 0.5.

Métricas: Mostrar tarjetas visuales con los valores de $R^2$ y $MSE$.

Exportación: Un botón "Exportar a PDF" que, mediante CSS (@media print) y window.print(), o usando jsPDF, limpie la interfaz de botones y prepare un reporte estructurado de una página con la tabla de datos, las ecuaciones y las gráficas.

🚀 Instrucciones para el Agente (Prompt Directives)

Inicia por el Backend: Construye y prueba (con datos mock) la clase/servicio matemático (Eliminación de Gauss, Polinomio, Derivadas y Errores). Es el núcleo del proyecto.

API Rest: Envuelve la lógica matemática en un controlador de Express.

Frontend State: Usa Context API de React o un estado levantado (Lifting State Up) para manejar los datos de la tabla, de manera que la gráfica se actualice cuando los datos cambien.

Resiliencia: Añade validación de datos en el frontend (ej. no permitir campos vacíos antes de enviar a calcular) y manejo de errores si la matriz resulta singular.

Paginación/Estructura: Crea una UI limpia (layout dashboard) donde a la izquierda estén los controles y datos, y a la derecha los resultados y gráficas (ver Planeación_Proyecto_Solar.md para la maqueta inicial).