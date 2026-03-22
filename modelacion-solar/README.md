# 🌞 SolarMotion Tracker & Modeler

Aplicación web Full-Stack diseñada para modelar matemáticamente el movimiento aparente del sol y la intensidad de la luz natural a lo largo del día.

## 🎯 Objetivo del Proyecto

El sistema procesa datos empíricos recolectados por usuarios mediante dos experimentos físicos distintos (Intensidad Lumínica y Longitud de Sombra). Utiliza análisis numérico (Mínimos Cuadrados y Eliminación de Gauss) para encontrar la curva que mejor se ajuste, y aplica cálculo diferencial para hallar la velocidad de los cambios observados (ej. rapidez de crecimiento de la sombra). 

Además, la plataforma incluye un **Laboratorio de Machine Learning (ML)** desarrollado desde cero, permitiendo a los usuarios entrenar, validar y comparar diversos algoritmos de regresión.

## 🏗️ Arquitectura y Patrones de Diseño

El proyecto cuenta con una arquitectura desacoplada y fuertemente tipada:

*   **Frontend (Cliente):** 
    *   **React 19 + Vite:** SPA (Single Page Application) rápida y moderna.
    *   **Tailwind CSS v4:** Sistema de diseño responsivo basado en clases utilitarias, implementando "Glassmorphism" y un esquema de colores "Solar".
    *   **Recharts:** Visualización interactiva de gráficas matemáticas (Scatter, Line, Bar, Composed).
    *   **React Router:** Enrutamiento del lado del cliente (`/predictions`, `/lab`).

*   **Backend (Servidor):**
    *   **Node.js + Express + TypeScript:** API RESTful robusta y tipada.
    *   **Motor Matemático Nativo:** Implementación desde cero de álgebra lineal (Eliminación de Gauss con pivoteo parcial) sin depender de librerías externas pesadas.
    *   **Patrón Strategy & Registry:** El módulo de Machine Learning (Lab) utiliza el patrón Strategy para definir un contrato común (`TrainableModel`) que es implementado por distintos algoritmos (Linear, Polynomial, Decision Tree, SVM, KNN). El patrón Singleton/Registry orquesta la inyección y disponibilidad de estos modelos.
    *   **Arquitectura en Capas:** Controladores (Controllers) -> Servicios de Negocio (Services) -> Acceso a Datos (Prisma Models).

*   **Base de Datos:**
    *   **SQLite:** Base de datos relacional ligera, ideal para desarrollo y fácil portabilidad de los experimentos guardados.
    *   **Prisma ORM:** Mapeo objeto-relacional tipado para asegurar la integridad de la base de datos y facilitar migraciones.

## ✨ Principales Features

### 1. 📊 Predicciones (Modelado Físico)
*   **Modo Luz:** Modela la intensidad de luz (curva convexa).
*   **Modo Sombra:** Modela la longitud de sombra proyectada por un gnomon (curva cóncava) y genera automáticamente un gráfico secundario con la **primera derivada** (velocidad de cambio de la sombra).
*   **Ajuste Dinámico:** Slider para elegir el grado del polinomio (1 al 6) ajustando la regresión por Mínimos Cuadrados en tiempo real.
*   **Historial y Exportación:** Panel para guardar experimentos en la base de datos, cargarlos posteriormente y opción de impresión/exportación a PDF optimizada.

### 2. 🧪 Laboratorio ML (Observabilidad y Entrenamiento)
*   **Algoritmos Nativos:** Modelos construidos internamente para propósitos educativos y de control absoluto:
    *   Regresión Lineal (OLS)
    *   Regresión Polinomial (Least Squares)
    *   Árboles de Decisión (CART - Decision Tree Regressor)
    *   Máquinas de Vectores de Soporte (SVR Linear & RBF)
    *   K-Vecinos Más Cercanos (KNN Regressor)
*   **Hiperparámetros Configurables:** Cada modelo expone sus hiperparámetros (ej. `maxDepth` en Árboles, `C` en SVM, `k` en KNN) ajustables mediante UI.
*   **Métricas Estándar:** Evaluación automática de R², MSE (Error Cuadrático Medio), MAE y RMSE.
*   **Panel de Comparación:** Permite seleccionar múltiples modelos simultáneamente, entrenarlos con el mismo dataset y obtener un Leaderboard comparativo de su precisión (R²) y errores (MSE).

## 🚀 Paso a Paso para la Ejecución Local

Para ejecutar el proyecto en tu máquina local, asegúrate de tener instalado **Node.js** (v18+) y seguir estos pasos:

### 1. Clonar e inicializar dependencias

El proyecto está dividido en dos directorios principales: `client/` y `server/`.
Deberás inicializar ambos de forma independiente.

**Terminal 1 (Backend):**
```bash
cd server
npm install
```

**Terminal 2 (Frontend):**
```bash
cd client
npm install
```

### 2. Configurar Base de Datos (Backend)

La base de datos SQLite se creará automáticamente cuando corras las migraciones de Prisma.

**En la Terminal del Backend (`server/`):**
```bash
npx prisma generate
npx prisma migrate dev --name init
```
Esto creará el archivo `dev.db` dentro de la carpeta `server/prisma/` y preparará el ORM.

### 3. Iniciar los Servidores de Desarrollo

Es necesario correr ambos servidores simultáneamente.

**En la Terminal del Backend (`server/`):**
```bash
npm run dev
```
*(El servidor de la API correrá por defecto en `http://localhost:3001`)*

**En la Terminal del Frontend (`client/`):**
```bash
npm run dev
```
*(El servidor web de Vite correrá normalmente en `http://localhost:5173`)*

Abre tu navegador y navega a `http://localhost:5173` para comenzar a usar **SolarMotion**.

---
*Desarrollado para el módulo de Modelación Semestre 2026-1.*
