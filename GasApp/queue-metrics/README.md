# QueueMetrics App 📊

QueueMetrics es una aplicación móvil desarrollada en React Native y Expo para la recolección de datos en campo y la simulación de modelos matemáticos de líneas de espera (Teoría de Colas). 

Cumple con todos los requisitos para calcular sistemas **M/M/1** y **M/M/S**, brindando métricas precisas, recomendaciones automatizadas y reportes en PDF.

## Características Principales ✨

- **Offline-First:** Funciona sin conexión gracias a una base de datos local (SQLite).
- **Múltiples Modelos:** Soporta cálculos para colas con un servidor (M/M/1) o múltiples servidores en paralelo (M/M/S).
- **Cálculo de Probabilidad (Pn):** Permite calcular la probabilidad exacta de que haya $n$ clientes en el sistema.
- **Análisis Inteligente:** Sugiere acciones de mejora operativas en base al factor de utilización ($\rho$).
- **Exportación Profesional:** Genera un informe detallado en formato PDF directamente en el dispositivo.

## Requisitos Previos 🛠️

Asegúrate de tener instalados los siguientes componentes antes de iniciar:

1. [Node.js](https://nodejs.org/) (LTS recomendado).
2. [Android Studio](https://developer.android.com/studio) instalado y configurado con un Emulador Virtual (AVD) en ejecución.
3. Git (opcional pero recomendado).

## Instrucciones de Instalación 📦

1. Clona el repositorio y navega al directorio del proyecto:
   ```bash
   cd queue-metrics
   ```
2. Instala las dependencias necesarias:
   ```bash
   npm install
   ```

## Instrucciones de Ejecución ▶️

Para lanzar la aplicación en modo desarrollo y verla en el emulador de Android Studio:

1. Asegúrate de tener el emulador de Android Studio abierto.
2. Inicia el servidor de Expo limpiando el caché (recomendado):
   ```bash
   npm start -- -c
   ```
3. En la terminal donde se está ejecutando Expo, presiona la tecla **`a`** para abrir la aplicación en tu emulador de Android (Expo Go se instalará automáticamente en el emulador si no lo tienes).

## Generación de la APK (Producción) 📱

Para generar un archivo instalable para dispositivos físicos, asegúrate de tener configurado [EAS CLI](https://docs.expo.dev/build/setup/):
```bash
npm install -g eas-cli
eas build -p android --profile preview
```

---
*Desarrollado para el proyecto de Modelación de Sistemas y Teoría de Colas.*