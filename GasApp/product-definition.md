Documento de Arquitectura y Producto: QueueMetrics App

FASE 1: Deconstrucción y Lógica de Negocio

Resumen Ejecutivo y Alcance del MVP

El objetivo es construir una aplicación móvil "Offline-First" para la toma de datos en campo, simulación y análisis de sistemas de líneas de espera (Teoría de Colas).

Must-Have (MVP): Ingreso manual o por cronómetro de tiempos (llegada/servicio), motor de cálculo para M/M/1 y M/M/S, visualización de resultados/fórmulas, y generación de reporte PDF descargable. Embalaje en formato APK.

Nice-to-Have (Post-MVP): Sincronización en la nube (Cloud Sync), análisis de sensibilidad mediante gráficos interactivos, importación masiva vía CSV, simulador visual de la cola (animaciones).

Historias de Usuario Principales (User Stories)

Como investigador de campo, quiero registrar los tiempos de llegada y servicio usando un cronómetro integrado en la app para evitar la doble digitación de datos.

Como analista de operaciones, quiero seleccionar entre los modelos M/M/1 y M/M/S y definir el número de servidores, para obtener estimaciones precisas de Lq, Wq, L, W y $\rho$.

Como consultor, quiero visualizar las ecuaciones matemáticas utilizadas y generar un informe automático en PDF con los resultados y gráficos, para presentarlo a mis clientes.

Diagrama de Flujo (Happy Path)

flowchart TD
    A([Inicio de Sesión / App Abierta]) --> B[Crear Nuevo Estudio de Campo]
    B --> C[Módulo de Recolección de Datos]
    C -->|Opción A| D[Ingreso Manual de Tasas λ, μ]
    C -->|Opción B| E[Cronómetro Integrado en Campo]
    D --> F{Configuración del Modelo}
    E --> F
    F --> G[M/M/1]
    F --> H[M/M/S]
    G --> I[Motor de Cálculo de Colas]
    H --> I
    I --> J[Dashboard de Resultados y Fórmulas]
    J --> K[Módulo de Propuestas de Mejora]
    K --> L[Generador de PDF]
    L --> M([Descargar/Compartir APK])


FASE 2: Arquitectura de Datos (Data Layer)

Para este caso de uso, donde la recolección en campo es crítica y puede ocurrir en zonas sin conectividad, la aplicación debe ser Offline-First.

Estrategia de Datos

Elección: Local SQL (SQLite) + File System.
Justificación: El modelo de datos es altamente estructurado y relacional (Un Estudio -> Muchas Observaciones -> Un Resultado). Las bases de datos NoSQL pueden ser más flexibles, pero SQLite garantiza la integridad referencial (ACID) en el dispositivo móvil y permite consultas de agregación rápidas (ej. promedios de tiempo) necesarias para calcular $\lambda$ (Lambda) y $\mu$ (Mu) directamente en el cliente sin depender de un servidor.

Modelo Entidad-Relación (ERD)

erDiagram
    STUDY ||--o{ DATA_POINT : records
    STUDY ||--o| QUEUE_MODEL : configures
    STUDY {
        string id PK
        string title
        string context_description
        datetime created_at
    }
    DATA_POINT {
        string id PK
        string study_id FK
        string event_type "ARRIVAL | SERVICE"
        float duration_seconds
        datetime timestamp
    }
    QUEUE_MODEL {
        string id PK
        string study_id FK
        string type "MM1 | MMS"
        int servers_count
        float lambda_calculated
        float mu_calculated
        float result_Lq
        float result_Wq
        float result_Rho
    }


FASE 3: Stack Tecnológico y Arquitectura del Sistema

Matriz de Decisión Tecnológica

Capa

Tecnología Seleccionada

Justificación (Costo, Velocidad, Escala)

Frontend/Móvil

React Native + Expo

Permite desarrollo rápido, acceso nativo al hardware y compilación fácil de un .APK sin necesidad de Android Studio complejo.

Base de Datos

Expo SQLite

Nativo en Expo, ligero, no requiere configuración de servidor. Costo $0. Ideal para recolección offline en campo.

Motor Matemático

JavaScript puro / Math.js

Los cálculos de colas requieren precisión (factoriales para M/M/S). JS es suficiente para ejecución en el cliente sin latencia de red.

UI Fórmulas

react-native-math-view

Renderizado de fórmulas LaTeX (ej. $\lambda / (\mu(\mu - \lambda))$) directamente en el móvil con alta calidad visual.

Generación PDF

expo-print

Convierte plantillas HTML/CSS en PDF nativo y permite invocar el menú "Compartir/Descargar" del OS instantáneamente.

Diagrama de Arquitectura del Sistema (C4 Context / Graph)

graph LR
    subgraph Dispositivo_Movil["📱 Aplicación Móvil (React Native)"]
        UI[Interfaz de Usuario UX/UI]
        MathEngine[Motor de Colas M/M/C]
        PDFGen[Servicio Generador PDF]
        
        subgraph Almacenamiento_Local
            DB[(SQLite)]
            FS[Expo File System]
        end
    end

    UI <--> MathEngine
    UI <--> DB
    UI --> PDFGen
    PDFGen --> FS
    FS -.-> Compartir[WhatsApp / Correo / Archivos]


FASE 4: UX/UI y Frontend

Estructura de Pantallas (Sitemap)

Home (Dashboard): Lista de estudios recientes y botón flotante (+ Nuevo Estudio).

Setup del Contexto: Formulario para describir el sistema (ej. "Cajero de Banco", "Autoservicio").

Data Collection (Modo Tracker): Interfaz con dos botones grandes tipo "Stopwatch" para registrar "Llegó Cliente" y "Atención Terminada".

Configuración del Modelo: Toggle M/M/1 vs M/M/S, y slider para definir $S$ (número de servidores).

Insights & Resultados: Tarjetas con métricas (Utilización $\rho$, Tiempo en Cola $W_q$), renderizado de la fórmula matemática usada y caja de texto para redactar conclusiones de mejora.

Vista Previa PDF: Visualizador del informe antes de la exportación final.

Componentes Clave y Librerías

UI Framework: React Native Paper (Material Design) para botones consistentes y tarjetas legibles, asegurando una estética profesional.

Animaciones: Lottie (React Native Lottie) para mostrar una pequeña animación de una "fila de espera" procesándose mientras calcula, aportando creatividad y estética.

Sonidos (Feedback): expo-av para pequeños beeps al momento de recolectar datos, confirmando el registro sin mirar la pantalla.

FASE 5: DevOps e Infraestructura

Dado que es una aplicación "Offline-First" cuyo entregable es un APK, la infraestructura se centra en el pipeline de construcción (CI/CD) más que en servidores en la nube.

Estrategia de Repositorio: Monorepo simple (GitHub).

CI/CD Pipeline: Utilizaremos GitHub Actions + EAS (Expo Application Services).

Paso 1: Push a la rama main.

Paso 2: GitHub Actions corre linting y pruebas unitarias del motor matemático.

Paso 3: EAS Build se dispara automáticamente creando el artefacto .APK (Android) y almacenándolo como "Release" en GitHub.

Hosting: Al no haber backend, el costo de infraestructura es $0/mes.

FASE 6: Plan de Ejecución (Roadmap Ágil)

Fases de Desarrollo (Sprints de 1 semana)

Sprint 1: Andamiaje y Datos (Foundation & Data)

Configuración del proyecto Expo.

Implementación de SQLite y operaciones CRUD para "Estudios" y "Observaciones".

UI de recolección de datos por botones (Cronómetro).

Sprint 2: El Cerebro (Math & Domain Logic)

Programación del Domain Model: Funciones de cálculo para tasas $\lambda$ y $\mu$.

Implementación de fórmulas M/M/1 y M/M/S (probabilidades y factoriales).

Pruebas unitarias rigorosas para asegurar cálculos exactos.

Sprint 3: Estética y Resultados (UI/UX)

Desarrollo de la pantalla de Resultados con renders LaTeX de las fórmulas.

Módulo de "Propuesta de Mejora" (campos de texto).

Integración de Lottie animations para delight del usuario.

Sprint 4: El Entregable (PDF & Release)

Diseño de la plantilla HTML corporativa para el PDF.

Integración de expo-print para generar el PDF y guardarlo en el celular.

Configuración de EAS Build, generación del .APK final y pruebas en dispositivo real.

Próximos Pasos Inmediatos (Hoy)

Validar las Fórmulas: Confirmar que usaremos notación de Kendall estandar (L, Lq, W, Wq, P0, Pn).

Inicializar Repositorio: Ejecutar npx create-expo-app queue-metrics -t expo-template-blank-typescript.

Instalar Dependencias Base: Añadir expo-sqlite, expo-print, y react-native-paper.