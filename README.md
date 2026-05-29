# Pueblos Mágicos — Sistema Híbrido de Recomendación Turística

Sistema de recomendación híbrido para destinos turísticos de México basado en reseñas de texto libre.
El proyecto utiliza embeddings semánticos multilingües, filtrado híbrido Content-Based + Collaborative Filtering y modelado de tópicos para explicabilidad.

Desarrollado sobre el dataset **Rest-Mex 2025**, compuesto por aproximadamente **150,000 reseñas** de **40 Pueblos Mágicos** de México.

---

# Objetivo

Diseñar un recomendador turístico capaz de interpretar consultas en lenguaje natural y generar recomendaciones personalizadas de Pueblos Mágicos utilizando información semántica derivada de reseñas reales.

Ejemplos de consultas:

* `"Quiero un pueblo tranquilo con buena comida"`
* `"Busco arquitectura colonial y hoteles románticos"`
* `"Destino para ecoturismo y naturaleza"`

---

# Arquitectura del Sistema

```text
Consulta del usuario (texto libre)
        │
        ▼
Embedding semántico
SentenceTransformer:
paraphrase-multilingual-mpnet-base-v2
(768 dimensiones)
        │
        ▼
Sistema híbrido CB + CF
score(pueblo) =
α × CB_score + (1 - α) × CF_score
        │
        ▼
Alpha dinámico:
f(volumen de reseñas,
 diversidad de polaridad)
        │
        ▼
Ranking Top-N
        │
        ├── Visualización UMAP
        └── Explicabilidad BERTopic
                │
                ▼
API Flask + Frontend Web
```

---

# Metodología

## 1. Content-Based Filtering (CB)

Cada pueblo se representa mediante embeddings semánticos generados a partir de las reseñas.

* Modelo:
  `paraphrase-multilingual-mpnet-base-v2`
* Similaridad:
  producto interno sobre vectores normalizados
* Motor de búsqueda:
  `FAISS IndexFlatIP`

---

## 2. Collaborative Filtering (CF)

Se construye un espacio de similitud entre pueblos utilizando patrones agregados de reseñas.

* Algoritmo:
  `NearestNeighbors`
* Métrica:
  similitud coseno
* Vecinos:
  `k = 20`

---

## 3. Score Híbrido

La recomendación final combina CB y CF:

```text
score(pueblo) =
α × CB_score + (1 - α) × CF_score
```

donde:

* `CB_score` = similitud semántica de la consulta
* `CF_score` = afinidad colaborativa entre pueblos
* `α` = peso dinámico dependiente de:

  * volumen de reseñas
  * diversidad de polaridad

Esto permite ajustar automáticamente la contribución de cada componente según la calidad y riqueza de información disponible por pueblo.

---

# Evaluación

Evaluación realizada mediante protocolo **Leave-One-Out** sobre aproximadamente **800 reseñas**.

## Métricas

| Método        | HR@1  | HR@3  | HR@5  | MRR   | NDCG@5 |
| ------------- | ----- | ----- | ----- | ----- | ------ |
| Híbrido CB+CF | 0.xxx | 0.xxx | 0.xxx | 0.xxx | 0.xxx  |
| Content-Based | 0.xxx | 0.xxx | 0.xxx | 0.xxx | 0.xxx  |

---

# Experimento BERTopic

Se evaluó BERTopic como posible mejora del sistema de ranking.

| Versión                  | Silhouette | HR@5    | Observación                           |
| ------------------------ | ---------- | ------- | ------------------------------------- |
| Baseline embeddings      | 0.255      | 0.407   | Modelo base                           |
| BERTopic v1 (5 tópicos)  | 0.315 ↑    | 0.351 ↓ | Tópicos similares a `Type`            |
| BERTopic v2 (19 tópicos) | 0.379 ↑    | 0.378 ↓ | Mejor clustering, menor recomendación |

## Conclusión

BERTopic mejora la separación temática de los documentos, pero no incrementa la calidad de recomendación.

La principal limitación observada es que **24 de los 40 pueblos comparten el mismo tópico dominante**, asociado principalmente a restaurantes.

Por esta razón, BERTopic se incorpora únicamente como mecanismo de:

* explicabilidad
* exploración temática
* interpretación de resultados

y no como componente del ranking principal.

---

# Estructura del Proyecto

```text
pueblos-magicos-recsys/
│
├── data/
│   └── Rest-Mex_2025_test_with_labels.csv
│
├── cache/
│   ├── model_cache/
│   ├── embeddings_cache.pkl
│   ├── eval_results.json
│   └── bertopic_v2_cache.pkl
│
├── 01_data_loading.py
├── 02_eda_basico.py
├── 03_eda_avanzado.py
├── 04_sistema_hibrido.py
├── 05_bertopic_experiment.py
├── 06_app_flask.py
│
├── requirements.txt
└── README.md
```

---

# Instalación

## 1. Clonar repositorio

```bash
git clone https://github.com/tu-usuario/pueblos-magicos-recsys.git
cd pueblos-magicos-recsys
```

## 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

## 3. Agregar dataset

Colocar el archivo:

```text
data/Rest-Mex_2025_test_with_labels.csv
```

---

# Uso

## Construcción de embeddings y evaluación

Primera ejecución: ~50 minutos
Ejecuciones posteriores usando caché: ~1 minuto

```bash
python 04_sistema_hibrido.py
```

---

## Lanzar aplicación web

```bash
python 06_app_flask.py
```

Aplicación disponible en:

```text
http://localhost:5000
```

---

## Ejecutar experimento BERTopic

```bash
python 05_bertopic_experiment.py
```

---

# API

## `GET /`

Interfaz web del sistema.

---

## `GET /metrics`

Devuelve métricas de evaluación en formato JSON.

### Ejemplo

```json
{
  "hr5": 0.407,
  "mrr": 0.298,
  "ndcg5": 0.331
}
```

---

## `POST /recommend`

Genera recomendaciones a partir de texto libre.

### Request

```json
{
  "query": "Quiero un pueblo tranquilo con buena comida",
  "top_n": 5
}
```

### Response

```json
{
  "recommendations": [...],
  "map_b64": "..."
}
```

---

# Tecnologías Utilizadas

| Componente              | Tecnología                            |
| ----------------------- | ------------------------------------- |
| Embeddings              | sentence-transformers                 |
| Modelo semántico        | paraphrase-multilingual-mpnet-base-v2 |
| Similaridad CB          | FAISS                                 |
| Collaborative Filtering | scikit-learn                          |
| Clustering              | KMeans                                |
| Visualización           | UMAP                                  |
| Topic Modeling          | BERTopic + HDBSCAN                    |
| Backend API             | Flask                                 |
| Frontend                | HTML/CSS/JavaScript                   |

---

# Dataset

## Rest-Mex 2025

Dataset de reseñas turísticas en español sobre Pueblos Mágicos de México.

### Variables principales

| Columna    | Descripción                               |
| ---------- | ----------------------------------------- |
| `Title`    | Título de la reseña                       |
| `Review`   | Texto de la reseña                        |
| `Town`     | Pueblo Mágico                             |
| `Region`   | Región geográfica                         |
| `Type`     | Categoría (Restaurant, Hotel, Attractive) |
| `Polarity` | Polaridad de sentimiento (1–5)            |

---

# Líneas Futuras

* Incorporación de reranking con LLMs
* Recomendación contextual basada en clima y temporada
* Fine-tuning de embeddings turísticos
* Integración de perfiles de usuario
* Evaluación online con usuarios reales
* Dashboard analítico interactivo

---

# Licencia

Proyecto académico y de investigación.

El uso del dataset Rest-Mex 2025 debe respetar sus términos y restricciones correspondientes.
