# RAG de incidencias logísticas

Proyecto didáctico inspirado en la [unidad 3 del curso de agentes de Hugging Face](https://huggingface.co/learn/agents-course/unit3/agentic-rag/invitees). Usa cinco incidencias de ejemplo para comparar búsqueda por palabras, búsqueda semántica y búsqueda híbrida. Los datos son ficticios y siguen los casos A–E del ejemplo.

## 1. Buscar por palabras (BM25)

Necesitas Python 3.10 o posterior. Crea un entorno virtual e instala las dependencias una vez:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

En PowerShell, desde esta carpeta, usa el Python del entorno:

```powershell
.\.venv\Scripts\python.exe retriever.py "material llegando tarde a línea" --mode bm25
.\.venv\Scripts\python.exe retriever.py "retraso entrega" --mode bm25 --tipo transporte
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

BM25 puntúa las palabras compartidas. Se eliminan artículos y preposiciones frecuentes y se ignoran tildes. Con la primera consulta, aquí devuelve B, E y C: E comparte «material» aunque trata de otro problema, y D queda fuera.

## 2. Buscar por significado (embeddings)

Ejecuta:

```powershell
.\.venv\Scripts\python.exe retriever.py "material llegando tarde a línea" --mode semantic
```

Se usa el modelo multilingüe [`paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2). La primera ejecución descarga el modelo; después calcula similitud coseno entre la pregunta y cada incidencia. En la prueba de este proyecto, los tres primeros fueron B (0,418), D (0,401) y C (0,383). Los valores pueden variar con la versión del modelo. Un score alto significa similitud textual, no confirma que la causa sea la misma.

## 3. Filtrar y combinar

```powershell
.\.venv\Scripts\python.exe retriever.py "material llegando tarde a línea" --mode hybrid --tipo transporte
.\.venv\Scripts\python.exe retriever.py "problemas con la referencia" --mode hybrid --referencia 8V0123456
```

El filtro de metadatos se aplica **antes** de buscar. En la primera consulta filtrada aparecen A y D. `hybrid` combina las posiciones de BM25 y embeddings con Reciprocal Rank Fusion (RRF), porque sus scores tienen escalas distintas. El score híbrido ya no es una similitud coseno. También se puede filtrar por `--zona` y `--causa`.

La salida muestra el ID y el texto de origen de cada resultado. La búsqueda no genera respuestas ni inventa acciones correctivas. Para responder qué medidas funcionaron necesitaremos datos reales con un campo de acciones y resultados.

Esta demostración recalcula los embeddings al buscar. Con miles de incidencias habrá que guardarlos en un índice para no repetir ese trabajo. El reranker y la generación de respuestas quedan para etapas posteriores.

## Estructura

- `data/incidencias_demo.json`: casos A–E, con tipo, zona, causa y referencia.
- `retriever.py`: carga, filtros, BM25, embeddings y fusión.
- `tests/test_retriever.py`: comprueba búsqueda y filtros sin descargar el modelo.

## Siguiente paso

Probar preguntas reales, revisar falsos positivos y decidir qué metadatos son fiables. Después añadiremos una herramienta que devuelva los resultados al agente con sus IDs como citas.
