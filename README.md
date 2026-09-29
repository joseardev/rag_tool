# RAG Tool: aprendiendo RAG agéntico paso a paso

Proyecto propio inspirado en la [unidad 3 del curso de agentes de Hugging Face](https://huggingface.co/learn/agents-course/unit3/agentic-rag/invitees). El objetivo es construir una herramienta que busque información verificable sobre invitados y, después, permitir que un agente decida cuándo usarla.

## Paso 1: recuperación

`retriever.py` convierte cada registro en texto y crea un índice BM25. Ante una pregunta, devuelve los registros con términos relevantes y su puntuación. Esta primera versión permite observar la recuperación por separado, antes de conectar un modelo. Los tres invitados de `data/invitados_demo.json` son **ficticios** y sus correos son de ejemplo.

Necesitas Python 3.10 o posterior. Desde esta carpeta:

```bash
python retriever.py "Marta Rios"
python retriever.py "platos vegetarianos"
python -m unittest discover -s tests
```

Para usar los datos del curso en lugar de los ficticios:

```bash
python -m venv .venv
# Activa el entorno virtual de tu sistema e instala las dependencias:
python -m pip install -r requirements.txt
python retriever.py "Ada Lovelace" --source huggingface
```

La primera consulta al conjunto de Hugging Face necesita conexión para descargarlo. Las consultas de demostración funcionan sin conexión ni token.

## Qué observar

1. Una búsqueda por nombre encuentra el registro, incluso sin escribir tildes.
2. Una búsqueda por descripción puede encontrar a alguien sin mencionar su nombre.
3. Un término ausente devuelve «No hay coincidencias».

BM25 compara palabras, no significados. Más adelante podremos contrastarlo con búsqueda semántica y conectar el recuperador a un agente.

## Próximos pasos

1. Convertir la búsqueda en una herramienta con una entrada y salida claras.
2. Dar la herramienta al agente y observar cuándo la llama.
3. Evaluar respuestas contra los registros y añadir fuentes.
