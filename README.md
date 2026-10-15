# EcoMarket — Fase 3 (Notebook)

Notebook único para RAG + Plan-and-Execute.

1. Python 3.10+; instalar `pip install -r requirements.txt`.
2. Copiar `.env.example` como `.env` y definir `HF_TOKEN` si vas a generar respuestas con Llama.
3. Abrir `EcoMarket_RAG_Plan_Execute.ipynb` desde esta carpeta y ejecutar celdas en orden.
4. La primera indexación descarga BAAI/bge-m3 y necesita recursos de memoria.
5. Descomentar `demo(...)` y los casos para visualizar estados.

El acceso mediante API a Llama 3.1 depende de habilitación y proveedor. Sin API, el notebook permite verificar recuperación, planificación determinista y evidencias, pero no simula generación del LLM. La herramienta HTTP opcional está desactivada.

**Nota:** las fechas y estados de `pedidos.txt` son datos académicos históricos.
