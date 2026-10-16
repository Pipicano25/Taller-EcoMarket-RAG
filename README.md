# EcoMarket — Sistema Inteligente de Atención al Cliente

**Autores:** Anderson Pipicano, Fredy Alvarez  
**Asignatura:** Inteligencia Artificial Generativa  
**Objetivo:** Extender las capacidades de **EcoBot** (desarrollado inicialmente en el Taller Práctico #1) mediante una arquitectura **RAG (*Retrieval-Augmented Generation*)**, permitiéndole consultar una base de conocimiento vectorial interna para responder con precisión cualquier solicitud de los clientes o informar de manera transparente cuando no cuenta con las herramientas o información para atenderla.



Proyecto académico de **Inteligencia Artificial Generativa** orientado a mejorar la atención al cliente de **EcoMarket**, una plataforma de comercio electrónico de productos sostenibles.

El proyecto parte de **EcoBot**, un asistente para consultas de clientes, y amplía sus capacidades mediante **generación aumentada por recuperación (RAG)**: antes de responder, el sistema consulta documentos de la empresa para fundamentar sus respuestas y reconocer cuándo no dispone de información o herramientas suficientes.

## Contenido del proyecto

| Sección | Descripción | Documentación |
| --- | --- | --- |
| **Fases 1 y 2** | Selección y justificación de embeddings y base vectorial; diseño de la base de conocimiento y estrategia de segmentación. | [Caso de estudio del Taller 2](fase1&2/caso_de_estudio_taller2.md) |
| **Fase 3** | Implementación del RAG y de un agente *Plan-and-Execute* con planificación, herramientas y replanificación, desarrollada en un notebook. | [Ver la Fase 3](fase3) |

## Tecnologías principales

**Python · LangChain · LangGraph · BAAI/bge-m3 · ChromaDB · GLM-5.3-Flash (Ollama Cloud)**

## Explorar la implementación

Para conocer la arquitectura, los documentos utilizados, la configuración del entorno de trabajo, la ejecución del notebook y las limitaciones del sistema, consulta el **[README de la Fase 3](fase3/README.md)**.

> **Nota:** El proyecto es un prototipo académico. Los registros de pedidos y demás datos de demostración no representan consultas operativas en tiempo real.
