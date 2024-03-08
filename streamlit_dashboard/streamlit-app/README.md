
## Introducción
`app.py` es el archivo principal de la aplicación, el cual ejecuta y utiliza módulos del directorio `src`. Los principales pasos del flujo de la app son:

- Configuración y estilo de la página con Streamlit (división MLFLOW - Evidently).
- Carga de archivos CSV para datos de entrada y valores reales.
- Interacción con MLflow realizar predicciones al modelo desplegado.
- Utilización de Evidently AI para crear proyectos y generar reportes a partir del dataset cargado en la app comparado con el dataset de referencia definido en el codigo de [ui.py](streamlit_dashboard/streamlit-app/src/ui.py)

## Uso

Para iniciar la aplicación, se debe ejecutar el siguiente comando:

```bash
streamlit run streamlit_dashboard/streamlit-app/app.py
```

Como el modelo fue cargado utilizando el servidor de Inria, en estos momentos no es posible solicitar predicciones al modelo del codigo.
