import os
from pathlib import Path
import streamlit as st
from typing import Dict
from typing import List
from typing import Text

from src.ui import calcular_correlacion, display_header, encontrar_y_extraer_correspondencia, load_model, main_predict_from_mlflow, make_data_quality_report, set_mlflow_config, solicitar_columna_tiempo, solicitar_columna_target
from src.ui import display_report
from src.ui import display_sidebar_header
from src.ui import select_project
from src.ui import select_report
from src.ui import set_page_container_style
from src.utils import EntityNotFoundError
from src.utils import get_reports_mapping
from src.utils import list_periods

import os
from pathlib import Path
import streamlit as st
from typing import List, Text
from datetime import datetime

import pandas as pd

from src.ui import (
    display_header, display_report, display_sidebar_header,
    set_page_container_style, subir_archivo_csv, subir_true_csv
)
from src.utils import EntityNotFoundError, get_reports_mapping, list_periods
from src.ui import make_project, extraer_fechas, select_current, column_stattest

PROJECTS_DIR: Path = Path("/home/mblanco/streamlit_dashboard/projects")
REPORTS_DIR_NAME: Text = "reports"

def format_datetime(fecha):
    """Formatea la fecha al formato 'YYYY-MM-DD HH:MM:SS'"""
    return fecha.strftime("%Y-%m-%d 00:00:00")

import time

if __name__ == "__main__":

    projects: List[Text] = [p for p in os.listdir(PROJECTS_DIR) if not p.startswith(".")]

    try:
        # Sidebar: Logo y enlaces
        set_page_container_style()
        mlflow, evidently = st.tabs(["MLFlow", "Evidently AI"])
        display_sidebar_header()
        archivo_csv = subir_archivo_csv()
        true_csv = subir_true_csv()

        if archivo_csv is not None:
            with mlflow:
                if st.button("Solicitar predicciones"):
                    set_mlflow_config()
                    model = load_model()
                    for file in archivo_csv:
                        main_predict_from_mlflow(file, true_csv, model)
            with evidently:
                pass
                # Sidebar: Subir archivo CSV

                boton_presionado = st.button("Generar reporte", key="generar_reporte_button")
                if boton_presionado and archivo_csv:
                    #barra de carga
                    bar = st.progress(0)
                    for i in range(100):
                        bar.progress(i + 1)
                        time.sleep(0.001)
                    file = archivo_csv
                    for f in file:
                        # Sidebar: Seleccionar periodo
                        train_path = "/home/mblanco/streamlit_dashboard/data4_parte_70.csv"
                        ref_inicio_str, ref_fin_str = extraer_fechas(train_path)
                        cur_inicio_str, cur_fin_str = extraer_fechas(f)
                        target = 'target'
                        # tambien es posibile solicitar con la funcion: solicitar_columna_target(archivo_csv)
                        time_col = "timestamp"
                        # tambien es posibile solicitar con la funcion: solicitar_columna_tiempo(archivo_csv)
                        # Generar proyecto y reportes
                        raw_data = pd.read_csv(f,sep=',',parse_dates=[time_col],index_col=time_col)
                        numerical_features = raw_data.columns[raw_data.dtypes == 'float64'].tolist()
                        categorical_features = raw_data.columns[raw_data.dtypes == 'object'].tolist()
                        #per_column_stattest = column_stattest(numerical_features, categorical_features)

                        make_project(f, ref_inicio_str, ref_fin_str, cur_inicio_str, cur_fin_str, target, time_col)
                        # Seleccionar proyecto y periodo
                        selected_project: Path = PROJECTS_DIR / select_project(projects)
                        reports_dir: Path = selected_project / REPORTS_DIR_NAME
                        periods: List[Text] = list_periods(reports_dir)
                        selected_period = f'{cur_inicio_str}_{cur_fin_str}'
                        period_dir: Path = reports_dir / selected_period

                        #display_header(selected_project.name, selected_report_name)
                        report_mapping: Dict[Text, Path] = get_reports_mapping(period_dir)
                        if report_mapping:
                            tab_titles = list(report_mapping.keys())
                            tabs = st.tabs(tab_titles)
                            for i, report_name in enumerate(tab_titles):
                                with tabs[i]:
                                    st.write(f"Periodo de referencia: {ref_inicio_str} - {ref_fin_str}")
                                    st.write(f"Periodo actual: {cur_inicio_str} - {cur_fin_str}")
                                    display_report(report_mapping[report_name])
                        else:
                            st.error("No se encontraron reportes")
    except EntityNotFoundError as e:
        st.error(e)
    except Exception as e:
        raise e
