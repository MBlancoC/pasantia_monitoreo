from datetime import datetime
from pathlib import Path

import os
from typing import Iterable
from typing import List
from typing import Text

import pandas as pd
from PIL import Image
import streamlit as st
import streamlit.components.v1 as components

from src.utils import EntityNotFoundError
from src.utils import get_report_name

from evidently.pipeline.column_mapping import ColumnMapping
from evidently.report import Report
from evidently.metrics import DataDriftTable
from evidently.metric_preset import DataDriftPreset
from evidently.metric_preset import TargetDriftPreset
from evidently.metric_preset import DataQualityPreset
from evidently.metric_preset.regression_performance import RegressionPreset

import time
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns

from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

def set_page_container_style() -> None:
    """Set report container style."""

    margins_css = """
    <style>
        /* Configuration of paddings of containers inside main area */
        .main > div {
            max-width: 100%;
            padding-left: 10%;
        }

        /*Font size in tabs */
        button[data-baseweb="tab"] div p {
            font-size: 18px;
            font-weight: bold;
        }
    </style>
    """
    st.markdown(margins_css, unsafe_allow_html=True)


def display_sidebar_header() -> None:

    # Logo
    logo = Image.open("/home/mblanco/streamlit_dashboard/streamlit-app/static/logo.png")
    #inria_logo = Image.open("static/inr_logo_rouge.png")
    with st.sidebar:
        #st.image(inria_logo, use_column_width=True)
        st.image(logo, use_column_width=True)
        st.header("")  # add space between logo and selectors

def select_project(projects: List[Text]) -> Path:
    """Select a project name form selectbox
    and return path to the project directory.

    Args:
        projects (List[Text]): List of available projects.

    Raises:
        EntityNotFoundError: If projects list is empty.

    Returns:
        Path: Path to the project.
    """

    if not projects:
        raise EntityNotFoundError("🔍 Projects not found")

    # Exclude "bike-sharing" project
    filtered_projects = [project for project in projects if project != "bike-sharing"]

    selected_project: Text = st.sidebar.selectbox(
        label="💼 Select project", options=filtered_projects,
    )

    return Path(selected_project)



def select_report(report_names: List[Text]) -> Text:
    """Select a report name from a selectbox.

    Args:
        report_names (List[Text]): Available report names.

    Raises:
        EntityNotFoundError: If report name list is empty.

    Returns:
        Text: Report name.
    """

    if not report_names:
        raise EntityNotFoundError("🔍 Reports not found")
    selected_report_name: Text = st.sidebar.selectbox(
        label="📈 Select report", options=report_names
    )

    return selected_report_name


def display_header(project_name: Text, report_name: Text) -> None:
    """Display report header.

    Args:
        project_name (Text): Project name.
        period (Text): Period.
        report_name (Text): Report name.
    """
    #dates_range: Text = period_dir_to_dates_range(period)
    st.caption(f"💼 Project: {project_name}")
    st.header(f"Report: {report_name}")
    #st.caption(f"Period: {dates_range}")


@st.cache_data
def display_report(report_path: Path) -> List[Text]:
    """Display report.

    Args:
        report (Path): Report path.

    Returns:
        List[Text]: Report parts content - list report part contents.
    """

    # If a report is file then read and display the report
    if report_path.is_file():
        with open(report_path, encoding="utf8") as report_f:
            report: Text = report_f.read()
            components.html(report, width=1000, height=1200, scrolling=True)
        return [report]

    # If a report is complex report (= directory) then
    elif report_path.is_dir():
        # list report parts
        report_parts: List[Path] = sorted(
            list(map(
                lambda report_part: report_path / report_part,
                os.listdir(report_path))
                )
            )
        tab_names: List[Text] = map(get_report_name, report_parts)
        tab_names_formatted = [f"📈 {name}" for name in tab_names]

        # create tabs
        tabs: Iterable[object] = st.tabs(tab_names_formatted)
        report_contents: List[Text] = []

        # read each report part and display in separate tab
        for tab, report_part_path in zip(tabs, report_parts):
            with tab:
                with open(report_part_path) as report_part_f:
                    report_part_content: Text = report_part_f.read()
                    report_contents.append(report_part_content)
                    components.html(
                        report_part_content, width=1000, height=1200, scrolling=True
                    )

        return report_contents

    else:
        return EntityNotFoundError("🔍 No reports found")

def column_stattest(numerical_features, categorical_features):
    """
    Definir las metricas de drift para cada columna dependiendo de si es numerica o categorica.
    Entregar las opciones en una selectbox en streamlit en la sidebar

    ej:

    per_column_stattest = {
    'air_main_temp': 'wasserstein',
    'hourly_precipitation': 'psi',
    'main_rel_humidity': 'psi',
    'air_pressure': 'wasserstein',
    'max_solar_rad': 'wasserstein',
    'max_wind_speed': 'wasserstein',
    'min_temp': 'wasserstein',
    'max_temp': 'wasserstein',
    'wind_DD': 'psi',
    'cold_hours_b7': 'psi',
    'id': 'psi',
    'name': 'jensenshannon'
    }

    """
    per_column_stattest = {}
    for feature in numerical_features:
        per_column_stattest[feature] = st.sidebar.selectbox(f"Select drift metric for {feature}", ['wasserstein', 'ks', 'psi', 'jensenshannon'])
    for feature in categorical_features:
        per_column_stattest[feature] = st.sidebar.selectbox(f"Select drift metric for {feature}", ['ks''psi', 'jensenshannon'])

    return per_column_stattest


def make_project(csv_path, ref_start, ref_end, cur_start, cur_end, target, time_column):
    """
    Genera reportes de Evidently para los datos actuales en comparación con un conjunto de referencia.
    Los reportes se guardan en la carpeta /reports.
    El nombre de la carpeta es el rango de fechas del periodo actual.
    Se guardan los archivos data_drift.html, target_drift.html y data_quality.html.

    Inputs:
    csv_path (str): Ruta al CSV con los datos actuales.
    ref_start (datetime): Fecha de inicio del periodo de referencia.
    ref_end (datetime): Fecha de fin del periodo de referencia.
    cur_start (datetime): Fecha de inicio del periodo actual.
    cur_end (datetime): Fecha de fin del periodo actual.
    target (str): Nombre de la columna objetivo.
    time_column (str): Nombre de la columna de tiempo.

    """

    train_path = "/home/mblanco/streamlit_dashboard/data4_parte_70.csv"
    ref_data = pd.read_csv(train_path, sep=",")
    ref_data[time_column] = pd.to_datetime(ref_data[time_column])
    ref_data.set_index(time_column, inplace=True)

    raw_data = pd.read_csv(csv_path, sep=',')
    raw_data[time_column] = pd.to_datetime(raw_data[time_column])
    raw_data.set_index(time_column, inplace=True)



    REF_TIME_START = ref_start
    REF_TIME_END = ref_end

    CURRENT_TIME_START = cur_start
    CURRENT_TIME_END = cur_end
    st.write(CURRENT_TIME_START)
    st.write(CURRENT_TIME_END)

    #current_duration = CURRENT_TIME_END - CURRENT_TIME_START
    #adjusted_ref_end = REF_TIME_START + current_duration

    target = "target"
    prediction = target

    numerical_features = raw_data.columns[raw_data.dtypes == 'float64'].tolist()
    categorical_features = raw_data.columns[raw_data.dtypes == 'object'].tolist()

    #per_column_stattest = per_column_stattest
    reports_dir = Path("/home/mblanco/streamlit_dashboard/projects/data_inria/reports") / f"{CURRENT_TIME_START}_{CURRENT_TIME_END}"
    reports_dir.mkdir(parents=True, exist_ok=True)
    ref_data.sort_index(inplace=True)
    raw_data.sort_index(inplace=True)
    st.write(raw_data)
    reference = ref_data.loc[REF_TIME_START:REF_TIME_END]
    current = raw_data


    # Make reports

    # Data drift
    data_drift_report = Report(metrics=[
    DataDriftTable(),
    ])
    data_drift_report.run(reference_data=reference, current_data=current.loc[CURRENT_TIME_START:CURRENT_TIME_END])

    data_drift_report_path = reports_dir / 'data_drift.html'
    data_drift_report.save_html(str(data_drift_report_path))

    #Target Drift

    target_drift_report = Report(metrics=[TargetDriftPreset()])
    target_drift_report.run(
        reference_data=reference,
        current_data=current.loc[CURRENT_TIME_START:CURRENT_TIME_END]
        #column_mapping=column_mapping
    )
    target_drift_report_path = reports_dir / 'target_drift.html'
    target_drift_report.save_html(str(target_drift_report_path))

    #Model Performance
    """
    column_mapping = ColumnMapping()

    column_mapping.target = target
    column_mapping.prediction = prediction
    column_mapping.numerical_features = numerical_features

    regression_performance_report = Report(metrics=[RegressionPreset()])

    regression_performance_report.run(
        reference_data=reference,
        current_data=current.loc[CURRENT_TIME_START:CURRENT_TIME_END],
        column_mapping=column_mapping
    )
    model_performance_report_path = reports_dir / 'model_performance.html'
    regression_performance_report.save_html(str(model_performance_report_path))
    """

    #Data Quality

    column_mapping = ColumnMapping()
    column_mapping.numerical_features = numerical_features

    data_quality_report = Report(metrics=[DataQualityPreset()])
    data_quality_report.run(
        reference_data=reference,
        current_data=current.loc[CURRENT_TIME_START:CURRENT_TIME_END],
        column_mapping=column_mapping
    )
    data_quality_report_path = reports_dir / 'data_quality.html'
    data_quality_report.save_html(str(data_quality_report_path))

def subir_archivo_csv() -> List[Path]:
    """Permite a los usuarios subir varios archivos CSV y los guarda temporalmente.

    Returns:
        List[Path]: Lista de rutas a los archivos CSV guardados.
    """
    st.sidebar.subheader("📂 CSV (Pred + Drift)")
    archivos_subidos = st.sidebar.file_uploader("Subir archivos CSV", type=["csv"], accept_multiple_files=True)
    rutas_temporales = []

    if archivos_subidos is not None and len(archivos_subidos) > 0:
        st.sidebar.write("✅ Archivos cargados correctamente")
        for archivo in archivos_subidos:
            # Guardar cada archivo temporalmente
            ruta_temporal = Path("temp") / archivo.name
            ruta_temporal.parent.mkdir(parents=True, exist_ok=True)
            with open(ruta_temporal, "wb") as f:
                f.write(archivo.getbuffer())
            rutas_temporales.append(ruta_temporal)

    return rutas_temporales

def subir_true_csv() -> Path:
    """Permite a los usuarios subir un archivo CSV y lo guarda temporalmente.

    Returns:
        Path: Ruta al archivo CSV guardado.
    """
    st.sidebar.subheader("📂 True CSV forescast")
    archivo_subido = st.sidebar.file_uploader("Subir archivo True CSV", type=["csv"])
    if archivo_subido is not None:
        st.sidebar.write("✅ Archivo cargado correctamente")
        # Guardar el archivo temporalmente
        ruta_temporal = Path("temp") / archivo_subido.name
        ruta_temporal.parent.mkdir(parents=True, exist_ok=True)
        with open(ruta_temporal, "wb") as f:
            f.write(archivo_subido.getbuffer())
        return ruta_temporal
    else:
        return None

def extraer_fechas(csv_path):
    """
    Extrae las fechas de inicio y fin del archivo CSV.

    Args:
        csv_path (Path): Ruta al archivo CSV.

    Returns:
        Tuple[datetime, datetime]: Fecha de inicio y fin.
    """
    # Leer el archivo CSV
    df = pd.read_csv(csv_path, sep=',')
    # Extraer las fechas de inicio y fin
    fecha_inicio = df["timestamp"][0]
    fecha_fin = df["timestamp"].iloc[-1]
    return (fecha_inicio, fecha_fin)

def select_current(fechas, key_prefix):
    """

    """
    inicio, fin = fechas
    st.sidebar.subheader("📆 Seleccionar periodo actual")
    fecha_inicio = st.sidebar.date_input(f"Fecha de inicio {key_prefix}", inicio.date(), min_value=inicio.date(), max_value=fin.date(), key=f"fecha_inicio_{key_prefix}")
    fecha_fin = st.sidebar.date_input(f"Fecha de fin {key_prefix}", fin.date(), min_value=fecha_inicio, max_value=fin.date(), key=f"fecha_fin_{key_prefix}")
    fecha_inicio = datetime.combine(fecha_inicio, datetime.min.time())
    fecha_fin = datetime.combine(fecha_fin, datetime.max.time())
    return fecha_inicio.strftime("%Y-%m-%d %H:%M"), fecha_fin.strftime("%Y-%m-%d %H:%M")

def set_mlflow_config():
    """
    Configura la URI de MLflow para el servidor de MLflow.
    """
    MLFLOW_TRACKING_URI = "http://192.168.2.92:8002"
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def load_model():
    """
    Se carga el modelo de MLflow, se debe indicar el ID del modelo que se quiere cargar.
    """
    return mlflow.pyfunc.load_model('runs:/df278906d83c417f817c4f462c51b242/model')

def make_predictions(model, input_data):
    """
    Se realiza la predicción con el modelo cargado.
    """
    predictions = model.predict(input_data)
    return predictions

def prepare_input(archivo_csv):
    """

    """
    df = pd.read_csv(archivo_csv)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    time_data = TimeSeriesDataFrame.from_data_frame(
        df,
        id_column="item_id",
        timestamp_column="timestamp"
    )
    return time_data

def plot_predictions(y_past, y_pred, true_csv=None):
    """
    Se grafican las series de tiempo pasadas y predichas.
    En caso de cargar un archivo CSV con valores reales, se grafican también.
    """
    item_id = "Temperatura del Aire Media-La Platina"
    sub_past = y_past[::1]

    fig, ax = plt.subplots(figsize=(15, 5))  # Ajusta el tamaño de la figura aquí (ancho, alto)
    ax.plot(sub_past[-100:], label="Past time series values")
    ax.plot(y_pred['mean'], label="Mean forecast")

    if true_csv is not None:
        df3 = pd.read_csv(true_csv)
        data3 = TimeSeriesDataFrame.from_data_frame(df3)
        data3_loc = data3.loc[item_id]["target"][1:45]
        sub3 = data3_loc
        ax.plot(sub3, label="Real Values")

        # Calcula la correlación entre y_pred y sub3
        correlation = y_pred['mean'].corr(sub3, method = "kendall")
        ax.text(0.5, 0.9, f'Correlation: {correlation:.2f}', transform=ax.transAxes, fontsize=15)

    ax.fill_between(
        y_pred.index, y_pred["0.1"], y_pred["0.9"], color="red", alpha=0.1, label=f"10%-90% confidence interval"
    )
    ax.legend()
    return fig


def main_predict_from_mlflow(archivo_csv, true_csv, model):
    """
    Se realiza la predicción con el modelo cargado y se grafican las series de tiempo pasadas y predichas.
    """

    st.subheader(f'Predicción {archivo_csv}')
    try:
        time_data = prepare_input(archivo_csv)
        y_pred = make_predictions(model, time_data)
        y_pred = y_pred.loc["Temperatura del Aire Media-La Platina"]

        y_past = time_data.loc["Temperatura del Aire Media-La Platina"]["target"]

        fig = plot_predictions(y_past, y_pred, true_csv)

        time.sleep(0.5)
        st.pyplot(fig)
        st.write(y_pred.head())
        y_pred = pd.DataFrame(y_pred)
        y_pred.to_csv(f"/home/mblanco/streamlit_dashboard/predictions/pred_{str(archivo_csv).split('/')[1]}")

        return y_pred

    except Exception as e:
        st.write(e)
        st.error(f'Error: {e}')

def solicitar_columna_tiempo(csv_path):
    """Solicita al usuario seleccionar la columna de tiempo del archivo CSV.

    Args:
        csv_path (Path): Ruta al archivo CSV.

    Returns:
        Text: Nombre de la columna de tiempo.
    """
    # Leer el archivo CSV
    df = pd.read_csv(csv_path, sep=',')
    # Solicitar al usuario seleccionar la columna de tiempo
    columna_tiempo = st.sidebar.selectbox("🕰️ Seleccionar columna de tiempo", df.columns, key="columna_tiempo")
    return columna_tiempo

def solicitar_columna_target(csv_path):
    """Solicita al usuario seleccionar la columna target del archivo CSV.

    Args:
        csv_path (Path): Ruta al archivo CSV.

    Returns:
        Text: Nombre de la columna target.
    """
    # Leer el archivo CSV
    df = pd.read_csv(csv_path, sep=',')
    # Solicitar al usuario seleccionar la columna target
    columna_target = st.sidebar.selectbox("🎯 Seleccionar columna target", df.columns, key="columna_target")
    return columna_target


def encontrar_y_extraer_correspondencia(fecha_inicio, fecha_fin, lista_csv_paths):
    """
    Encuentra un archivo CSV en una lista que contenga el rango de fechas dado y extrae ese rango.

    Args:
        fecha_inicio (datetime): Fecha de inicio del rango.
        fecha_fin (datetime): Fecha de fin del rango.
        lista_csv_paths (List[Path]): Lista de rutas a archivos CSV.

    Returns:
        DataFrame: Datos extraídos del archivo CSV encontrado que corresponde al rango de fechas.
    """
    st.write(f"Predicción inicio: {fecha_inicio}, fecha_fin: {fecha_fin}")
    #st.write(lista_csv_paths)
    for csv_path in lista_csv_paths:
        df = pd.read_csv("/home/mblanco/streamlit_dashboard/Data/"+csv_path, sep=',', parse_dates=['timestamp'])
        #

        # Comprobar si el rango de fechas se encuentra dentro de este archivo CSV
        if df["timestamp"].min() <= fecha_inicio and df["timestamp"].max() >= fecha_fin:
            # Extraer solo los datos dentro del rango de fechas
            datos_correspondientes = df[(df["timestamp"] >= fecha_inicio) & (df["timestamp"] <= fecha_fin)]
            st.write(csv_path)
            #st.write(f"Busqueda inicio: {df['timestamp'].min()}, fecha_fin: {df['timestamp'].max()}")
            return (csv_path, datos_correspondientes)

    # Si no se encuentra ningún archivo que coincida
    return None



def normalizar_formato_fecha(df, columna_fecha):
    """
    Normaliza y redondea el formato de la columna de fecha.

    Args:
        df (DataFrame): DataFrame de Pandas.
        columna_fecha (str): Nombre de la columna de fecha a normalizar.

    Returns:
        DataFrame: DataFrame con la columna de fecha normalizada.
    """
    df[columna_fecha] = pd.to_datetime(df[columna_fecha]).dt.round('1s')  # Redondea al segundo más cercano
    return df


def calcular_correlacion(csv_real, csv_prediccion):
    """
    Calcula la correlación entre los valores reales y los predichos.

    Args:
        csv_real (str): Ruta al CSV con los datos reales.
        csv_prediccion (str): Ruta al CSV con las predicciones.

    Returns:
        float: Valor de correlación.
    """
    # Cargar datos reales y normalizar formato de fecha
    datos_reales = pd.read_csv(csv_real)

    # Cargar datos de predicción y normalizar formato de fecha
    predicciones = pd.read_csv(csv_prediccion)


def make_data_quality_report(real_csv, pred_csv):
    """
    Genera un informe de calidad de datos para los datos actuales en comparación con un conjunto de referencia.

    Args:
        csv_path (str): Ruta al CSV con los datos actuales.
        ref_start (datetime): Fecha de inicio del periodo de referencia.
        ref_end (datetime): Fecha de fin del periodo de referencia.
        cur_start (datetime): Fecha de inicio del periodo actual.
        cur_end (datetime): Fecha de fin del periodo actual.
        target (str): Nombre de la columna objetivo.
        time_column (str): Nombre de la columna de tiempo.

    """


    target = "target"
    time_column = "timestamp"
    # Cargar datos de referencia
    ref_data = pd.read_csv(real_csv, sep=",")
    #st.write(ref_data.columns)
    ref_data[time_column] = pd.to_datetime(ref_data[time_column])
    ref_data.set_index(time_column, inplace=True)

    # Cargar datos actuales
    raw_data = pd.read_csv(pred_csv, sep=',')
    raw_data.drop(["item_id"])
    raw_data[time_column] = pd.to_datetime(raw_data[time_column])
    raw_data.set_index(time_column, inplace=True)

    # Definir periodos de tiempo
    REF_TIME_START, REF_TIME_END = extraer_fechas(real_csv)
    CURRENT_TIME_START, CURRENT_TIME_END = extraer_fechas(pred_csv)

    # Ajustar los datos de referencia al mismo período de tiempo que los datos actuales
    current_duration = CURRENT_TIME_END - CURRENT_TIME_START
    adjusted_ref_end = REF_TIME_START + current_duration

    # Filtrar datos de referencia y actuales según los periodos definidos
    reference = ref_data.loc[REF_TIME_START:adjusted_ref_end]
    current = raw_data.loc[CURRENT_TIME_START:CURRENT_TIME_END]

    # Identificar características numéricas y categóricas
    numerical_features = current.columns[current.dtypes == 'float64'].tolist()
    categorical_features = current.columns[current.dtypes == 'object'].tolist()

    # Preparar el mapeo de columnas para el informe
    column_mapping = ColumnMapping()
    column_mapping.numerical_features = numerical_features
    column_mapping.categorical_features = categorical_features

    # Generar el informe de calidad de datos
    data_quality_report = Report(metrics=[DataQualityPreset()])
    data_quality_report.run(
        reference_data=reference,
        current_data=current,
        column_mapping=column_mapping
    )

    # Guardar el informe de calidad de datos
    reports_dir = Path("/home/mblanco/streamlit_dashboard/projects/data_inria/reports") / f"{CURRENT_TIME_START}_{CURRENT_TIME_END}"
    reports_dir.mkdir(parents=True, exist_ok=True)
    data_quality_report_path = reports_dir / 'data_quality.html'
    data_quality_report.save_html(str(data_quality_report_path))

    display_report(data_quality_report_path)
