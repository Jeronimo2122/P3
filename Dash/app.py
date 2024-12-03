# Importar bibliotecas necesarias
import joblib
import pandas as pd
import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import plotly.express as px
import psycopg2
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf


# Configuración de la conexión a la base de datos
DATABASE_CONFIG = {
    'dbname': "postgres",
    'user': "postgres",
    'password': "Lab9sqljuan",
    'host': "database-juan.c139i8kqq4e8.us-east-1.rds.amazonaws.com",
    'port': '5432'
}

# Establecer conexión a la base de datos
engine = psycopg2.connect(**DATABASE_CONFIG)

# Función para cargar y ajustar el scaler con los datos de entrenamiento
def cargar_scaler(X_train):
    scaler = StandardScaler()
   
    scaler.fit_transform(X_train)
    return scaler

# Cargar el modelo entrenado
model = tf.keras.models.load_model("modelo.h5")
scaler = joblib.load("scaler.pkl")

# Crear una instancia de la aplicacion Dash
app = dash.Dash(__name__)

# Diseno de la aplicacion
app.layout = html.Div([
    html.Div([
        html.Img(
            src="/assets/icfes_logo.png",
            alt="Imagen local",
            style={"width": "350px", "height": "200px"}
        ),

        html.H1(
            children=[
                "Resultados del ICFES",
                html.Br(),
                "Departamento del Magdalena"
            ],
            style={
                "fontFamily": "Ubuntu",  # Fuente de Google Fonts
                "color": "#026ab0",  # Color del texto
                "fontWeight": "bold",  # Negrita opcional, puedes ajustarlo
                "fontSize": "48px",     # Tamaño grande
                "textAlign": "center",  # Centrar el texto
                "marginTop": "20px",    # Espacio superior
                "lineHeight": "1.4",    # Espaciado entre líneas
            }
        )
    ], style={
        "display": "flex",       # Flexbox para organizar en fila
        "flexDirection": "row",  # Dirección horizontal
        "justifyContent": "center",  # Centrar elementos horizontalmente
        "alignItems": "center",
        "border": "2px solid #026ab0",  # Borde azul
        "padding": "10px",  # Espaciado interno
        "borderRadius": "10px"  # Bordes redondeados
    }),

    html.P("El Instituto Colombiano para la Evaluación de la Educación, \
        es una entidad especializada en ofrecer servicios de evaluación \
        de la educación en todos sus niveles, y en particular apoya al \
        Ministerio de Educación Nacional en la realización de los exámenes \
        de Estado y en adelantar investigaciones sobre los factores que \
        inciden en la calidad educativa, para ofrecer información pertinente \
        y oportuna para contribuir al mejoramiento de la calidad de la educación. \
        En este caso solo se estudiaran los resultados de la evacualcion Saber \
        que se le hace a los estudiantes de grado 11 en el departamento del Magdalena.",
        style={
            "fontFamily": "Arial",
            "fontSize": "16px",
            "lineHeight": "1.6",
            "textAlign": "justify",
            "margin": "20px",
            "border": "2px solid #026ab0",  # Borde azul
            "padding": "10px",  # Espaciado interno
            "borderRadius": "10px"  # Bordes redondeados
        }),

    # Seccion de estadisticas sobre los resultados del ICFES
    html.Div([
        html.H2("Estadísticas de los resultados del ICFES", style={"textAlign": "center"}),
        html.Div([
            html.H2("Distribucion de putuaciones en las diferentes competencias", style={"textAlign": "center"}),
            html.P("En el siguinete histograma se mostrará la distribución de los puntajes \
                    obtenidos por los estudiantes en las diferentes competencias evaluadas \
                    por el ICFES. Selecciona una competencia para visualizar su distribución:",
                    style={
                        "fontFamily": "Arial",
                        "fontSize": "16px",
                        "lineHeight": "1.6",
                        "textAlign": "justify",
                        "margin": "20px"
                    }),
            dcc.Dropdown(
                id='competencia-dropdown',
                options=[
                    {'label': 'Matemáticas', 'value': 'punt_matematicas'},
                    {'label': 'Lectura Crítica', 'value': 'punt_lectura_critica'},
                    {'label': 'Sociales y Ciudadanas', 'value': 'punt_sociales_ciudadanas'},
                    {'label': 'Ciencias Naturales', 'value': 'punt_c_naturales'},
                    {'label': 'Inglés', 'value': 'punt_ingles'}
                ],
                value='punt_matematicas',
                style={
                    "fontFamily": "Arial",
                    "fontSize": "16px",
                    "margin": "20px"
                }
            ),
            dcc.Graph(id='competencia-graph')
        ]),

        # Correlacion enter el puntaje total y el nivel socio economico
        html.Div([
            html.Div([
                html.Div([
                    html.H2("Puntaje total promedio por nivel socioeconómico", style={"textAlign": "center"}),
                    html.P("En el siguiente gráfico se muestran algunos estadísticos descriptivos \
                            del puntaje total obtenido por los estudiantes en la prueba Saber 11, \
                            en función del nivel socioeconómico de los estudiantes. Selecciona un \
                            nivel socioeconómico para visualizar la distribución de los puntajes:",
                            style={
                                "fontFamily": "Arial",
                                "fontSize": "16px",
                                "lineHeight": "1.6",
                                "textAlign": "justify",
                                "margin": "20px"
                            }),
                    dcc.Dropdown(
                        id='nivel-socioeconomico-dropdown',
                        options=[
                            {'label': 'Estrato 1', 'value': 'fami_estratovivienda_estrato_1'},
                            {'label': 'Estrato 2', 'value': 'fami_estratovivienda_estrato_2'},
                            {'label': 'Estrato 3', 'value': 'fami_estratovivienda_estrato_3'},
                            {'label': 'Estrato 4', 'value': 'fami_estratovivienda_estrato_4'},
                            {'label': 'Estrato 5', 'value': 'fami_estratovivienda_estrato_5'},
                            {'label': 'Estrato 6', 'value': 'fami_estratovivienda_estrato_6'},
                            {'label': 'Sin Estrato', 'value': 'fami_estratovivienda_sin_estrato'}
                        ],
                        value='fami_estratovivienda_estrato_2',
                        style={
                            "fontFamily": "Arial",
                            "fontSize": "16px",
                            "margin": "20px"
                        }
                    ),
                    html.P(id='media', style={"fontWeight": "bold", "fontFamily": "Arial", "fontSize": "16px", "margin": "20px"}),
                    html.P(id='minimo', style={"fontWeight": "bold", "fontFamily": "Arial", "fontSize": "16px", "margin": "20px"}),
                    html.P(id='maximo', style={"fontWeight": "bold", "fontFamily": "Arial", "fontSize": "16px", "margin": "20px"})
                ]),
                dcc.Graph(id='scatter-nivel-socioeconomico')
            ], style={
                "display": "flex",       # Flexbox para organizar en fila
                "flexDirection": "row",  # Dirección horizontal
                "justifyContent": "center",  # Centrar elementos horizontalmente
                "alignItems": "center",
            })
        ]),

        html.Div([
            html.P("Seleccione la cantidad de varibales independientes que quiera para saber la correlacion de estas con el puntaje de un estudiante"),
            dcc.Checklist(
                id='checklist',
                options=[
                    {'label': 'Estudio en colegio bilingue', 'value': 'cole_bilingue'},
                    {'label': 'Tiene internet en casa', 'value': 'fami_tieneinternet'},
                    {'label': 'Tiene computador en casa', 'value': 'fami_tienecomputador'},
                    {'label': 'La familia tiene carro', 'value': 'fami_tieneautomovil'},
                    {'label': 'Estudio en jornada unica', 'value': 'cole_jornada_unica'},
                    {'label': 'Estudio en jornada matutina', 'value': 'cole_jornada_manana'},
                    {'label': 'Estudio en jornada nocturna', 'value': 'cole_jornada_noche'},
                    {'label': 'Estudio en jornada de la tarde', 'value': 'cole_jornada_tarde'},
                    {'label': 'Estudio en jornada unica', 'value': 'cole_jornada_unica'},
                    {'label': 'Estudio en jornada de sabado', 'value': 'cole_jornada_sabatina'},
                    {'label': 'Cantidad de personas en el hogar', 'value': 'fami_personashogar'},

                ],
                value=['cole_bilingue', 'fami_tieneinternet', 'fami_tienecomputador', 'cole_jornada_unica'],
                style={
                    "display": "flex",
                    "flexDirection": "row",
                    "flexWrap": "wrap",
                    "gap": "20px"  # Espacio entre los elementos
                }
            ),
            dcc.Graph(id='output-checklist')
        ]),
    ], style={
        "border": "2px solid #026ab0",  # Borde azul
        "padding": "10px",  # Espaciado interno
        "borderRadius": "10px",  # Bordes redondeados
        "margin": "20px"  # Margen externo
    }),

    html.Div([
        html.H2("Predicción del puntaje total en la prueba Saber 11", style={"textAlign": "center"}),
        html.P("Ingresa los datos del estudiante para predecir su puntaje total en la prueba Saber 11:",
               style={"fontFamily": "Arial", "fontSize": "16px", "lineHeight": "1.6", "textAlign": "justify", "margin": "20px"}),
        html.Div([
            html.Div([
                html.Label("Estudio en colegio bilingue:"),
                dcc.Dropdown(
                    id='cole_bilingue',
                    options=[
                        {'label': 'Sí', 'value': 1},
                        {'label': 'No', 'value': 0}
                    ],
                    value=1
                )
            ], style={"margin": "10px", "flex": "1"}),

            html.Div([
                html.Label("El colegio es oficial:"),
                dcc.Dropdown(
                    id='colegio_oficial',
                    options=[
                        {'label': 'Sí', 'value': 1},
                        {'label': 'No', 'value': 0}
                    ],
                    value=1
                )
            ], style={"margin": "10px", "flex": "1"}),

            html.Div([
                html.Label("Estudio en sede principal del colegio:"),
                dcc.Dropdown(
                    id='sede_principal',
                    options=[
                        {'label': 'Sí', 'value': 1},
                        {'label': 'No', 'value': 0}
                    ],
                    value=1
                )
            ], style={"margin": "10px", "flex": "1"}),

            html.Div([
                html.Label("Estuvo privado de la libertad:"),
                dcc.Dropdown(
                    id='privado_libertad',
                    options=[
                        {'label': 'Sí', 'value': 1},
                        {'label': 'No', 'value': 0}
                    ],
                    value=0
                )
            ], style={"margin": "10px", "flex": "1"}),

            html.Div([
                html.Label("¿Cuantas personas viven en su hogar?:"),
                dcc.Input(
                    id='num_personas_hogar',
                    type='number',
                    value=4
                )
            ], style={"margin": "10px", "flex": "1"}),

            html.Div([
                html.Label("La familia tiene carro:"),
                dcc.Dropdown(
                    id='tienen_carro',
                    options=[
                        {'label': 'Sí', 'value': 1},
                        {'label': 'No', 'value': 0}
                    ],
                    value=0
                )
            ], style={"margin": "10px", "flex": "1"}),

            html.Div([
                html.Label("La familia tiene computador:"),
                dcc.Dropdown(
                    id='tienen_computador',
                    options=[
                        {'label': 'Sí', 'value': 1},
                        {'label': 'No', 'value': 0}
                    ],
                    value=1
                )
            ], style={"margin": "10px", "flex": "1"}),

            html.Div([
                html.Label("La familia tiene acceso a internet:"),
                dcc.Dropdown(
                    id='tiene_internet',
                    options=[
                        {'label': 'Sí', 'value': 1},
                        {'label': 'No', 'value': 0}
                    ],
                    value=1
                )
            ], style={"margin": "10px", "flex": "1"}),

            html.Div([
                html.Label("El colegio esta en area urbana:"),
                dcc.Dropdown(
                    id='area_urbana',
                    options=[
                        {'label': 'Sí', 'value': 1},
                        {'label': 'No', 'value': 0}
                    ],
                    value=1
                )
            ], style={"margin": "10px", "flex": "1"}),

            html.Div([
                html.Label("El colegio es calendario A:"),
                dcc.Dropdown(
                    id='calendario_a',
                    options=[
                        {'label': 'Sí', 'value': 1},
                        {'label': 'No', 'value': 0}
                    ],
                    value=0
                )
            ], style={"margin": "10px", "flex": "1"}),

            html.Div([
                html.Label("¿De qué caracter es el colegio?:"),
                dcc.Dropdown(
                    id='caracter',
                    options=[
                        {'label': 'Tecnico', 'value': 'Tecnico'},
                        {'label': 'Academico', 'value': 'Academico'},
                        {'label': 'Tecnico/Academico', 'value': 'Tecnico/Academico'},
                        {'label': 'No aplica', 'value': 'No aplica'}
                    ],
                    value=0
                )
            ], style={"margin": "10px", "flex": "1"}),

            html.Div([
                html.Label("El colegio es:"),
                dcc.Dropdown(
                    id='genero',
                    options=[
                        {'label': 'Masculino', 'value': 'Masculino'},
                        {'label': 'Femenino', 'value': 'Femenino'},
                        {'label': 'Mixto', 'value': 'Mixto'}
                    ],
                    value = 'Masculino'
                )
            ], style={"margin": "10px", "flex": "1"}),

            html.Div([
                html.Label("La jornada del colegio es:"),
                dcc.Dropdown(
                    id='jornada_matutina_nocturna',
                    options=[
                        {'label': 'Matutina', 'value': 'Matutina'},
                        {'label': 'Nocturna', 'value': 'Nocturna'},
                        {'label': 'Sabatina', 'value': 'Sabatina'},
                        {'label': 'Tarde', 'value': 'Tarde'},
                        {'label': 'Unica', 'value': 'Unica'},

                    ],
                    value='Unica'
                )
            ], style={"margin": "10px", "flex": "1"}),

            html.Div([
                html.Label("El estudiante es:"),
                dcc.Dropdown(
                    id='genero_estudiante',
                    options=[
                        {'label': 'Hombre', 'value': 1},
                        {'label': 'Mujer', 'value': 0}
                    ],
                    value=1
                )
            ], style={"margin": "10px", "flex": "1"}),

            html.Div([
                html.Label("El estudiante es colombiano:"),
                dcc.Dropdown(
                    id='nacionalidad',
                    options=[
                        {'label': 'Sí', 'value': 1},
                        {'label': 'No', 'value': 0}
                    ],
                    value=1
                )
            ], style={"margin": "10px", "flex": "1"}),

            html.Div([
                html.Label("El estrato del estudiante es:"),
                dcc.Dropdown(
                    id='estrato',
                    options=[
                        {'label': '1', 'value': 1},
                        {'label': '2', 'value': 2},
                        {'label': '3', 'value': 3},
                        {'label': '4', 'value': 4},
                        {'label': '5', 'value': 5},
                        {'label': '6', 'value': 6},
                        {'label': 'Sin estrato', 'value': 0}

                    ],
                    value=3
                )
            ], style={"margin": "10px", "flex": "1"}),

            html.Div([
                html.Label("Desempeño del estudiante en ingles:"),
                dcc.Dropdown(
                    id='desempeno_ingles',
                    options=[
                        {'label': 'A-', 'value': 'A-'},
                        {'label': 'A1', 'value': 'A1'},
                        {'label': 'A2', 'value': 'A2'},
                        {'label': 'B+', 'value': 'B+'},
                        {'label': 'B1', 'value': 'B1'}
                    ],
                    value='A2'
                )
            ], style={"margin": "10px", "flex": "1"}),

        ], style={
            "display": "flex",
            "flexDirection": "row",
            "flexWrap": "wrap",
            "justifyContent": "space-around",
            "gap": "20px",  # Espacio entre los elementos
        }),

        html.Div([
            html.H2(id='output-prediccion', style={"textAlign": "center"})
        ], style={"textAlign": "center"})
    ], style={"border": "2px solid #026ab0",  # Borde azul
        "padding": "10px",  # Espaciado interno
        "borderRadius": "10px"  # Bordes redondeados
    })
])

# Callback para actualizar la gráfica de la competencia seleccionada
@app.callback(
    Output('competencia-graph', 'figure'),
    Input('competencia-dropdown', 'value')
)
def update_hist(selected_competencia):
    # Consulta SQL para obtener los datos de las competencias
    query = f"""
        SELECT {selected_competencia}
        FROM estudiantes
    """
    
    # Leer los datos de la base de datos
    df = pd.read_sql(query, engine)
    
    # Crear histograma con la distribución de la competencia seleccionada
    fig = px.histogram(df, x=selected_competencia, nbins=40)

    # Personalizar la estética de la gráfica
    fig.update_layout(
        plot_bgcolor='white',  # Fondo blanco
        xaxis_title='Puntaje',
        yaxis_title='Frecuencia',
        xaxis=dict(
            tickmode='linear',  # Modo de ticks lineal
            tick0=0,            # Primer tick en 0
            dtick=5           # Intervalo de ticks en 10
        )
    )

    return fig

# Callback para actualizar la gráfica de dispersión del nivel socioeconómico
@app.callback(
    Output('scatter-nivel-socioeconomico', 'figure'),
    Output('media', 'children'),
    Output('minimo', 'children'),
    Output('maximo', 'children'),
    Input('nivel-socioeconomico-dropdown', 'value')
)
def scatter_nivel_socioeconomico(nivel_socioeconomico):
    # Consulta SQL para obtener los datos de puntaje total y nivel socioeconómico
    query = ""
    if nivel_socioeconomico == 'fami_estratovivienda_estrato_1':
        query = """
            SELECT punt_global
            FROM estudiantes
            WHERE fami_estratovivienda_estrato_2 = 0 AND fami_estratovivienda_estrato_3 = 0 AND fami_estratovivienda_estrato_4 = 0 AND fami_estratovivienda_estrato_5 = 0 AND fami_estratovivienda_estrato_6 = 0 AND fami_estratovivienda_sin_estrato = 0
            """
    else:
        query = f"""
            SELECT punt_global
            FROM estudiantes
            WHERE {nivel_socioeconomico} = 1
            """
    
    # Leer los datos de la base de datos
    df = pd.read_sql(query, engine)

    fig = px.box(df, y="punt_global", labels={"y": "Puntaje Total"}, title="Puntaje total vs Nivel Socioeconomico {}".format(nivel_socioeconomico))
    fig.update_layout(
        plot_bgcolor='white',  # Fondo blanco
        yaxis_title='Puntaje Total'
    )

    media = html.P("El promedio de puntajes para este estrato fue: " + str(np.mean(df["punt_global"])))
    minimo = html.P("El puntaje minimo para este estrato fue: " + str(np.min(df["punt_global"])))
    maximo = html.P("El puntaje maximo para este estrato fue: " + str(np.max(df["punt_global"])))
    
    return fig, media, minimo, maximo

# Callback para la correlacion entre las variabales dependientes y la independiente
@app.callback(
    Output('output-checklist', 'figure'),
    Input('checklist', 'value')
)
def update_output(value):
    # Consulta SQL para obtener los datos de puntaje total y nivel socioeconómico
    query = f"""
        SELECT punt_global, {', '.join(value)}
        FROM estudiantes
        """
    
    # Leer los datos de la base de datos
    df = pd.read_sql(query, engine)

    correlacion = df.corr()['punt_global'].drop('punt_global')

    # Crear un gráfico de barras con la correlación de las variables seleccionadas
    fig = px.bar(x=correlacion.index, y=correlacion.values, labels={"x": "Variable", "y": "Correlación"}, title="Correlación entre variables independientes y puntaje total")
    fig.update_layout(
        plot_bgcolor='white',  # Fondo blanco
        yaxis_title='Puntaje Total'
    )

    return fig

# Callback para predecir el puntaje total en la prueba Saber 11
@app.callback(
    Output('output-prediccion', 'children'),
    Input('cole_bilingue', 'value'),
    Input('colegio_oficial', 'value'),
    Input('sede_principal', 'value'),
    Input('privado_libertad', 'value'),
    Input('num_personas_hogar', 'value'),
    Input('tienen_carro', 'value'),
    Input('tienen_computador', 'value'),
    Input('tiene_internet', 'value'),
    Input('area_urbana', 'value'),
    Input('calendario_a', 'value'),
    Input('caracter', 'value'),
    Input('genero', 'value'),
    Input('jornada_matutina_nocturna', 'value'),
    Input('genero_estudiante', 'value'),
    Input('nacionalidad', 'value'),
    Input('estrato', 'value'),
    Input('desempeno_ingles', 'value')
)
def predict_score(cole_bilingue, colegio_oficial, sede_principal, privado_libertad, 
                  num_personas_hogar, tienen_carro, tienen_computador, tiene_internet, 
                  area_urbana, calendario_a, caracter, genero, jornada_matutina_nocturna, 
                  genero_estudiante, nacionalidad, estrato, desempeno_ingles):
    datos = []
    datos.append(cole_bilingue)
    datos.append(colegio_oficial)
    datos.append(sede_principal)
    datos.append(privado_libertad)
    datos.append(num_personas_hogar)
    datos.append(tienen_carro)
    datos.append(tienen_computador)
    datos.append(tiene_internet)
    datos.append(area_urbana)
    datos.append(calendario_a)
    
    if caracter == 'No aplica':
        datos.append(1)
        datos.append(0)
        datos.append(0)
    elif caracter == 'Tecnico':
        datos.append(0)
        datos.append(1)
        datos.append(0)
    elif caracter == 'Tecnivo/Academico':
        datos.append(0)
        datos.append(0)
        datos.append(1)
    else:
        datos.append(0)
        datos.append(0)
        datos.append(0)
    
    if genero == 'Masculino':
        datos.append(1)
        datos.append(0)
    elif genero == 'Mixto':
        datos.append(0)
        datos.append(1)
    else:
        datos.append(0)
        datos.append(0)

    if jornada_matutina_nocturna == 'Matutina':
        datos.append(1)
        datos.append(0)
        datos.append(0)
        datos.append(0)
        datos.append(0)
    elif jornada_matutina_nocturna == 'Nocturna':
        datos.append(0)
        datos.append(1)
        datos.append(0)
        datos.append(0)
        datos.append(0)
    elif jornada_matutina_nocturna == 'Sabatina':
        datos.append(0)
        datos.append(0)
        datos.append(1)
        datos.append(0)
        datos.append(0)
    elif jornada_matutina_nocturna == 'Tarde':
        datos.append(0)
        datos.append(0)
        datos.append(0)
        datos.append(1)
        datos.append(0)
    elif jornada_matutina_nocturna == 'Unica':
        datos.append(0)
        datos.append(0)
        datos.append(0)
        datos.append(0)
        datos.append(1)
    else:
        datos.append(0)
        datos.append(0)
        datos.append(0)
        datos.append(0)
        datos.append(0)
    
    datos.append(genero_estudiante)
    datos.append(nacionalidad)
    
    if estrato == 1:
        datos.append(0)
        datos.append(0)
        datos.append(0)
        datos.append(0)
        datos.append(0)
        datos.append(0)
    elif estrato == 2:
        datos.append(1)
        datos.append(0)
        datos.append(0)
        datos.append(0)
        datos.append(0)
        datos.append(0)
    elif estrato == 3:
        datos.append(0)
        datos.append(1)
        datos.append(0)
        datos.append(0)
        datos.append(0)
        datos.append(0)
    elif estrato == 4:
        datos.append(0)
        datos.append(0)
        datos.append(1)
        datos.append(0)
        datos.append(0)
        datos.append(0)
    elif estrato == 5:
        datos.append(0)
        datos.append(0)
        datos.append(0)
        datos.append(1)
        datos.append(0)
        datos.append(0)
    elif estrato == 6:
        datos.append(0)
        datos.append(0)
        datos.append(0)
        datos.append(0)
        datos.append(1)
        datos.append(0)
    elif estrato == 0:
        datos.append(0)
        datos.append(0)
        datos.append(0)
        datos.append(0)
        datos.append(0)
        datos.append(1)
    
    if desempeno_ingles == 'A-':
        datos.append(1)
        datos.append(0)
        datos.append(0)
        datos.append(0)
        datos.append(0)
    elif desempeno_ingles == 'A1':
        datos.append(0)
        datos.append(1)
        datos.append(0)
        datos.append(0)
        datos.append(0)
    elif desempeno_ingles == 'A2':
        datos.append(0)
        datos.append(0)
        datos.append(1)
        datos.append(0)
        datos.append(0)
    elif desempeno_ingles == 'B+':
        datos.append(0)
        datos.append(0)
        datos.append(0)
        datos.append(1)
        datos.append(0)
    elif desempeno_ingles == 'B1':
        datos.append(0)
        datos.append(0)
        datos.append(0)
        datos.append(0)
        datos.append(1)
    else:
        datos.append(0)
        datos.append(0)
        datos.append(0)
        datos.append(0)
        datos.append(0)
    
    datos = np.array(datos).reshape(1, -1)
    datos = scaler.transform(datos)
    puntaje = model.predict(datos)

    resultado = html.P(" El puntaje pronosticado es: " + str(puntaje[0][0]))
    return resultado

# Ejecutar la aplicación
if __name__ == '__main__':
    app.run_server(host="0.0.0.0", debug=True)