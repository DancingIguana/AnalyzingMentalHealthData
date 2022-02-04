from turtle import width
import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import dash_table as dt

import plotly.graph_objs as go
import plotly.express as px
import numpy as np
import pandas as pd

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram

from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

from sklearn.decomposition import PCA
from yellowbrick.cluster import SilhouetteVisualizer

import mlxtend as mlx
import mlxtend.preprocessing as mlp
import mlxtend.frequent_patterns as FP
from mlxtend.frequent_patterns import association_rules

from sklearn.metrics import normalized_mutual_info_score

exS = [dbc.themes.BOOTSTRAP]

##==================================
##OBTENIENDO DATAFRAMES ESENCIALES
##==================================

df = pd.read_csv("Dataset/data.csv", sep='\t')

# Eliminando datos que no tienen información sobre algunas de las características generales del encuestado
df = df[df['age'] <= 120]
df = df[df['gender'] != 0]
df = df[df['familysize'] != 0]
df = df[df['familysize'] <= 17]
df = df[df['married'] != 0]
df = df[df['religion'] != 0]
df = df[df['hand'] != 0]
df = df[df['orientation'] != 0]
df = df[df['education'] != 0]
df = df[df['urban'] != 0]
df['race'] = (df['race']//10)
df[df['voted'] == 0] = 3 #
df[df['engnat'] == 0] = 3 #


dfAnswers = df.filter(regex=("Q\d*A"))
dfPersonality = df.filter(regex=("TIPI\d"))
dfGeneral = df[['education','urban','gender','engnat','age','hand','religion','orientation','race','voted','married','familysize',]]

##==================================
##         CÁLCULO DE DASS
##==================================

# Dictionary where the each answer column has a category (depression, anxiety, stress)
questionCategory = {"Depression": ["Q" + str(i) + "A" for i in [3,5,10,13,16,17,21,24,26,31,34,37,38,42]],
                   "Anxiety": ["Q" + str(i) + "A" for i in [2,4,7,9,15,19,20,23,25,28,30,36,40,41]],
                   "Stress": ["Q" + str(i) +"A" for i in [1,6,8,11,12,14,18,22,27,29,32,33,35,39]]}

# Calculating total DASS Score for each person
for category in questionCategory:
    col_list = questionCategory[category]
    df[category + "Points"] = dfAnswers[col_list].sum(axis = 1) - len(questionCategory[category])
    
#Classifying according to the DASS score of each one
conditions = [
        (df["Depression" + "Points"] <= 9),
        (df["Depression" + "Points"] >= 10) &  (df["Depression" + "Points"] <= 13),
        (df["Depression" + "Points"] >= 14) & (df["Depression" + "Points"] <= 20),
        (df["Depression" + "Points"] >= 21) & (df["Depression" + "Points"] <= 27),
        (df["Depression" + "Points"] > 27)]

values = [0,1,2,3,4]

df["Depression" + "Cat"] = np.select(conditions, values)

conditions = [
        (df["Anxiety" + "Points"] <= 7),
        (df["Anxiety" + "Points"] > 7) &  (df["Anxiety" + "Points"] <= 9),
        (df["Anxiety" + "Points"] > 9) & (df["Anxiety" + "Points"] <= 14),
        (df["Anxiety" + "Points"] > 14) & (df["Anxiety" + "Points"] <= 19),
        (df["Anxiety" + "Points"] > 19)]

values = [0,1,2,3,4]

df["Anxiety" + "Cat"] = np.select(conditions, values)

conditions = [
        (df["Stress" + "Points"] <= 14),
        (df["Stress" + "Points"] > 14) &  (df["Stress" + "Points"] <= 18),
        (df["Stress" + "Points"] > 18) & (df["Stress" + "Points"] <= 25),
        (df["Stress" + "Points"] > 25) & (df["Stress" + "Points"] <= 33),
        (df["Stress" + "Points"] > 33)]

values = [0,1,2,3,4]

df["Stress" + "Cat"] = np.select(conditions, values)

dfDASS = pd.DataFrame()

for category in questionCategory:
    dfDASS[category + "Cat"] = df[category + "Cat"]

##==================================
##     CÁLCULO DE PERSONALIDAD
##==================================
# Add personality types to data
personality_types = ['Extraversion', 'Agreeableness', 'Conscientiousness', 'EmotionalStability', 'Openness']

# Invert some entries
tipi = dfPersonality.copy()
tipi_inv = tipi.filter(regex='TIPI(2|4|6|8|10)').apply(lambda d: 7 - d)
tipi[tipi.columns.intersection(tipi_inv.columns)] = tipi_inv

# Calculate scores
for idx, pt in enumerate( personality_types ):
    df[pt] = tipi[['TIPI{}'.format(idx + 1), 'TIPI{}'.format(6 + idx)]].mean(axis=1)

personalities = df[personality_types]
personalities

##==================================
##     FUNCIONES ESENCIALES
##==================================
def pca2(data, pc_count = None):
    return PCA(n_components = 2).fit_transform(data)

def getDendrogram(dataF):
    cj = linkage(dataF, 'complete')
    
    dendrogram(cj, p = 5, truncate_mode = 'level')#, p = 5, truncate_mode='level')
    plt.ylabel("Distancia",fontsize = 16)
    plt.xlabel("Encuestado",fontsize = 16)
    plt.show()

def elbowMethod(dataF, n = 8):
    model = KMeans()
    
    visualizer = KElbowVisualizer(model, k = (1,n))
    visualizer.fit(dataF)
    visualizer.show()

    
def getClustersCentroidsDF(dataF, n = 2):
    km = KMeans(n_clusters = n, init = 'k-means++', n_init = 10, max_iter = 1000, random_state = 42)
    pg_cl = km.fit_predict(dataF)
    pg_centroids = km.cluster_centers_
    d = {}
    for i in range(len(pg_centroids)):
        d[f"Cluster {i}"] = pg_centroids[i]

    profiles_df = pd.DataFrame(data = d)
    profiles_df.index = [col for col in dataF.columns]

    return pg_cl, pg_centroids, profiles_df

def plotInPC12(dataMatrix, clusters):
    X = pca2(dataMatrix)
    cdict = {0: '#003f5c', 1: '#ffa600', 2: '#dd5182', 3: '#ff6e54', 4:'#444e86', 5:'#955196' }
    
    fig, ax = plt.subplots()
    for g in np.unique(clusters):
        ix = np.where(clusters == g)
        plt.scatter(X[:,0][ix], X[:,1][ix], c = cdict[g], label = "Cluster " + str(g))
    
    plt.scatter(X[:,0], X[:,1], c = pg_cl)
    plt.xlabel("CP 1", fontsize = 16)
    plt.ylabel("CP 2", fontsize = 16)
    plt.legend()

def getSilhouettePlots(dfData):
    from yellowbrick.cluster import SilhouetteVisualizer
    for i in [2,3,4,5,6]:
        model = KMeans(i, random_state=42)
        visualizer = SilhouetteVisualizer(model, colors='yellowbrick')

        visualizer.fit(dfData)        # Fit the data to the visualizer
        visualizer.show()

def getFP(dataF, minS = 0.1):
    dataFBin = pd.get_dummies(dataF, 
                    columns=dataF.columns,
                    dtype = bool)
    pf = FP.apriori(dataFBin, min_support=minS,use_colnames=True)
    
    return pf

def getAR(FP, minT = 0.8):
    return association_rules(FP, metric='confidence', min_threshold=minT)

def dfBin(dataF):
    return pd.get_dummies(dataF, 
                    columns=dataF.columns,
                    dtype = bool)

# ======================================
#              CÁLCULO DE CLUSTERS
# ======================================
clusterDFs = {"General": {1:None,2:None,3:None,4:None,5:None,6:None,7:None,8:None},
                "Personalities":{1:None,2:None,3:None,4:None,5:None,6:None,7:None,8:None},
                 "DASS": {1:None,2:None,3:None,4:None,5:None,6:None,7:None,8:None}}
clustersTables =  {"General": {1:pd.DataFrame({"Sin clusters": [0]}),2:None,3:None,4:None,5:None,6:None,7:None,8:None},
                "Personalities":{1:pd.DataFrame({"Sin clusters": [0]}),2:None,3:None,4:None,5:None,6:None,7:None,8:None},
                 "DASS": {1:pd.DataFrame({"Sin clusters": [0]}),2:None,3:None,4:None,5:None,6:None,7:None,8:None}}
print("Calculando clusters")
for cluster in ["General", "Personalities", "DASS"]:
    if cluster == "General":
        dfAux2 = dfGeneral.copy()
        dfAux2['age'] = dfAux2['age']//10
        dfAux = dfBin(dfAux2)
        #print(dfAux)
        dataMatrix = np.asarray(dfAux)*1
    elif cluster == "Personalities":
        dfAux = personalities.copy()
        dataMatrix = np.asarray(dfAux)
    elif cluster == "DASS":
        dfAux = dfDASS.copy()
        dataMatrix = np.asarray(dfAux)

    CPDF = pd.DataFrame()
    X = pca2(dataMatrix)
    CPDF["CP1"] = X[:,0]
    CPDF["CP2"] = X[:,1]

    for i in range(2,9):
        CPDF2 = CPDF.copy()
        GenCl, GenCentroids, GenClusDF = getClustersCentroidsDF(dfAux, n = i)
        CPDF2["Cluster"] = [str(i) for i in GenCl]
        clusterDFs[cluster][i] = CPDF2
        clustersTables[cluster][i] = GenClusDF
        #print("\n")
        #print(clustersTables[cluster][i])
        clustersTables[cluster][i].insert(loc = 0, column = "Atributo",value = clustersTables[cluster][i].index)
print("Listo")
# ======================================
#            PÁRRAFOS
# ======================================
p1 = "Algunos de los fenómenos más estudiados en áreas como la psicología son aquellos relacionados con las enfermedades mentales. Entre ellas se encuentran el estrés, la depresión y ansiedad. Con el objetivo de poder identificar sujetos que padezcan alguna de éstas y determinar qué tan grave es su estado, se hace del uso de herramientas como lo son los cuestionarios y las encuestas. A partir de los resultados de éstas es posible obtener una idea general acerca del estado de salud mental de una persona."
p2 = "También ocurre algo similar en el estudio de la personalidad de una persona. Dependiendo del tipo de clasificación de éstas que se esté haciendo, se hace un determinado cuestionario y a partir de los resultados se extraen las características de la personalidad de alguien."
p3 = "Cuando relacionamos a estos dos tipos de análisis o estudio, no sería extraño hacer preguntas como, ¿se puede determinar en cierta medida el nivel de depresión, estrés o ansiedad a partir de los resultados de un cuestionario de personalidad o viceversa? O, ¿qué tanta relación hay con la personalidad de una persona y la severidad de las enfermedades mentales que pueda tener? O incluso, ¿puede que también otros factores fuera de los perfiles de personalidad y de estados de salud mental, como lo son la edad, género, religión, raza, etc., tengan influencia sobre estos resultados?\nEl objetivo general de este proyecto será hacer una exploración de un conjunto de datos que cuenta con cuestionarios del perfil general, personalidad, niveles de estrés, depresión y ansiedad de varias personas. A partir del uso de técnicas y herramientas de minería de datos como patrones frecuentes, reglas de asociación y clustering trataremos de buscar respuestas a las preguntas planteadas hace un momento."
p4 = "Lo que se presenta en este Tablero es  un conjunto de gráficas que nos permiten darnos una idea visual de los datos con los cuáles estamos trabajando. Como gráfica adicional, también incluiremos algunos de los resultados de los agrupamientos discutidos en el reporte de este proyecto."

p5 = "Un tipo de datos con los que se trabajó en este proyecto fueron los relacionados a la información general de las personas."
p6 = "Estos incluyen la edad, el género, el nivel de educación, la raza, entre otros. En la figura de al lado se muestra cada uno de ellos y la distribución que tienen entre los encuestados. Estos mismos fueron utilizados en el proyecto para construir los perfiles generales de cada una de las personas. Al final resultó que los clusters que ayudaban a diferenciar más a la población estaban construidos principalmente con base en la raza, la religión y la lengua nativa de los encuestados."
p7 = "Uno de estos grupos constaba de personas asiáticas con religión musulmana y cuya lengua nativa no era el inglés; mientras que el otro grupo era de personas blancas, generalmente agnósticas o ateas cuya lengua nativa sí era el inglés. Esto nos llevó a la conclusión de que la encuesta probablemente se llevó a cabo principalmente en dos regiones y cada una tenía un perfil de personas de este tipo."
p8 = "Donde no se vieron cambios significativos entre los cluster fue en parámetros como la edad, notemos que ésta no es muy variada entre los encuestados, la mayoría de las personas está entre los 15 y 30 años."

p9 = "El otro tipo de datos con los que contábamos fueron las respuestas del cuestionario enfocadas a determinar niveles de estrés, depresión y ansiedad en categorías DASS (Depression Anxiety Stress Scales)"
p10 = "Cada una de las tres enfermedades mentales está clasificada en cinco niveles de intensidad. Los puntajes generales resultantes de todo eso se muestran en la figura de abajo a la derecha. Al lado izquierdo se muestra la frecuencia de las distintas respuestas directas que se obtuvieron en la encuesta (van del 1 al 4, representando de menor a mayor qué tan identificado se sentía el encuestado con la pregunta planteada)."

p11 = "El último tipo de datos fueron las respuestas a un cuestionario de personalidad. Éstas se encontraban en una escala del 1 al 7, donde 4 representa un punto neutro (no está de acuerdo ni en desacuerdo). La frecuencia con la que se vieron las respuestas se encuentran en la figura de abajo en la izquierda."
p12 = "A partir de eso se generan cuatro valores que caracterizan a la personalidad. Cada uno con un valor sobre una escala del mismo tipo que las respuestas de las preguntas. Esto se muestra en la gráfica de abajo en la derecha."

p13 = "Algunos de los resultados finales que se presentaron en este proyecto fue la creación de perfiles de personalidad, generales y de niveles de depresión, estrés y ansiedad. En la figura de abajo se muestran los distintos cluster que se pueden formar a partir de K-means usando estos datos. Además, la tabla que viene abajo de la figura ayuda a interpretar los cluster, ya que contiene los centroides correspondientes de cada uno. De esta manera podemos caracterizar los grupos de cada tipo de perfil y tratar de visualizarlos sobre los primeros dos componentes principales"
# ======================================
#            VALORES EN TEXTO
# ======================================
textValues = {
    'education': {1: "Less than high school",2:"High school", 3:"University degree", 4:"Graduate degree"},
    'urban' : {1:'Rural (country side)', 2:'Suburban', 3:'Urban (town, city)'}, 
    'gender': {1: 'Male', 2:'Female', 3:'Other'},
    'engnat': {1: 'Yes', 2:'No', 3: 'Unknown'},
    'hand': {1: 'Right', 2: 'Left', 3: 'Both'},
    'religion': {1:'Agnostic', 2:'Atheist', 3:'Buddhist', 4:'Christian (Catholic)', 5:'Christian (Mormon)', 6:'Christian (Protestant)', 7:'Christian (Other)', 8:'Hindu', 9:'Jewish', 10:'Muslim', 11:'Sikh', 12:'Other'},
    'orientation': {1:'Heterosexual', 2:'Bisexual', 3:'Homosexual', 4:'Asexual', 5:'Other'},
    'race': {1:'Asian', 2:'Arab', 3:'Black', 4:'Indigenous Australian', 50:'Native American', 60:'White', 70:'Other'},
    'voted': {1:'Yes', 2:'No', 3:'Unknown'},
    'married': {1:'Never married', 2:'Currently married', 3:'Previously married'}
    }

questionValues = {
    'Q1A':	'I found myself getting upset by quite trivial things.',
    'Q2A':	'I was aware of dryness of my mouth.',
    'Q3A':	'I couldnt seem to experience any positive feeling at all.',
    'Q4A':	'I experienced breathing difficulty (eg, excessively rapid breathing, breathlessness in the absence of physical exertion).',
    'Q5A':	'I just couldnt seem to get going.',
    'Q6A':	'I tended to over-react to situations.',
    'Q7A':	'I had a feeling of shakiness (eg, legs going to give way).',
    'Q8A':	'I found it difficult to relax.',
    'Q9A':	'I found myself in situations that made me so anxious I was most relieved when they ended.',
    'Q10A':	'I felt that I had nothing to look forward to.',
    'Q11A':	'I found myself getting upset rather easily.',
    'Q12A':	'I felt that I was using a lot of nervous energy.',
    'Q13A':	'I felt sad and depressed.',
    'Q14A':	'I found myself getting impatient when I was delayed in any way (eg, elevators, traffic lights, being kept waiting).',
    'Q15A':	'I had a feeling of faintness.',
    'Q16A':	'I felt that I had lost interest in just about everything.',
    'Q17A':	'I felt I wasnt worth much as a person.',
    'Q18A':	'I felt that I was rather touchy.',
    'Q19A':	'I perspired noticeably (eg, hands sweaty) in the absence of high temperatures or physical exertion.',
    'Q20A':	'I felt scared without any good reason.',
    'Q21A':	'I felt that life wasnt worthwhile.',
    'Q22A':	'I found it hard to wind down.',
    'Q23A':	'I had difficulty in swallowing.',
    'Q24A':	'I couldnt seem to get any enjoyment out of the things I did.',
    'Q25A':	'I was aware of the action of my heart in the absence of physical exertion (eg, sense of heart rate increase, heart missing a beat).',
    'Q26A':	'I felt down-hearted and blue.',
    'Q27A':	'I found that I was very irritable.',
    'Q28A':	'I felt I was close to panic.',
    'Q29A':	'I found it hard to calm down after something upset me.',
    'Q30A':	'I feared that I would be &quot;thrown&quot; by some trivial but unfamiliar task.',
    'Q31A':	'I was unable to become enthusiastic about anything.',
    'Q32A':	'I found it difficult to tolerate interruptions to what I was doing.',
    'Q33A':	'I was in a state of nervous tension.',
    'Q34A':	'I felt I was pretty worthless.',
    'Q35A':	'I was intolerant of anything that kept me from getting on with what I was doing.',
    'Q36A':	'I felt terrified.',
    'Q37A':	'I could see nothing in the future to be hopeful about.',
    'Q38A':	'I felt that life was meaningless.',
    'Q39A':	'I found myself getting agitated.',
    'Q40A':	'I was worried about situations in which I might panic and make a fool of myself.',
    'Q41A':	'I experienced trembling (eg, in the hands).',
    'Q42A':	'I found it difficult to work up the initiative to do things.'
}

personalityValues = {
    'TIPI1':	'Extraverted, enthusiastic.',
    'TIPI2':	'Critical, quarrelsome.',
    'TIPI3':	'Dependable, self-disciplined.',
    'TIPI4':	'Anxious, easily upset.',
    'TIPI5':	'Open to new experiences, complex.',
    'TIPI6':	'Reserved, quiet.',
    'TIPI7':	'Sympathetic, warm.',
    'TIPI8':	'Disorganized, careless.',
    'TIPI9':	'Calm, emotionally stable.',
    'TIPI10':	'Conventional, uncreative.'
}

# ======================================
#                 DASH
# ======================================

tablero = dash.Dash(__name__, external_stylesheets = exS)
# ======================================
#      DEFINIENDO LOS COMPONENTES
# ======================================

D1 = dcc.Dropdown(id = 'D1',
                 options =[{'label': c, 'value': c} for c in dfGeneral.columns],
                 value = 'age')
D2 = dcc.Dropdown(id = 'D2',
                 options =[{'label': c[:-1] + f": {questionValues[c]}", 'value': c} for c in dfAnswers.columns],
                 value = 'Q1A')

D3 = dcc.Dropdown(id = 'D3',
                 options =[{'label': c[:-3], 'value': c} for c in dfDASS.columns],
                 value = 'DepressionCat')

D4 = dcc.Dropdown(id = 'D4',
                 options =[{'label': c, 'value': c} for c in personalities.columns],
                 value = 'EmotionalStability')

D5 = dcc.Dropdown(id = 'D5',
                 options =[{'label': c + f": {personalityValues[c]}", 'value': c} for c in dfPersonality.columns],
                 value = 'TIPI1')


D6 = dcc.Dropdown(id = 'D6',
                 options =[{'label': c, 'value': c} for c in ["General", "Personalities", "DASS"]],
                 value = 'General')

D7 = dcc.Slider(id = 'D7',
               min = 1,
               max = 8,
               value = 2,
               step = 1,
               tooltip = {'placement':'bottom',
                         'always_visible':False})

G1 = dcc.Graph(id = 'G1')
G2 = dcc.Graph(id = 'G2')
G3 = dcc.Graph(id = 'G3')
G4 = dcc.Graph(id = 'G4')
G5 = dcc.Graph(id = 'G5')
G6 = dcc.Graph(id = 'G6')
Tabla1 = dcc.Graph(id = 'G7')
# ======================================
# Establecer el tablero
# ======================================
tablero.layout = html.Div([
    dbc.Row([
        html.H1(children = "Exploración de perfiles de datos generales, personalidad y niveles de estrés, depresión y ansiedad",
         style = {'text-align': 'center', 'fontSize': '50px'}),
        html.H4(children = "Juan Pablo Maldonado Castro", style = {'text-align': 'center'}),
        html.H4(children = "Minería de Datos", style = {"text-align": 'center'}),
        html.H4(children = "UNAM - ENES Morelia", style = {"text-align": 'center'}),
        html.H4(children = "maldonadocastrojp@gmail.com", style = {"text-align": 'center', "fontSize": "20px"}),
    ]),
    dbc.Row([
        html.H2(children = "Introducción",
        ),
        html.P(p1),
        html.P(p2),
        html.P(p3),
        html.P(p4)
    ]),
    #ESPACIO PARA TENER TÍTULO DE SECCIÓN Y MENÚ DROPDOWN DE PREGUNTAS GENERALES
    dbc.Row([
        dbc.Col(html.H2("Datos Generales"),
           width = 6,
           ),

        dbc.Col(html.P(""),
           width = 3,
           ),
        dbc.Col(D1,
           width = 3,
           ),
    ]),

    # GRÁFICA Y TEXTO DE LOS DATOS GENERALES
    dbc.Row([
        dbc.Col([html.P(p5),html.P(p6),html.P(p7),html.P(p8)]
           #width = 6,
           ),

        dbc.Col(G1,
           width = 6,
           ),
    ]),

    # ESPACIO BLANCO
    dbc.Row([
        dbc.Col(html.P(""),
            )
    ]),
    dbc.Row([
        dbc.Col(html.H2("Datos de estrés, depresión y ansiedad"),
           width = 6,
           ),
    ]),
    dbc.Row([
        dbc.Col([html.P(p9),html.P(p10)]
            )
    ]),
    # ESPACIO PARA MENÚ DROPDOWN DE PREGUNTAS DASS Y TÍTULO
    dbc.Row([
        dbc.Col(D2,
           width = 6,
           ),

        dbc.Col(D3,
           width = 6,
           ),
    ]),

    #GRÁFICA Y TEXTO DE LAS PREGUNTAS DASS
    dbc.Row([
        dbc.Col(G2,
           width = 6,
           ),

        dbc.Col(G3,
           width = 6,
           ),
    ]),

    # ESPACIO BLANCO
    dbc.Row([
        dbc.Col(html.P(""),
            )
    ]),

    dbc.Row([
        dbc.Col(html.H2("Datos de personalidad"),
           width = 6,
           ),
    ]),

    dbc.Row([
        dbc.Col([html.P(p11),html.P(p12)]
            )
    ]),
    # ESPACIO PARA MENÚ DROPDOWN DE PREGUNTAS PERSONALIDAD
    dbc.Row([
        dbc.Col(D5,
           width = 6),

        dbc.Col(D4,
           width = 6),
    ]),

    #GRÁFICA Y TEXTO DE LAS PREGUNTAS DASS
    dbc.Row([
        dbc.Col(G5,
           width = 6),

        dbc.Col(G4,
           width = 6),
    ]),
    
    # ESPACIO BLANCO
    dbc.Row([
        dbc.Col(html.P(""),
        )
    ]),

    dbc.Row([
        dbc.Col([html.H2("Agrupamiento de perfiles")]
            )
    ]),

    dbc.Row([
        dbc.Col([html.P(p13)]
            )
    ]),
    dbc.Row([
        dbc.Col(D6,
            width = 6),
        dbc.Col(html.P("K clusters"),
            width = 1
            ),
        dbc.Col(D7,
            width = 5),
    ]),
    dbc.Row([
        dbc.Col(G6,
    
            width = 12 )        
    ]),
    dbc.Row([dbc.Col(html.H3("Tabla de centroides por cluster")),]),
    dbc.Row([
        dbc.Col(
            dt.DataTable(
            id='tbl', data=clustersTables["Personalities"][2].to_dict('records'),
            columns=[{"name": i, "id": i} for i in clustersTables["Personalities"][2].columns],
            ),
        )
    ]),
    dbc.Row([
        dbc.Col(
            html.H2("Recreación y análisis de resultados")
        )
    ]),
    dbc.Row([
        dbc.Col(
            html.P("Para recrear estos y más resultados obtenidos en este proyecto puede utilizarse el Notebook anexo a la entrega del mismo. En caso de que se quiera ver la interpretación y análisis de estos, véase el reporte que también viene con esta entrega.")
        )
    ]),
    dbc.Row([
        dbc.Col(
            html.H2("Conjunto de datos utilizados")
        )
    ]),
    dbc.Row([
        dbc.Col(
            html.P("Peleg, Yam. 2021. Predicting Depression, Anxiety and Stress. Kaggle. https://www.kaggle.com/yamqwe/depression-anxiety-stress-scales")
        )
    ]),
], className = 'container')


# ======================================
# Definir los callbacks
# ======================================
@tablero.callback(Output('G1','figure'),
             Input('D1','value'))
             
def graficarValores(col):
    if col in ['age', 'familysize']:
        t1 = go.Histogram(x = df[col])
    else:
        t1 = go.Bar(x = [textValues[col][i] for i in df[col].unique()],
                    y = [df[col].value_counts()[i+1] for i in range(len(df[col].unique()))])
    F1 = go.Figure(data = [t1])
    F1.update_xaxes(title = col)
    F1.update_yaxes(title = f'Frecuencia')
    F1.update_layout(title=f'Frecuencia de {col}')
    return F1

@tablero.callback(Output('G2','figure'),
             Input('D2','value'))
             
def graficarValores(col):
    t1 = go.Bar(x = ['1','2','3','4'],
                y = [dfAnswers[col].value_counts()[i] for i in [1,2,3,4]],
                marker_color = ["#ef8400", "#dc6100", '#c73c00', '#b00000'])
    F1 = go.Figure(data = [t1])
    F1.update_xaxes(title = "Respuesta")
    F1.update_yaxes(title = f'Frecuencia')
    F1.update_layout(title=f'{questionValues[col]}')
    return F1

@tablero.callback(Output('G3','figure'),
             Input('D3','value'))


def graficarValores(col):
    t1 = go.Bar(x = ['Normal','Ligero','Moderado','Severo','Extr. severo'],
                y = [dfDASS[col].value_counts()[i] for i in [0,1,2,3,4]],
                marker_color = ["#ffa600", "#ef8400", '#dc6100', '#c73c00', '#b00000'])
    F1 = go.Figure(data = [t1])
    F1.update_xaxes(title = "Nivel de severidad")
    F1.update_yaxes(title = f'Frecuencia')
    F1.update_layout(title=f'Nivel de {col[:-3]} entre los encuestados')
    return F1


@tablero.callback(Output('G4','figure'),
             Input('D4','value'))

def graficarValores(col):
    t1 = go.Histogram(x = list(personalities[col]))
    F1 = go.Figure(data = [t1])
    F1.update_xaxes(title = "Nivel")
    F1.update_yaxes(title = f'Frecuencia')
    F1.update_layout(title=f'{col}')
    return F1

@tablero.callback(Output('G5','figure'),
             Input('D5','value'))
             
def graficarValores(col):
    t1 = go.Bar(x = ['1','2','3','4','5','6','7'],
                y = [dfPersonality[col].value_counts()[i] for i in [1,2,3,4,5,6,7]])
    F1 = go.Figure(data = [t1])
    F1.update_xaxes(title = "Nivel")
    F1.update_yaxes(title = f'Frecuencia')
    F1.update_layout(title= personalityValues[col])
    return F1

@tablero.callback(Output('G6','figure'),
                Output('tbl','data'),
                Output('tbl','columns'),
             Input('D6','value'),
             Input('D7', 'value'))
             
def graficarValores(group,k):
    if(k == 1):
        df = clusterDFs[group][2]
        t1 = go.Scatter(x = df["CP1"], y = df["CP2"], mode = "markers")
        F1 = go.Figure(data = [t1])
        F1.update_xaxes(title = "CP1")
        F1.update_yaxes(title = f'CP2')
        F1.update_layout(title=f'Datos de {group} sobre los primeros dos componentes principales')
    else:
        df = clusterDFs[group][k]
        F1 = px.scatter(df, x="CP1", y="CP2", color="Cluster",
                 title=f"Clusterings de {group} con k = {k} sobre los primeros dos componentes principales")
    return F1, clustersTables[group][k].to_dict('records'), [{"name": i, "id": i} for i in clustersTables[group][k].columns]



# ======================================
#           FUNCIÓN PRINCIPAL
# ======================================
if __name__ == "__main__":
    tablero.run_server(debug = True)