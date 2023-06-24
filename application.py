from dash import Dash, html, dcc, Input, Output,ctx  # pip install dash
import dash_bootstrap_components as dbc          # pip install dash-bootstrap-components
import pandas as pd                              # pip install pandas

import matplotlib                                # pip install matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import base64
from io import BytesIO

import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score            #pip install -U scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.cluster import KMeans

from fpdf import FPDF
from dash.exceptions import PreventUpdate
import calendar
import os
from datetime import datetime

# ========== Styles ============ #

plt.style.use('dark_background')
plt.xticks(rotation=30,fontweight="bold")

tab_card = {'height': '100%'}
tab_card_graph = {'height': '100%','background-color':'black'}
pathFile = os.path.join(os.getcwd(), 'src','Steel_industry_data.csv')

data = pd.read_csv(pathFile)
data_origem = pd.read_csv(pathFile)


## TRATAMENTO
## Converter dados para  formato numérico
data['date'] = pd.to_datetime(data['date'], dayfirst=True)
data['Usage_kWh'] = pd.to_numeric(data['Usage_kWh'])
data['Lagging_Current_Reactive.Power_kVarh'] = pd.to_numeric(data['Lagging_Current_Reactive.Power_kVarh'])
data['Leading_Current_Reactive_Power_kVarh'] = pd.to_numeric(data['Leading_Current_Reactive_Power_kVarh'])
data['CO2(tCO2)'] = pd.to_numeric(data['CO2(tCO2)'])
data['Lagging_Current_Power_Factor'] = pd.to_numeric(data['Lagging_Current_Power_Factor'])
data['Leading_Current_Power_Factor'] = pd.to_numeric(data['Leading_Current_Power_Factor'])
data['NSM'] = pd.to_numeric(data['NSM']) 
data['Month'] = pd.to_numeric(data['date'].apply(lambda x: str(x.month)))
data['Day'] = pd.to_numeric(data['date'].apply(lambda x: str(x.day)))


## Converter Dados Categóricos em dados Numerais
data['WeekStatus'].replace(['Weekday', 'Weekend'], [0, 1], inplace=True)
data['Day_of_week'].replace(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], [0, 1,2,3,4,5,6], inplace=True)
data['Load_Type'].replace(['Light_Load', 'Medium_Load', 'Maximum_Load'], [0,1,2], inplace=True)

## Apagar dados faltantes
data = data.dropna()

# To dict - para salvar no dcc.store
df_store = data.to_dict()
df_origem = data_origem.to_dict()

## Usar datetime (tempo) com index dos dados do dataset
data.reset_index()
data.set_index("date", inplace=True)

## Dividir conjuntos para treinamento e testes
X = data.drop('Usage_kWh', axis = 1)
y = data['Usage_kWh']
X = X.values
y = y.values
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 10, shuffle = False)

#### Regressão Linear
#model_SVR = SVR().fit(x_train, y_train)
model = LinearRegression().fit(x_train, y_train)

#Predições
y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)

#Predições_SVR
# y_train_pred_SVR = model_SVR.predict(x_train)
# y_test_pred_SVR = model_SVR.predict(x_test)

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = 'UEA - Indústria 4.0'
app.layout = dbc.Container([

# Armazenamento de dataset
    dcc.Store(id='dataset_origin', data=df_origem),
    dcc.Store(id='dataset_fixed', data=df_store),
    html.H1("UEA - Indústria 4.0", className='mb-2', style={'textAlign':'center'}),
    html.Hr(),

# Param 
    dbc.Row([
        dbc.Col([
            dbc.Card([                
                dbc.Row([
                    dbc.Col([
                        html.Legend('Months:.'),
                    ], sm=2, md=2, style={'margin-top': '5px'}),
                    dbc.Col([
                        dcc.RangeSlider(
                        id='rangeslider',
                        marks= {int(x): f'{x}' for x in data['Month'].unique()},
                        step=3,                
                        min=1,
                        max=12,
                        value=[1,12],   
                        dots=True,             
                        pushable=0,
                        tooltip={'always_visible':False, 'placement':'bottom'}, 
                        )
                    ], sm=10, md=10, style={'margin-top': '15px'}),
                ], className='g-1', style={'height': '20%', 'justify-content': 'center'}),

            ], style=tab_card)
        ])
    ], className='main_row g-2 my-auto'),
# Param 
    dbc.Row([
    dbc.Col([
        dbc.Card([    
                dbc.Row([
                    dbc.Col([
                        html.Legend('Field:.'),
                    ], sm=2, md=2, style={'margin-top': '1px','margin-botton': '15px'}),
                    dbc.Col([
                        dcc.Dropdown(
                        id='category',
                        value='Usage_kWh',
                        clearable=False,
                        options=data.columns[0:])
                    ], sm=4, md=4, style={'margin-top': '10px','margin-left': '15px','margin-botton': '15px'}),
                    dbc.Col([
                        dbc.Button("Reports", 
                        color="primary",
                        id="btn-reports"),
                        dcc.Download(id="download-pdf"),
                    ], sm=4, md=4, style={'margin-top': '8px'}),
                    dcc.Loading(
                        id="loading-1",
                        type="default",
                        color="rgb(42, 90, 72)",
                        children=html.Div(id="loading-output-1")
                    ),
                    dcc.Loading(
                        id="loading-2",
                        type="default",
                        color="rgb(42, 90, 72)",
                        children=html.Div(id="loading-output-2")
                    ),
                ], className='g-1', style={'height': '20%', 'justify-content':'left','margin-botton': '15px'})
            ], style=tab_card)
        ])
    ], className='main_row g-2 my-auto'),

# Graph01
    dbc.Row([
        dbc.Col([
            dbc.Card([                
                dbc.Row([
                    dbc.Col([
                         html.H4(id='titleGraph01', className='mb-2', style={'textAlign':'left'}),               
                        ], sm=9, md=9, style={'margin-top': '15px'}),
                ], className='g-1', style={'height': '20%', 'justify-content': 'center'}),

            ], style=tab_card,class_name='card-title')
        ])
    ], className='main_row g-2 my-auto'),
    dbc.Row([
        dbc.Col([
            dbc.Card([                
                dbc.Row([
                    dbc.Col([
                        html.Img(id='bar-graph-matplotlib1',style={'text-align': 'center'})
                    ], sm=9, md=9, style={'margin-top': '5px'}),
                ], className='g-1', style={'height': '20%', 'justify-content': 'center'}),

            ], style=tab_card_graph)
        ])
    ], className='main_row g-2 my-auto'),
    dbc.Row([
        dbc.Col([
            dbc.Card([                
                dbc.Row([
                    dbc.Col([
                        html.Img(id='bar-graph-matplotlib1_update')
                    ], sm=9, md=9, style={'margin-top': '5px'}),
                ], className='g-1', style={'height': '20%', 'justify-content': 'center'}),

            ], style=tab_card_graph)
        ])
    ], className='main_row g-2 my-auto'),
#Graph10
    dbc.Row([
        dbc.Col([
            dbc.Card([                
                dbc.Row([
                    dbc.Col([
                        html.Img(id='bar-graph-matplotlib10')
                    ], sm=9, md=9, style={'margin-top': '5px'}),
                ], className='g-1', style={'height': '20%', 'justify-content': 'center'}),

            ], style=tab_card_graph)
        ])
    ], className='main_row g-2 my-auto'),
#Heatmap
    dbc.Row([
        dbc.Col([
            dbc.Card([                
                dbc.Row([
                    dbc.Col([
                        html.H3("Heatmap - Gráfico de Correlação", className='mb-2', style={'textAlign':'left'}),
                        html.Plaintext("Mapa de calor para indicar, visualmente, a força da correlação entre as variáveis, lembrando que correlação:"),
                        html.Plaintext(" 1: Forte correção positiva (as variáveis crescem no mesmo sentido)"),
                        html.Plaintext("-1: Forte correção negativa (as variáveis crescem em sentidos opostos)"),
                        html.Plaintext(" 0: Correlação muito fraca/ inexistente (não é possível informar possível influencia entre as variáveis)."),
                    ], sm=9, md=9, style={'margin-top': '15px'}),
                ], className='g-1', style={'height': '20%', 'justify-content': 'center'}),

            ], style=tab_card,class_name='card-title')
        ])
    ], className='main_row g-2 my-auto'),

    dbc.Row([
        dbc.Col([
            dbc.Card([                
                dbc.Row([
                    dbc.Col([
                        html.Img(id='bar-graph-matplotlib2',style={'margin-left':'-50px'})
                    ], sm=8, md=8, style={'margin-top': '5px'}),
                ], className='g-1', style={'height': '20%', 'justify-content': 'center'}),

            ], style=tab_card_graph)
        ])
    ], className='main_row g-2 my-auto'),

    dbc.Row([
        dbc.Col([
            dbc.Card([                
                dbc.Row([
                    dbc.Col([
                        html.Img(id='bar-graph-matplotlib2_update',style={'margin-left': '-100px'})
                    ], sm=9, md=9, style={'margin-top': '5px'}),
                ], className='g-1', style={'height': '20%', 'justify-content': 'center'}),

            ], style=tab_card_graph)
        ])
    ], className='main_row g-2 my-auto'),

#Test Set
    dbc.Row([
        dbc.Col([
            dbc.Card([                
                dbc.Row([
                    dbc.Col([
                         html.H3("Gráfico - Test Set", className='mb-2'),                    
                        ], sm=9, md=9, style={'margin-top': '15px'}),
                ], className='g-1', style={'height': '20%', 'justify-content': 'center'}),

            ], style=tab_card,class_name='card-title')
        ])
    ], className='main_row g-2 my-auto'),
    dbc.Row([
        dbc.Col([
            dbc.Card([                
                dbc.Row([
                    dbc.Col([
                        html.Img(id='bar-graph-matplotlib3',style={'text-align': 'center'})
                    ], sm=12, md=12, style={'margin-top': '5px'}),
                ], className='g-1', style={'height': '20%', 'margin-left': '50px'}),

            ], style=tab_card_graph)
        ])
    ], className='main_row g-2 my-auto'),

#Training Set
    dbc.Row([
        dbc.Col([
            dbc.Card([                
                dbc.Row([
                    dbc.Col([
                         html.H3("Gráfico - Training Set", className='mb-2'),                    
                        ], sm=9, md=9, style={'margin-top': '15px'}),
                ], className='g-1', style={'height': '20%', 'justify-content': 'center'}),

            ], style=tab_card,class_name='card-title')
        ])
    ], className='main_row g-2 my-auto'),
    dbc.Row([
        dbc.Col([
            dbc.Card([                
                dbc.Row([
                    dbc.Col([
                        html.Img(id='bar-graph-matplotlib4',style={'text-align': 'center'})
                    ], sm=12, md=12, style={'margin-top': '5px'}),
                ], className='g-1', style={'height': '20%', 'margin-left': '50px'}),

            ], style=tab_card_graph)
        ])
    ], className='main_row g-2 my-auto'),

#Boxplot
    dbc.Row([
        dbc.Col([
            dbc.Card([                
                dbc.Row([
                    dbc.Col([
                         html.H3("Gráfico - Boxplot", className='mb-2'),                    
                        ], sm=9, md=9, style={'margin-top': '15px'}),
                ], className='g-1', style={'height': '20%', 'justify-content': 'center'}),

            ], style=tab_card,class_name='card-title')
        ])
    ], className='main_row g-2 my-auto'),
    dbc.Row([
        dbc.Col([
            dbc.Card([                
                dbc.Row([
                    dbc.Col([
                        html.Img(id='bar-graph-matplotlib5',style={'text-align': 'center'})
                    ], sm=12, md=12, style={'margin-top': '5px'}),
                ], className='g-1', style={'height': '20%', 'margin-left': '40px'}),

            ], style=tab_card_graph)
        ])
    ], className='main_row g-2 my-auto'),
    dbc.Row([
        dbc.Col([
            dbc.Card([                
                dbc.Row([
                    dbc.Col([
                         html.H3("Gráfico - Boxplot - Day of Week", className='mb-2'),                    
                        ], sm=9, md=9, style={'margin-top': '15px'}),
                ], className='g-1', style={'height': '20%', 'justify-content': 'center'}),

            ], style=tab_card,class_name='card-title')
        ])
    ], className='main_row g-2 my-auto'),
    dbc.Row([
        dbc.Col([
            dbc.Card([                
                dbc.Row([
                    dbc.Col([
                        html.Img(id='bar-graph-matplotlib5_update',style={'text-align': 'center'})
                    ], sm=12, md=12, style={'margin-top': '5px'}),
                ], className='g-1', style={'height': '20%', 'margin-left': '40px'}),

            ], style=tab_card_graph)
        ])
    ], className='main_row g-2 my-auto'),

#Training data prediction
    dbc.Row([
        dbc.Col([
            dbc.Card([                
                dbc.Row([
                    dbc.Col([
                         html.H3("Gráfico - Training data prediction", className='mb-2'),                    
                        ], sm=9, md=9, style={'margin-top': '15px'}),
                ], className='g-1', style={'height': '20%', 'justify-content': 'center'}),

            ], style=tab_card,class_name='card-title')
        ])
    ], className='main_row g-2 my-auto'),
    dbc.Row([
        dbc.Col([
            dbc.Card([                
                dbc.Row([
                    dbc.Col([
                        html.Img(id='bar-graph-matplotlib6',style={'text-align': 'center'})
                    ], sm=12, md=12, style={'margin-top': '5px'}),
                ], className='g-1', style={'height': '20%', 'margin-left': '25px'}),

            ], style=tab_card_graph)
        ])
    ], className='main_row g-2 my-auto'),

#Test data prediction
    dbc.Row([
        dbc.Col([
            dbc.Card([                
                dbc.Row([
                    dbc.Col([
                         html.H3("Gráfico - Test data prediction", className='mb-2'),                    
                        ], sm=9, md=9, style={'margin-top': '15px'}),
                ], className='g-1', style={'height': '20%', 'justify-content': 'center'}),

            ], style=tab_card,class_name='card-title')
        ])
    ], className='main_row g-2 my-auto'),
    dbc.Row([
        dbc.Col([
            dbc.Card([                
                dbc.Row([
                    dbc.Col([
                        html.Img(id='bar-graph-matplotlib7',style={'text-align': 'center'})
                    ], sm=12, md=12, style={'margin-top': '5px'}),
                ], className='g-1', style={'height': '20%', 'margin-left': '50px'}),

            ], style=tab_card_graph)
        ])
    ], className='main_row g-2 my-auto'),


#Training data prediction - SVR
    dbc.Row([
        dbc.Col([
            dbc.Card([                
                dbc.Row([
                    dbc.Col([
                         dbc.Button("Processar", 
                            color="primary",
                            id="btn-processar"),    
                         html.H3("Gráfico - Training data prediction - SVR", className='mb-2'),                    
                        ], sm=9, md=9, style={'margin-top': '15px'}),
                ], className='g-1', style={'height': '20%', 'justify-content': 'center'}),

            ], style=tab_card,class_name='card-title')
        ])
    ], className='main_row g-2 my-auto'),
    dbc.Row([
        dbc.Col([
            dbc.Card([                
                dbc.Row([
                    dbc.Col([
                        html.Img(id='bar-graph-matplotlib8',style={'text-align': 'center'})
                    ], sm=12, md=12, style={'margin-top': '5px'}),
                ], className='g-1', style={'height': '20%', 'margin-left': '25px'}),

            ], style=tab_card_graph)
        ])
    ], className='main_row g-2 my-auto'),

#Test data prediction - SVR
    dbc.Row([
        dbc.Col([
            dbc.Card([                
                dbc.Row([
                    dbc.Col([
                         html.H3("Gráfico - Test data prediction - SVR", className='mb-2'),                    
                        ], sm=9, md=9, style={'margin-top': '15px'}),
                ], className='g-1', style={'height': '20%', 'justify-content': 'center'}),

            ], style=tab_card,class_name='card-title')
        ])
    ], className='main_row g-2 my-auto'),
    dbc.Row([
        dbc.Col([
            dbc.Card([                
                dbc.Row([
                    dbc.Col([
                        html.Img(id='bar-graph-matplotlib9',style={'text-align': 'center'})
                    ], sm=12, md=12, style={'margin-top': '5px'}),
                ], className='g-1', style={'height': '20%', 'margin-left': '50px'}),

            ], style=tab_card_graph)
        ])
    ], className='main_row g-2 my-auto'),

])

def graph01(dataframe, selected_yaxis):
    fig, (ax1) = plt.subplots(figsize=(10,4)) 
    dataframe[selected_yaxis].plot(ax = ax1, label=selected_yaxis, style = '-', color = 'blue') 
    plt.xlim(pd.to_datetime(str(dataframe.index.min())),pd.to_datetime(str(dataframe.index.max())))
    plt.xticks(rotation=30)
    ax1.legend()
    #plt.tight_layout()

    return imageFig(fig,'Graph01')

def graph01_update(dataframe,selected_yaxis):
    fig, (ax) = plt.subplots(figsize=(10,4))


    dataframe[selected_yaxis].plot(ax = ax, style = '-', color = '#3F7F7F') 
    plt.ylabel(selected_yaxis)
    plt.xlim(pd.to_datetime(str(dataframe.index.min())),pd.to_datetime(str(dataframe.index.max())))
    plt.xticks(rotation=30)
    plt.title("Titulo - "+selected_yaxis,fontweight="bold")
    ax.legend(loc=6) #local da legenda
    
    plt.tight_layout()

    return imageFig(fig,'Graph01_update')

def graph02(dataframe):
    fig, (ax) = plt.subplots(figsize=(10,10))
    correlation_mat = dataframe.corr()
    sns.heatmap(correlation_mat, annot = True)
    fig.tight_layout()
    
    return imageFig(fig,'Graph02')

def graph02_update(dataframe):
    fig, (ax) = plt.subplots(figsize=(12,12))
    correlation_mat = dataframe.corr()
    sns.heatmap(correlation_mat, annot = True, fmt=".3f", linewidths=1, cmap="Blues")
    fig.tight_layout()

    return imageFig(fig,'graph02_update')

def graph03(dataframe,selection,y_test):

    fig, ax = plt.subplots(figsize=(12,4))
    plt.plot(y_test, label = 'Test Set - '+selection)
    ax.legend(['Test Set - '+selection])

    return imageFig(fig,'Graph03')

def graph04(dataframe,selection,y_train):
    fig, ax = plt.subplots(figsize=(12,4))
    plt.plot(y_train, label = 'Training Set - ' + selection )
    ax.legend(['Training Set - ' + selection])

    return imageFig(fig,'Graph04')

def graph05(dataframe,selected_yaxis):
    fig, ax = plt.subplots(figsize=(12,4))
    sns.boxplot(x=dataframe[selected_yaxis])
    ax.set_title(selected_yaxis,fontweight="bold")

    return imageFig(fig,'Graph05')

def graph05_update(dataframe,selected_yaxis):
    fig, ax = plt.subplots(figsize=(12,5))

    if(selected_yaxis != 'Day_of_week'):
        dataframe['Day_of_week'].replace([0, 1,2,3,4,5,6], ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], inplace=True)

    sns.boxplot(y=dataframe[selected_yaxis], x=dataframe['Day_of_week'])
    ax.set_title(selected_yaxis,fontweight="bold")

    if(selected_yaxis != 'Day_of_week'):
        dataframe['Day_of_week'].replace(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],[0, 1,2,3,4,5,6], inplace=True)

    return imageFig(fig,'Graph05_update')

def graph06(y_train, y_train_pred, selection):
    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(y_train, color = 'red', linewidth=2.0, alpha = 0.6)
    ax.plot(y_train_pred, color = 'blue', linewidth=0.8)
    ax.legend(['Atual','Predito'])
    score = r2_score(y_train, y_train_pred)
    complemento = (f'Resultado de R² para o conjunto treino: {score*100: 0.4f}')
    ax.set_title(selection+" - Training data prediction \n"+complemento,fontweight="bold")

    fig.tight_layout()

    return imageFig(fig,'Graph06')

def graph07(y_test, y_test_pred, selection):
    fig, ax =plt.subplots(figsize=(12,5))
    ax.plot(y_test, color = 'red', linewidth=2.0, alpha = 0.6)
    ax.plot(y_test_pred, color = 'blue', linewidth=0.8)
    ax.legend(['Atual','Predito'])
    score = r2_score(y_test, y_test_pred)
    complento = (f'Resultado de R² para o conjunto testes: {score*100: 0.4f}')

    ax.set_title(selection+" - Test data prediction \n" + complento,fontweight="bold" )


    return imageFig(fig,'Graph07')

def graph08(yy_train, y_train_pred_SVR_rbf, y_train_pred_SVR_poly, selection):
    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(yy_train, color = 'red', linewidth=2.0, alpha = 0.6)
    ax.plot(y_train_pred_SVR_rbf, color = 'blue', linewidth=1)
    ax.plot(y_train_pred_SVR_poly, color = 'green', linewidth=1)
    ax.legend(['Atual','rbf','poly'])
    score_rbf = r2_score(yy_train, y_train_pred_SVR_rbf)
    complemento_rbf = (f'Resultado de R² para o conjunto treino(rbf): {score_rbf*100: 0.4f}')
    
    score_poly = r2_score(yy_train, y_train_pred_SVR_poly)
    complemento_poly = (f'Resultado de R² para o conjunto treino(poly): {score_poly*100: 0.4f}')
    ax.set_title(selection+" - Training data prediction - SVR \n"+complemento_rbf+"\n"+complemento_poly,fontweight="bold")

    fig.tight_layout()
    return imageFig(fig,'Graph08')

def graph09(yy_test, y_test_pred_SVR_rbf,y_test_pred_SVR_poly, selection):
    fig, ax =plt.subplots(figsize=(12,5))
    ax.hist(yy_test, color = 'red', linewidth=2.0, alpha = 0.6)
    ax.hist(y_test_pred_SVR_rbf, color = 'blue', linewidth=1)
    ax.hist(y_test_pred_SVR_poly, color = 'green', linewidth=1)
    ax.legend(['Atual','rbf','poly'])
    score_rbf = r2_score(yy_test, y_test_pred_SVR_rbf)
    complemento_rbf = (f'Resultado de R² para o conjunto testes(rbf): {score_rbf*100: 0.4f}')

    score_poly = r2_score(yy_test, y_test_pred_SVR_poly)
    complemento_poly = (f'Resultado de R² para o conjunto testes(poly): {score_poly*100: 0.4f}')

    ax.set_title(selection+" - Training data prediction - SVR \n"+complemento_rbf+"\n"+complemento_poly,fontweight="bold")
    fig.tight_layout()

    return imageFig(fig,'Graph09')

def graph10(dataframe, selected_yaxis):

    colors = ["#bfd3e6", "#9b5b4f", "#4e4151", "#dbba78", "#bb9c55", "#909195","#dc1e1e","#a02933","#716807","#717cb4"]
    sns.palplot(sns.color_palette(colors))
    plt.style.use('dark_background')
    plt.figure(figsize=(5,5))
    sns.displot(x = dataframe[selected_yaxis],kde=False, bins = 50,color = "green", facecolor = "#3F7F7F",height = 4, aspect = 2.5)
    plt.title(selected_yaxis,fontweight="bold")
    plt.xlim()
    plt.tight_layout()

    return imageFig(plt,'Graph10')

def imageFig(fig,nm_grafico):

    pathFile = os.path.join(os.getcwd(), 'src',nm_grafico+'.png')
    fig.savefig(pathFile)

    buf = BytesIO()
    fig.savefig(buf, format="png")
    matplotlib.pyplot.close()
    # Embed the result in the html output.
    fig_data = base64.b64encode(buf.getbuffer()).decode("ascii")
    fig_matplotlib = f'data:image/png;base64,{fig_data}'
    buf.close()

    return fig_matplotlib

def dadosGraph(pdf,nm_titulo,nm_grafico):

    pathImage = (os.path.join(os.getcwd(), 'src',nm_grafico+'.png'))
    pdf.ln(10)
    pdf.set_font("Arial","B", 14)
    pdf.cell(w=0,txt=nm_titulo,align='L')
    pdf.ln(10)
    pdf.cell(w=0,link=pdf.image(pathImage,w=175), align='C')

def titleColum(selection):
    retorno = ''
    if(selection == 'Usage_kWh'): 
        retorno = 'Consumo de Energia (kWh)'
    elif(selection == 'Lagging_Current_Power_Factor'): 
        retorno = 'Fator de Potência Atrasado(%)'
    elif(selection == 'Leading_Current_Power_Factor'): 
        retorno = 'Fator de Potência Adiantado(%)'
    elif(selection == 'Lagging_Current_Reactive.Power_kVarh'): 
        retorno = 'Energia Reativa Atrasada(kVArh)'
    elif(selection == 'Leading_Current_Reactive_Power_kVarh'): 
        retorno = 'Energia Reativa Adiantada(kVArh)'
    elif(selection == 'CO2(tCO2)'): 
        retorno = 'Emissão de CO² (ppm)'
    elif(selection == 'NSM'): 
        retorno = 'N° de Segundos a partir de 00:00'
    elif(selection == 'WeekStatus'): 
        retorno = 'Fim de semana'
    elif(selection == 'Day_of_week'): 
        retorno = 'Dia da semana'
    elif(selection == 'Load_Type'): 
        retorno = 'Tipo de Consumo da Instalação: Light, Medium, Maximum'
    else:
        retorno = ""
    
    return (
        retorno
    )

def pdfText(self,texto,fonte,fonteSize,textoAlign,makeText,color,lineNumber):
    if(str.upper(color) in('R','RED')):
        self.set_text_color(255, 0, 0)
    elif(str.upper(color) in('O','ORANGE')):
        self.set_text_color(255, 128, 0)
    elif(str.upper(color) in('Y','YELLOW')):
        self.set_text_color(255, 255, 0)
    elif(str.upper(color) in('G','GREEN')):
        #self.set_text_color(0, 255, 0)
        self.set_text_color(42, 90, 72) #UEA-COLOR
    elif(str.upper(color) in('G-1','GREEN-1')):
        self.set_text_color(128, 255, 0)
    elif(str.upper(color) in('G+1','GREEN+1')):
        self.set_text_color(0, 255, 128)
    elif(str.upper(color) in('C','CYAN')):
        self.set_text_color(0, 255, 255)
    elif(str.upper(color) in('B','BLUE')):
        self.set_text_color(0, 0, 255)
    elif(str.upper(color) in('B-1','BLUE-1')):
        self.set_text_color(0, 128, 255)
    elif(str.upper(color) in('B+1','BLUE+1')):
        self.set_text_color(128,0, 255)
    elif(str.upper(color) in('M','MAGENTA')):
        self.set_text_color(255,0, 255)
    elif(str.upper(color) in('M+1','MAGENTA+1')):
        self.set_text_color(255,0, 128)
    else:
        self.set_text_color(0, 0, 0)
       
    self.set_draw_color(0, 80, 180)
    self.set_fill_color(230, 230, 0)
    self.set_font(fonte, makeText, fonteSize)
    self.cell(w=0,txt=texto,align=textoAlign)
    self.ln(lineNumber)

def pdfTable(self,dataFrame,fontfamily,fontSize,lineNumber,epwNumber):
        # Remember to always put one of these at least once.
        self.set_font(fontfamily,'',fontSize) 
        # Effective page width, or just epw
        epw = self.w - 2*self.l_margin
        # Set column width to 1/4 of effective page width to distribute content 
        # evenly across table and page
        col_width = epw/epwNumber
        # Text height is the same as current font size
        th = self.font_size
        self.ln(lineNumber)
        
        # Here we add more padding by passing 2*th as height
        for row in dataFrame.values.tolist():
            for datum in row:
                # Enter data in colums
                self.cell(col_width, 2*th, str(datum), border=1)
        
            self.ln(2*th)

        self.ln(6)

def pdfDataframeInfo(self,dataframe,fontfamily,fontSize,makeText):
    dtypes = dataframe.dtypes.to_dict()
    self.set_font(fontfamily,makeText,fontSize) 
    for col_name, typ in dtypes.items():
        self.ln(4)
        self.cell(w=0,txt=str(col_name)+" - "+str(typ),align='L')
    
    self.ln(12)

#Dados Media, Mean, Moda
def pdfDataframeValues(self,dataframe,select,fontfamily,fontSize,makeText):
    dtypes = dataframe.dtypes.to_dict()
    self.set_font(fontfamily,makeText,fontSize) 
    for col_name, typ in dtypes.items():
        self.ln(4)
        self.cell(w=0,txt=str("Column:" + str(col_name)),align='L')
        self.ln(8)
        self.cell(w=0,txt=str("Mean:" + str(round(dataframe[col_name].mean(),2))),align='L')
        self.ln(4)
        self.cell(w=0,txt=str("Max:" + str(round(dataframe[col_name].max(),2))),align='L')
        self.ln(4)
        self.cell(w=0,txt=str("Min:" + str(round(dataframe[col_name].min(),2))),align='L')
        self.ln(4)
        self.cell(w=0,txt=str("Median:" + str(round(dataframe[col_name].median(),2))),align='L')
        self.ln(4)
        self.cell(w=0,txt=str("Std:" + str(round(dataframe[col_name].std(),2))),align='L')
        self.ln(4)
        self.cell(w=0,txt=str("Var:" + str(round(dataframe[col_name].var(),2))),align='L')
        #self.ln(4)
        #self.cell(w=0,txt=str("Mode:" + str(round(dataframe[col_name].mode()[:3],2))),align='L')
        self.ln(10)
    
##Date Range
@app.callback(
    Output(component_id='bar-graph-matplotlib1', component_property='src'),
    Output(component_id='bar-graph-matplotlib1_update', component_property='src'),
    Output(component_id='bar-graph-matplotlib2', component_property='src'),
    Output(component_id='bar-graph-matplotlib2_update', component_property='src'),
    Output(component_id='bar-graph-matplotlib3', component_property='src'),
    Output(component_id='bar-graph-matplotlib4', component_property='src'),
    Output(component_id='bar-graph-matplotlib5', component_property='src'),
    Output(component_id='bar-graph-matplotlib5_update', component_property='src'),
    Output(component_id='bar-graph-matplotlib6', component_property='src'),
    Output(component_id='bar-graph-matplotlib7', component_property='src'),
    Output(component_id='bar-graph-matplotlib10', component_property='src'),
    Output("loading-output-2", "children"),
    [Input('rangeslider', 'value'),
    Input('dataset_fixed', 'data'),
    Input('category', 'value'),
    ], 
    #prevent_initial_call=True
)
def range_slider(range, df_store,selected):
    
    df_graph01 = pd.DataFrame(df_store)
    
    df_graph01 = df_graph01[(df_graph01['Month'] >= range[0]) & (df_graph01['Month'] <= range[1])]
    #df_graph01['date'] = pd.to_datetime(df_graph01['date'], format='ISO8601').dt.strftime('%Y-%m-%d')
 
    df_graph01['date'] = pd.to_datetime(df_graph01['date'])

    #reset_index
    df_graph01.reset_index()
    df_graph01.set_index("date", inplace=True)

    grafico01 = graph01(df_graph01,selected)
    grafico01_update = graph01_update(df_graph01,selected)

    grafico02 = graph02(df_graph01)
    grafico02_update = graph02_update(df_graph01)

    ## Dividir conjuntos para treinamento e testes
    X = df_graph01.drop(selected, axis = 1)
    y = df_graph01[selected]
    X = X.values
    y = y.values
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 10, shuffle = False)

    model = LinearRegression().fit(x_train, y_train)

    #Predições
    y_train_pred = model.predict(x_train)
    y_test_pred  = model.predict(x_test)

    grafico03 = graph03(df_graph01,selected,y_test)
    grafico04 = graph04(df_graph01,selected,y_train)
    grafico05 = graph05(df_graph01,selected)
    grafico05_update = graph05_update(df_graph01,selected)

    grafico06 = graph06(y_train, y_train_pred, selected)
    grafico07 = graph07(y_test, y_test_pred, selected)

    grafico10 = graph10(df_graph01,selected)


    return (
        grafico01,
        grafico01_update,
        grafico02,
        grafico02_update,
        grafico03,
        grafico04,
        grafico05,
        grafico05_update,
        grafico06,
        grafico07,
        grafico10,
        ""
    )

##Processar
@app.callback(
    Output(component_id='bar-graph-matplotlib8', component_property='src'),
    Output(component_id='bar-graph-matplotlib9', component_property='src'),
    [Input('rangeslider', 'value'),
    Input('dataset_fixed', 'data'),
    Input('category', 'value'),
    Input("btn-processar", "n_clicks"),
    ], 
    prevent_initial_call=True
)
def processar(range, df_store,selected,n_clicks):

    #print('starting..')
    if "btn-processar" == ctx.triggered_id:
    
        # print('starting..',ctx.triggered_id)
        # print('starting selected..',selected)
        # print('stat range',range)
        df_graph = pd.DataFrame(df_store)
        
        df_graph = df_graph[(df_graph['Month'] >= range[0]) & (df_graph['Month'] <= range[1])]
        #df_graph01['date'] = pd.to_datetime(df_graph01['date'], format='ISO8601').dt.strftime('%Y-%m-%d')
    
        df_graph['date'] = pd.to_datetime(df_graph['date'])

        #reset_index
        df_graph.reset_index()
        df_graph.set_index("date", inplace=True)

        ## Dividir conjuntos para treinamento e testes
        XX = df_graph.drop(selected, axis = 1)
        yy = df_graph[selected]
        XX = XX.values
        yy = yy.values
        xx_train, xx_test, yy_train, yy_test = train_test_split(XX, yy, test_size = 0.25, random_state = 10, shuffle = False)

        # print('rbf')
        # SVR_rbf = SVR(kernel='rbf' )
        # # print('linear')
        # # SVR_lin = SVR(kernel='linear')
        # print('poly')
        # SVR_poly = SVR(kernel='poly')

        # print('predict')
        # y_rbf = SVR_rbf.fit(XX, yy).predict(xx_train)
        # # y_lin = SVR_lin.fit(xx_train, yy_train).predict(xx_train)
        # y_poly = SVR_poly.fit(xx_train, yy_train).predict(xx_train)

        # grafico08 = graph08(yy_train, y_rbf,y_poly, selected)
        # grafico09 = ''

        #print('Iniciando SVR(rbf)')
        model_SRV_rbf = SVR(kernel='rbf',C=100,epsilon=0.01).fit(xx_train, yy_train)
        y_train_pred_SVR_rbf = model_SRV_rbf.predict(xx_train)
        y_test_pred_SVR_rbf  = model_SRV_rbf.predict(xx_test)

        #print('Iniciando SVR(poly)')
        model_SRV_poly = SVR(kernel='poly').fit(xx_train, yy_train)
        y_train_pred_SVR_poly = model_SRV_poly.predict(xx_train)
        y_test_pred_SVR_poly  = model_SRV_poly.predict(xx_test)

        #print('Gerando SVR ')
        grafico08 = graph08(yy_train, y_train_pred_SVR_rbf,y_train_pred_SVR_poly, selected)
        grafico09 = graph09(yy_test, y_test_pred_SVR_rbf,y_test_pred_SVR_poly, selected)
        #print('Finish ')

        return (
            grafico08,
            grafico09
        )
    else:
        raise PreventUpdate

#Title Graph01
@app.callback(
        Output("titleGraph01", "children"),
        Input('category', 'value'),
        Input('rangeslider', 'value'),
        prevent_initial_call=False,
)
def func(selection,range):

    periodo = calendar.month_abbr[range[0]] +' - '+calendar.month_abbr[range[1]]

    retorno = ''
    if(selection == 'Usage_kWh'): 
        retorno = 'Consumo de Energia (kWh)'
    elif(selection == 'Lagging_Current_Power_Factor'): 
        retorno = 'Fator de Potência Atrasado(%)'
    elif(selection == 'Leading_Current_Power_Factor'): 
        retorno = 'Fator de Potência Adiantado(%)'
    elif(selection == 'Lagging_Current_Reactive.Power_kVarh'): 
        retorno = 'Energia Reativa Atrasada(kVArh)'
    elif(selection == 'Leading_Current_Reactive_Power_kVarh'): 
        retorno = 'Energia Reativa Adiantada(kVArh)'
    elif(selection == 'CO2(tCO2)'): 
        retorno = 'Emissão de CO² (ppm)'
    elif(selection == 'NSM'): 
        retorno = 'N° de Segundos a partir de 00:00'
    elif(selection == 'WeekStatus'): 
        retorno = 'Fim de semana'
    elif(selection == 'Day_of_week'): 
        retorno = 'Dia da semana'
    elif(selection == 'Load_Type'): 
        retorno = 'Tipo de Consumo da Instalação: Light, Medium, Maximum'
    else:
        retorno = ""
    
    return (
        retorno + ":. " + periodo
    )

#Report PDF
@app.callback(
    Output("download-pdf", "data"),
    Output("loading-output-1", "children"),
    Input("btn-reports", "n_clicks"),
    Input('category', 'value'),
    Input('dataset_origin', 'data'),
    Input('rangeslider', 'value'),
    prevent_initial_call=True,
)
def func(n_clicks,selection,df_origem,range):

    df_origem = pd.DataFrame(df_origem)
    df_temp = data[(data['Month'] >= range[0]) & (data['Month'] <= range[1])]

    if "btn-reports" == ctx.triggered_id:
        pdf = FPDF()
        df_head = df_origem.head()

        pdf.add_page()

        pdfText(pdf,"UEA - Universidade do Estado do Amazonas","Arial",16,'C','B','Green',10)

        pdfText(pdf,"PÓS-Gradução Desenvolvimento de Software em Alto Desempenho","Arial",16,'C','B','Green',100)
        
        pdfText(pdf,"Indústria 4.0","Arial",16,'C','B','Green',70)

        pdfText(pdf,"Adriano Mourão","Arial",12,'R','B','Green',5)
        pdfText(pdf,"Salomão Calheiros","Arial",12,'R','B','Green',5)
        pdfText(pdf,"Thyago Lima","Arial",12,'R','B','Green',5)
        pdfText(pdf,"Willians Santos","Arial",12,'R','B','Green',1)
        pdf.set_y(-25)
        pdfText(pdf,"Manaus","Arial",10,'C','B','Green',4)
        pdfText(pdf,str(datetime.today().strftime('%d/%m/%Y %H:%M:%S')),"Arial",8,'C','B','Green',1)


#Typos de Dados Iniciais:
        periodo = calendar.month_abbr[range[0]] +' - '+calendar.month_abbr[range[1]]
        pdf.add_page()
        pdfText(pdf,"UEA - Universidade do Estado do Amazonas","Arial",16,'C','B','Green',10)
        pdfText(pdf,"PÓS-Gradução Desenvolvimento de Software em Alto Desempenho","Arial",16,'C','B','Green',10)


        pdfText(pdf,"Tipos de Dados Iniciais:."+periodo,"Arial",12,'C',"",'Black',10)
        pdfText(pdf,"Dataframe.info()","Arial",12,'L',"BI",'Green',10)
        pdfText(pdf,"Column - Types","Arial",8,'L',"",'Black',1)
        pdfDataframeInfo(pdf,df_origem,"Arial",8,"")

#dataframe.head()

        pdfText(pdf,"Dataframe.head()","Arial",12,'L',"BI",'Green',10)
        pdfTable(pdf,df_head,"Times",6,0.5,11)

#Dados após tratamento
        pdfText(pdf,"Dados após tratamento:.","Arial",12,'C',"",'Black',6) 
        pdfText(pdf,"Dataframe.info()","Arial",12,'L',"BI",'Green',10)     

#index
        pdfText(pdf,"Index","Arial",8,'L',"",'Orange',4)
        pdfText(pdf,str(data.index.name)+" - "+str(data.index.dtype),"Arial",8,'L',"",'Orange',6)
        pdfText(pdf,"Column - Types","Arial",8,'L',"",'Green',3)
        pdfDataframeInfo(pdf,data,"Arial",8,"")

#dataframe.head() 2

        pdfText(pdf,"Dataframe.head()","Arial",12,'L',"BI",'Green',10)
        pdfTable(pdf,data.head(),"Times",6,0.5,14)

#daframe.values
        pdf.add_page()
        pdfText(pdf,"Values:.","Arial",12,'L',"BI",'Green',10)
        pdfDataframeValues(pdf,df_temp,selection,"Arial",8,"")

#UEA COLOR
        pdf.set_text_color(42, 90, 72)
#Graph01
        pdf.add_page()
        dadosGraph(pdf,titleColum(selection),'graph01')
        dadosGraph(pdf,"",'graph01_update')
#Graph02
        pdf.add_page()
        dadosGraph(pdf,'Heatmap - Gráfico de Correlação','graph02')
#Graph02_update
        pdf.add_page()
        dadosGraph(pdf,'Heatmap - Gráfico de Correlação','graph02_update')
#Graph03
        pdf.add_page()
        dadosGraph(pdf,'Gráfico - Test Set','graph03')
#Graph04
        dadosGraph(pdf,'Gráfico - Training Set','graph04')
#Graph05
        pdf.add_page()
        dadosGraph(pdf,'Gráfico - Boxplot','graph05')
#Graph05_update
        dadosGraph(pdf,'Gráfico - Boxplot - Day of Week','graph05_update')
#Graph06
        pdf.add_page()
        dadosGraph(pdf,'Gráfico - Training data prediction - RL','graph06')
#Graph07
        dadosGraph(pdf,'Gráfico - Test data prediction - RL','graph07')
#Graph08        
        pdf.add_page()
        dadosGraph(pdf,'Gráfico - Training data prediction - SVR','graph08')
#Graph07
        dadosGraph(pdf,'Gráfico - Test data prediction - SVR','graph09')

        pdf.output("relatorio.pdf")
        return (
            dcc.send_file("relatorio.pdf"),
            ""
        )
    else:
        raise PreventUpdate

if __name__ == '__main__':
    app.run(debug=True)
