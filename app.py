from dash import Dash, html, dcc, Input, Output  # pip install dash
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
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

from fpdf import FPDF
from dash.exceptions import PreventUpdate

# ========== Styles ============ #
tab_card = {'height': '100%'}

data = pd.read_csv('src\\Steel_industry_data.csv')
data_origem = pd.read_csv('src\\Steel_industry_data.csv')


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
from sklearn.svm import SVR
#model = SVR().fit(x_train, y_train)
model = LinearRegression().fit(x_train, y_train)

#Predições
y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
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

            ], style=tab_card)
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

            ], style=tab_card)
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
                        html.Img(id='bar-graph-matplotlib2',style={'margin-top':'-10px'})
                    ], sm=8, md=8, style={'margin-top': '5px'}),
                ], className='g-1', style={'height': '20%', 'justify-content': 'center'}),

            ], style=tab_card)
        ])
    ], className='main_row g-2 my-auto'),

    dbc.Row([
        dbc.Col([
            dbc.Card([                
                dbc.Row([
                    dbc.Col([
                        html.Img(id='bar-graph-matplotlib2_update',style={'text-align': 'center'})
                    ], sm=9, md=9, style={'margin-top': '5px'}),
                ], className='g-1', style={'height': '20%', 'justify-content': 'center'}),

            ], style=tab_card)
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
                ], className='g-1', style={'height': '20%', 'justify-content': 'center'}),

            ], style=tab_card)
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
                ], className='g-1', style={'height': '20%', 'justify-content': 'center'}),

            ], style=tab_card)
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
                ], className='g-1', style={'height': '20%', 'justify-content': 'center'}),

            ], style=tab_card)
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
                ], className='g-1', style={'height': '20%', 'justify-content': 'center'}),

            ], style=tab_card)
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
                ], className='g-1', style={'height': '20%', 'justify-content': 'center'}),

            ], style=tab_card)
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
                ], className='g-1', style={'height': '20%', 'justify-content': 'center'}),

            ], style=tab_card)
        ])
    ], className='main_row g-2 my-auto'),

])

def graph01(dataframe, selected_yaxis):
    fig, (ax1) = plt.subplots(figsize=(10,4)) 
    dataframe[selected_yaxis].plot(ax = ax1, label=selected_yaxis, style = '-', color = 'blue') 
    plt.xlim(pd.to_datetime(str(dataframe.index.min())),pd.to_datetime(str(dataframe.index.max())))
    plt.xticks(rotation=90)
    ax1.legend()
    #plt.tight_layout()

    return imageFig(fig,'Graph01')

def graph01_update(dataframe,selected_yaxis):
    fig, (ax1) = plt.subplots(figsize=(10,4))
    dataframe[selected_yaxis].plot(ax = ax1, style = '-', color = 'green') 
    plt.ylabel(selected_yaxis)
    plt.xlim(pd.to_datetime(str(dataframe.index.min())),pd.to_datetime(str(dataframe.index.max())))
    plt.xticks(rotation=30)
    plt.title("Titulo - "+selected_yaxis)
    ax1.legend(loc=6) #local da legenda
    #plt.tight_layout()

    return imageFig(fig,'Graph01_update')

    fig, (ax) = plt.subplots(figsize=(15,15))
    correlation_mat = data.corr()
    sns.heatmap(correlation_mat, annot = True)
    plt.tight_layout()
    
    return imageFig(fig,'Graph02')

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
    ax.set_title(selected_yaxis)

    return imageFig(fig,'Graph05')

def graph05_update(dataframe,selected_yaxis):
    fig, ax = plt.subplots(figsize=(12,5))

    if(selected_yaxis != 'Day_of_week'):
        dataframe['Day_of_week'].replace([0, 1,2,3,4,5,6], ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], inplace=True)

    sns.boxplot(y=dataframe[selected_yaxis], x=dataframe['Day_of_week'])
    ax.set_title(selected_yaxis)

    if(selected_yaxis != 'Day_of_week'):
        dataframe['Day_of_week'].replace(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],[0, 1,2,3,4,5,6], inplace=True)

    return imageFig(fig,'Graph05_update')

def graph06(y_train, y_train_pred, selection):
    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(y_train, color = 'red', linewidth=2.0, alpha = 0.6)
    ax.plot(y_train_pred, color = 'blue', linewidth=0.8)
    ax.legend(['Atual','Predito'])
    score = r2_score(y_train, y_train_pred)
    ax.set_title("Training data prediction - "+selection)

    return imageFig(fig,'Graph06')

def graph07(y_test, y_test_pred, selection):
    fig, ax =plt.subplots(figsize=(12,5))
    ax.plot(y_test, color = 'red', linewidth=2.0, alpha = 0.6)
    ax.plot(y_test_pred, color = 'blue', linewidth=0.8)
    ax.legend(['Atual','Predito'])
    score = r2_score(y_test, y_test_pred)
    ax.set_title("Test data prediction - " + selection )

    return imageFig(fig,'Graph07')

def imageFig(fig,nm_grafico):

    #dadosGraph(fig,nm_titulo,nm_grafico)
    fig.savefig('src\\'+nm_grafico+'.png')

    buf = BytesIO()
    fig.savefig(buf, format="png")
    matplotlib.pyplot.close()
    # Embed the result in the html output.
    fig_data = base64.b64encode(buf.getbuffer()).decode("ascii")
    fig_matplotlib = f'data:image/png;base64,{fig_data}'
    buf.close()

    return fig_matplotlib

def dadosGraph(pdf,nm_titulo,nm_grafico):
    pdf.ln(10)
    pdf.cell(w=0,txt=nm_titulo,align='L')
    pdf.ln(10)
    pdf.cell(w=0,link=pdf.image('src\\'+nm_grafico+'.png',w=175), align='C')

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
    
    return (
        retorno
    )

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
        grafico07
    )

#Title Graph01
@app.callback(
        Output("titleGraph01", "children"),
        Input('category', 'value'),
        prevent_initial_call=False,
)
def func(selection):
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
    
    return (
        retorno
    )


#Report PDF
@app.callback(
    Output("download-pdf", "data"),
    Input("btn-reports", "n_clicks"),
    Input('category', 'value'),
    Input('dataset_origin', 'data'),
    prevent_initial_call=True,
)
def func(n_clicks,selection,df_origem):

    df_origem = pd.DataFrame(df_origem)

    if n_clicks is None:
        raise PreventUpdate
    else:
        pdf = FPDF()
        df_head = df_origem.head()

        pdf.add_page()
        pdf.set_draw_color(0, 80, 180)
        pdf.set_fill_color(230, 230, 0)
        pdf.set_text_color(42, 90, 72)
        pdf.set_font("Arial", "B", 16)

        pdf.cell(w=0,txt="UEA - Universidade do Estado do Amazonas",align='C')
        pdf.ln(10)

        pdf.cell(w=0,txt="PÓS-Gradução Desenvolvimento de Software em Alto Desempenho",align='C')

#Typos de Dados Iniciais:
        pdf.ln(12)
        pdf.set_font("Arial","", 10)
        pdf.cell(w=0,txt="Tipos de Dados Iniciais:. Dataframe.info()",align='C')
        pdf.ln(4)

        pdf.set_font("Arial","", 8)
        dtypes = df_origem.dtypes.to_dict()

        pdf.cell(w=0,txt="Column - Types",align='L')
        for col_name, typ in dtypes.items():
            pdf.ln(4)
            pdf.cell(w=0,txt=str(col_name)+" - "+str(typ),align='L')

#dataframe.head()

        # Remember to always put one of these at least once.
        pdf.set_font('Times','',10.0) 
        
        # Effective page width, or just epw
        epw = pdf.w - 2*pdf.l_margin
        
        # Set column width to 1/4 of effective page width to distribute content 
        # evenly across table and page
        col_width = epw/11

        # Text height is the same as current font size
        th = pdf.font_size

        pdf.ln(6)
        pdf.set_font("Arial","", 10)
        pdf.cell(epw, 0.0, 'Dataframe.head()', align='C')
        pdf.ln(4)
        pdf.set_font('Times','',6) 
        pdf.ln(0.5)
        
        # Here we add more padding by passing 2*th as height
        for row in df_head.values.tolist():
            for datum in row:
                # Enter data in colums
                pdf.cell(col_width, 2*th, str(datum), border=1)
        
            pdf.ln(2*th)

#Dados após tratamento
        pdf.ln(6)
        pdf.set_font("Arial","", 10)
        pdf.cell(w=0,txt="Dados após tratamento:.",align='C')
        

#index
        pdf.ln(2)
        pdf.cell(w=0,txt="Index",align='L')
        pdf.ln(4)
        pdf.cell(w=0,txt=str(data.index.name)+" - "+str(str(data.index.dtype)),align='L')

        pdf.ln(6)
        pdf.set_font("Arial","", 8)
        dtypes = data.dtypes.to_dict()

        pdf.cell(w=0,txt="Column - Types",align='L')
        for col_name, typ in dtypes.items():
            pdf.ln(4)
            pdf.cell(w=0,txt=str(col_name)+" - "+str(typ),align='L')

#Graph01
        pdf.add_page()

        dadosGraph(pdf,titleColum(selection),'graph01')

        dadosGraph(pdf,"",'graph01_update')

        pdf.add_page()

        dadosGraph(pdf,"Gráfico de Correlação - "+selection,'graph02')

        pdf.add_page()

        dadosGraph(pdf,"Gráfico de Correlação - "+selection,'graph02_update')

        pdf.add_page()

        dadosGraph(pdf,selection,'graph03')

        dadosGraph(pdf,selection,'graph04')

        pdf.add_page()

        dadosGraph(pdf,selection,'graph05')

        dadosGraph(pdf,selection,'graph05_update')

        pdf.add_page()

        dadosGraph(pdf,selection,'graph06')

        dadosGraph(pdf,selection,'graph07')

        pdf.output("relatorio.pdf")
        return dcc.send_file("relatorio.pdf")

if __name__ == '__main__':
    app.run_server(debug=True, port=8002,)