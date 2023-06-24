import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib                                # pip install matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
import os

from scipy.stats import skew

# ========== Styles ============ #
tab_card = {'height': '100%'}
pathFile = os.path.join(os.getcwd(), 'src','Steel_industry_data.csv')

data = pd.read_csv(pathFile)


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

# ## Apagar dados faltantes
data = data.dropna()

# # To dict - para salvar no dcc.store
# df_store = data.to_dict()
# df_origem = data_origem.to_dict()

# ## Usar datetime (tempo) com index dos dados do dataset
data.reset_index()
data.set_index("date", inplace=True)

select = 'Usage_kWh'

print('col',select)
print(f"\033[91m\033[1m")
#print("Skewness:",col,"=",round(skew(data[col]),3))
print("Kurtosis:",select,    "=",round(data[select].kurt(),2))
print("Mean:",select,    "=",round(data[select].mean(),2))
print("Max:",select,     "=",round(data[select].max(),2))
print("Min:",select,     "=",round(data[select].min(),2))
print("Median:",select,  "=",round(data[select].median(),2))
print("Std:",select,     "=",round(data[select].std(),2))
print("Var:",select,     "=",round(data[select].var(),2))
print("Mode:",select,    "=",round(data[select].mode(),2))

data[select]

plt.figure(figsize=(18,6))
sns.displot(x = data[select],kde=True,bins=50,color="green")
plt.title(select,fontweight="bold")
plt.show()

plt.tight_layout()
#print(f"\033[93m\033[1m")
#print("====="*25)

