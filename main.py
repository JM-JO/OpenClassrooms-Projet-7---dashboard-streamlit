# Imports
import streamlit as st
import joblib
import requests
import pandas as pd
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt

# Constantes
HOST = 'http://127.0.0.1:8000'     # développement sur serveur local
# HOST = 'https://project7-api-ml.herokuapp.com'     # serveur de production 




# Fonctions

## récupération du seuil_optimum (de la fonction de coût métier)
@st.cache
def seuil_optimum():
	return round(float(requests.post(HOST + '/seuil_optimum').content),3)
	
	
## récupération de la proba de défaut d'un client
@st.cache
def fetch_proba_defaut(id_client : int):
	return eval(requests.post(HOST + '/predict_id_client/' + str(id_client)).content)["probability"]


## récupération des données d'un client
@st.cache
def fetch_data(id_client : int):
	un_client_json = eval(requests.post(HOST + '/fetch_data_id_client/' + str(id_client)).content)
	un_client_pandas = pd.read_json(un_client_json, orient='index')    # format pandas.DataFrame
	return un_client_pandas
	
	
	
	

# Mise en place du dashboard
st.title('Home Credit Default Risk')


# Logs et debugs
"---------------------------"
st.header("Section consacrée aux logs et debugs")
st.write("seuil_optimum :", seuil_optimum())


# ID client
"---------------------------"
st.header('Identité du client')
id_client = st.text_input("Saisir l'identifiant du client", value="216030")
st.caption("Exemples de clients prédits positifs (défaut de crédit) : 215699") 
st.caption("Exemples de clients prédits négatifs (pas de défaut) : 216030")


# Données client
"---------------------------"
st.header('Données du client')
un_client_pandas = fetch_data(id_client)
with st.expander("Voir les données", expanded=False):
	st.dataframe(un_client_pandas.T)


# Résultat de la demande de crédit
"---------------------------" 
st.header('Résultat de la demande de crédit')
probability = fetch_proba_defaut(id_client)
left_column, right_column = st.columns([1.5,1])
with left_column:
	if probability < seuil_optimum(): 
		st.success(f"  \n __CREDIT ACCEPTE__  \n  \nLa probabilité de défaut de ce crédit est de __{round(100*probability,1)}__% (inférieure au seuil de {100*seuil_optimum()}% nécessaire pour obtenir un crédit).  \n ")
	else:
		st.error(f"__CREDIT REFUSE__  \nLa probabilité de défaut de ce crédit est de {round(100*probability,1)}% (supérieure au seuil de {100*seuil_optimum()}% nécessaire pour obtenir un crédit).")
with right_column:
	# pie chart
	plt.style.use('seaborn')
	fig = plt.figure(edgecolor='black', linewidth=4)
	labels = ['Défaut', 'Pas de défaut']
	colors = ["red", "limegreen"]
	probas = [probability*100, 100*(1-probability)]
	explode = (0.1, 0)
	plt.pie(x=probas, explode=explode, labels=labels, autopct='%.1f%%', pctdistance=0.7, shadow=True, startangle=45,
		   textprops={'size':14}, colors=colors)
	fig.suptitle("Risque de défaut du crédit", size=18)
	st.pyplot(fig)
# rectangles
plt.style.use('default')
fig, ax = plt.subplots(figsize=(10,1))
fig.suptitle("Risque de défaut du crédit (%)", size=15, y=1.1)
ax.add_patch(Rectangle((0,0), width=probability*100, height=1, color='red'))
ax.add_patch(Rectangle((probability*100,0), width=100-probability*100, height=1, color='limegreen'))
ax.set_xlim(0,100)
ax.set_ylim(0,1)
ax.set_yticks([])
ax.plot((seuil_optimum()*100, seuil_optimum()*100), (0,1), color='darkorange', ls=(0,(0.5,0.5)), lw=3)
st.pyplot(fig)


# Position du client vs les autres
"---------------------------" 
st.header('Positionnement du client par rapports aux autres clients (KDE proba défaut, version data scientist)')
figure_kde_distribution_proba_defaut_pour_datascientist = joblib.load('./src/figure_kde_distribution_proba_defaut_pour_datascientist.joblib') 
plt.annotate(text=f"Client {id_client}", xy=(probability,0), xytext=(probability,3), arrowprops=dict(facecolor='k', arrowstyle='simple'))
st.pyplot(figure_kde_distribution_proba_defaut_pour_datascientist)   

"---------------------------" 
st.header('Positionnement du client par rapports aux autres clients (KDE proba défaut, version chargé clientèle)')
figure_kde_distribution_proba_defaut_pour_client = joblib.load('./src/figure_kde_distribution_proba_defaut_pour_client.joblib') 
plt.annotate(text=f"Client {id_client}", xy=(probability,0), xytext=(probability,3), arrowprops=dict(facecolor='k', arrowstyle='simple'))
st.pyplot(figure_kde_distribution_proba_defaut_pour_client)   

"---------------------------" 
st.header('Positionnement du client par rapports aux autres clients (histogramme proba défaut, version data scientist)')
figure_hist_distribution_proba_defaut_pour_datascientist = joblib.load('./src/figure_hist_distribution_proba_defaut_pour_datascientist.joblib') 
plt.annotate(text=f"Client {id_client}", xy=(probability,0), xytext=(probability,1000), arrowprops=dict(facecolor='k', arrowstyle='simple'))
st.pyplot(figure_hist_distribution_proba_defaut_pour_datascientist)  

# Feature Importance
"---------------------------" 
st.header('Feature Importance Globale par Feature Permutation (version data scientist)')
figure_features_permutation_importances_pour_datascientist = joblib.load('./src/figure_features_permutation_importances_pour_datascientist.joblib') 
st.pyplot(figure_features_permutation_importances_pour_datascientist)  
st.caption("La feature importance par permutation est obtenue par mesure de la variation moyenne du score prédit après randomisation de cette feature dans le dataset.")
with st.expander("Voir les définitions des features", expanded=False):
	pass   # à compléter

def trace_kde(feature):
	"""Tracé KDE de la feature en argument. 
	Args :
	- feature (string).
	Returns :
	- tracé matplotlib via st.pyplot.
	"""
	figure = joblib.load('./src/figure_kde_distribution_' + feature + '_pour_datascientist.joblib') 
	y_max = plt.ylim()[1]
	x_client = un_client_pandas[feature].iloc[0]
	if str(x_client) == "nan":
		x_center = (plt.xlim()[1] + plt.xlim()[0]) / 2
		plt.annotate(text=f" Client {id_client}\n  donnée n.d.", xy=(x_center,0), xytext=(x_center,y_max/5))
		print(x_center)
	else:
		# plt.axline((x_client, 0), (x_client, 1e-5), c='k', ls='dashed', lw=2)
		plt.axvline(x=x_client, ymin=-1e10, ymax=1e10, c='k', ls='dashed', lw=2)
		plt.annotate(text=f" Client {id_client}\n  {round(x_client,4)}", xy=(x_client,0), xytext=(x_client,y_max/5))  #, arrowprops=dict(facecolor='k', arrowstyle='simple'))
	st.pyplot(figure)   
	st.caption(feature)
	print(str(x_client))


# Position du client vs les autres
"---------------------------" 
st.header('Positionnement du client par rapports aux autres clients dans les principales features')
left_column, middle_column, right_column = st.columns([1, 1, 1])
with left_column:
	trace_kde('EXT_SOURCE_2')
	trace_kde('PAYMENT_RATE')
with middle_column:
	trace_kde('EXT_SOURCE_3')
	trace_kde('ORGANIZATION_TYPE')
with right_column:
	trace_kde('EXT_SOURCE_1')
	trace_kde('AMT_ANNUITY')
