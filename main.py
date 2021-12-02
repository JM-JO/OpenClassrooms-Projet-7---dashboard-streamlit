# Imports
import streamlit as st
import joblib
import requests
import pandas as pd
from matplotlib.patches import Rectangle, FancyArrowPatch
import matplotlib.pyplot as plt


# Adress of the API server :
HOST = 'http://127.0.0.1:8000'     # developement on local server
# HOST = 'https://project7-api-ml.herokuapp.com'     # production server


# Functions

@st.cache
def optimum_threshold():
	"""Fetches the optimum threshold of the buisness cost function on the API server.
	Args :
	- None.
	Returns :
	- float.
	"""
	return round(float(requests.post(HOST + '/optimum_threshold').content),3)
	
	
@st.cache
def fetch_proba_default(id_client : int):
	"""Fetches the probability of default of a client on the API server.
	Args : 
	- id_client (int).
	Returns :
	- probability of default (float).
	"""
	return eval(requests.post(HOST + '/predict_id_client/' + str(id_client)).content)["probability"]


@st.cache
def fetch_data(id_client : int):
	"""Fetches the data of a client on the API server.
	Args : 
	- id_client (int).
	Returns :
	- pandas dataframe with a single line.
	"""
	one_client_json = eval(requests.post(HOST + '/fetch_data_id_client/' + str(id_client)).content)
	one_client_pandas = pd.read_json(one_client_json, orient='index')    # format: pandas.DataFrame
	return one_client_pandas
	
	
# Do not use @st.cache here.
def plot_kde(feature):
	"""Plots a KDE of the quantitative feature. 
	Args :
	- feature (string).
	Returns :
	- matplotlib plot via st.pyplot.
	"""
	figure = joblib.load('./src/figure_kde_distribution_' + feature + '_for_datascientist.joblib') 
	y_max = plt.ylim()[1]
	x_client = one_client_pandas[feature].iloc[0]
	if str(x_client) == "nan":
		x_center = (plt.xlim()[1] + plt.xlim()[0]) / 2
		plt.annotate(text=f" Client {id_client}\n  data not available", xy=(x_center,0), xytext=(x_center,y_max/5))
	else:
		plt.axvline(x=x_client, ymin=-1e10, ymax=1e10, c='k', ls='dashed', lw=2)
		plt.annotate(text=f" Client {id_client}\n  {round(x_client,4)}", xy=(x_client,y_max/5))
	st.pyplot(figure)   
	st.caption(feature)
	
	
# Do not use @st.cache here.
def barplot(feature):
	"""Barplot of a qualitative feature. 
	Args :
	- feature (string).
	Returns :
	- matplotlib plot via st.pyplot.
	"""
	figure = joblib.load('./src/figure_barplot_' + feature + '_for_datascientist.joblib') 
	x_client = one_client_pandas[feature].iloc[0]
	plt.axhline(y=optimum_threshold(), xmin=-1e10, xmax=1e10, c='darkorange', ls='dashed', lw=1)
	plt.axvline(x=x_client, ymin=-1e10, ymax=1e10, c='k', ls='dashed', lw=1)
	plt.annotate(text=f" Client {id_client}\n  Value : {x_client} (to be coded)", xy=(0,0.5)) 
	st.pyplot(figure)   
	st.caption(feature)	
	
	

# Implementation of the dashboard
st.title('Home Credit Default Risk')


# Logs et debugs
"---------------------------"
st.header("Section consacrée aux logs et debugs")
st.write("optimum_threshold :", optimum_threshold())


# ID client
"---------------------------"
st.header('Client ID')
id_client = st.text_input("Enter client ID", value="216030")
st.caption("Examples of clients predicted negative (no default) : 216030")
st.caption("Examples of clients predicted positive (credit default) : 215699") 


# Données client
"---------------------------"  
st.header('Client data')
one_client_pandas = fetch_data(id_client)
with st.expander("See data", expanded=False):
	st.dataframe(one_client_pandas.T)


# Result of credit application
"---------------------------" 
st.header('Result of credit application')
probability = fetch_proba_default(id_client)
if probability < optimum_threshold(): 
	st.success(f"  \n __CREDIT ACCEPTED__  \n  \nThe probability of default of the applied credit is __{round(100*probability,1)}__% (lower than the threshold of {100*optimum_threshold()}% for obtaining the credit).  \n ")
else:
	st.error(f"__CREDIT REFUSED__  \nThe probability of default of the applied credit is {round(100*probability,1)}% (higher than the threshold of {100*optimum_threshold()}% for obtaining the credit).  \n ")
# Rectangle Gauge
plt.style.use('default')
fig, ax = plt.subplots(figsize=(10,1))
fig.suptitle("Probability of credit default (%)", size=15, y=1.1)
ax.add_patch(Rectangle((0,0), width=optimum_threshold()*100, height=1, color=(0.5,0.9,0.5,0.5)))   
ax.add_patch(Rectangle((optimum_threshold()*100,0), width=100-optimum_threshold()*100, height=1, color=(1,0,0,0.5)))   
ax.plot((optimum_threshold()*100, optimum_threshold()*100), (0,1), color='#FF8C00', ls=(0,(0.5,0.5)), lw=6)
ax.add_patch(FancyArrowPatch((probability*100, 0.75), (probability*100, 0), mutation_scale=20))
ax.set_xlim(0,100)
ax.set_ylim(0,1)
ax.set_xticks(range(0,105,10))
ax.set_yticks([])
st.pyplot(fig)


# Position du client vs les autres
"---------------------------" 
st.header('Ranking of the client compared to other clients (KDE proba défaut, version data scientist)')
figure_kde_distribution_proba_default_for_datascientist = joblib.load('./src/figure_kde_distribution_proba_default_for_datascientist.joblib') 
plt.annotate(text=f"Client {id_client}", xy=(probability,0), xytext=(probability,3), arrowprops=dict(facecolor='k', arrowstyle='simple'))
st.pyplot(figure_kde_distribution_proba_default_for_datascientist)   



"---------------------------" 
st.header('Ranking of the client compared to other clients  (histogramme proba défaut, version data scientist)')
figure_hist_distribution_proba_default_for_datascientist = joblib.load('./src/figure_hist_distribution_proba_default_for_datascientist.joblib') 
plt.annotate(text=f"Client {id_client}", xy=(probability,0), xytext=(probability,1000), arrowprops=dict(facecolor='k', arrowstyle='simple'))
st.pyplot(figure_hist_distribution_proba_default_for_datascientist)  

# Feature Importance
"---------------------------" 
st.header('Global Feature Importance by Feature Permutation (version data scientist)')
figure_features_permutation_importances_for_datascientist = joblib.load('./src/figure_features_permutation_importances_for_datascientist.joblib') 
st.pyplot(figure_features_permutation_importances_for_datascientist)  
st.caption("For each feature, it is the average change of the predicted score after randomisation of the feature values in the dataset.")
with st.expander("See definitions of the features", expanded=False):
	pass   # to be completed   # en fait ce serait mieux que les définitions soient passées par un graphe plotly




# Position du client vs les autres
"---------------------------" 
st.header('Positionnement du client par rapports aux autres clients dans les principales features')
left_column, middle_column, right_column = st.columns([1, 1, 1])
with left_column:
	plot_kde('EXT_SOURCE_2') 
	plot_kde('PAYMENT_RATE')
with middle_column:
	plot_kde('EXT_SOURCE_3')
	barplot('ORGANIZATION_TYPE')
with right_column:
	plot_kde('EXT_SOURCE_1')
	plot_kde('AMT_ANNUITY')
