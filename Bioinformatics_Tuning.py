
# coding: utf-8

# In[702]:

## Handle necessary imports
# Pandas, numpy, matplotlib, seaborn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Bigram, Trigram combinations
import itertools
from itertools import combinations

# Biopython
from Bio import SeqIO, SeqUtils
from Bio.SeqUtils import ProtParamData
from Bio.SeqUtils.ProtParam import ProteinAnalysis

# Sklearn
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV

# Classsifiers
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier


# In[7]:

#Load fasta sequence data
cyto = 'data/cyto.fasta.txt'
mito = 'data/mito.fasta.txt'
nucleus = 'data/nucleus.fasta.txt'
secreted = 'data/secreted.fasta.txt'
blind = 'data/blind.fasta.txt'


# In[8]:

#Load 'cyto' data into dataframe and add label
with open(cyto) as file:  
    identifiers = []
    cytosequences = []
    for seq_record in SeqIO.parse(file, 'fasta'):  
        identifiers.append(seq_record.id)
        cytosequences.append(seq_record.seq)

cyto_df = pd.DataFrame.from_records(cytosequences)
cyto_df['label'] = 'cyto'

#Replace None entries with NaN: useful for features later
cyto_df.fillna(value=np.nan, inplace=True)


# In[9]:

#Load 'mito' data into dataframe and add label
with open(mito) as file:  
    identifiers = []
    mitosequences = []
    for seq_record in SeqIO.parse(file, 'fasta'):  
        identifiers.append(seq_record.id)
        mitosequences.append(seq_record.seq)

mito_df = pd.DataFrame.from_records(mitosequences)
mito_df['label'] = 'mito'

#Replace None entries with NaN: useful for features later
mito_df.fillna(value=np.nan, inplace=True)


# In[10]:

#Load 'nucleus' data into dataframe and add label
with open(nucleus) as file:  
    identifiers = []
    nucsequences = []
    for seq_record in SeqIO.parse(file, 'fasta'):  
        identifiers.append(seq_record.id)
        nucsequences.append(seq_record.seq)

nucleus_df = pd.DataFrame.from_records(nucsequences)
nucleus_df['label'] = 'nucleus'

#Replace None entries with NaN: useful for features later
nucleus_df.fillna(value=np.nan, inplace=True)


# In[11]:

#Load 'secreted' data into dataframe and add label
with open(secreted) as file:  
    identifiers = []
    secsequences = []
    for seq_record in SeqIO.parse(file, 'fasta'):  
        identifiers.append(seq_record.id)
        secsequences.append(seq_record.seq)

secreted_df = pd.DataFrame.from_records(secsequences)
secreted_df['label'] = 'secreted'

#Replace None entries with NaN: useful for features later
secreted_df.fillna(value=np.nan, inplace=True)


# In[12]:

#Load 'blind' data into dataframe and add label
with open(blind) as file:  
    blindidentifiers = []
    blindsequences = []
    for seq_record in SeqIO.parse(file, 'fasta'):  
        blindidentifiers.append(seq_record.id)
        blindsequences.append(seq_record.seq)

blind_df = pd.DataFrame.from_records(blindsequences)
blind_df['label'] = 'blind'

#Replace None entries with NaN: useful for features later
blind_df.fillna(value=np.nan, inplace=True)


# In[303]:

# All sequence data
sequences = cytosequences+mitosequences+nucsequences+secsequences+blindsequences


# In[304]:

#Concatanate the dataframes into one with all the sequence data
full_data=pd.concat([cyto_df,mito_df,nucleus_df,secreted_df,blind_df],0,ignore_index=True)


# In[305]:

#Reindex the label column as the last column
cols = full_data.columns.tolist()
cols.insert(np.shape(full_data)[1], cols.pop(cols.index('label')))
full_data = full_data.reindex(columns= cols)
full_data.head()


# In[306]:

# Split labels column from data
labels = full_data.pop('label')


# In[307]:

# Define the train test split point
train_idx = 9222
train = full_data[:train_idx]
test = full_data[train_idx:]


# In[308]:

# Feature engineering starts here 
# i.e. find relevant features for the sequence data

# 1. Sequence length
sequence_length = pd.DataFrame(full_data.count(axis=1),columns=['seq_len'])
sequence_length = sequence_length.astype(float)


# In[309]:

# 2. Global amino acid count i.e. no of amino acid per sequence
global_count = full_data.T

d = {}
for i in range(len(full_data)):
    series_global = global_count.groupby(i).size()
    d[str(i)]=pd.DataFrame({i:series_global.values},index = series_global.index+' global')

global_counts_feats = pd.concat([d[str(i)] for i in range(len(d))],axis=1)
global_counts_feats = global_counts_feats.fillna(0)
global_counts_feats = global_counts_feats.T


# In[311]:

# 3. Bigram amino counts i.e (AB, AC, AE etc)
aminos = pd.DataFrame(series_global.index)
aminos.columns = ['amino']

bigrams = pd.DataFrame(list(combinations(aminos.amino, 2)))
bigrams = bigrams.apply(lambda x: ''.join(x), axis=1)
bicnt=[]

for seq in sequences:
    for bi in bigrams:
        bicnt.append(seq.count(bi))
        
bigramcounts = [bicnt[i:i+190] for i  in range(0, len(bicnt), 190)]

bigram_count_df = pd.DataFrame(bigramcounts)
bigram_count_df.columns = bigrams


# In[290]:

# 4. Trigram amino counts i.e (ABC, ACE etc)
aminos = pd.DataFrame(series_global.index)
aminos.columns = ['amino']

from itertools import combinations
sequences = cytosequences+mitosequences+nucsequences+secsequences+blindsequences
trigrams = pd.DataFrame(list(combinations(aminos.amino, 3)))
trigrams = trigrams.apply(lambda x: ''.join(x), axis=1)
tricnt=[]

for seq in sequences:
    for tri in trigrams:
        tricnt.append(seq.count(tri))
trigramcounts = [tricnt[i:i+969] for i  in range(0, len(tricnt), 969)]

trigram_count_df = pd.DataFrame(trigramcounts)
trigram_count_df.columns = trigrams


# In[525]:

# Find the best split for local amino count
aminoFirstCount=[]
aminoLastCount=[]
cnt = np.arange(10,60,10)

for j in cnt:
    for i in range(len(sequences)):
            X=ProteinAnalysis(str(sequences[i][j:]))
            aminoFirstCount.append(X.count_amino_acids())
for j in cnt:
    for i in range(len(sequences)):
            X=ProteinAnalysis(str(sequences[i][:j]))
            aminoLastCount.append(X.count_amino_acids())

aminofirstchunk = [aminoFirstCount[i:i+len(sequences)] for i  in range(0, len(aminoFirstCount), len(sequences))]
aminofirst10 = pd.DataFrame(aminofirstchunk[0])
aminofirst10.columns = [str(cols)+'_first' for cols in aminofirst10.columns]
aminofirst20 = pd.DataFrame(aminofirstchunk[1])
aminofirst20.columns = [str(cols)+'_first' for cols in aminofirst20.columns]
aminofirst30 = pd.DataFrame(aminofirstchunk[2])
aminofirst30.columns = [str(cols)+'_first' for cols in aminofirst30.columns]
aminofirst40 = pd.DataFrame(aminofirstchunk[3])
aminofirst40.columns = [str(cols)+'_first' for cols in aminofirst40.columns]
aminofirst50 = pd.DataFrame(aminofirstchunk[4])
aminofirst50.columns = [str(cols)+'_first' for cols in aminofirst50.columns]

aminolastchunk = [aminoLastCount[i:i+len(sequences)] for i  in range(0, len(aminoLastCount), len(sequences))]
aminolast10 = pd.DataFrame(aminolastchunk[0])
aminolast10.columns = [str(cols)+'_last' for cols in aminolast10.columns]
aminolast20 = pd.DataFrame(aminolastchunk[1])
aminolast20.columns = [str(cols)+'_last' for cols in aminolast20.columns]
aminolast30 = pd.DataFrame(aminolastchunk[2])
aminolast30.columns = [str(cols)+'_last' for cols in aminolast30.columns]
aminolast40 = pd.DataFrame(aminolastchunk[3])
aminolast40.columns = [str(cols)+'_last' for cols in aminolast40.columns]
aminolast50 = pd.DataFrame(aminolastchunk[4])
aminolast50.columns = [str(cols)+'_last' for cols in aminolast50.columns]

acc=[]
feats=[]

for feat_first, featfirst_name in zip([aminofirst10, aminofirst20, aminofirst30, aminofirst40, aminofirst50], 
                      ['First 10%', 'First 20%', 'First 30%','First 40%', 'First 50%']):    
    for feat_last, featlast_name in zip([aminolast10, aminolast20, aminolast30, aminolast40, aminolast50], 
                      ['Last 10%', 'Last 20%', 'Last 30%','Last 40%', 'Last 50%']):   
        features_df = pd.concat([feat_first, feat_last],1)
        labels = labels[:train_idx]
        test_df = features_df[train_idx:]
        features_df = features_df[:train_idx]
        le = preprocessing.LabelEncoder()
        le.fit(labels.values)
        labels_enc=le.transform(labels)
        labels_df = pd.DataFrame(labels_enc,columns=['labels'])
        X_train, X_test, y_train, y_test = train_test_split(features_df, labels_enc, random_state=0, test_size=0.3)
        lr = LogisticRegression()
        y_pred_lr = lr.fit(X_train, y_train).predict(X_test)
        acc.append(metrics.accuracy_score(y_test, y_pred_lr))
        feats.append([featfirst_name, featlast_name])
results = pd.DataFrame()
results['Accuracy'] = acc
results['Combinations'] = feats
        

results = results.sort_values(by='Accuracy', ascending=False)
# results[0]


# In[ ]:

# 6. First 50 amino acid count 
local_count_first50 = global_count.drop(global_count.index[50:])

d = {}
for i in range(len(full_data)):
    series_global = local_count_first50.groupby(i).size()
    d[str(i)]=pd.DataFrame({i:series_global.values},index = series_global.index+' first50')

first50_counts_feats = pd.concat([d[str(i)] for i in range(len(d))],axis=1)
first50_counts_feats = first50_counts_feats.fillna(0)
first50_counts_feats = first50_counts_feats.T


# In[20]:

# 7. Last 50 amino acid count 
local_count_last50 = global_count.drop(global_count.index[:50])

d = {}
for i in range(len(full_data)):
    series_global = local_count_last50.groupby(i).size()
    d[str(i)]=pd.DataFrame({i:series_global.values},index = series_global.index+' last50')

last50_counts_feats = pd.concat([d[str(i)] for i in range(len(d))],axis=1)
last50_counts_feats = last50_counts_feats.fillna(0)
last50_counts_feats = last50_counts_feats.T


# In[22]:

## Biopython Protein Analysis Features
# 8. Isoelectric Point
# 9. Aromaticity
# 10. Secondary Structure Fraction
# 11. Gravy
# 12. Instability Index
# 13. Flexibility
# 14. Amino Percent
# 15. Molecular Weight
# 16. Protein Scale ~ Hydrophobicity
# 17. Protein Scale ~ Hydrophilicity
# 18. Protein Scale ~ Surface accessibility 

sequences = cytosequences+mitosequences+nucsequences+secsequences+blindsequences
isoelectricPt=[]
aromaticity=[]
aminoPercent=[]
secstruct=[]
hydrophob=[]
hydrophil=[]
surface=[]
gravy=[]
molweight=[]
instidx=[]
flex=[]

for seq in sequences:
        X=ProteinAnalysis(str(seq))
        isoelectricPt.append(X.isoelectric_point())
        aromaticity.append(X.aromaticity())  
        aminoPercent.append(X.get_amino_acids_percent())
        secstruct.append(X.secondary_structure_fraction())

# These features throw Key & Value Errors due to non standard amino acids
# (i.e. out of the 20 standard ones) e.g. X, U etc
        try:
            gravy.append(X.gravy())
            molweight.append(X.molecular_weight())
            instidx.append(X.instability_index())
            flex.append(X.flexibility())
            hydrophob.append(X.protein_scale(ProtParamData.kd, 9, 0.4))
            hydrophil.append(X.protein_scale(ProtParamData.hw, 9, 0.4))
            surface.append(X.protein_scale(ProtParamData.em, 9, 0.4))

        except (KeyError,ValueError):
            gravy.append(0)
            molweight.append(0)
            instidx.append(0)
            flex.append([0,0])
            hydrophob.append([0,0])
            hydrophil.append([0,0])
            surface.append([0,0])

isoelectricPt_df = pd.DataFrame(isoelectricPt,columns=['isoelectricPt'])
aromaticity_df = pd.DataFrame(aromaticity,columns=['aromaticity'])
aminoPercent_df = pd.DataFrame()
aminoPercent_df = aminoPercent_df.from_dict(aminoPercent)
aminoPercent_df.columns = [str(col) + '%' for col in aminoPercent_df.columns]
secstruct_df = pd.DataFrame(secstruct,columns=['helix','turn','sheet'])
instidx_df = pd.DataFrame(instidx, columns=['instabilityIdx'])
gravy_df = pd.DataFrame(gravy, columns=['gravy'])
molWeight_df = pd.DataFrame(molweight, columns=['molWeight'])
flex_df = pd.DataFrame(pd.DataFrame(flex).mean(axis=1), columns=['flexibility'])
hydrophob_df = pd.DataFrame(pd.DataFrame(hydrophob).mean(axis=1), columns=['hydrophobicity'])
hydrophil_df = pd.DataFrame(pd.DataFrame(hydrophil).mean(axis=1), columns=['hydrophilicity'])
surface_df = pd.DataFrame(pd.DataFrame(surface).mean(axis=1), columns=['surface_accesibility'])


# In[549]:

# Final feature dataframe to be used for model
features_df = pd.concat([sequence_length,secstruct_df, flex_df,isoelectricPt_df,aromaticity_df,instidx_df,
                         gravy_df,aminoPercent_df,hydrophob_df, hydrophil_df, surface_df, molWeight_df,
                         global_counts_feats,aminofirst20,aminolast40], 1)

# features_df.head(10)


# In[550]:

# Normalize the data
from sklearn.preprocessing import StandardScaler
features_df_scaled = features_df.copy()
scaler = StandardScaler().fit(features_df_scaled)
features_df_scaled = scaler.transform(features_df_scaled)
features_df_scaled = features_df_scaled


# In[551]:

# Split out the test set
labels = labels[:train_idx]
test_df = features_df_scaled[train_idx:]
features_df_scaled = features_df_scaled[:train_idx]


# In[552]:

# Encoding the 4 class labels
le = preprocessing.LabelEncoder()
le.fit(labels.values)

labels_enc=le.transform(labels)
labels_df = pd.DataFrame(labels_enc,columns=['labels'])


# In[732]:

# Ensemble of classifiers
X = features_df_scaled
y = labels_enc
weights_list = [8.0, 1.0, 3.0, 9.0, 8.0, 11.0, 7.0, 1.0]


# In[ ]:

# # Ensemble hyperparameter tuning
# # Utility function to report best scores
# optimization_df = pd.DataFrame(columns=['rank','accuracy','lr__C', 'rf__n_estimators', 'rf__max_depth', 
#                                         'rf__min_samples_split', 'rf__min_samples_leaf', 
                                        'rf__max_features','ada__n_estimators','ada__learning_rate',
                                       'gb__n_estimators','gb__max_depth','gb__min_samples_split',
                                       'gb__min_samples_leaf','gb__max_feature','xg__gamma',
                                       'xg__learning_rate','xg__reg_lambda','xg__reg_alpha',
                                       'xg__max_depth','xg__min_child_weight','SVC__C',
                                       'NN__hidden_layer_sizes','NN__activation','NN__solver',
                                       'NN__learning_rate','NN__alpha','KNN__n_neighbors','KNN__p'])
optimized_list = []
optimized_dict = {}

def report(results, n_top=2):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            optimized_dict = {'lr__C': results['param_lr__C'].data[candidate], 
                              'rf__n_estimators': results['param_rf__n_estimators'][candidate],
                              'rf__max_depth': results['param_rf__max_depth'][candidate],
                              'rf__min_samples_split': results['param_rf__min_samples_split'][candidate],
                              'rf__min_samples_leaf': results['param_rf__min_samples_leaf'][candidate],
                              'rf__max_feature': results['param_rf__max_feature'][candidate],
                              'ada__n_estimators': results['param_ada__n_estimators'][candidate],
                              'ada__learning_rate': results['param_ada__learning_rate'][candidate],
                              'gb__n_estimators': results['param_gb__n_estimators'][candidate],
                              'gb__max_depth': results['param_rf__max_depth'][candidate],
                              'gb__min_samples_split': results['param_gb__min_samples_split'][candidate],
                              'gb__min_samples_leaf': results['param_gb__min_samples_leaf'][candidate],
                              'gb__max_feature': results['param_gb__max_feature'][candidate],
                              'xg__gamma': results['param_xg__gamma'].data[candidate],
                              'xg__learning_rate': results['param_xg__learning_rate'].data[candidate],
                              'xg__reg_lambda': results['param_xg__reg_lambda'].data[candidate],
                              'xg__reg_alpha': results['param_xg__reg_alpha'].data[candidate],
                              'xg__max_depth': results['param_xg__max_depth'].data[candidate],
                              'xg__min_child_weight': results['param_xg__min_child_weight'].data[candidate],
                              'SVC__C': results['param_SVC__C'].data[candidate],
                              'NN__hidden_layer_sizes': results['param_NN__hidden_layer_sizes'].data[candidate],
                              'NN__activation': results['param_NN__activation'].data[candidate],
                              'NN__solver': results['param_NN__solver'].data[candidate],
                              'NN__learning_rate': results['param_NN__learning_rate'].data[candidate],
                              'NN__alpha': results['param_NN__alpha'].data[candidate],
                              'KNN__n_neighbors': results['param_KNN__n_neighbors'].data[candidate],
                              'KNN__p': results['param_KNN__p'].data[candidate],
                              'accuracy': results['mean_test_score'][candidate]
                             }
            optimized_list.append(optimized_dict)
            print("Model with rank: {0}".format(i))
            print("Mean accuracy score: {0:.3f})".format(
                  results['mean_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))            

from time import time
# specify parameters and distributions to sample from
params = {
            'lr__C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            'rf__n_estimators': [120,300,500,800,1200],
            'rf__max_depth':[5,8,15,25,30,None],
            'rf__min_samples_split': [1,2,5,10,15,100],
            'rf__min_samples_leaf': [1,2,5,10],
            'rf__max_features': ['log2','sqrt',None],
            'ada__n_estimators': [50,120,300,500,800],
            'ada__learning_rate': [0.01,0.015,0.025,0.05,0.1],
            'xg__learning_rate': [0.01,0.015,0.025,0.05,0.1],
            'xg__gamma': [0.05,0.1,0.3,0.5,0.7,0.9,1.0],
            'xg__max_depth': [3,5,7,9,12,15,17,25],
            'xg__min_child_weight':[1,3,5,7],
            'xg__reg_lambda': [0.001,0.01,0.1,1.0],
            'xg__reg_alpha': [0,0.1,0.5,1.0],
            'SVC__C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            'SVC__gamma': ['auto', 0.01, 0.1, 0.5, 1.0],
            'NN__hidden_layer_sizes':[(50,),(100,),(200,),(300,)],
            'NN__activation':['identity', 'logistic', 'tanh', 'relu'],
            'NN__solver':['lbfgs', 'sgd', 'adam'],
            'NN__learning_rate':['constant', 'invscaling', 'adaptive'],
            'NN__alpha': [0.001,0.01,0.1,1.0],
            'KNN__n_neighbors':[2,4,8,16],
            'KNN__p':[2,3]
          }
eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('ada', clf3), ('gbr', clf4), ('xgb', clf5),
                                   ('SVC', clf6), ('Neural', clf7), ('KNN', clf8)], voting='soft', 
                        weights=weights_list.tolist())

# run randomized search
n_iter_search = 2
random_search = RandomizedSearchCV(eclf, param_distributions=params,n_iter=n_iter_search)
start = time()
random_search.fit(X, y)
print("Randomized Search CV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
random_search.cv_results_

# Save params
optimization_df = optimization_df.append(optimized_list)
optimization_df.to_csv('OptizedEnsembleParams.csv')


# In[ ]:



