# LIANA+ signaling network inference
## I am using the TP7 data from the SCLC RPM dataset as an example here.
## The process is the same for the other datasets.
## Also, this network is specifically for A2 cells. 

# Import necessary libraries
import numpy as np
import pandas as pd
import scanpy as sc
import scipy as sc
from sklearn.preprocessing import MinMaxScaler
import plotnine as p9
import liana as li
import decoupler as dc 
import omnipath as op
import corneto as cn
import gurobipy as gb

# Read in L-R interaction results
LR_res = pd.read_csv("/Users/lopez/OneDrive/Documents/School/Research/SCLC/Inferring_mechanisms_heterogeneity_maintanence/clean_code/data/df7Net.txt", sep='\t')
# Select only the L-R interactions in which A2 cells are the receiving cell
lr_inter = LR_res[LR_res['target'] == 'A2']
# Group dataframe and sum the probabilities if they are the same receptor and interaction
grouped_intereactions = lr_inter.groupby(['receptor', 'interaction_name'])['prob'].sum().reset_index()
# Merge 
lr_inter_merged = lr_inter.merge(grouped_intereactions, on=['receptor', 'interaction_name'], suffixes=('', '_sum'), how='left')
# Obtain Receptor and probability score
input_scores_a2 = lr_inter_merged.set_index('receptor')['prob_sum'].to_dict()

# Read in TF activity results
inferred_TFs_a2 = pd.read_csv('/Users/lopez/OneDrive/Documents/School/Research/SCLC/Inferring_mechanisms_heterogeneity_maintanence/clean_code/data/active_A2_TFs_TP7.txt', sep='\t')
# Put TFs in dictionary
output_scores_a2 = inferred_TFs_a2.set_index('source')['score'].to_dict()
# Correct title case (only needed for mouse data)
output_scores_a2 = {key.capitalize(): value for key, value in output_scores_a2.items()} 

# Read in PPI network from OmniPath
ppis = pd.read_csv('/Users/lopez/OneDrive/Documents/School/Research/SCLC/Inferring_mechanisms_heterogeneity_maintanence/clean_code/data/ppi_with_KE_PE.txt', sep='\t')
# Reformat df
ppis['mor'] = ppis['is_stimulation'].astype(int) - ppis['is_inhibition'].astype(int)
ppis = ppis[(ppis['mor'] != 0) & (ppis['curation_effort'] > 1) & ppis['consensus_direction']]
input_pkn = ppis[['source_genesymbol', 'mor', 'target_genesymbol']]
input_pkn.columns = ['source', 'mor', 'target']

# Convert the PPI network into a knowledge graph
prior_graph = li.mt.build_prior_network(input_pkn, input_scores_a2, output_scores_a2, verbose=True)

# Calculate node weights
# min-max scaler to scale log-fold values between 0 and 1
scaler = MinMaxScaler()
# Read in log-fold values for A2 cells
markers_a2 = pd.read_csv('/Users/lopez/OneDrive/Documents/School/Research/SCLC/Inferring_mechanisms_heterogeneity_maintanence/clean_code/data/TP7_a2_markers.txt', sep='\t', index_col=5)
# Filter for significance
markers_a2 = markers_a2.loc[(markers_a2['p_val_adj'] < 0.05)]
# Reshape for Corneto
markers_a2 = markers_a2['avg_log2FC'].to_frame()
markers_a2['Norm_logFC'] = scaler.fit_transform(markers_a2['avg_log2FC'].values.reshape(-1,1))
markers_a2 = pd.Series(markers_a2.Norm_logFC.values, index=markers_a2.index).to_dict()

# Signaling network algorithm
df_res, problem = li.mt.find_causalnet(
    prior_graph,
    input_scores_a2,
    output_scores_a2,
    markers_a2,
    # penalize (max_penalty) nodes with counts in less than 0.1 of the cells
    node_cutoff=0.1,
    max_penalty=1,
    # the penaly of those in > 0.1 prop of cells set to:
    min_penalty=0.01,
    edge_penalty=0.01,
    missing_penalty=100,
    verbose=True,
    solver='GUROBI', 
    seed=1234
    )

# Visualize network
cn.methods.carnival.visualize_network(df_res)

# Save network
# df_res.to_csv('/path/to/file')