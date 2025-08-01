# scRNA-seq multiscale inference approach
This contains the code for the analysis in the multiscale inference approach 
manuscript.

![Multiscale Inference Overview](github_pipeline_image.png)

To perform the multiscale analysis, follow this code file order (the 
`RPM_stratified_comparison.Rmd` file only needs to be followed if doing 
timepoint comparisons):
1. [`RPM_cellchat_individual_TP_analysis.Rmd`](code/RPM_cellchat_individual_TP_analysis.Rmd)
2. [`RPM_stratified_comparison.Rmd`](code/RPM_stratified_comparison.Rmd)
3. [`RPM_DEA_decoupleR.Rmd`](code/RPM_DEA_decoupleR.Rmd)
4. [`LIANA_data_preparation.Rmd`](code/LIANA_data_preparation.Rmd)
5. [`liana_signaling_network_inference.py`](code/liana_signaling_network_inference.py)
6. [`network_creation.Rmd`](code/network_creation.Rmd)

The final markdown document will create two files that can be used in cytoscape 
to create the network. Some filtering is also done in cytoscape to remove any 
signaling intermediates and transcription factors with no downstream targets.

## **Summary of Steps**
This is a summary of the steps to follow in order to reproduce the inferred 
network results for the SCLC RPM dataset.

**Step 1**: Will take the gene expression matrix and follows the standard _CellChat_ 
workflow. This will infer the active signaling pathways and L-R interactions 
between the cell types in the data. The processed cellchat object will be saved 
to be used in the subsequent steps.
- Input files: [TP7 sparse matrix](data/TP7_norm_data.npz), [TP7 cell metadata](data/AD_norm_TP7_obs.csv),
[TP7 gene metadata](data/AD_norm_TP7_var.csv)
- Generated output file: [TP7 L-R interactions](data/df7net.txt)

**Step 2**: This step is only for comparing the inferred cell-cell communication 
networks in a temporal dataset. Will read in the processed cellchat objects for
each individual timepoint and follows the _CellChat_ comparison analysis 
workflow. **This step is not necessary to proceed with the following steps.**

**Step 3**: This will perform differential expression analysis and transcription
factor activity inference for each cell type present in the data. This will 
create a seurat object and follows the standard _Seurat_ workflow to obtain the 
markers/differentially expressed genes for each cell type. The log-fold values
are then used to infer the activity of TFs. The univariate linear model method
from _decoupleR_ is used to infer TF activity. The differentially expressed 
genes and inferred TFs are used in the next steps.
- Input files: [TP7 sparse matrix](data/TP7_norm_data.npz), [TP7 cell metadata](data/AD_norm_TP7_obs.csv),
[TP7 gene metadata](data/AD_norm_TP7_var.csv)
- Generated output files: [A2 DEA results](code/rds/TP7_markers_A2.rds), [A/N DEA results](code/rds/TP7_markers_AN.rds),
[P/Y DEA results](code/rds/TP7_markers_PY.rds), [A2 TF results](code/rds/TP7_TFs_activity_A2.rds), 
[A/N TF results](code/rds/TP7_TFs_activity_AN.rds), [P/Y TF results](code/rds/TP7_TFs_activity_PY.rds)

**Step 4**: This step will filter and format the files that are required for the
_LIANA+/CORNETO_ signaling network inference workflow/algorithm. The L-R 
interactions are extracted from the processed cellchat object. TFs are filtered
for activity, significance,  and if they are present in the expression data. 
Differentially expressed markers are filtered for significance. Additionally,
protein-protein interactions and transcriptional regulation interactions are
imported from _OmniPath_. 
- Input files: [TP7 sparse matrix](data/TP7_norm_data.npz), [TP7 cell metadata](data/AD_norm_TP7_obs.csv),
[TP7 gene metadata](data/AD_norm_TP7_var.csv), [TP11 sparse matrix](data/TP11_norm_data.npz), [TP11 cell metadata](data/AD_norm_count_TP11_obs.csv), [TP11 gene metadata](data/AD_norm_count_TP11_var.csv),
[RPM merged LR interactions](code/rds/mouse_cellchat_LR_int.rds)
- Generated output files: located in [`data/liana_preparation_outputs/`](data/liana_preparation_outputs/)

**Step 5**: This will follow the _LIANA+_ Intracellular Signaling network workflow.
For our analysis, we only use _CellChat_ to infer the L-R interactions.
Additionally, we also augment the node weights using log-fold values from the 
differential expression analysis. The output is a signaling network from 
receptor(s) to TF(s). This network is used in the next step to create the final 
network. This creates a network for one cell type within the data. Use the 
relevant L-R interactions, log-fold values and TFs for each cell type to infer
the signaling network for that cell type. In this example, we create a network for
the SCLC-A2 subtype at TP7. The networks for the other subtypes can be created by
replacing the relevant files.
- Input files: [TP7 split L-R interactions](data/liana_preparation_outputs/df7Net_split_LR.txt), 
[TP7 A2 TF activity results](data/liana_preparation_outputs/active_A2_TFs_TP7.txt),
[protein-protein interactions](data/liana_preparation_outputs/ppi_with_KE_PE.txt),
[TP7 A2 filtered DEA results](data/liana_preparation_outputs/TP7_a2_markers.txt)
- Generated output file: [TP7 A2 CORNETO network](data/TP7_A2_network.csv)

**Step 6**: This will create the final network connecting the L-R interactions to 
the downstream TFs and regulons. The transcriptional regulation interactions 
from _OmniPath_ are used to add the target genes to the TFs. A dataframe 
consisting of source, target and interaction type is created as well as an 
attribute table for each gene/node present in the network. These two are 
exported and used in Cytoscape to create the network. In this example, the 
generated output files are for the A2 subtype. Replace the relevant input files
for the other subtypes to generate the other networks.
- Input files: [TP7 sparse matrix](data/TP7_norm_data.npz), [TP7 cell metadata](data/AD_norm_TP7_obs.csv),
[TP7 gene metadata](data/AD_norm_TP7_var.csv), [TP7 split L-R interactions](data/liana_preparation_outputs/df7Net_split_LR.txt), [TP7 A2 CORNETO network](data/TP7_A2_network.csv), [SCLC gene list](genelist/sclc_gene_sig_clustered.txt),
[EMT gene list](genelist/EMTGenesUpdateHGNCNames.txt), [TP7 A2 TF results](code/rds/TP7_TFs_activity_A2.rds),
[TP7 A2 DEA results](code/rds/TP7_markers_A2.rds)
- Generated output files (these files are used in cytoscape): [Final network for the A2 subtype](data/final_network_A2.txt), [Network attributes for A2 subtype](data/attributes_table_A2.txt)

The code for the mathematical modeling and parameter scanning can be found in [`/code/model/`](code/model/).
