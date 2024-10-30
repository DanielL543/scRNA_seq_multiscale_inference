# scRNA-seq multiscale inference approach
This contains the code for the analysis in the multiscale inference approach 
manuscript.

To perform the multiscale analysis, follow this code file order:
1. `RPM_cellchat_individual_TP_analysis.Rmd`
2. `RPM_stratified_comparison.Rmd`
3. `RPM_DEA_decoupleR.Rmd`
4. `LIANA_data_preparation.Rmd`
5. `liana_signaling_network_inference.py`
6. `network_creation.Rmd`

The final markdown document will create two files that can be used in cytoscape 
to create the network. Some filtering is also done in cytoscape to remove any 
signaling intermediates and transcription factors with no downstream targets.

## **Summary of Steps**
**Step 1**: Will take the gene expression matrix and follows the standard _CellChat_ 
workflow. This will infer the active signaling pathways and L-R interactions 
between the cell types in the data. The processed cellchat object will be saved 
to be used in the subsequent steps.

**Step 2**: This step is only for comparing the inferred cell-cell communication 
networks in a temporal dataset. Will read in the processed cellchat objects for
each individual timepoint and follows the _CellChat_ comparison analysis 
workflow. This step is not necessary to proceed with the following steps.

**Step 3**: This will perform differential expression analysis and transcription
factor activity inference for each cell type present in the data. This will 
create a seurat object and follows the standard _Seurat_ workflow to obtain the 
markers/differentially expressed genes for each cell type. The log-fold values
are then used to infer the activity of TFs. The univariate linear model method
from _decoupleR_ is used to infer TF activity. The differentially expressed 
genes and inferred TFs are used in the next steps.

**Step 4**: This step will filter and format the files that are required for the
_LIANA+/CORNETO_ signaling network inference workflow/algorithm. The L-R 
interactions are extracted from the processed cellchat object. TFs are filtered
for activity, significance,  and if they are present in the expression data. 
Differentially expressed markers are filtered for significance. Additionally,
protein-protein interactions and transcriptional regulation interactions are
imported from _OmniPath_. 

**Step 5**: This will follow the _LIANA+_ Intracellular Signaling network workflow.
For our analysis, we only use _CellChat_ to infer the L-R interactions.
Additionally, we also augment the node weights using log-fold values from the 
differential expression analysis. The output is a signaling network from 
receptor(s) to TF(s). This network is used in the next step to create the final 
network. This creates a network for one cell type within the data. Use the 
relevant L-R interactions, log-fold values and TFs for each cell type to infer
the signaling network for that cell type.

**Step 6**: This will create the final network connecting the L-R interactions to 
the downstream TFs and regulons. The transcriptional regulation interactions 
from _OmniPath_ are used to add the target genes to the TFs. A dataframe 
consisting of source, target and interaction type is created as well as an 
attribute table for each gene/node present in the network. These two are 
exported and used in Cytoscape to create the network.