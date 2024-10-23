# scRNA-seq multiscale inference approach
This contains the code for the analysis in the multiscale inference approach manuscript.

To perform the multiscale analysis, follow this code file order:
1. `RPM_cellchat_individual_TP_analysis.Rmd`
2. `RPM_stratified_comparison.Rmd`
3. `RPM_DEA_decoupleR.Rmd`
4. `LIANA_data_preparation.Rmd`
5. `liana_signaling_network_inference.py`
6. `network_creation.Rmd`

The final markdown document will create two files that can be used in cytoscape to create the network. Some filtering is also done in cytoscape to remove any signaling intermediates and transcription factors with no downstream targets.
