# Functions for CellChat

# Create color palettes
spirited_cols <- c('#833437FF','#67B9E9FF','#8F8093FF','#C3AF97FF','#44A57CFF','#F0D77BFF') # Colors for cells
ghibli_div <- c('#0E84B4FF', '#B50A2AFF') # diverging colors

######################## Draw heatmaps #########################################

signaling_heatmaps <- function(CC_objects, pattern, width, height){
  # Initialize empty vector to store pathways
  pathways.union <- c()
  # Combine the identified signaling pathways from different datasets
  for(i in 1:length(chat_objects)){
    pathways.union <- unique(append(pathways.union,
                                    chat_objects[[i]]@netP$pathways))
  }
  # Initialize empty heatmap list
  hmaps <- list()
  # Iterate over cellchat objects and create heatmap for each
  for(i in 1:length(CC_objects)){
    hmaps[[i]] <- 
      netAnalysis_signalingRole_heatmap(CC_objects[[i]],
                                        pattern = pattern,
                                        signaling = pathways.union,
                                        title = names(CC_objects)[i],
                                        width = width,
                                        height = height, 
                                        color.heatmap = ifelse(pattern == 'outgoing', 'BuGn',
                                                               ifelse(pattern == 'incoming', 'GnBu',
                                                                      ifelse(pattern == 'all', 'OrRd', 'BuGn'))))
  }
  # Reduce heatmap lists into one combined heatmap
  combined_heatmap <- Reduce('+', hmaps)
  # Plot heatmap
  draw(combined_heatmap, ht_gap = unit(0.5,'cm'))
}

################# Aggregated signaling for specific pathway ####################

plot_agg_pathway <- function(timepoints, pathway){
  ## Need to know the timepoints in which the pathway is present
  CC_objects <- chat_objects[names(chat_objects) %in% timepoints]
  
  # Calculate the the edge weight
  edge.weight <- getMaxWeight(CC_objects, slot.name = 'netP', attribute = pathway)
  
  # Plot signaling
  par(mfrow=rev(n2mfrow(length(CC_objects))))
  for(i in 1:length(CC_objects)){
    netVisual_aggregate(CC_objects[[i]], 
                        signaling = pathway, 
                        layout = 'circle', 
                        edge.weight.max = edge.weight,
                        edge.width.max = 10,
                        signaling.name = paste(pathway, names(CC_objects)[i]),
                        vertex.weight = as.numeric(table(CC_objects[[i]]@idents)),
                        vertex.label.cex = 2.2,
                        color.use = spirited_cols)
  }
}

