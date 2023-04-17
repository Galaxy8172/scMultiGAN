generate_data <- function(expression.matrix){
	
	ltpm <- log2(t(t(expression.matrix)/colSums(expression.matrix)) * 1e6 + 1)
	max_reads_count <- apply(ltpm,2,max)
	ltpm_norm <- t(t(ltpm) / max_reads_count)


	upSample_zero <- function(mtx, rowNum){
 		 mRows <- dim(mtx)[1]
  		 mCols <- dim(mtx)[2]
  		 if(mRows>=rowNum){
    			return(mtx)
  		   } else{
    			zero_matrix = matrix(rep(0, mCols*(rowNum-mRows)),rowNum-mRows, mCols)
			colnames(zero_matrix) = colnames(mtx)
    			return(rbind(mtx,zero_matrix))
  			}

		}

	genecount <- dim(ltpm_norm)[1]
	fig_h <- ceiling(sqrt(genecount))
	matrix_upsample <- upSample_zero(ltpm_norm,fig_h^2)
  rownames(matrix_upsample)[(genecount+1):fig_h^2] <- paste("gene",seq(fig_h^2-genecount),sep="_")

	return(matrix_upsample)
}

generate_label <- function(expression.matrix,use_label=T,label=NULL,k=NULL){

  get_k_cells_expression <- function(expression,distance,cell,k){
    k_cells_expression <- expression[,order(distance[cell,])[2:(k+1)]]
    k_cells_mean <- as.data.frame(apply(k_cells_expression,1,mean))
    return(k_cells_mean)
  }
	if(use_label){
		#cell_labels <- unique(label[,1])
		#for(cell_label in cell_labels){
			#i = 1
			#cell_index <- grep(cell_label,label[,1])
			#bulk_part <- apply(expression.matrix[,cell_index],1,mean)
			#target_part <- matrix( rep(bulk_part,length(cell_index)),length(bulk),length(cell_index))
			#colnames(target_part) <- colnames(expression.matrix[,cell_index])
			#if(i == 1){
				#bulk <- bulk_part
				#target <- target_part
			#}
			#else{
				#bulk <- cbind(bulk,bulk_part)
			#}
			#i = i + 1
      cl <- makeCluster(32)
      registerDoParallel(cl)
      expression.label <- foreach(cell_label=label[,1],.combine = "cbind") %dopar% as.data.frame(apply(expression.matrix[,label[,1]==cell_label],1,mean))
      stopCluster(cl)
      colnames(expression.label) <- colnames(expression.matrix)
      return(expression.label)
		}
   else{
      cell_distance <- distance(t(expression.matrix),method="euclidean")
      rownames(cell_distance) <- colnames(expression.matrix)
      colnames(cell_distance) <- colnames(expression.matrix)
      cl <- makeCluster(30)
      registerDoParallel(cl)
      expression.label <- foreach(cell=colnames(expression.matrix),.combine = "cbind") %dopar% get_k_cells_expression(expression.matrix,cell_distance,cell,k)
      stopCluster(cl)
      colnames(expression.label) <- colnames(expression.matrix)
      return(expression.label)
   }
	}
