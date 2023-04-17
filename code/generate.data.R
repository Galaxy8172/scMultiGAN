library(argparser)
parser <- arg_parser("R script")
parser <- add_argument(parser, "--expression_matrix_path", default = NULL, help = "raw data path" )
parser <- add_argument(parser, "--file_suffix", default = NULL, help = "suffix of the input file")
parser <- add_argument(parser, "--label_file_path", default = NULL, help = "label file path")
args <- parse_args(parser)
if(args$file_suffix == "csv")
	{
	raw = read.csv(args$expression_matrix_path, header=T)
}else{
	raw = read.table(args$expression_matrix_path, header=T)
}

label <- read.table(args$label_file_path,header=T)
cell_type_num <- length(unique(label[,1]))

generate_data <- function(expression.matrix){
	
	ltpm <- log2(t(t(expression.matrix)/colSums(expression.matrix)) * 1e6 + 1)
	max_reads_count <- apply(ltpm,2,max)
	ltpm_norm <- t(t(ltpm) / max_reads_count)


	impute_extra_zero <- function(mat, rowNum){
 		 n_row <- dim(mat)[1]
  		 n_cols <- dim(mat)[2]
  		 if(n_row>=rowNum){
    			return(mat)
  		   } else{
    			impute_zero_matrix = matrix(rep(0, n_cols*(rowNum-n_row)),rowNum-n_row, n_cols)
			colnames(impute_zero_matrix) = colnames(mat)
    			return(rbind(mat,impute_zero_matrix))
  			}

		}

	genecount <- dim(ltpm_norm)[1]
	fig_h <- ceiling(sqrt(genecount))
	matrix_upsample <- impute_extra_zero(ltpm_norm,fig_h^2)
  rownames(matrix_upsample)[(genecount+1):fig_h^2] <- paste("gene",seq(fig_h^2-genecount),sep="_")

	return(list(matrix_upsample = matrix_upsample, fig_h = fig_h))
}
result <- generate_data(raw)
scMultiGAN <- result$matrix_upsample
fig_h <- result$fig_h
sprintf("n cell labels used for training is %d",cell_type_num)
sprintf("image size used for training is %d", fig_h)
write.csv(scMultiGAN,file="scMultiGAN.csv",quote=F)
