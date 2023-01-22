source("autoencode.R")

library(purrr)

heads <- as.matrix(read.csv('heads.csv'))
groups <- read.csv("groups.csv")

reduce.encode <- function(x,y){
	autoencode(heads,c(x,y),groups,
		   paste0('encode',as.character(y)),
			  epochs=10)
	return(c(x,y))
}

encodings <- Reduce(reduce.encode,c(64,32,16,8,4,2),128)

encode2 <- autoencode(heads,c(128,64,32,16,8,4,2),groups,'2',epochs=10)
encode2 <- autoencode(heads,c(128,64,32,16,8,4),groups,'4',epochs=10)
