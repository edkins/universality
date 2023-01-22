source('autoencode.R')

library(dirfns)
library(moreComplexHeatmap)

library(purrr)
library(optparse)
library(igraph)
library(umap)
library(ggplot2)

parser <- OptionParser()
parser <- add_option(parser, '--dir', action = 'store',
		     default = '2023-01-21')
parser <- add_option(parser, '--k', action = 'store',
		     default = 9,type = 'integer')
opts <- parse_args(parser)
k <- opts$k

heads <- as.matrix(read.csv('heads.csv'))

colnames(heads) <- mapply(paste,'token',
			  do.call(c,
				  lapply(as.character(1:16),
					 rep,16)),
			  rep(as.character(1:16)),
			  sep='_')

groups <- read.csv("groups.csv")
zeros <- apply(heads,2,purrr::compose(all,partial(`==`,0)))

corHeatmap(heads[,!zeros])

models <- lrtab(opts$dir,
		read.model,
		pattern='model',
		heads)

aic <- aic.table(models)
dir.tab(aic,'aic')

model <- models[[which.min(aic[,'aic'])]]$model
encoded <- models[[which.min(aic[,'aic'])]]$encoded

outmodel <- keras_model(
	inputs=get_layer(
		model, 
		index=length(model$layers)/2+1)$input, 
	outputs=get_layer( 
		model, 
		index=length(model$layers))$output)

predict(outmodel,diag(1,2,2))
embedding <- do.call(rbind,
		     lapply(runif(100,-1,1),
			    cbind,
			    runif(100,-1,1)))
pred <- predict(outmodel,embedding)

dat <- setNames(as.data.frame(cbind(embedding,pred)),
		  c('embedding1',
		    'embedding2',
		    colnames(heads)))
dat <- dat[seq(10,10000,10),]

plots <- lapply(names(dat)[-1:-2],
		function(y) ggplot( 
			dat, 
			aes_string(x=names(dat)[1], 
				   y=y, 
				   col=names(dat)[2]))+
		scale_color_viridis_c()+geom_point())

arrange.stats(plots,'embeddings.all',height=128,width=16)
arrange.stats(plots[!zeros],'embeddings.nonzero',height=128,width=16)
arrange.stats(plots[1:12],'embeddings',height=12,width=16)

encode.head <- do.call(rbind,split(encoded,
		     apply(groups[,-1],1,
			   purrr::compose(partial(paste,collapse='-'),
					  as.character))))

modelid <- sub('-.*','',row.names(encode.head))
headid <- sub('.*-','',row.names(encode.head))

dists <- as.matrix(dist(encode.head))

dir.pdf('dist')
draw(Heatmap(dists))
dev.off()

g <- get.knn(dists,k)

umap.pos <- umap(encode.head,n_neighbors=k)$layout
colnames(umap.pos) <- c("UMAP1","UMAP2")

plot.edge(umap.pos,g,'knn',1:9,'','')

leidens <- res.unif(c(0.01,3),k,dists,
		    modelid,
		    1000)
dir.csv(leidens,'leiden')

which.clust <- which.max(leidens[,2])
clusts <- leidens[which.clust,-1:-6]


arrange.stats(mapply(dot.col,
		     col=list(as.character(clusts),
			      modelid),
		     id=c('cluster','model'),
		     MoreArgs=list(y="UMAP2",
				   dat=as.data.frame(umap.pos)),
		     SIMPLIFY=F),
	      'clusters',2)

plots <- lapply(colnames(leidens)[2:6],
		dot.stat,as.data.frame(leidens))
arrange.stats(plots,'optimization')

spl <- paste0(as.character(groups[,2]),'-',
	     as.character(groups[,3]))

dir.pdf('modelhm')
Heatmap(heads,show_row_names=F,show_column_names=F,
	split=spl)
dev.off()

dir.pdf('clusthm')
Heatmap(heads,show_row_names=F,show_column_names=F,
	split=rep(clusts,nrow(heads)/length(clusts)))
dev.off()
