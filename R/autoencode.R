model.layers <- function(layers,activation){
	require(keras)
	if(length(activation)==1) {
		activation <- rep(activation,
				  length(layers)-2)
	}
	model <- keras_model_sequential()
	model %>% layer_dense(units = layers[2], 
			      activation = activation[1], 
			      input_shape = layers[1])
	layerfn <- function(x,i){
		layer_dense(x, units=layers[i+1],
			    activation=activation[i])
	}
	model <- Reduce(layerfn,2:length(activation),model)
	#         for (i in 2:length(activation)){
	#                 model %>% layer_dense(units=layers[i+1],
	#                                       activation=activation[i])
	#         }
	model %>% layer_dense(units=layers[length(layers)])
	return(model)
}

get.model <- function(x,y,layers,activation='tanh',
		      loss='mean_squared_error',
		      optimizer='adam',epochs=10000,
		      batch_size=256){
	require(keras)
	model <- model.layers(c(ncol(x),layers),activation)
	model %>% compile(
	  loss = loss, 
	  optimizer = optimizer
	)

	# fit model
	model %>% fit(
	  x = as.matrix(x), 
	  y = as.matrix(y), 
	  epochs = epochs,
	  batch_size=batch_size,
	  verbose = 0
	)
	return(model)
}

denoise <- function(x,y,layers,out,...){
	require(keras)
	require(dirfns)
	out <- mkdate(out,'')
	model <- get.model(x,y,layers,'tanh',...)
	save_model_hdf5(model,paste0(out,'.model'))
	return(model)
}

autoencode <- function(dat,layers,groups,out,...){
	require(dirfns)
	require(keras)
	out <- mkdate(out,'')
	layersfull <- c(layers,
			rev(layers[-length(layers)]),
			ncol(dat))
	model <- get.model(dat,dat,layersfull,'tanh',...)
	model.eval(model,as.matrix(dat),as.matrix(dat),
		   groups,length(layers),out)
	save_model_hdf5(model,paste0(out,'.model'))
	return(model)
}

plot.clust <- function(dat,clust,out,x=V1,y=V2,labs=names(cols),title='',width=20,ncols=3,...){
	require(ggplot2)
	require(ggpubr)

	clust <- as.data.frame(clust)
	cols <- apply(clust,2,as.character)
	cols <- as.data.frame(cols)
	names(cols) <- make.names(names(clust))

	nrows <- ceiling(ncol(cols)/ncols)
	height <- width/ncols*nrows

	dot <- function(z,dat,cols) {
		dat <- cbind(as.data.frame(dat),cols)
		ggplot(dat,
		       aes_string(x=names(dat)[1],
				  y=names(dat)[2],
				  col=z)) + geom_point()
	}
	if(!is.null(dim(dat))){
		plots <- lapply(names(cols),dot,dat,cols)
	}else{
		plots <- mapply(dot,names(cols),dat,MoreArgs = list(cols=cols),SIMPLIFY = F)
	}
	g <- ggarrange(plotlist=plots,
		       labels=labs,ncol=ncols,
		       nrow=nrows)

	g <- annotate_figure(g, top = text_grob(title,
		       face = "bold", size = 14))

	ggexport(g,filename=paste0(out,'.pdf'),
		 width=width,height=height,...)
}


model.eval <- function(model,x,y,groups,outix,out){
	require(keras)
	#         require(umap)
	mse <- evaluate(model, x, y)

	#         outmodel <- keras_model(inputs = model$input, 
	#                                 outputs = get_layer(model, 
	#                                                     index=outix)$output)
	#         outlayer <- predict(outmodel, x)
	outlayer <- model.out(model,x,outix)
	#         model.umap <- umap(outlayer)$layout
	#         colnames(model.umap) <- c("UMAP1","UMAP2")
	#         dir.tab(model.umap,out)
	plot.clust(outlayer[,1:2],#model.umap,
		   groups,out,
	       title=paste('MSE =',
			   as.character(signif(mse,6))),
	       width=20)
}


model.out <- function(model,x,index){
	require(keras)
	outmodel <- keras_model(inputs = model$input, 
				outputs = get_layer(model, 
						    index=index)$output)
	res <- predict(outmodel, x)
	return(res)
}

read.model <- function(m,dat){
	require(keras)
	res <- list(model=load_model_hdf5(m))
	res$err <- evaluate(res$model,dat,dat)
	res$nlayer <- length(res$model$layers)
	res$encoded <- model.out(res$model,dat,res$nlayer/2)
	res$bottleneck <- ncol(res$encoded)
	res$aic <- 2*res$bottleneck-2*log(res$err)
	#res$dists <- as.matrix(dist(res$encoded))
	return(res)
}

aic.table <- function(models) 
	sapply(c('bottleneck','nlayer','err','aic'), 
	       function(x) sapply(models,'[[',x))

get.knn <- function(dists,k,mode='directed'){
	require(igraph)
	neighbors <- apply(dists,2,order)
	adj <- sapply(1:ncol(neighbors),function(i){
			      r <- dists[,i]
			      sel <- neighbors[-1:-k,i]
			      r[sel] <- 0
			      return(r)
	})
	g <- graph_from_adjacency_matrix(adj,mode,T,F)
}

res.unif <- function(range,k,dists,groups,reps=1000){
	require(parallel)
	require(igraph)

	g <- get.knn(dists,k,'plus')

	res <- runif(reps,range[1],range[2])
	out <- par.apply(res,test.leiden,
		k=k,g=g,dists=dists,reps=1000,groups=groups)
	return(cbind(res,t(out)))
}


test.leiden <- function(res,k,g,dists,reps,groups,...){
	require(cluster)
	require(leiden)
	require(purrr)


	clust <- leiden(g,resolution_parameter=res)
	if(all(clust==1)) return(c(rep(0,5),clust))
	
	clust.group <- split(groups,clust)

	tp <- sapply(clust.group,purrr::compose(length,unique))
	fp <- sapply(clust.group,purrr::compose(sum,duplicated))
	precision <- mapply('/',tp,
			    sapply(clust.group,length))
	recall <- tp/length(unique(groups))
	f <- 2*((precision*recall)/(precision+recall))

	sil <- mean(silhouette(clust,dists)[,3])

	return(c(f=mean(f),
		 precision=mean(precision),
		 recall=mean(recall),
		 mean_silhouette=sil,
		 nclust=max(clust),clust))
}

par.apply <- function(...,f=parSapply){
	require(parallel)
	ncore <- detectCores()-2
	cl <- makeCluster(ncore,"FORK")
	res <- f(cl,...)
	stopCluster(cl)
	return(res)
}

dot.stat <- function(y,dat,...){
	require(ggplot2)
	ggplot(dat,
	       aes_string(x=names(dat)[1],
			  y=y,...))+geom_point()
}

dot.col <- function(y,dat,cols='',id='group',...){
	require(ggplot2)
	dat <- do.call(cbind,append(list(dat),setNames(list(cols),id)))
	dot.stat(y,dat,col=id,...)
}


dir.f <- function(f,file.arg='filename'){
	require(dirfns)
	function(...,filename='',ext='',path='.',append.date=T){
		out <- mkdate(filename,ext=ext,path=path,append.date=append.date)
		arglist <- append(list(...),setNames(out,file.arg))
		do.call(f,arglist)
	}
}

arrange.stats <- function(plots,filename,ncols=3,labels=NULL,...){
	require(ggpubr)

	nrows <- ceiling(length(plots)/ncols)

	plots <- ggarrange(plotlist=plots,
			   labels=labels,
			   ncol=ncols,nrow=nrows)

	dir.f(ggexport)(plots,
			filename=paste0(filename,
					'.pdf'),...)
}

plot.edge <- function(dat,g,out,cols=NULL,
		      xlab=names(dat[1]),
		      ylab=names(dat[2])){
	require(igraph)
	dists <- get.edgelist(g)

	dir.pdf(out)
	plot(NULL, xlim=range(dat[,1]), ylim=range(dat[,2]),
	     xlab=xlab,ylab=ylab) 
	apply(dists,1,function(x) lines(dat[x,]))
	if(!is.null(cols)){
		   pts <- split(as.data.frame(dat),cols)
		   col <- rainbow(length(pts))
		   mapply(points,pts,col=col,pch=19)
	     }
	dev.off()
}

