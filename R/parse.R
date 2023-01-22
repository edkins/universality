library(purrr)

dat <- sapply(readLines('data.csv'),
	      partial(mapply,
		      compose(partial(mapply,
				      as.numeric),
			      partial(strsplit,
				      split=','))))

write.csv(dat,'dat.t.csv')

dat <- read.csv('data-transposed.csv',header=F)

inputs <- split(dat,
		do.call(c,
			lapply(1:(nrow(dat)/256),
			       rep,256)))


heads <- do.call(rbind,lapply(inputs,t))

input.id <- do.call(c,lapply(1:length(inputs),
			     rep,
			     nrow(heads)/length(inputs)))
model.id <- rep(do.call(c,
			lapply(1:(ncol(dat)/8),
			       rep,8)),
		length(inputs))
head.id <- rep(1:8,nrow(heads)/8)

groups <- data.frame(input=input.id,
		    model=model.id,
		    head=head.id)

write.csv(heads,'heads.csv',row.names=F)
write.csv(groups,'groups.csv',row.names=F)
