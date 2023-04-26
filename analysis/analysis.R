# ----------------------------------------------------------
install.packages('ggplot2')
install.packages('reshape2')
# ----------------------------------------------------------

set.seed(42)


library(ggplot2)
library(reshape2)

# ----------------------------------------------------------
# Reading data
# ----------------------------------------------------------

rf.df   = read.csv("results/RandomForest_f1_scores.csv") 
unet.df = read.csv("results/Unet_f1_scores.csv")

# ----------------------------------------------------------
# Organize data RF data
# ----------------------------------------------------------

run1 = rf.df[,1:2]
run2 = rf.df[,3:4]
run3 = rf.df[,5:6]
run4 = rf.df[,7:8]
run5 = rf.df[,9:10]

run1$run = 1 
run2$run = 2 
run3$run = 3 
run4$run = 4 
run5$run = 5 

colnames(run1) = c("fscore", "id", "run")
colnames(run2) = c("fscore", "id", "run")
colnames(run3) = c("fscore", "id", "run")
colnames(run4) = c("fscore", "id", "run")
colnames(run5) = c("fscore", "id", "run")

rf.full = rbind(run1, run2, run3, run4, run5)

#removing png extension
rf.full$id = as.numeric(gsub(rf.full$id, pattern = ".png", replacement = ""))
rf.full$algo = "RF"

# ----------------------------------------------------------
# Organize data UNET data
# ----------------------------------------------------------

unet.run1 = unet.df[,1:2]
unet.run2 = unet.df[,3:4]
unet.run3 = unet.df[,5:6]
unet.run4 = unet.df[,7:8]
unet.run5 = unet.df[,9:10]

unet.run1$run = 1 
unet.run2$run = 2 
unet.run3$run = 3 
unet.run4$run = 4 
unet.run5$run = 5 

colnames(unet.run1) = c("fscore", "id", "run")
colnames(unet.run2) = c("fscore", "id", "run")
colnames(unet.run3) = c("fscore", "id", "run")
colnames(unet.run4) = c("fscore", "id", "run")
colnames(unet.run5) = c("fscore", "id", "run")

unet.full = rbind(unet.run1, unet.run2, unet.run3, unet.run4, unet.run5)

#removing png extension
unet.full$id = as.numeric(gsub(unet.full$id, pattern = ".png", replacement = ""))
unet.full$algo = "UNET"

# ----------------------------------------------------------
# Violin + Boxplot
# ----------------------------------------------------------

df.complete = rbind(rf.full, unet.full)

# g = ggplot(df.complete, aes(x = algo, y = fscore, group = algo))
# g = g + geom_boxplot()
# g 

g = ggplot(df.complete, aes(x = algo, y = fscore, group = algo))
g = g + geom_violin() + geom_boxplot(width=0.1)
g = g + labs(x = "Algorithm", y = "FScore")
# g 
ggsave(g, filename = "violinPlot.pdf", width = 3.54, heigth = 2.75)
# Saving 3.54 x 2.75 in image


# ----------------------------------------------------------
# Wilcoxon test - paired test
# ----------------------------------------------------------

# ordering by image id
rf.full   = rf.full[order(as.numeric(rf.full$id)),]
unet.full = unet.full[order(as.numeric(unet.full$id)),]

obj = wilcox.test(x = rf.full$fscore, y = unet.full$fscore, paired = TRUE, conf.level = 0.95)

print(obj$p.value)
if(obj$p.value < 0.05) {
	cat("There is statistical difference between the methods!")
} else {
	cat("There is no statistical difference between the methods!")
}

# ----------------------------------------------------------
# Heatmap
# ----------------------------------------------------------

# agregate by image (mean fscore)
# sort from best to worst

# ------------
# RF
# ------------

rf.ids = unique(rf.full$id)
aux.rf = lapply(rf.ids, function(id) {
	# print(id)
	sel = rf.full[which(rf.full$id == id),]
	return(mean(sel$fscore))
})

rf.avg = data.frame(cbind(rf.ids, unlist(aux.rf)))
colnames(rf.avg) = c("id", "fscore")
rf.avg = rf.avg[order(rf.avg$fscore),]

# ------------
# UNET
# ------------

unet.ids = unique(unet.full$id)
aux.unet = lapply(unet.ids, function(id) {
	# print(id)
	sel = unet.full[which(unet.full$id == id),]
	return(mean(sel$fscore))
})

unet.avg = data.frame(cbind(unet.ids, unlist(aux.unet)))
colnames(unet.avg) = c("id", "fscore")
unet.avg = unet.avg[order(unet.avg$fscore),]

# ------------
# ------------

scoresByImage = cbind(rf.avg, unet.avg)
colnames(scoresByImage) = c("RF.id", "RF.fscore", "Unet.id", "Unet.fscore")
write.csv(scoresByImage, "./scoresByImage.csv")


# 12 images with fscore < 0.5
print(scoresByImage[1:12,])

# ------------
# heatmp
# ------------

rf.avg$algo   = "RF"
unet.avg$algo = "UNET"

rf.avg$id   = as.factor(rf.avg$id)
unet.avg$id = as.factor(unet.avg$id)


df.avg = rbind(rf.avg, unet.avg)

g2 = ggplot(df.avg, aes(x = id, y = algo, fill = fscore))
g2 = g2 + geom_tile() + theme_bw()
g2 = g2 + scale_fill_gradient2(low = "red", high = "blue", mid = "white", 
	midpoint = 0.5, na.value = "grey50")
g2 = g2 +theme(axis.text.x=element_text(size=7))
g2 = g2 + theme(axis.text.x=element_text(angle = 90, hjust = 1))
g2 = g2 + labs(x = "Image ID", y = "Algorithm")
g2 
ggsave(g2,filename = "predictionsPlot.pdf", width = 6.87, heigth = 1.92)
# Saving 6.87 x 1.92 in image
# g3 + scale_x_discrete(guide = guide_axis(n.dodge = 2))

# ----------------------------------------------------------
# ----------------------------------------------------------
