# ----------------------------------------------------------
# Uncomment below if the packages are not installed in the machine
#install.packages('ggplot2')
#install.packages('reshape2')
#install.packages('dplyr')
# ----------------------------------------------------------

set.seed(42)

library(ggplot2)
library(reshape2)

# ----------------------------------------------------------
# Reading data
# ----------------------------------------------------------

cat(" - Reading data\n")
rf.df   = read.csv("../results/RandomForest_f1_scores.csv") 
unet.df = read.csv("../results/Unet_f1_scores.csv")

rf.jac   = read.csv("../results/RandomForest_jaccard_scores.csv")
unet.jac = read.csv("../results/Unet_jaccard_scores.csv")

# ----------------------------------------------------------
# Organize data RF data
# ----------------------------------------------------------

cat(" - Organizing RF data\n")

run1 = rf.df[,1:2]
run1$run = 1 
colnames(run1) = c("value", "id", "run")

run2 = rf.df[,3:4]
run2$run = 2 
colnames(run2) = c("value", "id", "run")

run3 = rf.df[,5:6]
run3$run = 3 
colnames(run3) = c("value", "id", "run")

run4 = rf.df[,7:8]
run4$run = 4 
colnames(run4) = c("value", "id", "run")

run5 = rf.df[,9:10]
run5$run = 5 
colnames(run5) = c("value", "id", "run")

rf.full.fsc = rbind(run1, run2, run3, run4, run5)

#removing png extension
rf.full.fsc$id = as.numeric(gsub(rf.full.fsc$id, pattern = ".png", replacement = ""))
rf.full.fsc$algo = "RF"
rf.full.fsc$measure = "FScore"

#organize jaccard (IoU)

run1 = rf.jac[,1:2]
run1$run = 1
colnames(run1) = c("value", "id", "run")

run2 = rf.jac[,3:4]
run2$run = 2 
colnames(run2) = c("value", "id", "run")

run3 = rf.jac[,5:6]
run3$run = 3 
colnames(run3) = c("value", "id", "run")

run4 = rf.jac[,7:8]
run4$run = 4 
colnames(run4) = c("value", "id", "run")

run5 = rf.jac[,9:10]
run5$run = 5 
colnames(run5) = c("value", "id", "run")

rf.full.jac = rbind(run1, run2, run3, run4, run5)

#removing png extension
rf.full.jac$id = as.numeric(gsub(rf.full.jac$id, pattern = ".png", replacement = ""))
rf.full.jac$algo = "RF"
rf.full.jac$measure = "IoU"

# join them (results with both measures)
rf.full = rbind(rf.full.fsc, rf.full.jac)

# ----------------------------------------------------------
# Organize data UNET data
# ----------------------------------------------------------

cat(" - Organizing UNET data\n")

unet.run1 = unet.df[,1:2]
unet.run1$run = 1 
colnames(unet.run1) = c("value", "id", "run")

unet.run2 = unet.df[,3:4]
unet.run2$run = 2 
colnames(unet.run2) = c("value", "id", "run")

unet.run3 = unet.df[,5:6]
unet.run3$run = 3 
colnames(unet.run3) = c("value", "id", "run")

unet.run4 = unet.df[,7:8]
unet.run4$run = 4 
colnames(unet.run4) = c("value", "id", "run")

unet.run5 = unet.df[,9:10]
unet.run5$run = 5 
colnames(unet.run5) = c("value", "id", "run")

unet.full.fsc = rbind(unet.run1, unet.run2, unet.run3, unet.run4, unet.run5)

#removing png extension
unet.full.fsc$id = as.numeric(gsub(unet.full.fsc$id, pattern = ".png", replacement = ""))
unet.full.fsc$algo = "UNET"
unet.full.fsc$measure = "FScore"

# unet jaccard (IoU)

unet.run1 = unet.jac[,1:2]
unet.run1$run = 1 
colnames(unet.run1) = c("value", "id", "run")

unet.run2 = unet.jac[,3:4]
unet.run2$run = 2 
colnames(unet.run2) = c("value", "id", "run")

unet.run3 = unet.jac[,5:6]
unet.run3$run = 3 
colnames(unet.run3) = c("value", "id", "run")

unet.run4 = unet.jac[,7:8]
unet.run4$run = 4 
colnames(unet.run4) = c("value", "id", "run")

unet.run5 = unet.jac[,9:10]
unet.run5$run = 5 
colnames(unet.run5) = c("value", "id", "run")

unet.full.jac = rbind(unet.run1, unet.run2, unet.run3, unet.run4, unet.run5)

#removing png extension
unet.full.jac$id = as.numeric(gsub(unet.full.jac$id, pattern = ".png", replacement = ""))
unet.full.jac$algo = "UNET"
unet.full.jac$measure = "IoU"

# join them (results with both measures)
unet.full = rbind(unet.full.fsc, unet.full.jac)

# ----------------------------------------------------------
# Violin + Boxplot
# ----------------------------------------------------------

cat(" @Plot: boxplot + violin\n")

df.complete = rbind(rf.full, unet.full)
g = ggplot(df.complete, aes(x = algo, y = value, group = algo))
g = g + geom_violin() + geom_boxplot(width=0.1)
g = g + labs(x = "Algorithm", y = "Value") + facet_grid(~measure)
ggsave(g, filename = "violinPlot.pdf", width = 5.96, height = 2.61)

# ----------------------------------------------------------
# Wilcoxon test - paired test
# ----------------------------------------------------------


#wilcoxon + fscore

# ordering by image id
sub.rf.fsc   = dplyr::filter(rf.full, measure == "FScore" & algo == "RF")
sub.unet.fsc = dplyr::filter(unet.full, measure == "FScore" & algo == "UNET")

sub.rf.fsc   = sub.rf.fsc[order(as.numeric(sub.rf.fsc$id)),]
sub.unet.fsc = sub.unet.fsc[order(as.numeric(sub.unet.fsc$id)),]

cat("-----------------------------------\n")
cat(" @ Performances (Fscore): \n")
cat(" - RF mean:", mean(sub.rf.fsc$value), "\n")
cat(" - RF sd:", sd(sub.rf.fsc$value), "\n")
cat(" - RF median" , median(sub.rf.fsc$value), "\n")
cat(" - UNET mean:", mean(sub.unet.fsc$value), "\n")
cat(" - UNET sd:", sd(sub.unet.fsc$value), "\n")
cat(" - UNET median:", median(sub.unet.fsc$value), "\n")
cat("-----------------------------------\n")


obj = wilcox.test(x = sub.rf.fsc$value, y = sub.unet.fsc$value, 
	paired = TRUE, conf.level = 0.95)

print(obj$p.value)
cat("@ Wilcoxon (FScore): ")
if(obj$p.value < 0.05) {
	cat("There is statistical difference between the methods!\n")
} else {
	cat("There is no statistical difference between the methods!\n")
}

# -----------------
#wilcoxon + jaccard (IoU)
# -----------------

sub.rf.jac   = dplyr::filter(rf.full, measure == "IoU" & algo == "RF")
sub.unet.jac = dplyr::filter(unet.full, measure == "IoU" & algo == "UNET")

sub.rf.jac   = sub.rf.jac[order(as.numeric(sub.rf.jac$id)),]
sub.unet.jac = sub.unet.jac[order(as.numeric(sub.unet.jac$id)),]

cat("-----------------------------------\n")
cat(" @ Performances (IoU): \n")
cat(" - RF mean:", mean(sub.rf.jac$value), "\n")
cat(" - RF sd:", sd(sub.rf.jac$value), "\n")
cat(" - RF median" , median(sub.rf.jac$value), "\n")
cat(" - UNET mean:", mean(sub.unet.jac$value), "\n")
cat(" - UNET sd:", sd(sub.unet.jac$value), "\n")
cat(" - UNET median:", median(sub.unet.jac$value), "\n")
cat("-----------------------------------\n")

obj.jac = wilcox.test(x = sub.rf.jac$value, y = sub.unet.jac$value, 
	paired = TRUE, conf.level = 0.95)

print(obj.jac$p.value)
cat("@ Wilcoxon (IoU): ")
if(obj.jac$p.value < 0.05) {
	cat("There is statistical difference between the methods!\n")
} else {
	cat("There is no statistical difference between the methods!\n")
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
	sel.fsc = rf.full.fsc[which(rf.full.fsc$id == id),]
	sel.jac = rf.full.jac[which(rf.full.jac$id == id),]
	return(c(mean(sel.fsc$value), mean(sel.jac$value)))
})

do.call("rbind", aux.rf)

rf.avg = data.frame(cbind(rf.ids, do.call("rbind", aux.rf)))
colnames(rf.avg) = c("id", "FScore", "IoU")
rf.avg = rf.avg[order(rf.avg$FScore),]

# # ------------
# # UNET
# # ------------

unet.ids = unique(unet.full$id)
aux.unet = lapply(unet.ids, function(id) {
	sel.fsc = unet.full.fsc[which(unet.full.fsc$id == id),]
	sel.jac = unet.full.jac[which(unet.full.jac$id == id),]
	return(c(mean(sel.fsc$value), mean(sel.jac$value)))
})

unet.avg = data.frame(cbind(unet.ids, do.call("rbind", aux.unet)))
colnames(unet.avg) = c("id", "FScore", "IoU")
unet.avg = unet.avg[order(unet.avg$FScore),]

# ------------
# ------------

scoresByImage = cbind(rf.avg, unet.avg)
colnames(scoresByImage) = c("RF.id", "RF.FScore", "RF.IoU", "UNet.id", "UNet.FScore", "UNet.Iou")
write.csv(scoresByImage, "./scoresByImage.csv")

# 12 images with fscore < 0.5
print(scoresByImage[1:12,])

# ------------
# heatmap
# ------------

cat(" @Plot: heatmap\n")
rf.avg$algo   = "RF"
unet.avg$algo = "UNET"

rf.avg$id   = as.factor(rf.avg$id)
unet.avg$id = as.factor(unet.avg$id)

df.avg = rbind(rf.avg, unet.avg)
df.hm  = melt(df.avg, id.vars = c(1, 4))

g2 = ggplot(df.hm, aes(x = (id), y = algo, fill = value))
g2 = g2 + geom_tile() + theme_bw() + facet_grid(variable~.)
g2 = g2 + scale_fill_gradient2(low = "red", high = "blue", mid = "white", 
	midpoint = 0.5, na.value = "grey50")
g2 = g2 +theme(axis.text.x=element_text(size=7))
g2 = g2 + theme(axis.text.x=element_text(angle = 90, hjust = 1))
g2 = g2 + labs(x = "Image ID", y = "Algorithm")
ggsave(g2,filename = "predictionsPlot.pdf", width = 6.78, height = 2.44)


cat("Finished :) \n")
# ----------------------------------------------------------
# ----------------------------------------------------------