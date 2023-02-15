install.packages('ggplot2')
library(ggplot2)

df_Unet = read.csv('boxplots/Unet_dice_scores.csv')

df_names = df_Unet[c(2,4,6,8,10)]
df_boxplot = df_Unet[,-c(2,4,6,8,10)]

df_boxplot = stack(df_boxplot, select=(1:5))
names(df_boxplot) = c("dice_scores", "runs")

boxplot = ggplot(df_boxplot, aes(x=runs, y=dice_scores, fill=runs)) + geom_boxplot()
ggsave("Unet_boxplot.png", boxplot, path="/boxplots")