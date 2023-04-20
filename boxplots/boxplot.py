import pandas as pd

import matplotlib.pyplot as plt

df_RF = pd.read_csv('boxplots/RandomForest_f1_scores.csv')
df_Unet = pd.read_csv('boxplots/Unet_f1_scores.csv')

df_RF.drop(df_RF.columns[[1, 3, 5, 7, 9]], axis=1, inplace=True)
df_Unet.drop(df_Unet.columns[[1, 3, 5, 7, 9]], axis=1, inplace=True)

df_RF = pd.Series(df_RF.stack().values)
df_Unet = pd.Series(df_Unet.stack().values)

df_boxplot = pd.concat([df_RF, df_Unet], axis=1) 
df_boxplot.columns = ['RF', 'Unet']

violin = plt.violinplot(df_boxplot, showextrema=False, showmeans=False)
boxplot = plt.boxplot(df_boxplot, labels=['RF', 'UNet'], patch_artist=True, medianprops={'color': 'black'}, boxprops={'facecolor':'white'})

for body in violin['bodies']:
    body.set_facecolor('cyan')
    body.set_alpha(0.9)

plt.title('Box plot and Violin plot')
plt.xlabel('Algorithms')
plt.ylabel('F1 Scores')

plt.grid()

plt.savefig('boxplots/boxplot_RF_UNet.pdf')
plt.close()