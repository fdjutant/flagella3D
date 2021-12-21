import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd
import numpy as np
import math

sns.set(style="white")

df = pd.read_csv('/Users/Jakob/Documents/python_notebooks/data/tips.csv')

#calculate standard error of the mean

std = df['total_bill'].std()
mean = df['total_bill'].mean()
count = df['total_bill'].count()
sem = std/math.sqrt(count)


#define sd and sem
mean = tips.groupby('day').total_bill.mean()
sem = tips.groupby('day').total_bill.std() / np.sqrt(tips.groupby('day').total_bill.count())
plt.errorbar(range(len(mean)), mean, yerr=sem, capsize=5, color='black', alpha=1,
             linewidth=2, linestyle='', marker='o')

#sns.barplot(x="day", y="total_bill", data=tips, capsize=0.1, ci="sd",
            #errwidth=1, linewidth=5, palette = 'Blues', alpha=0.3)
sns.swarmplot(x="day", y="total_bill", data=tips, color="black", alpha=1, palette='rainbow', zorder=1)
#sns.pointplot(x='day', y='total_bill', data=tips, #ci=95, linestyles='None',
              #color="grey", capsize=0.1, errwidth=1.5, opacity=0.1, estimator=np.mean)


sns.despine(left=True, bottom=True)
rcParams['figure.figsize'] = 10,8
plt.show()
print(sem)