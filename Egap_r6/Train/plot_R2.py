import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import ticker
matplotlib.rcParams['axes.linewidth'] = 1.5
import seaborn as sns
from sklearn.metrics import r2_score



#Reading variable data----------------------------------------
y = np.loadtxt('y_train.dat', unpack=True)
pred_y = np.loadtxt('predicted_yval.dat', unpack=True)

#Calculating R**2 value---------------------------------------
corr_matrix = np.corrcoef(y, pred_y)
corr = corr_matrix[0,1]
R_sq = r2_score(y, pred_y)

#Find a line of best fit--------------------------------------
a, b = np.polyfit(y, pred_y, 1)

#Plot cost function-------------------------------------------
fig, ax = plt.subplots()
sns.regplot(x=y, y=pred_y, scatter_kws = {'color': 'g'}, line_kws = {'color': 'r', 'lw':2})
ax.tick_params(axis='both', which='major', labelsize=17)
ax.tick_params(axis='both', which='minor', labelsize=17)
ax.set_ylabel(r'$\mathrm{Predict}$', fontsize=20)
ax.set_xlabel(r'$\mathrm{Actual}$', fontsize=20)
ax.set_xlim(0.8,3.0)
ax.set_ylim(0.8,3.0)
ax.text(0.9, 2.80, r'$y~=~{:.2f}~+~{:.2f}x$'.format(b,a), fontsize = 17)
ax.text(0.83, 2.60, r'$R^{{2}}~=~{:.2f}$'.format(R_sq), fontsize = 17)
ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(1))
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(1))
plt.savefig('R2.png', dpi=720, bbox_inches = "tight")
plt.show()
