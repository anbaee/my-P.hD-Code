# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 07:19:02 2020

@author: Novin
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#accuracy	precision	recall

data = pd.read_excel('variable time frame shuffeled.xlsx')
y1 = data['f1_micro']
y2 = data['recall']
y3 = data['precision']
y4 = data['accuracy']
y5 = data['f1_macro']

x = data['time frame']

csfont = {'fontname':'Comic Sans MS'}
hfont = {'fontname':'Helvetica'}

fig, ax = plt.subplots()

ax.plot(x, y1 , color='black', linewidth=1.0, linestyle=':'  , label = 'F1 Micro')  
ax.plot(x, y2, label = 'Recall' , color='red', linewidth=1.0, linestyle='--'  )
ax.plot(x, y3, label = 'Precision', color='green', linewidth=1.0, linestyle='-.'  )
ax.plot(x, y4, label = 'Accuracy', color='yellow', linewidth=1.0, linestyle='-'  )
ax.plot(x, y5, label = 'F1 Macro', color='blue', linewidth=1.0  )

legend = ax.legend(loc='upper right',shadow=True, fontsize='small')

# Put a nicer background color on the legend.
legend.get_frame().set_facecolor('C0')


plt.xlabel("∆t in minutes",**csfont)
plt.ylabel("Scores",**csfont)


plt.show()
