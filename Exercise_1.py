# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.cm as cm
from sklearn import linear_model
import os

#--------------------------------------------#
#////////////////////////////////////////////#
#//////////////////Journal01/////////////////#
#////////////////////////////////////////////#
#--------------------------------------------#

#////////////////////////////////////////////#
#/////////////////Lesson01Qa/////////////////#
#////////////////////////////////////////////#

print("#########################\nEXERCISE LESSON01 Qa\n#########################")
for i in range(2):
    print()

datapath = os.path.join("datasets", "lifesat", "")

oecd_bli = pd.read_csv(datapath + "oecd_bli_2015.csv", thousands=',')
oecd_bli = oecd_bli[oecd_bli["INEQUALITY"]=="TOT"]
oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")
#oecd_bli.head(2)

gdp_per_capita = pd.read_csv(datapath+"gdp_per_capita.csv", thousands=',', delimiter='\t',
                             encoding='latin1', na_values="n/a")
gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)
gdp_per_capita.set_index("Country", inplace=True)
#gdp_per_capita.head(2)

full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita, left_index=True, right_index=True)
full_country_stats.sort_values(by="GDP per capita", inplace=True)
#full_country_stats

remove_indices = [0, 1, 6, 8, 33, 34, 35]
keep_indices = list(set(range(36)) - set(remove_indices))

sample_data = full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[keep_indices]
#missing_data = full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[remove_indices]

sample_data.plot(kind='scatter', x="GDP per capita", y='Life satisfaction', figsize=(5,3))
plt.axis([0, 60000, 0, 10])
position_text = {
    "Hungary": (5000, 1),
    "Korea": (18000, 1.7),
    "France": (29000, 2.4),
    "Australia": (40000, 3.0),
    "United States": (52000, 3.8),
}
for country, pos_text in position_text.items():
    pos_data_x, pos_data_y = sample_data.loc[country]
    country = "U.S." if country == "United States" else country
    plt.annotate(country, xy=(pos_data_x, pos_data_y), xytext=pos_text,
            arrowprops=dict(facecolor='black', width=0.5, shrink=0.1, headwidth=5))
    plt.plot(pos_data_x, pos_data_y, "ro")
#save_fig('money_happy_scatterplot')
plt.show()

lin1 = linear_model.LinearRegression()
Xsample = np.c_[sample_data["GDP per capita"]]
ysample = np.c_[sample_data["Life satisfaction"]]
lin1.fit(Xsample, ysample)

t0 = 4.8530528
t1 = 4.91154459e-05

sample_data.plot(kind='scatter', x="GDP per capita", y='Life satisfaction', figsize=(5,3))
plt.axis([0, 60000, 0, 10])
M=np.linspace(0, 60000, 1000)
plt.plot(M, t0 + t1*M, "b")
plt.text(5000, 3.1, r"$\theta_0 = 4.85$", fontsize=14, color="b")
plt.text(5000, 2.2, r"$\theta_1 = 4.91 \times 10^{-5}$", fontsize=14, color="b")
#save_fig('best_fit_model_plot')
plt.show()

theta0 = lin1.intercept_
theta1 = lin1.coef_

print(theta0)
print(theta1)

R2 = lin1.score(Xsample, ysample)
print(R2)

#////////////////////////////////////////////#
#/////////////////Lesson02Qa/////////////////#
#////////////////////////////////////////////#

for i in range(50):
    if i == 48:
        print("#########################\nEXERCISE LESSON02 Qa\n#########################")
    print()

X = np.array([[1,2,3], [4,2,1], [3,8,5], [-9,-1,0]])
print(X)

#////////////////////////////////////////////#
#/////////////////Lesson02Qb/////////////////#
#////////////////////////////////////////////#

for i in range(4):
    if i == 2:
        print("#########################\nEXERCISE LESSON02 Qb\n#########################")
    print()

def L1(matrix):
    temp = 0
    temp2 = 0
    for i in range(len(matrix)):
        for x in range(len(matrix[0])):
            temp2 = matrix[i][x]
            if temp2 < 0:
                temp += temp2*-1
            else:
                temp += temp2
    return temp

def L2(matrix):
    temp = 0
    for i in range(len(matrix)):
        for x in range(len(matrix[0])):
            temp += (matrix[i][x])*(matrix[i][x])
    return np.sqrt(temp)

def L2Dot(vector1, vector2):
    temp = np.dot(vector1[0], vector2[0])
    return temp
        

tx=np.array([[1, 2, 3, -1]])
ty=np.array([[3,-1, 4,  1]])

expected_d1=8.0
expected_d2=4.242640687119285

d1=L1(tx-ty)
d2=L2(tx-ty)

print("L1 =", d1)
print("L2 =", d2)

print("tx-ty=",tx-ty,", d1-expected_d1=",d1-expected_d1,", d2-expected_d1=",d2-expected_d2)

for i in range(2):
    print()
    
print("Test with linalg.norm =", np.linalg.norm(tx-ty))

for i in range(2):
    print()
    
print(np.dot(tx[0],ty[0]))

print(L2Dot(tx,ty))