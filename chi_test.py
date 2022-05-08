import numpy as np
from sklearn import datasets
from sklearn.datasets import load_breast_cancer
from scipy.stats import chi2 as chis
from sklearn.feature_selection import SelectKBest, chi2
import matplotlib.pyplot as plt


#setting
dataset_filename = 'titanic/train.csv'

#create chi squared table
p = np.array([0.995, 0.99, 0.975, 0.95, 0.90, 0.10, 0.05, 0.025, 0.01, 0.005])
df = np.arange(1,101).reshape(-1, 1)
table = chis.isf(p, df)
#print(table, np.shape(table))
print(table[0,6])

dataset_open = open(dataset_filename)
dataset_str = dataset_open.read()
dataset_array = dataset_str.split('\n')
if dataset_array[-1] == '':
    dataset_array = dataset_array[:-1]

gender_list = []
survivor_stat = []
for i in range(len(dataset_array)):
    dataset_array[i] = dataset_array[i].split(',')
    if i == 0:
        continue
    gender_list.append(dataset_array[i][5])
    survivor_stat.append(dataset_array[i][1])

print(gender_list)

total_male = 0
total_female = 0
total_survive = 0
total_died = 0
total_male_survive = 0
total_female_survive = 0
total_male_died = 0
total_female_died = 0
total_data = 0
for i in range(len(gender_list)):
    if gender_list[i].lower() == 'male' and survivor_stat[i] == '1':
        total_male += 1
        total_male_survive += 1
        total_survive += 1
    if gender_list[i].lower() == 'male' and survivor_stat[i] == '0':
        total_male += 1
        total_male_died += 1
        total_died += 1
    elif gender_list[i].lower() == 'female' and survivor_stat[i] == '1':
        total_female += 1
        total_female_survive += 1
        total_survive += 1
    elif gender_list[i].lower() == 'female' and survivor_stat[i] == '0':
        total_female += 1
        total_female_died += 1
        total_died += 1
    total_data += 1

print(total_male, total_female, len(gender_list))
print(total_female_died, total_female_survive, total_male_died, total_male_survive)

#get expected value
E_male_survive = round((total_male / total_data) * (total_survive / total_data) * total_data)
E_female_survive = round((total_female / total_data) * (total_survive / total_data) * total_data)
E_male_died = round((total_male / total_data) * (total_died / total_data) * total_data)
E_female_died = round((total_female / total_data) * (total_died / total_data) * total_data)

print(E_male_survive, total_male_survive)
print(E_female_survive, total_female_survive)
print(E_male_died, total_male_died)
print(E_female_died, total_female_died)

#get chi squared
chi_male_survive = ((total_male_survive - E_male_survive)**2) / E_male_survive
chi_male_died = ((total_male_died - E_male_died)**2) / E_male_died
chi_female_survive = ((total_female_survive - E_female_survive)**2) / E_female_survive
chi_female_died = ((total_female_died - E_female_died)**2) / E_female_died

print(chi_male_survive, chi_male_died)
print(chi_female_survive, chi_female_died)

#total chi squared
total_chi = chi_male_survive + chi_male_died + chi_female_survive + chi_female_died

#compare with alpha = 0.05
#degree of freedom = (r-1) * (c-1), r = number of row, c = number of column
#degree of freedom = (2-1) * (2-1) = 1
convert_chi = table[0, 6]
print('compare chi of 0.05 and chi in this case', convert_chi, total_chi)

if total_chi > convert_chi:
    print('We can reject the null hypothesis and accept the alt hypothesis: Survival in Titanic depends on Gender')
else:
    print('We fail to reject the null hypothesis i.e., Survival in Titanic is independent of Gender')

#exit()

#plot
barWidth = 0.25
male = [total_male_survive, total_male_died]
female = [total_female_survive, total_female_died]

br1 = np.arange(len(male))
br2 = [x + barWidth for x in br1]

plt.bar(br1, male, color ='r', width = barWidth,
        edgecolor ='grey', label ='male')
plt.bar(br2, female, color ='g', width = barWidth,
        edgecolor ='grey', label ='female')
plt.xticks([r + (barWidth/2) for r in range(len(male))],
        ['survive', 'died'])

plt.legend()
plt.show()