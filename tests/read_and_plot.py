from contextlib import contextmanager
from matplotlib.cbook import flatten
import numpy as np
from numpy.core.fromnumeric import shape
import pytest
import socialforce
import pandas as pd
from matplotlib import pyplot as plt

layouts = ["benchmark", "single", "pillars", "horizontal", "angled"]
peop_num = np.linspace(20,120,20)
peop_num = [int(x) for x in peop_num]
sim_num = 50
columns = ["layout", "peop_num", "run", "generated", "succes", "time",
           "avrg_time", "std_time", "velocity", "avrg_velocity", "std_velocity"]

# data = []

# for layout in layouts:
#     data.append(pd.read_csv('data/df_{}'.format(layout)))

# data2 = pd.concat(data)
# data2.to_csv('data/complete_data')

#for layout in enumerate(layouts) : 
#    for peop in peop_num :
#        for run in range(sim_num) :
#            print(df.loc[df["layout"] == layout].loc[df['peop_num'] == peop].loc[df["run"]].loc[df['std_velocity']>4])

df = pd.read_csv('data/complete_data')
velocities = df['velocity'].values

# print(df['velocity'])

# print('old')
# print(velocities[0][0:20], velocities[-1][0:20])
for x, velocity in enumerate(velocities) :
    velocities[x] = velocity[1:-1].split(',')
    velocities[x] = [1/float(y) for y in velocities[x]]

# print('new')
# print(velocities[0][0:3], velocities[-1][0:3])

avrg = [np.mean(vels) for vels in velocities]
stds = [np.std(vels) for vels in velocities]

# print(df['velocity'])
df = df.rename(columns={'avrg_velocity': 'avrg_velocity_old', 'std_velocity': 'std_velocity_old'})
df['avrg_velocity'] = avrg
df['std_velocity'] = stds
# print(df['avrg_velocity_old'], df['avrg_velocity'], df['std_velocity_old'], df['std_velocity'])

del df['avrg_velocity_old']
del df['std_velocity_old']

df.to_csv('data/complete_fixed')
# df.loc[df['avrg_velocity']>10, 'avrg_velocity'] = np.mean(df.loc[df['avrg_velocity']<=10])
# # df = df.replace(df.loc[df['avrg_velocity']>10], np.mean(df.loc[df['avrg_velocity']<=10]))
# print(df.loc[df['avrg_velocity']>10])

# print(data)


# population = 90
# avr_times = []
# std_times = []
# avr_velocities = []
# std_velocities = []
# for ind, layout in enumerate(layouts):
#     df = data[ind]
#     # print(layout,df)
#     avr_times.append([])
#     std_times.append([])
#     avr_velocities.append([])
#     std_velocities.append([])
#     for pop_size in peop_num:
#         avr_times[ind].append(
#             df.loc[df['peop_num'] == pop_size]['avrg_time'].mean())
#         std_times[ind].append(
#             df.loc[df['peop_num'] == pop_size]['std_time'].mean())
#         avr_velocities[ind].append(
#             df.loc[df['peop_num'] == pop_size]['avrg_velocity'].mean())
#         std_velocities[ind].append(
#             df.loc[df['peop_num'] == pop_size]['std_velocity'].mean())

# print(np.shape(avr_times), np.shape(std_times), np.shape(avr_velocities), np.shape(std_velocities))

# for ind, layout in enumerate(layouts):
#     plt.clf()
#     line = plt.plot(peop_num, avr_times[ind], label="Lin")
#     plt.fill_between(peop_num, avr_times[ind], [x+2*y for x,y in zip(avr_times[ind], std_times[ind])], alpha=0.4, facecolor=line[0].get_color())
#     plt.fill_between(peop_num, avr_times[ind], [x-2*y for x,y in zip(avr_times[ind], std_times[ind])], alpha=0.4, facecolor=line[0].get_color())

#     plt.title("Average passing time - {} layout".format(layout))
#     plt.xlabel("Population size")
#     plt.ylabel("Passing time")
#     plt.legend(loc="upper right")
#     plt.grid()
#     plt.savefig("plots/times_{}png".format(layout), dpi=300, bbox_inches='tight')  #'horizontal', run=39, 112 people

#     plt.clf()
#     line = plt.plot(peop_num, avr_velocities[ind], label="Lin")
#     plt.fill_between(peop_num, avr_velocities[ind], [x+2*y for x,y in zip(avr_velocities[ind], std_velocities[ind])], alpha=0.4, facecolor=line[0].get_color())
#     plt.fill_between(peop_num, avr_velocities[ind], [x-2*y for x,y in zip(avr_velocities[ind], std_velocities[ind])], alpha=0.4, facecolor=line[0].get_color())

#     plt.title("Average velocity - {} layout".format(layout))
#     plt.xlabel("Population size")
#     plt.ylabel("Passing time")
#     plt.legend(loc="upper right")
#     plt.grid()
#     plt.savefig("plots/velocities_{}png".format(layout), dpi=300, bbox_inches='tight')
#     # plt.show()
# #print(avr_times, std_times, avr_velocities, std_velocities)

#     # times=df.loc[df['run']==0].loc[df['peop_num']==population]
#     # print(times)
#     # times=times['time'].values
#     # print(times)
#     #plt.hist(times, bins=20)
#     #plt.savefig('times_hist_{}_{}.png'.format(layout, population), dpi=300, bbox_inches='tight')
#     # plt.show()
