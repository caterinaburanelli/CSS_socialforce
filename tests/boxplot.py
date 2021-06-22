#%%
import pandas as pd
from matplotlib import pyplot as plt

bench = pd.read_csv("../data_validation/df_benchmark")
tot_h = pd.read_csv("../data_validation/df_tot_horizontal")
pil = pd.read_csv("../data_validation/df_pillars")

#%%
av_vel_bench = bench["avrg_velocity"]
av_vel_horiz = tot_h["avrg_velocity"]
av_vel_pil = pil["avrg_velocity"]

#%%
plt.boxplot([av_vel_bench, av_vel_horiz, av_vel_pil], 
labels=["Benchmark", "Total horizontal", "Pillars"])
plt.title("Average velocity with 70 agents")
plt.show()
#%%