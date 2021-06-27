from contextlib import contextmanager
from matplotlib.cbook import flatten
import numpy as np
from numpy import random
import pytest
import socialforce
import pandas as pd
from math import floor, ceil


@contextmanager
def visualize(states, space, output_filename):
    import matplotlib.pyplot as plt

    print('')
    with socialforce.show.animation(
            len(states),
            output_filename,
            writer='imagemagick') as context:
        ax = context['ax']
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')

        yield ax

        for s in space:
            ax.plot(s[:, 0], s[:, 1], '-o', color='black', markersize=2.5)

        actors = []
        for ped in range(states.shape[1]):
            speed = np.linalg.norm(states[0, ped, 2:4])
            radius = 0.2 + speed / 2.0 * 0.3
            p = plt.Circle(states[0, ped, 0:2], radius=radius,
                           facecolor='black' if states[0, ped, 4] > 0 else 'white',
                           edgecolor='black')
            actors.append(p)
            ax.add_patch(p)

        def update(i):
            for ped, p in enumerate(actors):
                # p.set_data(states[i:i+5, ped, 0], states[i:i+5, ped, 1])
                p.center = states[i, ped, 0:2]
                speed = np.linalg.norm(states[i, ped, 2:4])
                p.set_radius(0.2 + speed / 2.0 * 0.3)

        context['update_function'] = update


@pytest.mark.parametrize('n', [4])
@pytest.mark.parametrize('half_len', [6])
@pytest.mark.parametrize('half_width', [30])
@pytest.mark.parametrize('mode', ["single"])
def test_walkway_benchmark(n, half_len, half_width, mode, run=-1, visual=True):

   # pos_left = ((np.random.random((n, 2)) - 0.5) * 2.0) * np.array([half_width, half_len])
   # pos_right = ((np.random.random((n, 2)) - 0.5) * 2.0) * np.array([half_width, half_len])

    n_20 = ceil(n*0.2)
    n_80 = floor(n*0.8)
    
    pos_left_20 = np.transpose(np.array([np.random.uniform(-half_width, half_width, n_20),np.random.uniform(-half_len, 0, n_20)]))
    pos_left_80 = np.transpose(np.array([np.random.uniform(-half_width, half_width, n_80),np.random.uniform(0, half_len, n_80)]))
    pos_left = np.concatenate((pos_left_80, pos_left_20))
    
    pos_right_20 = np.transpose(np.array([np.random.uniform(-half_width, half_width, n_20),np.random.uniform(0, half_len, n_20)]))
    pos_right_80 = np.transpose(np.array([np.random.uniform(-half_width, half_width, n_80),np.random.uniform(-half_len, 0, n_80)]))
    pos_right = np.concatenate((pos_right_80, pos_right_20))

    generated = 2*n

    x_vel_left = np.random.normal(1.34, 0.26, size=(n, 1))
    x_vel_right = np.random.normal(-1.34, 0.26, size=(n, 1))
    x_destination_left = 100.0 * np.ones((n, 1))
    x_destination_right = -100.0 * np.ones((n, 1))

    zeros = np.zeros((n, 1))
    minus_one = -1 * np.ones((n, 1))

    state_left = np.concatenate(
        (pos_left, x_vel_left, zeros, x_destination_left, zeros), axis=-1)
    state_right = np.concatenate(
        (pos_right, x_vel_right, zeros, x_destination_right, zeros), axis=-1)
    initial_state = np.concatenate((state_left, state_right))

    # save the initial position and the time the agents enter and exit the system
    initial_state_left = np.concatenate((pos_left, zeros, minus_one), axis=-1)
    initial_state_right = np.concatenate((pos_right, zeros, minus_one), axis=-1)
    new_initial = np.concatenate((initial_state_left, initial_state_right))

    if (mode=="benchmark"):
        space = [
            np.array([(x, half_len) for x in np.linspace(-half_width, half_width, num=5000)]),
            np.array([(x, -half_len) for x in np.linspace(-half_width, half_width, num=5000)])
        ]
    elif (mode=="single"):
        space = [
            np.array([(x, half_len) for x in np.linspace(-half_width, half_width, num=5000)]),
            np.array([(x, -half_len) for x in np.linspace(-half_width, half_width, num=5000)]),
            np.array([(i, i) for i in np.linspace(1, -1.0)])
        ]
    elif (mode=="pillars"):
        a_values = np.arange(-half_width, half_width, 3)[1:]
        b = 0
        radius = 0.2
        stepSize = 0.1
        t = 0

        space = [
            np.array([(x, half_len) for x in np.linspace(-half_width, half_width, num=500)]),
            np.array([(x, -half_len) for x in np.linspace(-half_width, half_width, num=500)])
        ]

        # create pillars
        for a in a_values:
            positions = []
            t = 0
            while t < 2 * np.pi:
                for r in np.linspace(0, radius, 500):
                    coord = (r * np.cos(t) + a, r * np.sin(t) + b)
                    positions.append(coord)
                    t += stepSize
            space.append(np.array(positions))
    elif (mode=="horizontal"):
        space = [
            np.array([(x, half_len) for x in np.linspace(-half_width, half_width, num=500)]),
            np.array([(x, -half_len) for x in np.linspace(-half_width, half_width, num=500)]),
            np.array([(x, 0) for x in np.linspace(-half_width/4, half_width/4, num=500)])
        ]
    elif (mode=="angled"):
        space = [
            np.array([(x, half_len) for x in np.linspace(-half_width, half_width, num=5000)]),
            np.array([(x, -half_len) for x in np.linspace(-half_width, half_width, num=5000)]),
            np.array([(i, i+half_width/2) for i in np.linspace(-half_width/2-1, -half_width/2+1)]),
            np.array([(i, i-half_width/2) for i in np.linspace(half_width/2-1, half_width/2+1)])
        ]
    elif (mode=="tighter"):
        space = [
            np.array([(x, half_len) for x in np.linspace(-half_width, half_width, num=5000)]),
            np.array([(x, -half_len) for x in np.linspace(-half_width, half_width, num=5000)]),
            np.array([(-half_width/4-2 + x, half_len - x) for x in np.linspace(0, 2, num=5000)]),
            np.array([(half_width/4 + x, half_len - 2 + x) for x in np.linspace(0, 2, num=5000)]),
            np.array([(x, half_len-2) for x in np.linspace(-half_width/4, half_width/4, num=5000)]),
            np.array([(x, -half_len+2) for x in np.linspace(-half_width/4, half_width/4, num=5000)]),
            np.array([(-half_width/4-2 + x, -half_len + x) for x in np.linspace(0, 2, num=5000)]),
            np.array([(half_width/4 + x, -half_len + 2 - x) for x in np.linspace(0, 2, num=5000)])
        ]
    elif (mode=="two_lines"):
        space = [
            np.array([(x, half_len) for x in np.linspace(-half_width, half_width, num=5000)]),
            np.array([(x, -half_len) for x in np.linspace(-half_width, half_width, num=5000)]),
            np.array([(x, -half_len*2/3) for x in np.linspace(-half_width, half_width, num=5000)]),
            np.array([(x, half_len*2/3) for x in np.linspace(-half_width, half_width, num=5000)])
        ]
    s = socialforce.Simulator(initial_state, socialforce.PedSpacePotential(space))
    states = []
    times=[]
    paths=[]
    total=0
    out_l=0
    out_r=0
    for i in range(500):
        generated += out_l + out_r
        state = s.step().state
        # periodic boundary conditions

        indicess = [ind for ind, x in enumerate(state) if (state[ind, 0] > half_width)]
    #    print('new')
    #    print(indicess)

        new_initial[indicess, 3] = [i]*(len(indicess))
        out_l = len(indicess)
        p = random.random()
        if (p >= 0.5):
            pos_left_20 = np.transpose(np.array([np.random.uniform(-half_width, -half_width+1, floor(out_l*0.2)),np.random.uniform(-half_len, 0, floor(out_l*0.2))]))
            pos_left_80 = np.transpose(np.array([np.random.uniform(-half_width, -half_width+1, ceil(out_l*0.8)),np.random.uniform(0, half_len, ceil(out_l*0.8))]))
        else:
            pos_left_20 = np.transpose(np.array([np.random.uniform(-half_width, -half_width+1, ceil(out_l*0.2)),np.random.uniform(-half_len, 0, ceil(out_l*0.2))]))
            pos_left_80 = np.transpose(np.array([np.random.uniform(-half_width, -half_width+1, floor(out_l*0.8)),np.random.uniform(0, half_len, floor(out_l*0.8))]))
        pos_left = np.concatenate((pos_left_80, pos_left_20))
        x_vel_left = np.random.normal(1.34, 0.26, size=(out_l, 1))
        state[indicess, 0:3] = np.concatenate((pos_left, x_vel_left), axis=-1)
        new_times = i - new_initial[indicess,2]
        new_destinations = half_width - new_initial[indicess, 0]

        new_times = list(new_times)
        new_destinations = list(new_destinations)
        if len(new_times)>0 :
            times.append(new_times)
            paths.append(new_destinations)


        # enter in a new position at i time
        new_initial[indicess, 3] = [-1]*len(indicess)
        new_initial[indicess, 2] = [i]*len(indicess)
        for j in range(out_l):
            new_initial[indicess, 0:1] = pos_left[j,0:1]


        indicess = [ind for ind, x in enumerate(state) if (state[ind, 0] < -half_width)]

        new_initial[indicess, 3] = i
        out_r = len(indicess)
        p = random.random()
        if (p >= 0.5):
            pos_right_20 = np.transpose(np.array([np.random.uniform(half_width, half_width-1, floor(out_r*0.2)),np.random.uniform(0, half_len, floor(out_r*0.2))]))
            pos_right_80 = np.transpose(np.array([np.random.uniform(half_width, half_width-1, ceil(out_r*0.8)),np.random.uniform(-half_len, 0, ceil(out_r*0.8))]))
        else: 
            pos_right_20 = np.transpose(np.array([np.random.uniform(half_width, half_width-1, ceil(out_r*0.2)),np.random.uniform(0, half_len, ceil(out_r*0.2))]))
            pos_right_80 = np.transpose(np.array([np.random.uniform(half_width, half_width-1, floor(out_r*0.8)),np.random.uniform(-half_len, 0, floor(out_r*0.8))]))
        pos_right = np.concatenate((pos_right_80, pos_right_20))

        x_vel_right = np.random.normal(1.34, 0.26, size=(out_r, 1))
        state[indicess, 0:3] = np.concatenate((pos_right, x_vel_right), axis=-1)

        new_times = i - new_initial[indicess, 2]
        new_destinations =  new_initial[indicess, 0]  + half_width

        new_times = list(new_times)
        new_destinations = list(new_destinations)
        if len(new_times)>0 :
            times.append(new_times)
            paths.append(new_destinations)


        new_initial[indicess, 3] = [-1]*len(indicess)
        new_initial[indicess, 2] = [i]*len(indicess)


        for j in range(out_r):
            new_initial[(indicess)[j], 0:1] = pos_right[j,0:1]

        k = out_l + out_r
        total+=k

        states.append(state.copy())
    states = np.stack(states)

    times = list(flatten(times))
    nonzero = [ind for ind, x in enumerate(times) if (x > 0)]
    times = [times[ind] for ind in nonzero]
    paths = list(flatten(paths))
    paths = [paths[ind] for ind in nonzero]

    velocities = [i / j for i, j in zip(paths, times)]

#    print(total, times, np.mean(times), velocities, np.mean(velocities)) # np.mean(times),, np.mean(velocities)

    # skip creating the visual when running the simulations.
    if (visual == True ) :
        with visualize(states, space, 'docs/walkway_{}_{}.gif'.format(mode, n)) as _:
            pass

    return mode, 2*n, run, generated, total, times, np.mean(times), np.std(times), velocities, np.mean(velocities), np.std(velocities)


# for each layout
 # for different number of people 20 to 120, maybe with steps
  # run 50 simulations
   # save the data
    # make plots
""""
layouts = ["horizontal"]
peop_num = [20, 40, 60, 80, 100]
sim_num = 30

columns = ["layout", "peop_num", "run", "generated", "succes", "time", "avrg_time", "std_time", "velocity", "avrg_velocity", "std_velocity"]

dfs_layout = []
for layout in layouts :
    dfs_num = []
    for numb in peop_num :
        print('people_num: ', numb)
        dfs_runs = []
        for run in range(sim_num) :
            print('sim_num:', run)
            res = test_walkway_benchmark(int(numb), 6, 30, layout, run, False)
            res_df = pd.DataFrame(data=[res], columns=columns)
            dfs_runs.append(res_df)
        new_df=pd.concat(dfs_runs)
        dfs_num.append(new_df)
    layout_df = pd.concat(dfs_num)
    layout_df.to_csv('data_preference_avoidance/df_{}'.format(layout))
    # layout_df.to_csv('data/df_{}_test'.format(layout))
    print('{} DONE'.format(layout))
    dfs_layout.append(layout_df)
final_df = pd.concat(dfs_layout)
""""