import numpy as np
import random

from math import sin, cos, tau, hypot, atan2

import matplotlib
matplotlib.use('QT5Agg')
import matplotlib.pyplot as plt
import imageio

from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator


plt_figsize = (3, 3)


# equal probability population of agents
def uniform(agents):
    n = len(agents)
    return list(zip(agents, [1.0 / n] * n))


# disk game, limited to |a| <= 1
class Disk:
    name = 'disk'

    def random(self, n, mag=(0,1)):
        agents = []
        for i in range(n):
            theta = random.uniform(0, tau)
            r = random.uniform(mag[0], mag[1])
            agents.append((r * sin(theta), r * cos(theta)))
        return agents

    def zero(self):
        return (0, 0)

    def payoff(self, a, b):
        return a[0] * b[1] - a[1] * b[0]

    def support(self, mag=1, rand_mag=0):
        mags = np.clip(mag + rand_mag * np.random.randn(6), 0, 1)
        return [(m * sin(theta), m * cos(theta))
                for m, theta in zip(mags, np.arange(0, tau, tau/6))]

    def grad(self, a, b):
        return np.array([b[1], -b[0]])

    def apply_grad(self, a, g):
        x, y = a + g
        mag = max(hypot(x, y), 1)
        return x / mag, y / mag

    def figure(self):
        plt.figure(figsize=plt_figsize)
        plt.axis([-1.02, 1.02, -1.02, 1.02])
        plt.axis('off')
        ax = plt.gca()
        ax.add_artist(plt.Circle((0, 0), 1.0, color='lightgray', fill=False, zorder=-1))
        ax.set_aspect(1.0)

    def agent_coords(self, agents):
        coords = np.array(agents)
        return coords[:,0], coords[:,1]


tri_x = [ 0.866025, 0.000000, -0.866025]
tri_y = [-0.500000, 1.000000, -0.500000]
tri_points = np.array([tri_x, tri_y]).T

def make_triangle():
    return Line2D(xdata=(tri_x + tri_x[:1]), ydata=(tri_y + tri_y[:1]), color='lightgray', zorder=-1)

rps_payoff = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], dtype='float64')
rps_payoff_r = rps_payoff.clip(min=0)


class RPS:
    name = 'rps'

    def random(self, n, mag=(1,1)):
        agents = []
        while len(agents) < n:
            r = random.uniform(0, 1)
            p = random.uniform(0, 1)
            s = 1 - r - p
            if s <= 0: continue
            m = random.uniform(mag[0], mag[1])
            a = [z * m + 1/3 * (1 - m) for z in (r, p, s)]
            agents.append(a)
        return agents

    def zero(self):
        return (1/3, 1/3, 1/3)

    def support(self, mag=1, rand_mag=0):
        z1 = np.clip(mag + rand_mag * np.random.randn(3), 0, 1)
        z2 = (1 - z1) / 2
        return [
            (z1[0], z2[0], z2[0]),
            (z2[0], z1[0], z2[0]),
            (z2[0], z2[0], z1[0])
        ]

    def payoff(self, a, b):
        return a @ rps_payoff @ b

    def grad(self, a, b):
        return rps_payoff @ b

    def apply_grad(self, a, g):
        a = np.array(a, dtype='float64')
        a += g
        a = a.clip(0, 1)
        a = a / a.sum()
        return a[0], a[1], a[2]

    def figure(self):
        plt.figure(figsize=plt_figsize)
        ax = plt.gca()
        ax.add_line(make_triangle())
        ax.xaxis.set_major_locator(NullLocator())
        ax.yaxis.set_major_locator(NullLocator())
        for x, y, l in zip(tri_x, tri_y, 'RPS'):
            ax.text(x*0.9, y*0.92, l, horizontalalignment='center', verticalalignment='center')
        plt.axis([-0.9, 0.9, -0.6, 1.02])
        plt.axis('off')
        ax.set_aspect(1.0)

    def agent_coords(self, agents):
        coords = np.array(agents) @ tri_points
        return coords[:, 0], coords[:, 1]


# these are for finding the (a) nash equilibrium of the empirical meta-game

# fast: use fictitious play
fp_horizon = 5000
def get_nash_1(A):
    n = A.shape[0]
    counts = np.zeros(n)
    counts2 = np.zeros(n)
    for i in range(fp_horizon):
        utilities = A @ counts
        best_response, = np.nonzero(utilities == np.max(utilities))
        play = np.random.choice(best_response)
        counts[play] += 1
        if i > fp_horizon/10: counts2[play] += 1
    # drop all but the 16 highest prob agents:
    if n > 16:
        bottom = counts2.argsort()[:-16]
        counts2[bottom] = 0
    # drop agents that are less than 5% of the nash:
    # cutoff = min(counts2.max() / 20, fp_horizon / 20)
    # counts2[counts2 < cutoff] = 0
    return counts2 / counts2.sum()

# these other implementations are super slow and can fail in the case of
# degenerate games...
# comment out the import line if you wish to use them (you'll need to pip install nashpy)
# import nashpy

def get_nash_2(A):
    return nashpy.Game(A).lemke_howson_enumeration().__next__()[0]

def get_nash_3(A):
    return nashpy.Game(A).vertex_enumeration().__next__()[0]

def get_nash_4(A):
    return nashpy.Game(A).support_enumeration().__next__()[0]


class Population(object):
    def __init__(self, game, init_pop=[]):
        self.game = game
        self.agents = []
        self._eval_matrix = np.zeros([4 * len(init_pop)] * 2, dtype='float64')
        self.eval_matrix = None
        self.init_size = len(init_pop)
        self.last_size = 0
        self.last_agent = None
        for p in init_pop:
            self.append(p)

    def append(self, a):
        self.last_agent = a
        self.agents.append(a)
        n = len(self.agents) - 1
        if n == self._eval_matrix.shape[0]:
            old_matrix, self._eval_matrix = self._eval_matrix, np.zeros((2 * n, 2 * n))
            self._eval_matrix[:n, :n] = old_matrix
        for i, b in enumerate(self.agents):
            score = self.game.payoff(a, b) + random.uniform(0, 1e-10)
            self._eval_matrix[i, n] = score
            self._eval_matrix[n, i] = -score
        self.eval_matrix = self._eval_matrix[:n+1, :n+1]

    def nash_equilibrium(self):
        if len(self.agents) == 1:
            return zip(self.agents, [1.0])
        a_probs = get_nash_1(self.eval_matrix)
        return [(a, p) for a, p in zip(self.agents, a_probs) if p > 1e-3]

    def plot_agents(self):
        self.game.figure()
        x, y = self.game.agent_coords(self.agents)
        # c = (['orange'] * self.init_size) + [str(0.8 * (1 - (t / max_t))) for t in range(max_t)]
        c = ['orange'] * self.init_size
        c += ['gray'] * (self.last_size - len(c))
        c += ['green'] * (len(self.agents) - len(c))
        plt.scatter(x=x, y=y, c=c, s=15)
        # if max_t > 0: plt.text(x[-1], y[-1]-0.05, s=str(max_t), verticalalignment='top', horizontalalignment='center')

    def plot_nash(self, show_weights=False):
        ax = plt.gca()
        agents, probs = zip(*self.nash_equilibrium())
        x, y = self.game.agent_coords(agents)
        center = (x @ probs, y @ probs)
        # order the points to be clockwise around their center
        order = np.argsort(list(map(atan2, y - center[1], x - center[0])))
        x, y = x[order].tolist(), y[order].tolist()
        ax.add_artist(Line2D(xdata=x + x[:1], ydata=y + y[:1], color='purple', alpha=0.2))
        if show_weights:
            sizes = np.array(probs)[order] * 20 * len(probs)
            ax.scatter(x=x, y=y, s=sizes, color='purple', facecolors='none')
        ax.scatter(x=[center[0]], y=[center[1]], marker='x', color='purple')

    def oracle_step(self, agent, opponent_mixture, rectified=False):
        for i in range(10):
            grad = np.zeros(len(agent))
            prob_sum = 0
            for opponent, prob in list(opponent_mixture):
                if rectified and self.game.payoff(agent, opponent) < -1e-4: continue
                g = self.game.grad(agent, opponent)
                prob_sum += prob
                grad += prob * g
            if prob_sum == 0: break
            agent = self.game.apply_grad(agent, (0.1 / prob_sum) * grad)
        self.append(agent)

    # play against the initial population
    def fixed_play(self):
        self.oracle_step(self.last_agent, uniform(self.agents[:self.init_size]))

    # play against the most recent agent
    def self_play(self):
        self.oracle_step(self.last_agent, uniform([self.last_agent]))

    # play against the nash
    def psro(self):
        self.oracle_step(self.last_agent, self.nash_equilibrium())

    # play against all previous agents
    def psro_uniform(self):
        self.oracle_step(self.last_agent, uniform(self.agents))

    # play nash agents against each other, ignoring stronger players
    def psro_rectified(self):
        nash = self.nash_equilibrium()
        for agent, prob in nash:
            self.oracle_step(agent, nash, rectified=True)

    # run with a given step function
    def run(self, fn_name, num_steps, callback):
        step_fn = getattr(self, fn_name)
        callback(0)
        for t in range(num_steps):
            self.last_size = len(self.agents)
            if self.last_size < 1024:
                step_fn()
            callback(t+1)


# interactive animation of the trajectory
def animate(pop, fn_name, num_steps=5, plot_nash=False):

    def callback(t):
        pop.plot_agents()
        if plot_nash: pop.plot_nash()
        plt.show()

    pop.run(fn_name, num_steps, callback)


# produce a list of ndarrays of the trajectory
def animate_to_frames(pop, fn_name, num_steps, plot_nash=False):
    frames = []
    name = fn_name.replace('_', ' ')

    def callback(t):
        pop.plot_agents()
        if plot_nash: pop.plot_nash()
        fig = plt.gcf()
        fig.canvas.draw()
        plt.gca().set_title(f'{name} t={t}', fontsize=10)
        fig.tight_layout(pad=0.1, h_pad=0, w_pad=0)
        plt.savefig('temp.png')#, bbox_inches='tight')
        image = imageio.imread('temp.png')
        frames.append(image)

    pop.run(fn_name, num_steps, callback)
    return frames


# save the animation to a gif file
def animate_to_gif(file, *args, **kwargs):
    print(f'saving to {file}... ', end='')
    frames = animate_to_frames(*args, **kwargs)
    imageio.mimsave(file, frames, fps=3)
    print('done')


# animate to a single ndarray
def animate_to_np(*args, **kwargs):
    frames = animate_to_frames(*args, **kwargs)
    return np.stack(frames, axis=0)


# create a grid of animations for both games over all 4 algols
def animate_game_grid(file, init_pop_fn, seed=1, num_steps=12):
    rows = []
    print(f'writing to {file}...')
    for game in [Disk(), RPS()]:
        cols = []
        for method in ['fixed_play', 'psro_uniform', 'psro', 'psro_rectified']:
            random.seed(seed)
            np.random.seed(seed)
            print('\t', game.name, method)
            pop = Population(game, init_pop=init_pop_fn(game))
            plot_nash = 'psro' in method and 'uniform' not in method
            arr = animate_to_np(pop, method, num_steps=num_steps, plot_nash=plot_nash)
            cols.append(arr)
        item = np.concatenate(cols, axis=2)
        rows.append(item)
    item = np.concatenate(rows, axis=1)
    imageio.mimsave(file, item, fps=3)


animate_game_grid('grid_single.gif', lambda game: game.random(1, mag=(0.1, 0.4)))
animate_game_grid('grid_simple.gif', lambda game: [game.zero(), game.random(1, mag=(0.1, 0.4))[0]])
animate_game_grid('grid_random.gif', lambda game: game.random(3, mag=(0.2, 0.7)))
animate_game_grid('grid_support.gif', lambda game: game.support(0.5, rand_mag=0.02))
