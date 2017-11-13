import numpy as np
import matplotlib.pyplot as plt
from ..src import environment as env
# from matplotlib import cm


def plot3d_over_states(f, zlabel="", ):
    a = np.arange(0, env.JacksCarRentalEnvironment.MAX_CAPACITY + 1)
    b = np.arange(0, env.JacksCarRentalEnvironment.MAX_CAPACITY + 1)
    # b, a !!!
    b, a = np.meshgrid(b, a)
    v = f.reshape(env.JacksCarRentalEnvironment.MAX_CAPACITY + 1, -1)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax = fig.gca(projection='3d')
    # surf = ax.plot_surface(a, b, v, rstride=1, cstride=1, cmap=cm.coolwarm,
    #                   linewidth=0, antialiased=False)
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.scatter(a, b, v, c='b', marker='.')
    ax.set_xlabel("cars at A")
    ax.set_ylabel("cars at B")
    ax.set_zlabel(zlabel)

    # ax.view_init(elev=10., azim=10)

    plt.show()


def plot_policy(policy):
    a = np.arange(0, env.JacksCarRentalEnvironment.MAX_CAPACITY+1)
    b = np.arange(0, env.JacksCarRentalEnvironment.MAX_CAPACITY+1)
    a, b = np.meshgrid(a, b)
    po = policy.reshape(env.JacksCarRentalEnvironment.MAX_CAPACITY+1,-1)
    levels = range(-5, 6, 1)
    plt.figure(figsize=(7, 6))
    cs = plt.contourf(a, b, po, levels)
    cbar = plt.colorbar(cs)
    cbar.ax.set_ylabel('actions')
    # plt.clabel(cs, inline=1, fontsize=10)
    plt.title('Policy')
    plt.xlabel("cars at B")
    plt.ylabel("cars at A")