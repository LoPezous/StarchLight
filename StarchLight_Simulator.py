# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 12:10:27 2022

@author: marti
"""
from pickle import FALSE
from molecules import Mobile
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from molecules import SIRState, Starch, Ecoli, Shewanella, Electron, Lactate, Glucose, Amylase
from params import *

sns.set()


def distance(x1, y1, z1, x2, y2, z2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)


def to_matrix(people):
    size = len(people)
    out = np.zeros((size, size))
    for i in range(size):
        for j in range(i + 1, size):
            dist = distance(people[i].x, people[i].y, people[i].z, people[j].x, people[j].y, people[j].z)
            out[i][j] = dist
            out[j][i] = dist

    return out




def display_map(people, ax=None):
    x = [p.x for p in people]
    y = [p.y for p in people]
    z = [p.z for p in people]
    h = [p.state.name[0] for p in people]
    horder = ["A", "S", "G", "E", "L"]
    ax = sns.scatterplot(x, y, z, hue=h, hue_order=horder, ax=ax, size=z)

    ax.set_xlim((0.0, 1.0))
    ax.set_ylim((0.0, 1.0))

    ax.set_aspect(224 / 145)
    ax.set_axis_off()
    ax.set_frame_on(True)
    # ax.legend(loc=1, bbox_to_anchor=(0, 1))


count_by_population = None


def plot_population(people, ax=None):
    global count_by_population

    states = np.array([p.state.value for p in people], dtype=int)
    counts = np.bincount(states, minlength=7)
    entry = {
        "Starch": counts[SIRState.STARCH.value],
        "Glucose": counts[SIRState.GLUCOSE.value],
        "lactate": counts[SIRState.LACTATE.value],
        "Electron": counts[SIRState.ELECTRON.value],
        "e.coli": counts[SIRState.ECOLI.value],
        "shewanella": counts[SIRState.SO.value],
        "amylase": counts[SIRState.AMYLASE.value]

    }

    if count_by_population is None:
        count_by_population = pd.DataFrame(entry, index=[0.])
    else:
        count_by_population = count_by_population.append(entry, ignore_index=True)
    if ax is not None:
        count_by_population.index = np.arange(len(count_by_population)) / 24
        sns.lineplot(data=count_by_population, ax=ax)


'''
Main loop function, that is called at each turn
'''


def next_loop_event(t, liste_molecules):
    print("Time =", t)

    # Move each person
    for molecule in liste_molecules:
        molecule.move()

    liste_molecules = update_graph(liste_molecules)

    # if t % FRAME_RATE == 0:
    #     # fig.clf()
    #     # ax1, ax2 = fig.subplots(1, 2)
    #     display_map(liste_molecules, ax1)
    #     plot_population(liste_molecules, ax2)
    # else:
    #     plot_population(liste_molecules, None)
    return update_liste(liste_molecules)

'''
Function that crate the initial population
'''


def find_molecule(liste_molecules, _id):
    for i, molecule in enumerate(liste_molecules):
        if _id == molecule._id:
            return i


def update_graph(liste_molecules):
    # Reset
    for p in liste_molecules:
        p.voisin = []
        p.status = UNSEEN

    adjacency_matrix = to_matrix(liste_molecules)
    size = len(liste_molecules)
    for i in range(size):
        for j in range(i + 1, size):
            if adjacency_matrix[i][j] >= SOCIAL_DISTANCE:
                continue  # Skip

            liste_molecules[i].voisin.append(liste_molecules[j])
            liste_molecules[j].voisin.append(liste_molecules[i])

    return liste_molecules


def update_liste(liste_molecules):
    new_list = []
    for p in liste_molecules:
        p.voisin = []
        p.status = UNSEEN

    adjacency_matrix = to_matrix(liste_molecules)
    size = len(liste_molecules)
    for i in range(size):
        for j in range(i + 1, size):
            if adjacency_matrix[i][j] >= SOCIAL_DISTANCE:
                continue  # Skip

            liste_molecules[i].voisin.append(liste_molecules[j])
            liste_molecules[j].voisin.append(liste_molecules[i])

    for molecule in liste_molecules:
        if molecule.voisin and molecule.status == UNSEEN:
            produit = molecule.reaction()
            molecule.status = SEEN
            # print(molecule, molecule.voisin, produit)
            if isinstance(produit, list):  # If Lactate
                new_list += produit
            else:
                new_list.append(produit)
        else:
            new_list.append(molecule)
    return new_list


def create_data(): #->list(Mobile)
    data = []

    x = np.random.rand()
    y = np.random.rand()
    z = np.random.rand()
    for starch_i in range(starch_density):
        to_add = Starch(x, y, z)
        data.append(to_add)

    for ecoli_i in range(ecoli_density):
        to_add = Ecoli(x, y, z)
        data.append(to_add)

    for shewanella_i in range(so_density):
        to_add = Shewanella(x, y, z)
        data.append(to_add)

    for e in range(e_density):
        to_add = Electron(x, y, z)
        data.append(to_add)

    for l in range(l_density):
        to_add = Lactate(x, y, z)
        data.append(to_add)

    for g in range(g_density):
        to_add = Glucose(x, y, z)
        data.append(to_add)

    for a in range(ecoli_density):
        to_add = Amylase(x, y, z)
        data.append(to_add)

    return data


import matplotlib.animation as animation

molecules_list = create_data()

"""fig = plt.figure(1, figsize=(20., 10.))
duration = 1  # in days
anim = animation.FuncAnimation(fig, next_loop_event, frames=np.arange(duration * 24), interval=100, repeat=False)
"""
# To save the animation as a video

# anim.save("simulation.gif", writer='ffmpeg')

states = np.array([p.state.value for p in molecules_list], dtype=int)
counts = np.bincount(states, minlength=7) #liste states en chiffre pour le bincount
entry = {
    "Starch": counts[SIRState.STARCH.value],
    "Glucose": counts[SIRState.GLUCOSE.value],
    "lactate": counts[SIRState.LACTATE.value],
    "Electron": counts[SIRState.ELECTRON.value],
    "e.coli": counts[SIRState.ECOLI.value],
    "shewanella": counts[SIRState.SO.value],
    "amylase": counts[SIRState.AMYLASE.value]

}
donnees = [entry]

for time in np.arange(duration * 24):
    molecules_list = next_loop_event(time, molecules_list)
    states = np.array([p.state.value for p in molecules_list], dtype=int)
    counts = np.bincount(states, minlength=7)
    entry = {
        "Starch": counts[SIRState.STARCH.value],
        "Glucose": counts[SIRState.GLUCOSE.value],
        "lactate": counts[SIRState.LACTATE.value],
        "Electron": counts[SIRState.ELECTRON.value],
        "e.coli": counts[SIRState.ECOLI.value],
        "shewanella": counts[SIRState.SO.value],
        "amylase": counts[SIRState.AMYLASE.value]

    }
    donnees.append(entry)

df = pd.DataFrame(donnees)
Amperes = ((df.at[df.index[-1],'Electron']/(6.2*10**18)))/3600
Power = Amperes * Volts
upscale_factor = (1/starch_mass)*1000
print(df)
print(Amperes, 'A.h')
print('Mean intensity: ', Amperes/((duration*24)/60), 'A ')
print('Mean power: ', Power/((duration*24)/60), 'W ')
print('glucose mass: ', starch_mass, ' g')
print('Prediction for 1kg: ', Amperes*upscale_factor, 'A.h ')
print('Prediction for 1kg: ', Power*upscale_factor, 'W.h ')
#plt.figure(figsize=(20, 10)) 
df.plot()
plot_population(molecules_list)
plt.savefig('graph2.png')
plt.show()
