# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 12:10:27 2022

@author: marti
"""

from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()

AREA_SIZE = 1000
SPEED = 50 
SIGMA = (SPEED * 1000. / 24. / AREA_SIZE) / (3. * np.sqrt(2))
UNSEEN = 0
DONE = 1
SOCIAL_DISTANCE = 0.0000007
BETA1 = 1 # amylase rate
BETA2 = 1 # glucose to lactate rate
BETA3 = 1 # lactate to electron rate

FRAME_RATE = 10          # Refresh graphics very FRAME_RATE hours
#DENSITY = 100
starch_density = 10
ecoli_density = 10
so_density = 10
e_density = 0
l_density = 0
g_density = 0


# Get a and b of the line equation (ax + b) from 2 points
def compute_line_parameters(p1, p2):
    a = (p1[1] - p2[1]) / (p1[0] - p2[0])
    b = p1[1] - a * p1[0]
    return a, b

## The locations of borders on the map
A = (0., 1.)
B = (1.,0.)
C = (0.,1.)
D = (1.,0.)
LINE1 = compute_line_parameters(A, B)
LINE2 = compute_line_parameters(B, C)
LINE3 = compute_line_parameters(C, D)

class SIRState(Enum):
    STARCH = 0
    GLUCOSE = 1
    LACTATE = 2
    ELECTRON = 3
    AMYLASE = 4
    ECOLI = 5
    SO = 6

    
def distance(x1, y1, x2, y2):
    return np.sqrt(np.square(x1 - x2) + np.square(y1 - y2))
"""
@dataclass
class Fix:
    x: float        # Normalized x position
    y: float        # normalized y position
    original_x: float
    original_y: float
    succ = []       # neighbors
    status = UNSEEN
    

    def __init__(self, x, y):
        self.x = x
        self.original_x = x
        self.y = y
        self.original_y = y
        

    # Function that tells if the person has infected_neightbors
    def has_amylase_neighbor(self):
        for neighbor in self.succ:
            
            if (neighbor.state == SIRState.AMYLASE):
                return True
            

        
    
    def move(self, sigma=SIGMA):
        


        self.x = self.x
        self.y = self.y

    def update(self):
        return self
"""
@dataclass
class Mobile:
    x: float        # Normalized x position
    y: float        # normalized y position
    original_x: float
    original_y: float
    succ = []       # neighbors
    status = UNSEEN
    

    def __init__(self, x, y):
        self.x = x
        self.original_x = x
        self.y = y
        self.original_y = y
        

    # Function that tells if the person has infected_neightbors
    def has_amylase_neighbor(self):
        for neighbor in self.succ:
            if (neighbor.state == SIRState.AMYLASE):
                
                return True
            
        
        
    
    def has_ecoli_neighbor(self):
        for neighbor in self.succ:
            if (neighbor.state == SIRState.ECOLI):
                return True
        
    
    def has_shewanella_neighbor(self):
        for neighbor in self.succ:
            if (neighbor.state == SIRState.SO):
                return True
        

    def move(self, sigma=SIGMA):
        dx = np.random.normal(0, SIGMA)
        dy = np.random.normal(0, SIGMA)

        # Clip to borders
        new_x = np.clip(self.x + dx, 0.0, 1.0)
        new_y = np.clip(self.y + dy, 0.0, 1.0)


        self.x = new_x
        self.y = new_y

    def update(self):
        return self

class starch(Mobile):
    state = SIRState.STARCH

    def update(self):
        turn_glucose = self.has_amylase_neighbor()

        if turn_glucose:
            return glucose(self.x, self.y)
        
        return self

class glucose(Mobile):
    state = SIRState.GLUCOSE

    def update(self):
        turn_lactate = self.has_ecoli_neighbor()

        if (turn_lactate and np.random.rand() < BETA2):
            for i in range(2): # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                return lactate(self.x, self.y)
        return self

class ecoli(Mobile):
    state = SIRState.ECOLI

    def update(self):
        return self

class shewanella(Mobile):
    state = SIRState.SO

    def update(self):
        return self

class amylase(Mobile):
    state = SIRState.AMYLASE
    
    def update(self):
        return self
    
    
class lactate(Mobile):
    state = SIRState.LACTATE

    def update(self):
        turn_electron = self.has_shewanella_neighbor()

        if (turn_electron and np.random.rand() < BETA3):
            return electron(self.x, self.y)
        return self

class electron(Mobile):
    state = SIRState.ELECTRON

def to_matrix(starch_molecules):
    size = len(people)
    out = np.zeros((size, size))
    for i in range(size):
        for j in range(i + 1, size):
            dist = distance(people[i].x, people[i].y, people[j].x, people[j].y)
            out[i][j] = dist
            out[j][i] = dist

    return out

def update_graph(people):
    # Reset
    for p in people:
        p.succ = []
        p.status = UNSEEN
        

    adjacency_matrix = to_matrix(people)
    size = len(people)
    for i in range(size):
        for j in range(i + 1, size):
            if adjacency_matrix[i][j] >= SOCIAL_DISTANCE:
                continue # Skip
            
            people[i].succ.append(people[j])
            people[j].succ.append(people[i])





def display_map(people, ax = None):
    x = [ p.x for p in people]
    y = [ p.y for p in people]
    h = [ p.state.name[0] for p in people]
    horder = ["A", "S", "G", "E","L"]
    ax = sns.scatterplot(x, y, hue=h, hue_order=horder, ax=ax)
    ax.set_xlim((0.0,1.0))
    ax.set_ylim((0.0,1.0))
    ax.set_aspect(224/145)
    ax.set_axis_off()
    ax.set_frame_on(True)
    ax.legend(loc=1, bbox_to_anchor=(0, 1))


count_by_population = None
def plot_population(people, ax = None):
    global count_by_population

    states = np.array([p.state.value for p in people], dtype=int)
    counts = np.bincount(states, minlength = 7)
    entry = {
        "Starch" : counts[SIRState.STARCH.value],
        "Glucose" : counts[SIRState.GLUCOSE.value],
        "lactate" : counts[SIRState.LACTATE.value],
        "Electron" : counts[SIRState.ELECTRON.value],
        "e.coli" : counts[SIRState.ECOLI.value],
        "shewanella" : counts[SIRState.SO.value],
        "amylase" : counts[SIRState.AMYLASE.value]
        
    }
    
    if count_by_population is None:
        count_by_population = pd.DataFrame(entry, index=[0.])
    else:
        count_by_population = count_by_population.append(entry, ignore_index=True)
    if ax != None:
        count_by_population.index = np.arange(len(count_by_population)) / 24
        sns.lineplot(data=count_by_population, ax = ax)


'''
Main loop function, that is called at each turn
'''
def next_loop_event(t):
    print("Time =",t)

    # Move each person
    for p in people:
        p.move()
        

    update_graph(people)

    # Update the state of people
    for i in range(len(people)):
        people[i] = people[i].update()
        
    if t % FRAME_RATE == 0:
        fig.clf()
        ax1, ax2 = fig.subplots(1,2)
        display_map(people, ax1)
        plot_population(people, ax2)
    else:
        plot_population(people, None)
        

'''
Function that crate the initial population
'''
def create_data():
    data = []
    
    x = np.random.rand()
    y = np.random.rand()
    for starch_i in range(starch_density):
        to_add = starch(x, y)
        data.append(to_add)
        
    for ecoli_i in range(ecoli_density):
        to_add = ecoli(x, y)
        data.append(to_add)
        
    for shewanella_i in range(so_density):
        to_add = shewanella(x, y)
        data.append(to_add)
    
    for e in range(e_density):
        to_add = electron(x, y)
        data.append(to_add)
    
    for l in range(l_density):
        to_add = lactate(x, y)
        data.append(to_add)
    
    for g in range(g_density):
        to_add = glucose(x, y)
        data.append(to_add)
    
    for a in range(ecoli_density):
        to_add = amylase(x, y)
        data.append(to_add)
        
        
        

    return data


import matplotlib.animation as animation
people = create_data()

fig = plt.figure(1, figsize = (20.,10.))
duration = 2 # in days
anim = animation.FuncAnimation(fig, next_loop_event, frames=np.arange(duration*24), interval=100, repeat=False)

# To save the animation as a video
anim.save("simulation.gif", writer = "ffmpeg")
plt.savefig('graph1.png')
plt.show()
