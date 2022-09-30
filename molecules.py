"""File containing the classes."""
import random

import numpy as np
from dataclasses import dataclass
from enum import Enum
from params import *

def wrap_into_box(x, box):
    """Subroutine to wrap the coordinates into a box renvoi un x entre 0 et L"""
    if (x > box):
        x -= box

    if (x < 0):
        x += box
    
    return x

class SIRState(Enum):
    STARCH = 0
    GLUCOSE = 1
    LACTATE = 2
    ELECTRON = 3
    AMYLASE = 4
    ECOLI = 5
    SO = 6


@dataclass
class Mobile:
    x: float  # Normalized x position
    y: float  # normalized y position
    z: float
    voisin = []  # neighbors
    status = UNSEEN

    def __post_init__(self):
        self._id = random.randint(1, 1_000_000)

    def has_neighbor_of_type(self, molecule_type):
        for neighbor in self.voisin:
            if neighbor.state == molecule_type:
                return True

    def move(self):
        dx = np.random.uniform(-MAX_MOVE, MAX_MOVE) 
        dy = np.random.uniform(-MAX_MOVE, MAX_MOVE)
        dz = np.random.uniform(-MAX_MOVE, MAX_MOVE)

        # # Clip to borders
        # new_x = np.clip(self.x + dx, 0.0, L)
        # new_y = np.clip(self.y + dy, 0.0, L)
        # new_z = np.clip(self.z + dz, 0.0, L)

        # self.x = new_x
        # self.y = new_y
        # self.z = new_z
    
        new_x = self.x + dx
        self.x = wrap_into_box(new_x, L)
        new_y = self.y + dy
        self.y = wrap_into_box(new_y, L)
        new_z = self.z + dz
        self.z = wrap_into_box(new_z, L)
        #print(self.x, self.y, self.z)
    

    def reaction(self):
        return self


class Starch(Mobile):
    state = SIRState.STARCH

    def reaction(self):
        turn_glucose = self.has_neighbor_of_type(SIRState.AMYLASE)
        if turn_glucose:
            self.status = SEEN
            return Glucose(self.x, self.y, self.z)
        return self


class Glucose(Mobile):
    state = SIRState.GLUCOSE

    def reaction(self):
        turn_lactate = self.has_neighbor_of_type(SIRState.ECOLI)
        if turn_lactate and np.random.rand() < BETA2:
            self.status = SEEN
            return [Lactate(self.x, self.y, self.z), Lactate(self.x, self.y, self.z)]
        return self


class Ecoli(Mobile):
    state = SIRState.ECOLI


class Shewanella(Mobile):
    state = SIRState.SO


class Amylase(Mobile):
    state = SIRState.AMYLASE


class Lactate(Mobile):
    state = SIRState.LACTATE

    def reaction(self):
        turn_electron = self.has_neighbor_of_type(SIRState.SO)
        if turn_electron and np.random.rand() < BETA3:
            self.status = SEEN
            return [Electron(self.x, self.y, self.z), Electron(self.x, self.y, self.z), Electron(self.x, self.y, self.z), Electron(self.x, self.y, self.z), Electron(self.x, self.y, self.z)]
        return self


class Electron(Mobile):
    state = SIRState.ELECTRON
