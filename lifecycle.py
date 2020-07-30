#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import numba as nb
from numba import njit, prange, int32, int64, types, float64, deferred_type, typed, typeof
from numba.experimental import jitclass
import time


# In[2]:


#include diapause
#include more biological data
#write up one/two pages what is assumed and how is it justified, 
#include basic analysis of behaviour. Are results reasonable?
#compare with other models
#How could it be used to replicate experiments? How would I simulate a truck in a field?
    #give coords. Eggs hatch with same coords as adult. adult moves. if adult in
    #fixed range collect etc.
#Or a lab experiment?


# In[3]:


spec = [('stage', types.string),
        ('T', int32),
        ('age', float64),
        ('stage_age', float64),
        ('p_death', float64),
        ('stage_length', float64),
        ('egg_rate', float64),
       ]

@jitclass(spec)
class Mosquito:
    def __init__(self, stage, T, age):
        self.T = T
        self.stage = stage
        self.age = age
        self.stage_age = 0.0
        self.p_death= self.get_p_death()
        self.stage_length: float = self.get_stage_length()
        self.egg_rate = self.get_egg_rate()
    
    def get_p_death(self):
        if self.stage == 'egg':
            return 1 - 0.91 * (1/(1 + np.exp(-5-self.T))) *(1/(1 + np.exp(self.T-40)))
        elif self.stage == 'larva':
            return 1 - 0.96 * (1/(1 + np.exp(4-self.T))) *(1/(1 + np.exp(self.T-36)))
        elif self.stage == 'pupa':
            return 1 - 0.999 * (1/(1 + np.exp(13-self.T))) *(1/(1 + np.exp(self.T-36)))
        elif self.stage == 'naive' or self.stage == 'adult':
            return 1 - 0.98 * (1/(1 + np.exp(-5-self.T))) *(1/(1 + np.exp(self.T-38)))
        
    def get_stage_length(self):
#         Values taken from 'Development and calibration of a model for the potential 
#         establishment and impact of Aedes albopictus in Europe' - Pasquali et al
        if self.stage == 'egg':
            a = 0.0000416657
            T_sup = 37.3253
            if self.T < T_sup:
                mean_rate = a * self.T ** 2 * (T_sup - self.T)
                mean = 1 / mean_rate
            else:
                mean = np.nan
        else:
            if self.stage == 'larva':
                a = 0.00008604
                T_inf = 8.2934
                T_sup = 36.0729
            elif self.stage == 'pupa':
                a = 0.0003102
                T_inf = 11.9433
                T_sup = 40
            elif self.stage == 'naive':
                if np.random.randint(2) == 1:
                    a = 0.0001812
                    T_inf = 7.7804
                    T_sup = 35.2937
                else:
                    return np.inf
            elif self.stage == 'adult':
                return np.inf
            else:
                raise Exception('Invalid Stage Name')
            if T_inf < self.T < T_sup:
                mean_rate = a * self.T * (self.T - T_inf) * np.sqrt(T_sup - self.T)
                mean = 1 / mean_rate
            else:
                mean = np.nan
        if mean is np.nan:
            return np.inf
        else:
            return np.random.gamma(mean * 10, 0.1)
    
    def get_egg_rate(self):
        if self.stage == 'adult':
            return max(0, -0.04 * (self.T - 14) * (self.T - 52))
        else:
            return 0
    
    def develop(self):
        if self.stage == 'egg':
            return Mosquito('larva', self.T, self.age)
        elif self.stage == 'larva':
            return Mosquito('pupa', self.T, self.age)
        elif self.stage == 'pupa':
            return Mosquito('naive', self.T, self.age)
        elif self.stage == 'naive':
            return Mosquito('adult', self.T, self.age)
        else:
            raise Exception('Stage Invalid for Development')

    def update_stage_length(self, density):
        #larvae and pupae develop slower at higher density
        if self.stage == 'larva':
            a = 0.00008604
            T_inf = 8.2934
            T_sup = 36.0729
        elif self.stage == 'pupa':
            a = 0.0003102
            T_inf = 11.9433
            T_sup = 40
        else:
            return
        mean_rate = a * self.T * (self.T - T_inf) * np.sqrt(T_sup - self.T)
        mean = 0.2789 * density ** 0.2789 / mean_rate
        self.stage_length = np.random.gamma(mean * 10, 0.1)


# In[22]:


mosquito_type = typeof(Mosquito('egg', 25,0))
mosquito_list_type = typeof([Mosquito('egg', 25,0)])

specs = [('egg_numbers', types.ListType(int64)),
         ('larva_numbers', types.ListType(int64)),
         ('pupa_numbers', types.ListType(int64)),
         ('naive_numbers', types.ListType(int64)),
         ('adult_numbers', types.ListType(int64)),
         ('total_numbers', types.ListType(int64)),
         ('Temps', types.ListType(float64)),
         ('T', float64),
         ('mosquitos', mosquito_list_type)
        ]

@jitclass(specs)
class Population:
    def __init__(self, egg_no, larva_no, pupa_no, naive_no, adult_no, Temps):
        self.egg_numbers = typed.List([egg_no])
        self.larva_numbers = typed.List([larva_no])
        self.pupa_numbers = typed.List([pupa_no])
        self.naive_numbers = typed.List([naive_no])
        self.adult_numbers = typed.List([adult_no])
        self.total_numbers = typed.List([egg_no + larva_no + pupa_no + naive_no + adult_no])
        self.Temps = typed.List(Temps)
        self.T = self.Temps[0]
        
        mosquitos = [Mosquito('egg', self.T, 0.0) for i in range(egg_no)]
        mosquitos += [Mosquito('larva', self.T, 0.0) for i in range(larva_no)]
        mosquitos += [Mosquito('pupa', self.T, 0.0) for i in range(pupa_no)]
        mosquitos += [Mosquito('naive', self.T, 0.0) for i in range(naive_no)]
        mosquitos += [Mosquito('adult', self.T, 0.0) for i in range(adult_no)]
        self.mosquitos = mosquitos
    
    def propagate(self, iterations, timestep):
        for day in range(iterations):
            survivors = [Mosquito('egg', 25,0.0) for i in range(0)]
            density = self.larva_numbers[-1] + self.pupa_numbers[-1]
            for i, mosquito in enumerate(self.mosquitos):
                #density dependent p_death
#                 p_death = 1 - (1 - mosquito.p_death) * self.density_dependence(mosquito)
                death_value = np.random.random()
                if 1 - ((1 - mosquito.p_death) * self.density_dependence(mosquito, density)) ** timestep < death_value:
                    mosquito.update_stage_length(density)
                    mosquito.age += timestep
                    mosquito.stage_age += timestep
                    if mosquito.stage == 'adult':
                        survivors += self.lay_eggs(mosquito, timestep)
                    if mosquito.stage_age > mosquito.stage_length:
                        mosquito = mosquito.develop()
                    survivors.append(mosquito)    
            self.mosquitos = list(survivors)
            self.update_numbers()
    
    def density_dependence(self, mosq, density):
        if mosq.stage == 'larva' or mosq.stage == 'pupa':
            return np.exp(-3.819 * 10 ** (-5) * density)
        else:
            return 1
        
    
    def update_numbers(self):
        eggs, larvae, pupae, naives, adults = 0,0,0,0,0
        for mosquito in self.mosquitos:
            if mosquito.stage == 'egg':
                eggs += 1
            elif mosquito.stage == 'larva':
                larvae += 1
            elif mosquito.stage == 'pupa':
                pupae += 1
            elif mosquito.stage == 'naive':
                naives += 1
            elif mosquito.stage == 'adult':
                adults += 1
        self.egg_numbers.append(eggs)
        self.larva_numbers.append(larvae)
        self.pupa_numbers.append(pupae)
        self.naive_numbers.append(naives)
        self.adult_numbers.append(adults)
        self.total_numbers.append(eggs+larvae+pupae+naives+adults)
    
    def lay_eggs(self, mosquito, timestep):
        number = np.random.poisson(mosquito.egg_rate * timestep)
        return [Mosquito('egg', self.T, 0) for i in range(number)]

def plot(pop, timestep):
    xs = timestep * np.array(range(len(pop.egg_numbers)))
    plt.figure(figsize = (20,10))
    plt.plot(xs, pop.egg_numbers, label = 'eggs')
    plt.plot(xs, pop.larva_numbers, label = 'larvae')    
    plt.plot(xs, pop.pupa_numbers, label = 'pupae')    
    plt.plot(xs, pop.naive_numbers, label = 'naive')    
    plt.plot(xs, pop.adult_numbers, label = 'adult')    
    plt.plot(xs, pop.total_numbers, label = 'total')
    plt.legend()
    plt.show()


# In[27]:


test = Population(0,0,0,0,100,[25.0,2.0])
test.propagate(500,0.05)
plot(test,0.5)


# In[26]:


test = Population(0,0,0,0,100,[25.0,2.0])
test.propagate(400,0.5)
plot(test,0.5)

