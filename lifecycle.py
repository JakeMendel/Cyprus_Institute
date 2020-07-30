#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import numba as nb
from numba import njit, prange, int32, int64, types, float64, deferred_type, typed, typeof
from numba.experimental import jitclass
import time


# In[42]:


spec = [('stage', types.string),
        ('initial_T', float64),
        ('age', float64),
        ('stage_age', float64),
        ('p_death', float64),
        ('stage_length', float64),
        ('egg_rate', float64),
        ('birth_density', int32)
       ]

@jitclass(spec)
class Mosquito:
    def __init__(self, stage, T, age, density):
        self.initial_T = T
        self.stage = stage
        self.age = age
        self.stage_age = 0.0
        self.birth_density = density
        self.p_death= self.get_p_death(self.initial_T)
        self.stage_length = self.get_stage_length()
        self.egg_rate = self.get_egg_rate()
    
    def get_p_death(self, T):
        if self.stage == 'egg':
            return 1 - 0.91 * (1/(1 + np.exp(-5-T))) *(1/(1 + np.exp(T-40)))
        elif self.stage == 'larva':
            return 1 - 0.96 * (1/(1 + np.exp(4-T))) *(1/(1 + np.exp(T-36)))
        elif self.stage == 'pupa':
            return 1 - 0.999 * (1/(1 + np.exp(13-T))) *(1/(1 + np.exp(T-36)))
        elif self.stage == 'naive' or self.stage == 'adult':
            return 1 - 0.98 * (1/(1 + np.exp(-5-T))) *(1/(1 + np.exp(T-38)))
        
    def get_stage_length(self):
#         Values taken from 'Development and calibration of a model for the potential 
#         establishment and impact of Aedes albopictus in Europe' - Pasquali et al
        if self.stage == 'egg':
            a = 0.0000416657
            T_sup = 37.3253
            if self.initial_T < T_sup:
                mean_rate = a * self.initial_T ** 2 * (T_sup - self.initial_T)
                mean = 1 / mean_rate
            else:
                mean = None
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
            if T_inf < self.initial_T < T_sup:
                mean_rate = a * self.initial_T * (self.initial_T - T_inf) * np.sqrt(T_sup - self.initial_T)
                if self.stage in ['larva', 'pupa']:
                    mean = (1 + 0.2789 * self.birth_density ** 0.2789) / mean_rate
                else:
                    mean = 1 / mean_rate
            else:
                mean = None
        if mean is None:
            return np.inf
        else:
            return np.random.gamma(mean * 10, 0.1)
    
    def get_egg_rate(self):
        if self.stage == 'adult':
            return max(0, -0.04 * (self.initial_T - 14) * (self.initial_T - 52))
        else:
            return 0
    
    def develop(self, T, density):
        if self.stage == 'egg':
            return Mosquito('larva', T, self.age, density)
        elif self.stage == 'larva':
            return Mosquito('pupa', T, self.age, density)
        elif self.stage == 'pupa':
            return Mosquito('naive', T, self.age, density)
        elif self.stage == 'naive':
            return Mosquito('adult', T, self.age, density)
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
        mean_rate = a * self.initial_T * (self.initial_T - T_inf) * np.sqrt(T_sup - self.initial_T)
        mean = (1 + 0.2789 * density ** 0.2789) / mean_rate
        self.stage_length = np.random.gamma(mean * 10, 0.1)


# In[43]:


mosquito_list_type = typeof([Mosquito('egg', 25,0, 1000)])

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
        
        density = self.larva_numbers[-1] + self.pupa_numbers[-1]
        mosquitos = [Mosquito('egg', self.T, 0.0, density) for i in range(egg_no)]
        mosquitos += [Mosquito('larva', self.T, 0.0, density) for i in range(larva_no)]
        mosquitos += [Mosquito('pupa', self.T, 0.0, density) for i in range(pupa_no)]
        mosquitos += [Mosquito('naive', self.T, 0.0, density) for i in range(naive_no)]
        mosquitos += [Mosquito('adult', self.T, 0.0, density) for i in range(adult_no)]
        self.mosquitos = mosquitos
    
    def propagate(self, iterations, timesteps_in_day):
        timestep = 1 / timesteps_in_day
        if len(self.Temps) < iterations:
            raise Exception("Not enough temperature data for that many iterations")
        for iteration in range(iterations):
            survivors = [Mosquito('egg', 25,0.0, 1000) for i in range(0)]
            density = self.larva_numbers[-1] + self.pupa_numbers[-1]
            Temp = self.Temps[iteration]
            for i, mosquito in enumerate(self.mosquitos):
                death_value = np.random.random()
                if 1 - ((1 - mosquito.get_p_death(Temp)) * self.death_density_dependence(mosquito, density)) ** timestep < death_value:
#                     mosquito.update_stage_length(density)
                    mosquito.age += timestep
                    mosquito.stage_age += timestep
                    if mosquito.stage == 'adult':
                        survivors += self.lay_eggs(mosquito, timestep, Temp, density)
                    if mosquito.stage_age > mosquito.stage_length:
                        mosquito = mosquito.develop(Temp, density)
                    survivors.append(mosquito)    
            self.mosquitos = list(survivors)
            self.update_numbers()
    
    def death_density_dependence(self, mosq, density):
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
    
    def lay_eggs(self, mosquito, timestep, Temp, density):
        number = np.random.poisson(mosquito.egg_rate * timestep)
        return [Mosquito('egg', Temp, 0.0, density) for i in range(number)]

def plot(pop, timestep, include = None):
    if include is None:
        include = ['egg', 'larva', 'pupa', 'naive', 'adult', 'total']
    xs = timestep * np.array(range(len(pop.egg_numbers)))
    plt.figure(figsize = (20,10))
    if 'egg' in include:
        plt.plot(xs, pop.egg_numbers, label = 'eggs')
    if 'larva' in include:
        plt.plot(xs, pop.larva_numbers, label = 'larvae')    
    if 'pupa' in include:
        plt.plot(xs, pop.pupa_numbers, label = 'pupae')    
    if 'naive' in include:
        plt.plot(xs, pop.naive_numbers, label = 'naive')    
    if 'adult' in include:
        plt.plot(xs, pop.adult_numbers, label = 'adult')    
    if 'total' in include:
        plt.plot(xs, pop.total_numbers, label = 'total')
    plt.legend()
    plt.show()



# In[45]:


test = Population(0,0,0,0,100,[25.0] * 600)
test.propagate(400,2)
plot(test,0.5)


# In[56]:


a = [25+ 10 * np.sin(2 * np.pi * x / 365) for x in np.linspace(0,1000,1000)]
test = Population(0,0,0,0,100,a)
get_ipython().run_line_magic('time', 'test.propagate(600,1)')
plot(test,1)


# In[57]:


plt.figure(figsize = (20,10))
plt.plot(a[:600])
plt.show()


# In[58]:


plot(test,1,include = ['pupa', 'naive', 'adult'])

