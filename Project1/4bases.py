#!/usr/bin/env python
# coding: utf-8

# In[31]:


import numpy as np
import scipy.optimize as opt
import scipy.misc as misc
from scipy.integrate import solve_ivp
import ctypes
import matplotlib.pyplot as plt
import numpy.random as rn
from numba import njit, jit
sns.reset_defaults()

def base(a: int, von: int, zu: int):
    zahl = 0
    multiplikator = 1
    while a > 0:
        zahl += (a%zu)*multiplikator
        a=a//zu
        multiplikator*=von
    return zahl
base = np.vectorize(base)  

def nt(zahl: int, N: int):
    zahl = str(zahl).zfill(N)
    Seq=''
    seqdic={'0':'A', '1': 'G', '2': 'C', '3': 'T'}
    for i in range(len(zahl)):
        Seq = Seq +  seqdic[zahl[i]]
    return Seq
nt = np.vectorize(nt)                   
    
class solver:
    def __init__(self, N: int, initial_conditions: np.ndarray, r: float, eta: float, u: float, approximate_mutation: bool, max_peak: int, second_peak: int, mu_0: float):

        # length of sequence
        self.N = N
        # dimension of space
        self.D = 4 ** self.N

        # check dimensionality
        if len(initial_conditions) != self.D:
            raise TypeError('Number of initial conditions does not match dimension.')
        
        # set initial conditions
        self.initial_conditions = initial_conditions

        # fittest peak
        self.r = r

        if eta > r:
            raise TypeError('Height of secondary peak greater than main peak.')
        else:
            self.eta = eta

        # mutation probability
        self.u = u

        # position of peaks
        self.peak_r = max_peak
        self.peak_eta = second_peak

        self.mu_0 = mu_0
        # fitness landscape
        self.mu = np.ones(self.D) * mu_0
        self.mu[self.peak_r] *= self.r
        self.mu[self.peak_eta] *= self.eta

        # mutation matrix
        if approximate_mutation:
            self.q_matrix = self.q_matrix_approximation()
        else:
            self.q_matrix = self.q_matrix_full()


    #@njit
    def Hamming_dinstance(self, a: int, b: int):
        """Function to calculate the Hamming distance between two sequences using the bit representation formalism

        Args:
            a (int): sequence one
            b (int): sequence two

        Returns:
            int: Hamming distance between sequences
        """
        # performing bit representation
        a = str(base(a,10,4)).zfill(self.N)
        b = str(base(b,10,4)).zfill(self.N)
        h_ab = 0
        for i in range(self.N):
            # bitwise difference makes Hamming distance (does not work for anything other than bit representation)
            if a[i]!= b[i]:
                h_ab+=1
        return h_ab
    
    def q_matrix_approximation(self):
        """q matrix for nearest neighbour mutations

        Returns:
            np.ndarray: q matrix in bit representation
        """
        
        q = np.zeros((4 ** self.N, 4 ** self.N))
        # off diagonals
        for i in range(4 ** self.N):
            for j in range(i+1, 4 ** self.N):
                if self.Hamming_dinstance(i, j) == 1:
                    q[i,j] = self.u
                    q[j,i] = self.u
        
        # diagonal terms
        for i in range(self.D):
            s = 0
            for j in range(self.D):
                if j != i:
                    s+=q[i, j]
            q[i, i] = 1 - s
        return q
    
    #@njit       
    def q_matrix_full(self):
        """q matrix for full mutation scenario

        Returns:
            np.ndarray: q matrix in bit representation
        """
        q = np.zeros((4 ** self.N, 4 ** self.N))
        # off diagonals
        for i in range(4 ** self.N):
            for j in range(i+1, 4 ** self.N):
                h = self.Hamming_dinstance(i, j)
                q[i,j] = self.u ** h * (1 - self.u) ** (self.N - h)
                q[j,i] = q[i, j]
        # diagonal terms
        for i in range(self.D):
            s = 0
            for j in range(self.D):
                if j != i:
                    s+=q[i, j]
            q[i, i] = 1 - s

        return q


    def system_equation(self, t:float, system:np.ndarray):
        """_summary_

        Args:
            t (float): array
            system (np.ndarray): system array

        Returns:
            np.ndarray: system equation at time t
        """
        dsystem_dt = np.zeros(self.D)

        # sum of mu with respective f
        mu_bar = np.dot(self.mu, system)

        # loop over system
        for i in range(self.D):
            for j in range(self.D):
                # mutations
                dsystem_dt[i] += self.mu[j] * self.q_matrix[i, j] * system[j]
            dsystem_dt[i] -= mu_bar * system[i]
        
        return dsystem_dt

    def solve(self, time_span, n_time):
        """Solver for system equation

        Args:
            time_span (_type_): _description_
            n_time (_type_): _description_

        Returns:
            _type_: _description_
        """
        time = np.linspace(time_span[0], time_span[1], n_time)

        result = solve_ivp(fun=self.system_equation, t_span=time_span, y0=self.initial_conditions, t_eval=time)
        print(result.y)
        return result

    def mu_wide(self, radius:int):
        """Function to implement wide mu region around eta peak

        Args:
            radius (int): radius of area
        """
        self.omega = radius
        Omega = []
        for i in range(self.D):
            if self.Hamming_dinstance(self.peak_eta, i) <= radius:
                Omega.append(i)
                self.mu[i] = self.eta * self.mu_0
        self.Omega = np.array(Omega)


    def order_parameter_F(self, f):
        F = 0
        for i in self.Omega:
            F += f[i]
        
        return F / f[self.peak_r]
    
    def order_parameter_A(self, f):
        return self.D * f[self.peak_r] / np.sum(f)
        



if __name__ == "__main__":
    """
    Just plotting some stuff to see whether it looks reasonable.... looks reasonable....
    """
    N = 2
    d = 4 ** N
    in_con = np.ones(d) * 1 / d

    tmax = 100

    test2 = solver(N, in_con, 2, 1.9, 1e-1, True, 1, 7, 1)
    print(test2)
    test2.mu_wide(1)
    result2 = test2.solve([0, tmax], 100)
    liste=result2.y
    f = plt.subplots()
    print( nt(base(1,10,4),N), nt(base(7,10,4),N))
    plt.plot(result2.t, result2.y.T)
    plt.ylabel(r"$f$")
    plt.xlabel(r"$t$")
    plt.yscale("log")
    plt.legend([ nt(base(i,10,4),N) for i in np.arange(d)], shadow=True)
    plt.savefig("rn_4.pdf", dpi=500, bbox_inches="tight")

    f2 = plt.subplots()
    plt.bar(np.arange(0, d, 1), result2.y[:,-1])
    plt.xticks(np.arange(0, d, 1),nt(base(np.arange(0, d, 1),10,4),N))
    plt.xlabel("Sequence")
    plt.ylabel(r"$f_\mathrm{final}$")
    plt.savefig("rn_5.pdf", dpi=500, bbox_inches="tight")

    f3 = plt.subplots()
    plt.bar(np.arange(0, d, 1), test2.mu)
    plt.xticks(np.arange(0, d, 1),nt(base(np.arange(0, d, 1),10,4),N))
    plt.xlabel("Sequence")
    plt.ylabel(r"$\mu$")
    plt.savefig("rn_6.pdf", dpi=500, bbox_inches="tight")
    plt.show()


# In[ ]:





# In[ ]:





# In[32]:


from matplotlib import animation
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
fig = plt.figure(figsize=(8,6))
axes = fig.add_subplot(1,1,1)
axes.set_ylim(0, 0.2)
plt.style.use("seaborn")
palette = list(reversed(sns.color_palette("Spectral", 16).as_hex()))

y1, y2, y3, y4 ,y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16 = [], [], [],[], [], [],[], [], [],[], [], [],[], [], [],[]
def animate(i):
    y1=liste[0,i]
    y2=liste[1,i]
    y3=liste[2,i]
    y4=liste[3,i]
    y5=liste[4,i]
    y6=liste[5,i]
    y7=liste[6,i]
    y8=liste[7,i]
    y9=liste[8,i]
    y10=liste[9,i]
    y11=liste[10,i]
    y12=liste[11,i]
    y13=liste[12,i]
    y14=liste[13,i]
    y15=liste[14,i]
    y16=liste[15,i]
    
    plt.bar(nt(base(np.arange(0, d, 1),10,4),N), [y1, y2, y3, y4,y5, y6,y7, y8,y9, y10,y11, y12,y13, y14,y15, y16], color=palette)

plt.title("Developement: ", color=("blue"))
ani = FuncAnimation(fig, animate, interval=100,frames=100)
ani.save('test3.gif', writer='Pillow')


# In[1]:


def base(a: int, von: int, zu: int):
    zahl = 0
    multiplikator = 1
    while a > 0:
        zahl += (a%zu)*multiplikator
        a=a//zu
        multiplikator*=von
    print(zahl)


# In[2]:


base(10,10,4)


# In[4]:


base(22,4,10)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




