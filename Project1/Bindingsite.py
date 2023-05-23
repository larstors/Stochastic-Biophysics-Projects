import numpy as np
import scipy.optimize as opt
import scipy.misc as misc
from scipy.integrate import solve_ivp
import ctypes
import matplotlib.pyplot as plt
import numpy.random as rn
from numba import njit, jit
import seaborn as sns
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
def Prob_bound(seq,E_list):
    E=0
    for s in range(len(seq)):
        E+=E_list[s][seq[s]]
    a=np.exp(-E)
    return a/(1+a)

class solver:
    def __init__(self, N: int, initial_conditions: np.ndarray, u: float, approximate_mutation: bool, En_matrix):

        # length of sequence
        self.N = N
        # dimension of space
        self.D = 4 ** self.N

        # check dimensionality
        if len(initial_conditions) != self.D:
            raise TypeError('Number of initial conditions does not match dimension.')

        # set initial conditions
        self.initial_conditions = initial_conditions

        # mutation probability
        self.u = u

        # mutation matrix
        if approximate_mutation:
            self.q_matrix = self.q_matrix_approximation()
        else:
            self.q_matrix = self.q_matrix_full()
        self.en_matrix = En_matrix

        self.mu = np.ones(self.D)
        for i in range(self.D):
            j = base(i,10,4)
            m=str(nt(j,self.N))
            mu=Prob_bound(m,self.en_matrix)
            self.mu[i]=mu


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





if __name__ == "__main__":
    """
    Just plotting some stuff to see whether it looks reasonable.... looks reasonable....
    """
    N = 2
    d = 4 ** N
    in_con = np.ones(d) * 1 / d
    En_matrix=[{'A':0,'G':4,'C':4,'T':3},{'A':0,'G':4,'C':2,'T':1}]#,{'A':0,'G':2,'C':1,'T':1},{'A':3,'G':2,'C':1,'T':0}]
    tmax = 100
    test2 = solver(N, in_con, 0.1, True, En_matrix)
    print(test2)
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
