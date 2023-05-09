import numpy as np
import scipy.optimize as opt
import scipy.misc as misc
from scipy.integrate import solve_ivp
import ctypes
import matplotlib.pyplot as plt
import numpy.random as rn
from numba import njit, jit


class solver:
    def __init__(self, N: int, initial_conditions: np.ndarray, r: float, eta: float, u: float, approximate_mutation: bool, max_peak: int, second_peak: int, mu_0: float):

        # length of sequence
        self.N = N
        # dimension of space
        self.D = 2 ** self.N

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
        a = bin(a)[2:].zfill(self.N)
        b = bin(b)[2:].zfill(self.N)
        h_ab = 0
        for i in range(self.N):
            h_ab += np.abs(int(a[i]) - int(b[i]))
        return h_ab
    #@njit
    def q_matrix_approximation(self):
        """q matrix for nearest neighbour mutations

        Returns:
            np.ndarray: q matrix in bit representation
        """
        q = np.zeros((2 ** self.N, 2 ** self.N))
        for i in range(2 ** self.N):
            for j in range(i+1, 2 ** self.N):
                if self.Hamming_dinstance(i, j) == 1:
                    q[i,j] = self.u
                    q[j,i] = self.u
            
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
        q = np.zeros((2 ** self.N, 2 ** self.N))
        for i in range(2 ** self.N):
            for j in range(i+1, 2 ** self.N):
                h = self.Hamming_dinstance(i, j)
                q[i,j] = self.u ** h * (1 - self.u) ** (self.N - h)
                q[j,i] = q[i, j]

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

        mu_bar = np.dot(self.mu, system)

        # loop over system
        for i in range(self.D):
            for j in range(self.D):
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

        return result

    def mu_wide(self, radius:int):
        """Function to implement wide mu region around eta peak

        Args:
            radius (int): radius of area
        """
        for i in range(self.D):
            if self.Hamming_dinstance(self.peak_eta, i) <= radius:
                self.mu[i] = self.eta * self.mu_0
        



if __name__ == "main":
    """
    Just plotting some stuff to see whether it looks reasonable.... looks reasonable....
    """
    N = 3
    d = 2 ** N
    in_con = np.ones(d) * 1 / d

    tmax = 100

    test = solver(N, in_con, 2, 1.9, 1e-2, True, 1, 5, 1)
    result = test.solve([0, tmax], 100)

    f = plt.subplots()
    plt.plot(result.t, result.y.T)
    plt.ylabel(r"$f$")
    plt.xlabel(r"$t$")
    plt.yscale("log")
    plt.legend(['%d' % i for i in np.arange(d)], shadow=True)
    plt.savefig("rn_1.pdf", dpi=500, bbox_inches="tight")

    print(test.q_matrix)

    f2 = plt.subplots()
    plt.bar(np.arange(0, d, 1), result.y[:,-1])
    plt.xlabel("Sequence")
    plt.ylabel(r"$f_\mathrm{final}$")
    plt.savefig("rn_2.pdf", dpi=500, bbox_inches="tight")

    f3 = plt.subplots()
    plt.bar(np.arange(0, d, 1), test.mu)
    plt.xlabel("Sequence")
    plt.ylabel(r"$\mu$")
    plt.savefig("rn_3.pdf", dpi=500, bbox_inches="tight")

    test2 = solver(N, in_con, 2, 1.9, 1e-1, False, 1, 5, 1)
    result2 = test.solve([0, tmax], 100)

    f = plt.subplots()
    plt.plot(result2.t, result2.y.T)
    plt.ylabel(r"$f$")
    plt.xlabel(r"$t$")
    plt.yscale("log")
    plt.legend(['%d' % i for i in np.arange(d)], shadow=True)
    plt.savefig("rn_4.pdf", dpi=500, bbox_inches="tight")

    print(test2.q_matrix)

    f2 = plt.subplots()
    plt.bar(np.arange(0, d, 1), result2.y[:,-1])
    plt.xlabel("Sequence")
    plt.ylabel(r"$f_\mathrm{final}$")
    plt.savefig("rn_5.pdf", dpi=500, bbox_inches="tight")

    f3 = plt.subplots()
    plt.bar(np.arange(0, d, 1), test2.mu)
    plt.xlabel("Sequence")
    plt.ylabel(r"$\mu$")
    plt.savefig("rn_6.pdf", dpi=500, bbox_inches="tight")

