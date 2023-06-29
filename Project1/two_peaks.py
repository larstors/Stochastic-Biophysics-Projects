import numpy as np
import scipy.optimize as opt
import scipy.misc as misc
from scipy.integrate import solve_ivp
import ctypes
import matplotlib.pyplot as plt
import numpy.random as rn
from numba import njit, jit
from mpl_toolkits.axes_grid1 import make_axes_locatable

class solver:
    def __init__(
        self,
        N: int,
        initial_conditions: np.ndarray,
        r: float,
        eta: float,
        u: float,
        approximate_mutation: bool,
        max_peak: int,
        second_peak: int,
        mu_0: float,
    ):

        # length of sequence
        self.N = N
        # dimension of space
        self.D = 2**self.N

        # check dimensionality
        if len(initial_conditions) != self.D:
            raise TypeError("Number of initial conditions does not match dimension.")

        # set initial conditions
        self.initial_conditions = initial_conditions

        # fittest peak
        self.r = r

        if eta > r:
            raise TypeError("Height of secondary peak greater than main peak.")
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

    # @njit
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
            # bitwise difference makes Hamming distance (does not work for anything other than bit representation)
            h_ab += np.abs(int(a[i]) - int(b[i]))
        return h_ab

    def q_matrix_approximation(self):
        """q matrix for nearest neighbour mutations

        Returns:
            np.ndarray: q matrix in bit representation
        """

        q = np.zeros((2**self.N, 2**self.N))
        # off diagonals
        for i in range(2**self.N):
            for j in range(i + 1, 2**self.N):
                if self.Hamming_dinstance(i, j) == 1:
                    q[i, j] = self.u
                    q[j, i] = self.u

        # diagonal terms
        for i in range(self.D):
            s = 0
            for j in range(self.D):
                if j != i:
                    s += q[i, j]
            q[i, i] = 1 - s
        return q

    # @njit
    def q_matrix_full(self):
        """q matrix for full mutation scenario

        Returns:
            np.ndarray: q matrix in bit representation
        """
        q = np.zeros((2**self.N, 2**self.N))
        # off diagonals
        for i in range(2**self.N):
            for j in range(i + 1, 2**self.N):
                h = self.Hamming_dinstance(i, j)
                q[i, j] = self.u**h * (1 - self.u) ** (self.N - h)
                q[j, i] = q[i, j]
        # diagonal terms
        for i in range(self.D):
            s = 0
            for j in range(self.D):
                if j != i:
                    s += q[i, j]
            q[i, i] = 1 - s

        return q

    def system_equation(self, t: float, system: np.ndarray):
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

        result = solve_ivp(
            fun=self.system_equation,
            t_span=time_span,
            y0=self.initial_conditions,
            t_eval=time,
        )

        return result

    def mu_wide(self, radius: int):
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
    N = 8
    d = 2**N
    in_con = np.ones(d) * 1 / d

    tmax = 50

    main_peak = 1
    second_peak = d - 1

    test2 = solver(N, in_con, 2, 1.9, 1e-1, True, main_peak, second_peak, 1)

    kappa = test2.Hamming_dinstance(main_peak, second_peak) - 1

    omega = np.arange(0, kappa + 1)

    eta = np.linspace(1, 1.99, 20, endpoint=True)

    F = np.zeros((len(omega), len(eta)))
    A = np.zeros((len(omega), len(eta)))
    # inefficient way of doing this...
    for i in range(len(omega)):
        for j in range(len(eta)):
            print("omega = %d\t eta = %f" % (omega[i], eta[j]))
            test2.eta = eta[j]
            test2.mu_wide(omega[i])
            result2 = test2.solve([0, tmax], 100)
            F[i, j] = test2.order_parameter_F(result2.y[:, -1])
            A[i, j] = test2.order_parameter_A(result2.y[:, -1])

    y, x = np.meshgrid(omega + 0.5 * np.ones_like(omega), eta)

    fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(10, 5))

    c1 = ax[0].pcolormesh(x, y, F.T)
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(c1, cax=cax)
    c2 = ax[1].pcolormesh(x, y, A.T)
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(c2, cax=cax)
    

    ax[0].set_title(r"$F$")
    ax[1].set_title(r"$A$")
    ax[0].set_xlabel(r"$\eta$")
    ax[1].set_xlabel(r"$\eta$")
    ax[0].set_ylabel(r"$\omega$")

    plt.savefig("heatmap.pdf", dpi=500, bbox_inches="tight")
