import numpy as np
from scipy.integrate import solve_ivp

import matplotlib.pyplot as plt

from abc import ABC, abstractmethod

## SOURCES
# https://archimede.dm.uniba.it/~testset/CWI_reports/testset2006r23.pdf
#  - Most ODE problems implemented
# https://www.unige.ch/~hairer/testset/testset.html

class test_problem(ABC):
    def __init__(self):
        self.times = None
        self.y0 = None
        self.yT = None

        self.yp = None

    @abstractmethod
    def dxdt(self, y, t):
        raise NotImplementedError


class chemical_akzo_nobel(test_problem):
    def __init__(self):
        self.times = [0, 180]    
    
        self.k_1 = 18.7
        self.k_2 = 0.58
        self.k_3 = 0.09
        self.k_4 = 0.42

        self.K = 34.4
        self.klA = 3.3
        self.p_co2 = 0.9
        self.H = 737

        self.y0 = np.array([0.437, 0.00123, 0, 0, 0, 0.367])
        self.yT = np.array([0.1161602274780192,
                           0.1119418166040848e-2,
                           0.1621261719785814,
                           0.3396981299297459e-2,
                           0.1646185108335055,
                           0.198953327594281])

        self.yp = np.zeros_like(self.y0)

    def dxdt(self, y, t):
        r1 = self.k_1 * y[0]**4 * y[1]**.5
        r2 = self.k_2 * y[2] * y[3]
        r3 = self.k_2/self.K * y[0] * y[4]
        r4 = self.k_3 * y[0] * y[3]**2
        r5 = self.k_4 * y[5]**2 * y[1]**.5
        F = self.klA * (self.p_co2 / self.H - y[1])

        self.yp[0] =  -2*r1 + r2 - r3   - r4
        self.yp[1] = -.5*r1             - r4 - .5*r5 + F
        self.yp[2] =     r1 - r2 + r3
        self.yp[3] =        - r2 + r3 - 2*r4
        self.yp[4] =          r2 - r3           + r5
        self.yp[5] =                             -r5

        return self.yp

class hires(test_problem):
    def __init__(self):
        self.times = [0, 321.8122]
        self.y0 = np.array([1, 0, 0, 0, 0, 0, 0, 0.0057])
        self.yT = np.array([0.7371312573325668e-3,
                            0.1442485726316185e-3,
                            0.5888729740967575e-4,
                            0.1175651343283149e-2,
                            0.2386356198831331e-2,
                            0.6238968252742796e-2,
                            0.2849998395185769e-2,
                            0.2850001604814231e-2])

        self.yp = np.zeros_like(self.y0)

    def dxdt(self, y, t):
        self.yp[0] = -1.71*y[0] + 0.34*y[1] + 8.32*y[2] + 0.0007
        self.yp[1] = 1.71*y[0] - 8.75 * y[1]
        self.yp[2] = -10.03*y[2] + 0.43*y[3] + 0.035*y[4]
        self.yp[3] = 8.32*y[1] + 1.71*y[2] - 1.12*y[3]
        self.yp[4] = -1.745*y[4] + 0.43*y[5] + 0.43*y[6]
        self.yp[5] = -280 * y[5]*y[7] + 0.69*y[3] + 1.71*y[4] - 0.43*y[5] + 0.69*y[6]
        self.yp[6] = 280 * y[5]*y[7] - 1.81*y[6]
        self.yp[7] = -280  * y[5]*y[7] + 1.81*y[6]

        return self.yp

class pollutions(test_problem):
    def __init__(self):
        self.times = [0, 60.]
        self.y0 = np.array([0, 0.2, 0, 0.04, 0, 0, 0.1, 0.3, 0.01, 0, 0, 0, 0, 0, 0, 0, 0.007, 0, 0, 0])
        self.yT = np.array([0.5646255480022769e-1,
                            0.1342484130422339,
                            0.4139734331099427e-8,
                            0.5523140207484359e-2,
                            0.2018977262302196e-6,
                            0.1464541863493966e-6,
                            0.7784249118997964e-1,
                            0.3245075353396018,
                            0.7494013383880406e-2,
                            0.1622293157301561e-7,
                            0.1135863833257075e-7,
                            0.2230505975721359e-2,
                            0.2087162882798630e-3,
                            0.1396921016840158e-4,
                            0.8964884856898295e-2,
                            0.4352846369330103e-17,
                            0.6899219696263405e-2,
                            0.1007803037365946e-3,
                            0.1772146513969984e-5,
                            0.5682943292316392e-4])

        self.yp = np.zeros_like(self.y0)

        self.k = np.array([0.350,
                           0.266e2,
                           0.123e5,
                           0.860e-3,
                           0.820e-3,
                           0.150e5,
                           0.130e-3,
                           0.240e5,
                           0.165e5,
                           0.900e4,
                           0.220e-1,
                           0.120e5,
                           0.188e1,
                           0.163e5,
                           0.480e7,
                           0.350e-3,
                           0.175e-1,
                           0.100e9, 
                           0.444e12,
                           0.124e4,
                           0.210e1,
                           0.578e1,
                           0.474e-1,
                           0.178e4,
                           0.312e1])

    def dxdt(self, y, t):
        k = self.k
        r = np.zeros_like(k)
        
        r[0 ] = k[0 ] * y[0 ] 
        r[1 ] = k[1 ] * y[1 ] * y[ 3]
        r[2 ] = k[2 ] * y[4 ] * y[ 1]
        r[3 ] = k[3 ] * y[6 ]     
        r[4 ] = k[4 ] * y[6 ]     
        r[5 ] = k[5 ] * y[6 ] * y[ 5]
        r[6 ] = k[6 ] * y[8 ]    
        r[7 ] = k[7 ] * y[8 ] * y[ 5]
        r[8 ] = k[8 ] * y[10] * y[ 1]
        r[9 ] = k[9 ] * y[10] * y[ 0]
        r[10] = k[10] * y[12]    
        r[11] = k[11] * y[9 ] * y[ 1]
        r[12] = k[12] * y[13]    
        r[13] = k[13] * y[0 ] * y[ 5]
        r[14] = k[14] * y[2 ]    
        r[15] = k[15] * y[3 ]    
        r[16] = k[16] * y[3 ]     
        r[17] = k[17] * y[15]   
        r[18] = k[18] * y[15]     
        r[19] = k[19] * y[16] * y[ 5]
        r[20] = k[20] * y[18]     
        r[21] = k[21] * y[18]     
        r[22] = k[22] * y[0 ] * y[ 3]
        r[23] = k[23] * y[18] * y[ 0]
        r[24] = k[24] * y[19] 

        self.yp[0 ] = -r[0]-r[9]-r[13] - r[22] - r[23] + r[1]+r[2]+r[8]+r[10]+r[11]+r[21]+r[23]
        self.yp[1 ] = -r[1 ] -   r[2 ] - r[8 ] - r[11] + r[0 ] +   r[20]
        self.yp[2 ] = -r[14] +   r[0 ] + r[16] + r[18] + r[21]
        self.yp[3 ] = -r[1 ] -   r[15] - r[16] - r[22] + r[14]
        self.yp[4 ] = -r[2 ] + 2*r[3 ] + r[5 ] + r[6 ] + r[12] +   r[19]
        self.yp[5 ] = -r[5 ] -   r[7 ] - r[13] - r[19] + r[2 ] + 2*r[17]
        self.yp[6 ] = -r[3 ] -   r[4 ] - r[5 ] + r[12]
        self.yp[7 ] =  r[3 ] +   r[4 ] + r[5 ] + r[6 ]
        self.yp[8 ] = -r[6 ] -   r[7 ]
        self.yp[9 ] = -r[11] +   r[6 ] + r[8]
        self.yp[10] = -r[8 ] -   r[9 ] + r[7 ] + r[10]
        self.yp[11] =  r[8 ]              
        self.yp[12] = -r[10] +   r[9 ]            
        self.yp[13] = -r[12] +   r[11]            
        self.yp[14] =  r[13]                      
        self.yp[15] = -r[17] -   r[18] + r[15]    
        self.yp[16] = -r[19]                      
        self.yp[17] =  r[19]                      
        self.yp[18] = -r[20] -   r[21] - r[23] + r[22] + r[24]
        self.yp[19] = -r[24] +   r[23]

        return self.yp

#Verify correctness - radau, rk45 cant handle it
class ring_modulator(test_problem):
    def __init__(self):
        self.y0 = np.zeros(15)
        self.yT = np.array([-0.17079903291846e-1,
                            -0.66609789784834e-2,
                            0.27531919254370,
                            -0.39115731811511,
                            -0.38851730770493,
                            0.27795920295388,
                            0.11146002811043,
                            0.29791296267403e-6,
                            -0.31427403451731e-7,
                            0.70165883118556e-3,
                            0.85207537676917e-3,
                            -0.77741454302426e-3,
                            -0.77631966493048e-3,
                            0.78439425971261e-4,
                            0.25232278361831e-4])
        self.yp = np.zeros_like(self.y0)

        self.times = [0, 1e-3]

        self.C = 1.6e-8
        self.Cs = 2e-12
        self.Cp = 1e-8
        self.Lh = 4.45
        self.Ls1 = 0.002
        self.Ls2 = 5e-4
        self.Ls3 = 5e-4
        self.gamma = 40.67286402e-9
        self.R = 25000
        self.Rp = 50
        self.Rg1 = 36.3
        self.Rg2 = 17.3
        self.Rg3 = 17.3
        self.Ri = 50
        self.Rc = 600
        self.delta = 17.7493332

    def dxdt(self, y, t):
        Uin1 = 0.5 * np.sin(2000*np.pi*t)
        Uin2 = 2*np.sin(2000*np.pi*t)

        UD1 = y[2] - y[4] - y[6] - Uin2
        UD2 = -y[3] + y[5] - y[6] - Uin2
        UD3 = y[3] + y[4] + y[6] + Uin2
        UD4 = -y[2] - y[5] + y[6] + Uin2

        q = lambda U: self.gamma * (np.exp(self.delta*U)  - 1)

        self.yp[0 ] = self.C**-1*(y[7] - 0.5*y[9] + 0.5*y[10] + y[13] - self.R**-1*y[0])
        self.yp[1 ] = self.C**-1*(y[8] - 0.5*y[11] + 0.5*y[12] + y[14] - self.R**-1*y[1])
        self.yp[2 ] = self.Cs**-1*(y[9] - q(UD1) + q(UD4))
        self.yp[3 ] = self.Cs**-1*(-y[10] + q(UD2) - q(UD3))
        self.yp[4 ] = self.Cs**-1*(y[11] + q(UD1) - q(UD3))
        self.yp[5 ] = self.Cs**-1*(-y[12] - q(UD2) + q(UD4))
        self.yp[6 ] = self.Cp**-1*(-self.Rp**-1*y[6] + q(UD1) + q(UD2) - q(UD3) - q(UD4))
        self.yp[7 ] = -self.Lh**-1*y[0]
        self.yp[8 ] = -self.Lh**-1*y[1]
        self.yp[9 ] = self.Ls2**-1*(0.5*y[0] - y[2] - self.Rg2*y[9])
        self.yp[10] = self.Ls3**-1*(-0.5*y[0] + y[3] - self.Rg3*y[10])
        self.yp[11] = self.Ls2**-1*(0.5*y[1] - y[4] - self.Rg2*y[11])
        self.yp[12] = self.Ls3**-1*(-0.5*y[1] + y[5] - self.Rg3*y[12]) 
        self.yp[13] = self.Ls1**-1*(-y[0] + Uin1 - (self.Ri+self.Rg1)*y[13])
        self.yp[14] = self.Ls1**-1*(-y[1] - (self.Rc + self.Rg1)*y[14])

        return self.yp

#TODO
class medical_akzo_nobel(test_problem):
    def __init__(self):
        raise NotImplementedError

#TODO
class emep(test_problem):
    def __init__(self):
        raise NotImplementedError

class pleiades(test_problem):
    def __init__(self):
        self.y0 = np.array([3, 3, -1, -3, 2, -2, 2,
                            3, -3, 2, 0, 0, -4, 4,
                            0, 0, 0, 0, 0, 1.75, -1.5,
                            0, 0, 0, -1.25, 1, 0, 0])
        self.yT = np.array([0.3706139143970502,
                            0.3237284092057233e1,
                            -0.3222559032418324e1,
                            0.6597091455775310,
                            0.3425581707156584,
                            0.1562172101400631e1,
                            -0.7003092922212495,
                            -0.3943437585517392e1,
                            -0.3271380973972550e1,
                            0.5225081843456543e1,
                            -0.2590612434977470e1,
                            0.1198213693392275e1,
                            -0.2429682344935824,
                            0.1091449240428980e1,
                            0.3417003806314313e1,
                            0.1354584501625501e1,
                            -0.2590065597810775e1,
                            0.2025053734714242e1,
                            -0.1155815100160448e1,
                            -0.8072988170223021,
                            0.5952396354208710,
                            -0.3741244961234010e1,
                            0.3773459685750630,
                            0.9386858869551073,
                            0.3667922227200571,
                            -0.3474046353808490,
                            0.2344915448180937e1,
                            -0.1947020434263292e1])
        self.yp = np.zeros_like(self.y0)

        self.times = [0, 3]

    def dxdt(self, y, t):
        self.yp *= 0

        self.yp[:14] = y[14:]

        for i in range(7):
            for j in range(7):
                if i != j:
                    rij = ((y[i] - y[j])**2 + (y[i + 7] - y[j + 7])**2)**(3/2)

                    self.yp[i + 14] += (j+1)*((y[j] - y[i]))/rij
                    self.yp[i + 21] += (j+1)*((y[j+7] - y[i+7]))/rij

        return self.yp

class van_der_pol(test_problem):
    def __init__(self):
        self.y0 = np.array([2.0086198608748431365094, 0])
        self.yT = np.array([2.0086198608748431365094, 0])
        self.yp = np.zeros_like(self.y0)

        self.times = [0, 6.663286859323130]

        self.mu = 1.

    def dxdt(self, y, t):
        self.yp[0] = y[1]
        self.yp[1] = self.mu*(1-y[0]**2)*y[1]-y[0]

        return self.yp

class harmonic(test_problem):
    def __init__(self):
        self.y0 = np.array([-1, 1])
        self.yT = np.array([-1, 1])
        self.yp = np.zeros_like(self.y0)

        self.times = [0, 2*np.pi]

    def dxdt(self, y, t):
        self.yp[0] = y[1]
        self.yp[1] = y[0]

        return self.yp

class arenstorf(test_problem):
    def __init__(self):
        self.y0 = np.array([0.994, 0, 0, -2.0015851063790855224])
        self.yT = np.array([0.994, 0, 0, -2.0015851063790855224])
        self.yp = np.zeros_like(self.y0)

        self.mu = 0.012277471
        self.times = [0, 17.06521656015796]

    def dxdt(self, y, t):
        d1 = ((y[0] + self.mu)**2 + y[2]**2)**(3/2)
        d2 = ((y[0] - (1 - self.mu))**2 + y[2]**2)**(3/2)

        self.yp[0] = y[1]
        self.yp[1] = y[0] + 2*y[3] - (1-self.mu)*(y[0] + self.mu)/d1 - self.mu*(y[0] - (1 - self.mu))/d2
        self.yp[2] = y[3]
        self.yp[3] = y[2] - 2*y[1] - (1-self.mu)*y[2]/d1 - self.mu*y[2]/d2

        return self.yp

class rober(test_problem):
    def __init__(self):
        self.y0 = np.array([1.0, 0, 0])
        self.yT = np.array([0.2083340149701284E-07,
                            0.8333360770334744E-13,
                            0.9999999791665152E+00])
        self.yp = np.zeros_like(self.y0)

        self.times = [0, 1e11]

    def dxdt(self, y, t):
        self.yp[0] = -.04*y[0] + 1e4*y[1]*y[2]
        self.yp[2] = 3e7*y[1]**2
        self.yp[1] = -self.yp[0]-self.yp[2]

        return self.yp

def main():
    #problem = chemical_akzo_nobel()
    #problem = hires()
    #problem = arenstorf()
    #problem = harmonic()
    #problem = pleiades()
    #problem = van_der_pol()
    #problem = pollutions()
    #problem = ring_modulator()
    problem = rober()

    sol = solve_ivp(lambda t, y: problem.dxdt(y, t),
                 problem.times,
                 problem.y0,
                 'Radau',
                 #t_eval=np.linspace(problem.times[0], problem.times[1], 10000),
                 rtol=1e-10,
                 atol=1e-10)

    traj = sol.y
    t = sol.t

    print(traj[:, -1] - problem.yT)

    #fig = plt.figure()
    #for i in range(3):
    #    ax = fig.add_subplot(3, 5, i+1)
    #    ax.plot(t, traj[i, :])

    #for i in range(7):
    #    plt.plot(traj[i, :], traj[i+7, :])
    
    #plt.plot(traj[0, :], traj[2, :])
    
    #plt.show()

if __name__ == "__main__":
    main()
