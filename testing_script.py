'''
to run:

python3 testing_scirpt.py [length of test in seconds]
'''



import numpy as np
import scipy
import time
import random
from random import sample
from scipy import optimize, fft
import math
import sys

class runScipy(object):
    def __init__(self):
        super(runScipy, self).__init__()
        # set functions here
        self.funcs = [self.arrayMath,self.funcMinimize,self.funcOptimize, self.roots]


    def callSoft(self,height,width):
        print("scipy")
        func_calls = random.randint(1,len(self.funcs))
        arr = self.createArray(height,width)
        funcs_to_run = sample(self.funcs,func_calls)
        for i in funcs_to_run:
            # i should be a function call
            i(arr)
        print("scipy")

    def createArray(self,h,w):
        return np.random.rand(h,w)

    def f(self,x):
        return (x - 2) * x * (x + 2)**2

    def fun(self,x):
        return [x[0]  + 0.5 * (x[0] - x[1])**3 - 1.0, 0.5 * (x[1] - x[0])**3 + x[1]]

    def jac(self,x):
        return np.array([[1 + 1.5 * (x[0] - x[1])**2, -1.5 * (x[0] - x[1])**2], [-1.5 * (x[1] - x[0])**2, 1 + 1.5 * (x[1] - x[0])**2]])

    def roots(self, arr=None):
        x0 = arr.flatten()
        sol = optimize.root(self.fun, [0, 0], jac=self.jac, method='hybr')
        sol = optimize.root(self.fun, [0, 0], jac=self.jac, method='lm')
        sol = optimize.root(self.fun, [0, 0], method='broyden1')
        sol = optimize.root(self.fun, [0, 0], method='broyden2')
        sol = optimize.root(self.fun, [0, 0], method='anderson')
        sol = optimize.root(self.fun, [0, 0], method='linearmixing')
        sol = optimize.root(self.fun, [0, 0], method='diagbroyden')
        sol = optimize.root(self.fun, [0, 0], method='excitingmixing')
        sol = optimize.root(self.fun, [0, 0], method='krylov')
        sol = optimize.root(self.fun, [0, 0], method='df-sane')

    def arrayMath(self, arr=None):
        x = fft.fft(arr)
        x = fft.ifft(arr)

    def funcMinimize(self, arr=None):
        res = optimize.minimize_scalar(self.f)
        # x = res.x
        res = optimize.minimize_scalar(self.f, bounds=(-3, -1), method='bounded')
        # x = res.x
        res = optimize.minimize_scalar(self.f, method='brent')
        # x = res.x
        res = optimize.minimize_scalar(self.f, method='golden')
        # x = res.x

    def funcOptimize(self, arr=None):
        from scipy.optimize import minimize, rosen, rosen_der
        x0 = arr.flatten()
        res = minimize(rosen, x0, method='Nelder-Mead', tol=1e-6)
        # res.x
        res = minimize(rosen, x0, method='Powell', tol=1e-6)
        # res.x
        res = minimize(rosen, x0, method='CG', tol=1e-6)
        # res.x
        res = minimize(rosen, x0, method='BFGS', tol=1e-6)
        # res.x
        # res = minimize(rosen, x0, method='Newton-CG', tol=1e-6)
        # res.x
        res = minimize(rosen, x0, method='L-BFGS-B', tol=1e-6)
        # res.x
        res = minimize(rosen, x0, method='TNC', tol=1e-6)
        # res.x
        res = minimize(rosen, x0, method='COBYLA', tol=1e-6)
        # res.x
        res = minimize(rosen, x0, method='SLSQP', tol=1e-6)
        # res.x
        res = minimize(rosen, x0, method='trust-constr', tol=1e-6)
        # res.x
        # res = minimize(rosen, x0, method='dogleg', tol=1e-6)
        # res.x
        # res = minimize(rosen, x0, method='trust-ncg', tol=1e-6)
        # res.x
        # res = minimize(rosen, x0, method='trust-exact', tol=1e-6)
        # res.x
        # res = minimize(rosen, x0, method='trust-krylov', tol=1e-6)
        # res.x



class runNumpy(object):
    def __init__(self):
        super(runNumpy, self).__init__()
        # set functions here
        self.funcs = [self.arrayMath,self.moreMath,self.arrayTrig,self.arrayAggregate,self.arraySort,self.arrayManipulation]


    def callSoft(self,height,width):
        print("numpy")
        func_calls = random.randint(1,len(self.funcs))
        arr = self.createArray(height,width)
        funcs_to_run = sample(self.funcs,func_calls)
        for i in funcs_to_run:
            # i should be a function call
            i(arr)
        print("numpy")

    def createArray(self,h,w):
        return np.random.rand(h,w)

    def arrayMath(self, arr=None):
        x = np.subtract(arr,arr)
        x = np.add(arr,arr)
        x = np.divide(arr,arr)
        x = np.multiply(arr,arr)
        a = np.linspace(0,arr.shape[0], arr.shape[0])
        a = np.linspace(0,arr.shape[1], arr.shape[1])

    def moreMath(self, arr=None):
        x = np.exp(arr)
        x = np.sqrt(arr)

    def arrayTrig(self, arr=None):
        x = np.sin(arr)
        x = np.cos(arr)
        x = np.log(arr)

    def arrayAggregate(self, arr=None):
        x = np.sum(arr)
        x = np.amin(arr)
        x = np.amax(arr)
        x = np.cumsum(arr, axis=0)
        x = np.mean(arr)
        x = np.median(arr)
        x = np.corrcoef(arr)
        x = np.std(arr)

    def arraySort(self, arr=None):
        x = np.sort(arr)
        x = np.sort(arr, axis=0)

    def arrayManipulation(self, arr=None):
        x = np.transpose(arr)
        x = np.ravel(arr)
        x = arr.reshape(arr.shape[0], (-1 * arr.shape[1]))
        x = np.concatenate((arr, arr), axis=0)
        x = np.vstack((arr, arr))
        x = np.hstack((arr, arr))
        x = np.column_stack((arr, arr))

class runPython(object):
    def __init__(self):
        super(runPython, self).__init__()
        # set functions here
        self.funcs = [self.arrayMath,self.arrayCat,self.arrayTotals] #self.moreMath,self.pythonTrig


    def callSoft(self,height,width):
        func_calls = random.randint(1,len(self.funcs))
        arr = self.createArray(height,width)
        funcs_to_run = sample(self.funcs,func_calls)
        print("python")
        for i in funcs_to_run:
            # i should be a function call
            i(arr)
        print("python")


    def createArray(self,h,w):
        arr = []
        for h_count in range(h):
            inner_arr = []
            for w_count in range(w):
                inner_arr.append(random.random())
            arr.append(inner_arr)
        return arr

    def arrayMath(self, arr=None):
        for i in arr:
            for j in i:
                x = j + j
                x = j - j
                x = j * j
                x = j / j

    def arrayCat(self, arr=None):
        for i in arr:
            x = i + i

    def arrayTotals(self,arr=None):
        total = 0
        sub_total = 0
        for i in arr:
            for j in i:
                total+=j
                sub_total-=j

    # def moreMath(self, arr=None):
    #     for i in arr:
    #         for j in i:
    #             x = math.ceil(j)
    #             x = math.copsign(j,-1*j)
    #             x = math.fabs(j)
    #             x = math.factorial(j)
    #             x = math.floor(j)
    #             x = math.fmod(j,j//2)
    #             x = math.frexp(j)
    #             x = math.isfinite(j)
    #             x = math.ininf(j)
    #             x = math.isnan(j)
    #             x = math.idexp(j)
    #             x = math.modf(j)
    #             x = math.trunc(j)
    #             x = math.exp(j)
    #             x = math.expm1(j)
    #             x = math.log1p(j)
    #             x = math.log2(j)
    #             x = math.log10(j)
    #             x = math.pow(j,4)
    #             x = math.sqrt(j)

    # def pythonTrig(self, arr=None):
    #     for i in arr:
    #         for j in i:
    #             print("python")
    #             x = math.acos(j)
    #             x = math.asin(j)
    #             x = math.atan(j)
    #             x = math.cos(j)
    #             x = math.sin(j)
    #             x = math.tan(j)
    #             x = math.degrees(j)
    #             x = math.radians(j)
    #             x = math.acosh(j)
    #             x = math.asinh(j)
    #             x = math.atanh(j)
    #             x = math.cosh(j)
    #             x = math.sinh(j)
    #             x = math.tanh(j)
    #             x = math.erf(j)
    #             x = math.erfc(j)
    #             x = math.gamma(j)
    #             x = math.lgamma(j)


modes = [runNumpy(), runScipy(), runPython()]

start = time.time()

length = int(sys.argv[-1])
max_dim = 100

# loop for "length" seconds
while((time.time() - start) < length):


    software = sample(modes,1)[0]
    heights, widths = random.randint(2,max_dim), random.randint(2,max_dim)

    for h in range(1,heights):
        for w in range(1,widths):
            software.callSoft(h,w)
