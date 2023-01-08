from scipy.optimize import linprog
from scipy.optimize import minimize
import numpy as np

# https://www.kaggle.com/code/vijipai/lesson-5-mean-variance-optimization-of-portfolios

def MaximizeReturns(MeanReturns, PortfolioSize):
    c = (np.multiply(-1, MeanReturns))
    A = np.ones([PortfolioSize, 1]).T
    b = [1]
    res = linprog(c, A_ub = A, b_ub = b, bounds = (0,1), method='simplex')

    return res


def MinimizeRisk(CovarReturns, PortfolioSize):

    def f(x, CovarReturns):
        func = np.matmul(np.matmul(x, CovarReturns), x.T)
        return func

    def constraintEq(x):
        A = np.ones(x.shape)
        b = 1
        constraintVal = np.matmul(A, x.T)-b
        return constraintVal
    
    xinit = np.repeat(0.1, PortfolioSize)
    cons = ({'type': 'eq', 'fun': constraintEq})
    lb = 0
    ub = 1
    bnds = tuple([(lb, ub) for x in xinit])

    opt = minimize(f, x0=xinit, argus=(CovarReturns), bounds=bnds, constraints=cons, tol=10**-3)

    return opt


def MinimizeRiskConstr(MeanReturns, CovarReturns, PortfolioSize, R):
    
    def  f(x, CovarReturns):
         
        func = np.matmul(np.matmul(x, CovarReturns ), x.T)
        return func

    def constraintEq(x):
        AEq = np.ones(x.shape)
        bEq = 1
        EqconstraintVal = np.matmul(AEq, x.T) - bEq 
        return EqconstraintVal
    
    def constraintIneq(x, MeanReturns, R):
        AIneq = np.array(MeanReturns)
        bIneq = R
        IneqconstraintVal = np.matmul(AIneq, x.T) - bIneq
        return IneqconstraintVal
    
    xinit=np.repeat(0.1, PortfolioSize)
    cons = ({'type': 'eq', 'fun':constraintEq},
            {'type':'ineq', 'fun':constraintIneq, 'args':(MeanReturns,R)})
    lb = 0
    ub = 1
    bnds = tuple([(lb, ub) for x in xinit])

    opt = minimize (f, args=(CovarReturns), method='trust-constr', x0=xinit, bounds=bnds, constraints=cons, tol=10**-3)

    return  opt