import numpy as np
from scipy.special import gamma, hyp2f1
import matplotlib.pyplot as plt
import pandas as pd
import os

def f_trap_coupled(f, alpha, tspan, x0, h):
    """
    Solve a system of coupled fractional-order differential equations using a trapezoidal-type predictor-corrector method.

    This method approximates the solution to the system:
        D^α₁ x(t) = F1(t, x, y)
        D^α₂ y(t) = F2(t, x, y)
    with initial conditions x(t₀) = x₀ and y(t₀) = y₀, where D^α denotes the fractional derivative
    in the Caputo sense, and α₁, α₂ are the fractional orders.

    Args:
        f (dict): A dictionary containing the right-hand side functions with keys:
            - "F1": function representing dx/dt = F1(t, x, y)
            - "F2": function representing dy/dt = F2(t, x, y)
        alpha (tuple): A tuple (α₁, α₂) of fractional orders for x and y, respectively.
        tspan (tuple): Time interval (t₀, t_final) over which to solve the system.
        x0 (tuple): Initial conditions (x₀, y₀).
        h (float): Step size for time discretization.

    Returns:
        tuple: 
            - tvals (np.ndarray): Array of time points.
            - xvals (np.ndarray): Array of solution values at each time step, shape (N, 2),
            where column 0 is x(t) and column 1 is y(t).
    """
    N = np.ceil((tspan[-1] - tspan[0])/h) + 1
    N = np.int32(N)
    f1 = f["F1"]
    f2 = f["F2"]
    
    alpha_1 = alpha[0]
    alpha_2 = alpha[1]
    
    t0 = tspan[0]
    t1 = t0 + h
    
    x_0 = x0[0]
    y_0 = x0[1]
    f1_0 = f1(t0, x_0, y_0)
    f2_0 = f2(t0, x_0, y_0)
    
    x1_pred = x_0 + alpha_1 * h**alpha_1 * f1_0 / gamma(alpha_1 + 1)
    y1_pred = y_0 + alpha_2 + h**alpha_2 * f2_0 / gamma(alpha_2 + 1)
    
    f1_pred = f1(t1, x1_pred, y1_pred)
    f2_pred = f2(t1, x1_pred, y1_pred)
    
    x1_corr = x_0 + (1 + alpha_1) * ( (h**alpha_1 * f1_pred)/gamma(alpha_1 + 2) )
    
    y1_corr = y_0 + (1 + alpha_2) * ((h**alpha_1 * f2_pred)/gamma(alpha_2 + 2))
    
    fvals = np.zeros((N-1, 2), dtype = np.float64)
    svals = np.zeros((N-1, 2), dtype = np.float64)
    
    xvals = np.zeros((N, 2), dtype = np.float64)
    tvals = np.linspace(tspan[0] + h, tspan[-1], N, dtype = np.float64)
    
    xvals[0, 0] = x1_corr
    xvals[0, 1] = y1_corr
    
    
    i = 1
    for j in range(2, N+1):
        fvals[i-1, 0] = f1(tvals[i-1], xvals[i-1, 0], xvals[i-1, 1])
        fvals[i-1, 1] = f2(tvals[i-1], xvals[i-1, 0], xvals[i-1, 1])
        
        bx = (j-1)**(alpha_1 + 1) - (j - alpha_1 - 1)*j**alpha_1
        by = (j-1)**(alpha_2 + 1) - (j - alpha_2 - 1)*j**alpha_2
        
        svals[i-1, 0] = j**(alpha_1 + 1) - 2 * (j - 1)**(alpha_1 + 1) + (j - 2)**(alpha_1 + 1)
        svals[i-1, 1] = j**(alpha_2 + 1) - 2 * (j - 1)**(alpha_2 + 1) + (j - 2)**(alpha_2 + 1)
        
        
        s = np.sum(np.multiply(svals[:i+1, :], fvals[i::-1, :]), axis = 0)
        
        s1 = s[0]
        s2 = s[1]


        x_j = xvals[j-2, 0] + h**alpha_1/gamma(alpha_1 + 1) * f1(tvals[j-2], xvals[j-2, 0], xvals[j-2, 1])
        y_j = xvals[j-2, 1] + h**alpha_2/gamma(alpha_2 + 2) * f2(tvals[j-2], xvals[j-2, 0], xvals[j-2, 1])
        
        i += 1
        
        xvals[j-1, 0] = (
            h**alpha_1 / gamma(alpha_1 + 2) * bx * f1_0 +
            x_0 +
            h**alpha_1 / gamma(alpha_1 + 2) * s1 +
            h**alpha_1 / gamma(alpha_1 + 2) * f1(tvals[j-1], x_j, y_j)
        )

        xvals[j-1, 1] = (
            h**alpha_2 / gamma(alpha_2 + 2) * by * f2_0 +
            x_0 +
            h**alpha_2 / gamma(alpha_2 + 2) * s2 +
            h**alpha_2 / gamma(alpha_2 + 2) * f2(tvals[j-1], x_j, y_j)
        )

    tvals = np.concatenate(([t0], tvals))
    xvals = np.concatenate(([[x_0, y_0]], xvals))
    return tvals, xvals

def f_trap(f, alpha, tspan, x0, h):
    if len(alpha) != len(f.values()):
        raise ValueError("Incorrect Dimensions, the ith ODE of the system must correspond to an order alpha[i]")
    N = np.ceil((tspan[-1] - tspan[0])/h) + 1
    N = np.int32(N)
    f_list = list(f.values())
    alpha = np.array(alpha)
    dof = len(f_list)
    
    
    t0 = tspan[0]
    t1 = t0 + h
    
    
    f0 = np.array([f_list[_](t0, x0) for _ in range(dof)])

    
    pred = x0 + alpha * h**alpha * np.array([f_list[_](t0, x0) / gamma(alpha[_] + 1) for _ in range(dof)])
    
    
    f_pred_1 = np.array([f_list[_](t1, pred) for _ in range(dof)])
    
    corr_lambda = lambda i : x0[i] + (1 + alpha[i]) *  ( (h**alpha[i] * f_pred_1[i])/gamma(alpha[i] + 2) )
    
    corr_1 = np.array([corr_lambda(_) for _ in range(dof)])
    
    
    fvals = np.zeros((N-1, dof), dtype = np.float64)
    svals = np.zeros((N-1, dof), dtype = np.float64)
    
    xvals = np.zeros((N, dof), dtype = np.float64)
    tvals = np.linspace(tspan[0] + h, tspan[-1], N, dtype = np.float64)
    
    
    
    xvals[0,:] = corr_1
    
    b_lambda = lambda x : (j - 1)**(x + 1) - (j - x - 1)*j**x
    s_lambda = lambda x : j**(x + 1) - 2 * (j - 1)**(x + 1) + (j - 2)**(x + 1)
    i = 1
    for j in range(2, N+1):
        fvals[i-1, :] = np.array([f_list[_](tvals[i-1], xvals[i-1, :]) for _ in range(dof)])
        
        
        b_arr = b_lambda(alpha)
        
        
        svals[i-1, :] = s_lambda(alpha)
        
        s = np.sum(np.multiply(svals[:i+1, :], fvals[i::-1, :]), axis = 0)
        # if j == 5:
        #     print(svals[:i, :])
        #     print(fvals[i:1:-1, :])
        #     print("mult", np.multiply(svals[:i+1, :], fvals[i::-1, :]))
        #     print("s", s)
        #     import sys; sys.exit()
        pred_x = xvals[j-2, :] + h**alpha/gamma(alpha + 1) * np.array([f_list[_](tvals[j-2], xvals[j-2, :]) for _ in range(dof)])
        
        i += 1

        
        xvals[j-1, :] = h**alpha / gamma(alpha+2) * b_arr * f0 + x0 + h**alpha/gamma(alpha + 2) * s + h**alpha/gamma(alpha + 2) * np.array([f_list[_](tvals[j-1], pred_x) for _ in range(dof)])

    tvals = np.concatenate(([t0], tvals))
    xvals = np.concatenate(([x0], xvals))
    return tvals, xvals


A_lambda = lambda h, tau, k, j : -h**tau / tau * ((k - 1 - j) * h)**(1 - tau) * (
    -tau + (tau - 1) * hyp2f1(1, 1, 1 + tau, h / ((k + 1 - j)*h))
)


def f_adams_bashforth(g, tau, u0, J_AB, t_last, h):
    """
    Solve a fractional-order ODE of the form

        D^tau u(t) = g(t, u(t)),   u(0) = u0,   0 < tau <= 1

    using the generalised Adams-Bashforth (explicit) method with a
    linear-interpolation of the history load function
    
    The operator D is defined in the Caputo-sense. Algorithm is derived in: https://www.sciencedirect.com/science/article/pii/S0021999116300870

    Parameters
    ----------
    g : callable
        Right-hand side function g(t, u) of the fractional ODE.
    tau : float
        Fractional order of the derivative, 0 < tau <= 1.
    u0 : float
        Initial condition y(0).
    J_AB : int
        Order of the Adams-Bashforth scheme (0, 1, or 2).
    t_last : float
        End time of the integration interval [0, t_last].
    h : float
        Time step size.

    Returns
    -------
    t_vals : ndarray, shape (N,)
        Uniformly spaced time grid from 0 to t_last.
    u_vals : ndarray, shape (N,)
        Approximate solution at each time point.
    g_vals : ndarray, shape (N,)
        Values of g(t_k, u_k) at each time point.
    """
    gamma_reciprocals = np.array([1/gamma(i + tau) for i in range(1, 4)])
    N = np.ceil(t_last/h) + 1
    N = N.astype(int)
    t_vals = np.linspace(0, t_last, N)
    u_vals = np.zeros(N)
    g_vals = np.zeros(N)
    
    u_vals[0] = u0
    g_vals[0] = g(t_vals[0], u_vals[0])

    beta_AB = np.zeros(J_AB+1, dtype = np.float64)
    
    if J_AB == 0:
        beta_AB[0] = gamma_reciprocals[0]
        
    elif J_AB == 1:
        beta_AB[0] = gamma_reciprocals[1]
        beta_AB[1] = gamma_reciprocals[0] - gamma_reciprocals[1]
        
    elif J_AB == 2:
        beta_AB[0] = 0.5 * gamma_reciprocals[1] + gamma_reciprocals[2]
        beta_AB[1] = gamma_reciprocals[0] - 2 * gamma_reciprocals[2]
        beta_AB[2] = -0.5 * gamma_reciprocals[1] + gamma_reciprocals[2]
        
    
    
    for k in range(1, N):
        AB_sum = 0.0
        for j in range(J_AB + 1):
            if j > k:
                break
            AB_sum += beta_AB[j] * g_vals[k - 1 - j]
        
        load_k = 0
        for j in range(k-2):
            load_k += ((u_vals[j+1] - u_vals[j]) / h) * (A_lambda(h, tau, k, j) - A_lambda(h, tau, k, j + 1))
        
        load_k += ((u_vals[k-1] - u_vals[k-2]) / h) * (A_lambda(h, tau, k, k-2) - (-h * (tau - 1) * np.pi / np.sin(np.pi * tau)))
        load_k *= 1 / (gamma(tau) * gamma(2 - tau))
        u_vals[k] = u_vals[k-1] + h**tau * AB_sum - load_k


        g_vals[k] = g(t_vals[k], u_vals[k])
    
    return t_vals, u_vals, g_vals


def f_adams_bashforth_sys(g_dict, tau_list, u0_list, J_AB, t_last, h):
    N = np.ceil(t_last/h) + 1
    N = N.astype(int)
    
    dof = len(tau_list)
    
    gamma_reciprocals = np.zeros((dof, 3))
    for j in range(dof):
        gamma_reciprocals[j] = np.array([1/gamma(i + tau_list[j]) for i in range(1, 4)])
    
    t_vals = np.linspace(0, t_last, N)
    u_vals = np.zeros((N, dof))
    g_vals = np.zeros((N, dof))
    g_funcs = list(g_dict.values())
    u_vals[0, :] = np.array(u0_list, dtype = np.float64)
    g_vals[0, :] = np.array([g_funcs[i](t_vals[0], u_vals[0,:]) for i in range(dof)], dtype = np.float64)
    

    beta_AB = np.zeros((dof, J_AB+1), dtype = np.float64)
    
    if J_AB == 0:
        beta_AB[:, 0] = gamma_reciprocals[:, 0]
        
    elif J_AB == 1:
        beta_AB[:, 0] = gamma_reciprocals[:, 1]
        beta_AB[:, 1] = gamma_reciprocals[:, 0] - gamma_reciprocals[:, 1]
        
    elif J_AB == 2:
        beta_AB[:, 0] = 0.5 * gamma_reciprocals[:, 1] + gamma_reciprocals[:, 2]
        beta_AB[:, 1] = gamma_reciprocals[:, 0] - 2 * gamma_reciprocals[:, 2]
        beta_AB[:, 2] = -0.5 * gamma_reciprocals[:, 1] + gamma_reciprocals[:, 2]
        
    for k in range(1, N):
        for i in range(len(tau_list)):
            AB_sum = 0.0
            for j in range(J_AB + 1):
                if j > k:
                    break
                AB_sum += beta_AB[i][j] * g_vals[k - 1 - j, i]
            
            load_k = 0
            tau = tau_list[i]
            for j in range(k-2):
                load_k += ((u_vals[j+1][i] - u_vals[j][i]) / h) * (A_lambda(h, tau, k, j) - A_lambda(h, tau, k, j + 1))
                
            load_k += ((u_vals[k-1][i] - u_vals[k-2][i]) / h) * (A_lambda(h, tau, k, k-2) - (-h * (tau - 1) * np.pi / np.sin(np.pi * tau)))
            load_k *= 1 / (gamma(tau) * gamma(2 - tau))
            u_vals[k][i] = u_vals[k-1][i] + h**tau * AB_sum - load_k

        for i in range(len(tau_list)):
            g_vals[k, i] = g_funcs[i](t_vals[k], u_vals[k,:])
    
    return t_vals, u_vals, g_vals

