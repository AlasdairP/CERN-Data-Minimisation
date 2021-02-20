# -*- coding: utf-8 -*-
"""
DAH Project - Study of B+- mesons.
This script uses two Gaussians and an exponential to model the data.
"""

#Imports
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.signal import find_peaks
import scipy.optimize
import math

"""
This function creates a composite function consisting of
    a two Gaussians and an exponential. 
    
param x: the independent variable, the mass values of the bins.
param sigma1: the standard deviation of the first Gaussian.
param sigma2: the standard deviation of the second Gaussian.
param mu: the mean of the Gaussians. They are fixed to be the same.
param A: the amplitude of the exponential.
param rho: the decay constant of the exponential.
returns y: the resulting function.
    
Note: I realise it is bad practise to hard-code the amplitude of the Gaussian,
    but scipy.optimize.curve_fit does not seem able to take fixed parameters.
"""

def gauss_exp(x, sigma1, sigma2, mu, A, rho):
    
    y = []
    
    for xi in x:
        
        pref1 = (19594/2)*(1/(sigma1*np.sqrt(2*np.pi)))
        gauss1 = pref1*np.exp((-1/2)*((xi - mu)/sigma1)**2)
        
        pref2 = (19594/2)*(1/(sigma2*np.sqrt(2*np.pi)))
        gauss2 = pref2*np.exp((-1/2)*((xi - mu)/sigma2)**2)
        
        exp = A*np.exp(-xi/rho)
        
        yi = gauss1 + gauss2 + exp
        y.append(yi)
        
    return y


def main():
    
    # Import data
    with open("kdata-small.bin") as f:
        datalist = np.fromfile(f, dtype=np.float32)
    
    # Number of events (each event has 5 observables)
    nevent = int(len(datalist)/5)
    
    #xdata is a list where each element is an event, itself a list of the 5 observables
    xdata = np.split(datalist, nevent)    

    
    # Make list of invariant mass of events
    xmass = []
    for i in range(nevent):
        xmass.append(xdata[i][0])
         
    # Choose number of bins
    bins = 150    

    # Create histogram of masses
    n, bin_edges, patches = plt.hist(xmass, bins)
    
    """
    For the fit, the x value corresponding to the number of counts
    in a bin should be the centre of that bin.
    """
    
    centres = []
    for i in range(bins):
        
        centre = (bin_edges[i] + bin_edges[i+1])/2
        centres.append(centre)     
    
    # Rename as x and y for simplicity
    x = centres
    y = n
    
    # Initial guesses for the fit 
    sigma1 = 4
    sigma2 = 8
    mu = 5279
    rho = 825
    A = 32000
    
    # Fit using Scipy's Curve Fit
    fit = scipy.optimize.curve_fit(gauss_exp, xdata=x, ydata=y, p0=[sigma1, sigma2, mu, A, rho])

    # Fitted parameters
    sigma1 = fit[0][0]
    sigma2 = fit[0][1]
    mu = fit[0][2]
    A = fit[0][3]
    rho = fit[0][4]
    
    # Retrieve errors from fit
    perr = np.sqrt(np.diag(fit[1]))
    
    # Plot fitted curve on top of original histogram
    y_fitted = gauss_exp(x, sigma1, sigma2, mu, A, rho)
    
    plt.plot(x, y_fitted, color='red')
    plt.xlabel("Mass (MeV/c^2)")
    plt.ylabel("Number  of events")
    plt.title("Histogram of mass")
    plt.show()
    
    #### Residuals ####
    
    # Calculate errors as the Poissonic errors on counts in a histogram bin.
    error_bars = np.sqrt(y)
    
    # Calculate residuals as the difference between the fit and the histogram.
    residuals = y_fitted - y
    
    # Calculate how many error bars cross the x axis
    crosses_x = 0
    for i in range(bins-1):
        
        if residuals[i] > 0:
            if residuals[i] - error_bars[i] < 0:
                crosses_x += 1     
        else:
            if residuals[i] + error_bars[i] > 0:
                crosses_x += 1             
    
    # Plot residuals
    plt.plot(x, residuals, 'bo', markersize=4)
    plt.errorbar(x, residuals, yerr=error_bars, fmt='none', ecolor='red', elinewidth=1)
    plt.title("Residuals")
    plt.ylabel("Residual (counts)")
    plt.xlabel("Mass (MeV/c^2)")
    plt.show()
    
    #### Chi-squared ####
    
    chi_sq, p = scipy.stats.chisquare(y, y_fitted)
    
    # Reduced chi-squared
    fit_params = 5
    dof = bins - fit_params
    reduced_chi_sq = chi_sq/dof
    
    #### Combining the two sigmas ####
    
    sigma = np.sqrt((1/2)*sigma1**1 + (1/2)*sigma2**2)
    
    ######### Write output data to text file ##########
    
    outfile = open("data_2gaussians", "w")
    
    outfile.write("Output data for B+- combined. \n" + 
        "Function used in fit: Two Gaussians + exponential \n")
    
    outfile.write("sigma1 = " + str(fit[0][0]) + "\n" +
        "sigma2 = " + str(fit[0][1]) + "\n" +
        "mu = " + str(fit[0][2]) + "\n" +
        "A = " + str(fit[0][3]) + "\n" +
        "rho = " + str(fit[0][4]) + "\n")
    
    outfile.write("sigma1 error = " + str(perr[0]) + "\n" +
        "sigma1 error = " + str(perr[1]) + "\n" +
        "mu error = " + str(perr[2]) + "\n" +
        "A error = " + str(perr[3]) + "\n" +
        "rho error = " + str(perr[4]) + "\n") 
    
    outfile.write("Combined sigma = " + str(sigma) + "\n")
    outfile.write(str(crosses_x) + "/" + str(bins) + " error bars cross the x axis \n")
    outfile.write("Chi-squared = " + str(chi_sq) + "\n")
    outfile.write("Reduced chi-squared = " + str(reduced_chi_sq) + "\n")
     
main()




