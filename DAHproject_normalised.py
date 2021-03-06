# -*- coding: utf-8 -*-
"""
DAH Project - Study of B+ mesons
"""

#Imports
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import curve_fit
from scipy.stats import chisquare
import math

def gauss_exp(x, sigma, mu, rho, A):
    
    y = []
    
    a = x[0]
    length = len(x)
    b = x[length-1]
    
    B = 19594 - A
    
    for xi in x:
        
        pref = (A/(sigma*np.sqrt(2*np.pi)))
        gauss = pref*np.exp((-1/2)*((xi - mu)/sigma)**2)
        
        exp_pref = B/(rho*(np.exp(-a/rho) - np.exp(-b/rho)))
        
        exp = exp_pref*np.exp(-xi/rho)
        yi = gauss + exp
        y.append(yi)
        
    return y


def main():
    
    # Import data from binary file

    with open("kdata-small.bin") as f:
        datalist = np.fromfile(f, dtype=np.float32)
    
    # Number of events (each event has 5 observables)
    
    nevent = int(len(datalist)/5)
    
    #xdata is a list where each element is an event, itself a list of the 5 observables
    xdata = np.split(datalist, nevent)    

    
    # Make list of invariant mass of events
    
    xmass = []
    for i in range (nevent):
        xmass.append(xdata[i][0])
    
    """
    The other four variables can be collected in exactly the same way,
    but are not used here so separate lists for each are not needed.
    """
    
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
        
    signal_counts = 0
    lower_counts = 0
    upper_counts = 0
    extra_low_counts = 0
    extra_high_counts = 0
    
    """
    hist_data[0] contains all the counts, and has length n
    hist_data[1] contains the bin edges, and therefore has length n+1.
    """
    bin_edges = bin_edges[:-1]
    
    for i, value in enumerate(bin_edges):
        
        if value < 5229.1:
            extra_low_counts += n[i]
            
        elif 5229.1 < value < 5254.1:
            lower_counts += n[i]       
             
        elif 5254.1 < value < 5304.1:
            signal_counts += n[i]     
            
        elif 5304.1 < value < 5329.1:
            upper_counts += n[i]  
         
        elif value > 5329.1:
            extra_high_counts += n[i]
    
    background_in_signal = ((upper_counts + lower_counts)/2)*2
    gaussian_counts = signal_counts - background_in_signal    
    
    # Rename as x and y for simplicity
    x = centres
    y = n
    
    # Initial guesses of the parameters for the fit
    
    sigma = 8.3
    mu = 5279
    rho = 800
    A = gaussian_counts
    
    # Fit using Scipy's Curve Fit
    
    fit = scipy.optimize.curve_fit(gauss_exp, xdata=x, ydata=y, p0=[sigma, mu, rho, A])
    
    # Fitted parameters
    
    sigma = fit[0][0]
    mu = fit[0][1]
    rho = fit[0][2]
    A = fit[0][3]
    
    # Retrieve errors from output of the fit, perr is a list of the errors
    perr = np.sqrt(np.diag(fit[1]))
    
    # Plot the fit using the fitted parameters
    
    y_fitted = gauss_exp(x, sigma, mu, rho, A)
    
    plt.plot(x, y_fitted, color='red')
    plt.xlabel("Mass (Mev/c^2)")
    plt.ylabel("Number of counts")
    plt.title("Histogram of mass")
    plt.xlim([5200,5350])
    plt.show()
    
    ######## Residuals ########
    
    error_bars = np.sqrt(y)
    
    residuals = y_fitted - y
    
    crosses_x = 0
    for i in range(bins-1):
        
        if residuals[i] > 0:
            if residuals[i] - error_bars[i] < 0:
                crosses_x += 1     
        else:
            if residuals[i] + error_bars[i] > 0:
                crosses_x += 1            
    
    plt.plot(x, residuals, 'bo', markersize=4)
    plt.errorbar(x, residuals, yerr=error_bars, fmt='none', ecolor='red', elinewidth=1)
    plt.title("Residuals")
    plt.ylabel("Residual (counts)")
    plt.xlabel("Mass (MeV/c^2)")
    plt.show()
    
    #### Chi-squared ####
    
    chi_sq, p = scipy.stats.chisquare(y, y_fitted)
    
    fit_params = 3
    dof = bins - fit_params
    reduced_chi_sq = chi_sq/dof

    ######### Write output data to text file ##########
    
    outfile = open("data_initial_normalised", "w")
    
    outfile.write("Output data for B+- combined. \n" + 
        "Function used in fit: single Gaussian + exponential \n")
    
    outfile.write("sigma = " + str(fit[0][0]) + "\n" +
        "mu = " + str(fit[0][1]) + "\n" +
        "rho = " + str(fit[0][2]) + "\n" +
        "A = " + str(fit[0][3]) + "\n")
    
    outfile.write("sigma error = " + str(perr[0]) + "\n" +
        "mu error = " + str(perr[1]) + "\n" +
        "rho error = " + str(perr[2]) + "\n" +
        "A error = " + str(perr[3]) + "\n") 
    
    outfile.write(str(crosses_x) + "/" + str(bins) + " error bars cross the x axis \n")
    outfile.write("Chi-squared = " + str(chi_sq) + "\n")
    outfile.write("Reduced chi-squared = " + str(reduced_chi_sq) + "\n")
    
main()

"""

    This is old code, giving an initial crude estimate of the mass
    by finding the bin with the maximum number of counts
    
    mass_index_list = [i for i, value in enumerate(n) if value == n.max()]
    mass_index = mass_index_list[0]

    startofbin = bin_edges[mass_index]
    endofbin = bin_edges[mass_index+1]
    mass_estimate = (startofbin + endofbin)/2
    print(mass_estimate)
    
    ###################

    This is more old code, used to count the total numbers of counts
    in the peak and sideband regions, which could then be used for sideband subtraction.
    
"""       


