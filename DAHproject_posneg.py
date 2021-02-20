# -*- coding: utf-8 -*-
"""
DAH Project - Study of B+- mesons
This script estimates the masses of B+ and B- individually.
It consists of a main function and two functions for modelled function,
one for B+ and one for B-.
"""

#Imports
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.optimize
import scipy.stats
import math

"""
These two functions create a composite function consisting of
    a single Gaussian and an exponential. 
    
param x: the independent variable, the mass values of the bins.
param sigma: the standard deviation of the Gaussian.
param mu: the mean of the Gaussian.
param A: the amplitude of the exponential.
param rho: the decay constant of the exponential.
returns y: the resulting function.
    
Note: I realise it is bad practise to hard-code the amplitude of the Gaussian,
    but scipy.optimize.curve_fit does not seem able to take fixed parameters.
"""

def gauss_exp_pos(x, sigma, mu, A, rho):
    
    y = []
    
    for xi in x:
        
        pref = (9932/(sigma*np.sqrt(2*np.pi)))
        gauss = pref*np.exp((-1/2)*((xi - mu)/sigma)**2)
        exp = A*np.exp(-xi/rho)
        yi = gauss + exp
        y.append(yi)
        
    return y
    
def gauss_exp_neg(x, sigma, mu, A, rho):
    
    y = []
    
    for xi in x:
        
        pref = (9662/(sigma*np.sqrt(2*np.pi)))
        gauss = pref*np.exp((-1/2)*((xi - mu)/sigma)**2)
        exp = A*np.exp(-xi/rho)
        yi = gauss + exp
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

    # Make lists of invariant mass and charge of events
    xmass = []
    for i in range (nevent):
        xmass.append(xdata[i][0])   
        
    xcharge = []
    for i in range (nevent):
        xcharge.append(xdata[i][3])
         
    # Choose number of bins
    bins = 150
    
    # Collect B+ and B- events, splitting data by charge.
    xmass_pos = []
    xmass_neg = []
    
    for i in range(nevent):
        
        if xcharge[i] > 0:
            xmass_pos.append(xmass[i])
        
        else:
            xmass_neg.append(xmass[i]) 
    
    # Record number of B+ and B- events 
    n_event_pos = len(xmass_pos)
    n_event_neg = len(xmass_neg)
    
    # Bin the data into histograms
    n_pos, bin_edges_pos, patches_pos = plt.hist(xmass_pos, bins)
    n_neg, bin_edges_neg, patches_neg = plt.hist(xmass_neg, bins)
    
    """
    For the fit, the x value corresponding to the number of counts
    in a bin should be the centre of that bin.
    """
    
    centres_pos = []
    centres_neg = []
    
    for i in range(bins):
        
        centre_pos = (bin_edges_pos[i] + bin_edges_pos[i+1])/2
        centres_pos.append(centre_pos)     
        centre_neg = (bin_edges_neg[i] + bin_edges_neg[i+1])/2
        centres_neg.append(centre_neg) 
        
    # Rename as x and y for simplicity
    x_pos = centres_pos
    y_pos = n_pos
    x_neg = centres_neg
    y_neg = n_neg    

    # Initial guesses for parameters
    sigma_pos = 8.3
    mu_pos = 5279
    rho_pos = 825
    A_pos = 16000
    
    sigma_neg = 8.3
    mu_neg = 5279
    rho_neg = 825
    A_neg = 16000
      
    # Fit using Scipy's Curve Fit
    fit_pos = scipy.optimize.curve_fit(gauss_exp_pos, xdata=x_pos, ydata=y_pos, p0=[sigma_pos, mu_pos, A_pos, rho_pos])
    fit_neg = scipy.optimize.curve_fit(gauss_exp_neg, xdata=x_neg, ydata=y_neg, p0=[sigma_neg, mu_neg, A_neg, rho_neg])  
    
    # Fitted parameters
    sigma_pos = fit_pos[0][0]
    mu_pos = fit_pos[0][1]
    A_pos = fit_pos[0][2]
    rho_pos = fit_pos[0][3]
    
    sigma_neg = fit_neg[0][0]
    mu_neg = fit_neg[0][1]
    A_neg = fit_neg[0][2]
    rho_neg = fit_neg[0][3]
    
    # Retrieve errors from fits
    perr_pos = np.sqrt(np.diag(fit_pos[1]))
    perr_neg = np.sqrt(np.diag(fit_neg[1]))
    
    y_fitted_pos = gauss_exp_pos(x_pos, sigma_pos, mu_pos, A_pos, rho_pos)
    y_fitted_neg = gauss_exp_neg(x_neg, sigma_neg, mu_neg, A_neg, rho_neg) 
    
    # Clear the initial histograms, since I don't wish to show both simultaneously.
    plt.clf()
    
    # Plot histogram with fit for B+
    plt.hist(xmass_pos, bins)
    plt.xlabel("Mass (MeV/c^2)")
    plt.ylabel("Number  of events")
    plt.title("Mass histogram of B+")
    plt.plot(x_pos, y_fitted_pos, color='red')
    plt.show()  
    
    # Plot histogram with fit for B- 
    plt.hist(xmass_neg, bins)
    plt.xlabel("Mass (MeV/c^2)")
    plt.ylabel("Number  of events")
    plt.title("Mass histogram of B-")       
    plt.plot(x_neg, y_fitted_neg, color='red')   
    plt.show()
    
    ######## Residuals ########
    
    # Calculate residuals as the difference between the fit and the histogram.
    residuals_pos = y_fitted_pos - y_pos
    residuals_neg = y_fitted_neg - y_neg
    
    # Calculate errors as the Poissonic errors on counts in a histogram bin.
    error_bars_pos = np.sqrt(y_pos)
    error_bars_neg = np.sqrt(y_neg)
    
    # Record the numbers of error bars crossing the y=0 line
    crosses_x_pos = 0
    crosses_x_neg = 0
    
    for i in range(bins-1):
        
        if residuals_pos[i] > 0:
            if residuals_pos[i] - error_bars_pos[i] < 0:
                crosses_x_pos += 1     
        else:
            if residuals_pos[i] + error_bars_pos[i] > 0:
                crosses_x_pos += 1
                
        if residuals_neg[i] > 0:
            if residuals_neg[i] - error_bars_neg[i] < 0:
                crosses_x_neg += 1     
        else:
            if residuals_neg[i] + error_bars_neg[i] > 0:
                crosses_x_neg += 1         
                
    # Plot residuals
    plt.plot(x_pos, residuals_pos, 'bo', markersize=4)
    plt.errorbar(x_pos, residuals_pos, yerr=error_bars_pos, fmt='none', ecolor='red', elinewidth=1)
    plt.title("Residuals for B+")
    plt.ylabel("Residual (counts)")
    plt.xlabel("Mass (MeV/c^2)")
    plt.show()

    plt.plot(x_neg, residuals_neg, 'bo', markersize=4)
    plt.errorbar(x_neg, residuals_neg, yerr=error_bars_neg, fmt='none', ecolor='red', elinewidth=1)
    plt.title("Residuals for B-")
    plt.ylabel("Residual (counts)")
    plt.xlabel("Mass (MeV/c^2)")
    plt.show()   
    
    #### Chi-squared ####
    
    # Calculate chi-squared, the "p" value is not used
    chi_sq_pos, p_pos = scipy.stats.chisquare(y_pos, y_fitted_pos)    
    chi_sq_neg, p_neg = scipy.stats.chisquare(y_neg, y_fitted_neg)
    
    # Calculate reduced chi-squared by dividing by the degrees of freedom
    fit_params = 4
    dof = bins - fit_params
    
    reduced_chi_sq_pos = chi_sq_pos/dof    
    reduced_chi_sq_neg = chi_sq_neg/dof
    
    ######### Write output data to text file ##########
    
    outfile = open("data_posneg", "w")
    
    outfile.write("Output data for B+ and B- separated. \n" + 
        "Function used in fits: single Gaussian + exponential \n")
        
    # Total numbers of events
    outfile.write(str(n_event_pos))
    outfile.write(str(n_event_neg))
    
    # B+
    
    outfile.write("B+ \n" +
        "sigma = " + str(fit_pos[0][0]) + "\n" +
        "mu = " + str(fit_pos[0][1]) + "\n" +
        "A = " + str(fit_pos[0][2]) + "\n" +
        "rho = " + str(fit_pos[0][3]) + "\n")
    
    outfile.write("sigma error = " + str(perr_pos[0]) + "\n" +
        "mu error = " + str(perr_pos[1]) + "\n" +
        "A error = " + str(perr_pos[2]) + "\n" +
        "rho error = " + str(perr_pos[3]) + "\n")  
    
    outfile.write(str(crosses_x_pos) + "/" + str(bins) + " error bars cross the x axis \n")
    outfile.write("Chi-squared = " + str(chi_sq_pos) + "\n")
    outfile.write("Reduced chi-squared = " + str(reduced_chi_sq_pos) + "\n")
    
    # B-
    
    outfile.write("B- \n" +
        "sigma = " + str(fit_neg[0][0]) + "\n" +
        "mu = " + str(fit_neg[0][1]) + "\n" +
        "A = " + str(fit_neg[0][2]) + "\n" +
        "rho = " + str(fit_neg[0][3]) + "\n")
    
    outfile.write("sigma error = " + str(perr_neg[0]) + "\n" +
        "mu error = " + str(perr_neg[1]) + "\n" +
        "A error = " + str(perr_neg[2]) + "\n" +
        "rho error = " + str(perr_neg[3]) + "\n")
        
    outfile.write(str(crosses_x_neg) + "/" + str(bins) + " error bars cross the x axis \n")
    outfile.write("Chi-squared = " + str(chi_sq_neg) + "\n")
    outfile.write("Reduced chi-squared = " + str(reduced_chi_sq_neg) + "\n")
    
main()




