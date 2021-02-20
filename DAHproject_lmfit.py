# -*- coding: utf-8 -*-
"""
DAH Project - Study of B+ mesons
"""

#Imports
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.signal import find_peaks
import math
import lmfit
from lmfit.models import GaussianModel, ExponentialModel, ConstantModel


def main():
    
    # xmass = np.loadtxt(sys.argv[1])
    # import data

    with open("kdata-small.bin") as f:
        datalist = np.fromfile(f, dtype=np.float32)
    
    # number of events (each event has 5 observables)
    
    nevent = int(len(datalist)/5)
    
    #xdata is a list where each element is an event, itself a list of the 5 observables
    xdata = np.split(datalist, nevent)    

    
    # make list of invariant mass of events
    xmass = []
    for i in range (nevent):
        xmass.append(xdata[i][0])
    
    xmomentum = []
    for i in range(nevent):
        xmomentum.append(xdata[i][1])
      
    xpseudorapidity = []
    for i in range(nevent):
        xpseudorapidity.append(xdata[i][1])
        
    xcharge = []
    for i in range(nevent):
        xcharge.append(xdata[i][1])
        
    xlifetime = []
    for i in range(nevent):
        xlifetime.append(xdata[i][1]) 
         
    bins = 150
    
    plt.hist(xmass, bins)
    plt.xlabel("Mass (MeV/c^2)")
    plt.ylabel("Number  of events")
    plt.title("Histogram of mass")
    #plt.show()  
    
    """
    plt.hist(xmomentum, bins)
    plt.xlabel("Transverse momentum (MeV/c)")
    plt.ylabel("Number of events")
    plt.title("Histogram of transvere momentum")
    plt.show()
    
    plt.hist(xpseudorapidity, bins)
    plt.xlabel("Pseudorapidity (MeV/c)")
    plt.ylabel("Number of events")
    plt.title("Histogram of pseudorapidity")
    plt.show()
    
    plt.hist(xcharge, bins)
    plt.xlabel("Charge (C)")
    plt.ylabel("Number of events")
    plt.title("Histogram of charge")
    plt.show()
    
    plt.hist(xlifetime, bins)
    plt.xlabel("Lifetime (ps)")
    plt.ylabel("Number of events")
    plt.title("Histogram of lifetimes")
    plt.show()
    """
    n, bin_edges, patches = plt.hist(xmass, bins)
    
    # hist_data[0] is the number of counts in each bin
    # hist_data[1] is the mass of each bin
    
    mass_index_list = [i for i, value in enumerate(n) if value == n.max()]
    mass_index = mass_index_list[0]

    startofbin = bin_edges[mass_index]
    endofbin = bin_edges[mass_index+1]
    mass_estimate = (startofbin + endofbin)/2
    print(mass_estimate)
    print((endofbin - startofbin)/math.sqrt(12))
    
    # hist_data[0] contains all the counts, and has length n
    # hist_data[1] contains the bin edges, and therefore has length n+1.
    
    
    # remove right hand edge of right hand bin

    print(len(bin_edges))
    
    bin_edges = bin_edges[:-1]
    
    print(len(bin_edges))
    
    signal_counts = 0
    lower_counts = 0
    upper_counts = 0
    extra_low_counts = 0
    extra_high_counts = 0
    """    
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
            
    print("signal counts" + str(signal_counts))
    print("lower counts" + str(lower_counts))    
    print("upper counts" + str(upper_counts)) 
    print("extra_low_counts" + str(extra_low_counts))
    print("extra_high_counts" + str(extra_high_counts))
    """
    #peaks = sp.signal.find_peaks(plt.hist(xmass, bins)[0])
    #print(peaks)        
    
    x = bin_edges
    y = n
    
    gauss_mod = GaussianModel(prefix= 'gauss_')
    pars = gauss_mod.guess(y, x=x)

    #pars['gauss_center'].set(value=5278.3)
    #pars['gauss_sigma'].set(value=8.3)
    #pars['gauss_height'].set(value=900)
    
    exp_mod = ExponentialModel(prefix='exp_')
    pars.update(exp_mod.guess(y, x=x))
    
    pars['exp_amplitude'].set(value=32000)
    pars['exp_decay'].set(value=825)
    
    model = gauss_mod + exp_mod
    
    out = model.fit(y, pars, x=x)
    
    print(out.fit_report(min_correl=0.25))
    
    plt.plot(x, out.init_fit)
    plt.plot(x, out.best_fit)
    plt.show()

main()

def gauss_exp():
    
    x = np.arange(100)
    y = []
    sigma = 10
    mu = 50
    rho = 10
    
    for xi in x:
        
        
        pref = (1/(sigma*np.sqrt(2*np.pi)))
        gauss = (1/pref)*np.exp((-1/2)*((xi - mu)/sigma)**2)
        exp = np.exp(-xi/rho)
        yi = gauss + exp
        y.append(yi)
        
    return y

gauss_exp()
