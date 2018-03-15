import os
import numpy as np
import sys
import glob
import re
import h5py
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
if __package__ is None:
    sys.path.append(os.path.realpath("/data/shared/CMS_Deep_Learning"))
    sys.path.append(os.path.realpath("/data/shared/RegressionLCD"))
    sys.path.append(os.path.realpath("/nfshome/mliu/LCDstudies/Scripts"))
from CMS_Deep_Learning.io import gen_from_data, retrieve_data, simple_grab
from CMS_Deep_Learning.postprocessing.metrics import distribute_to_bins
from model import *
from preprocessing import *

def dif(target, predicted):
    dif = target - predicted
    dif = dif.reshape((dif.shape[0],))
    return dif

def rDif(target, predicted):
    dif = target - predicted
    rDif = (dif / target)*100
    rDif = rDif.reshape((rDif.shape[0],))
    return rDif

def findAB(the_target, E_summed, H_summed):
    m = None
    sol = None
    for A in np.linspace(40, 150, 500):
        for B in np.linspace(40, 150, 500):
            res = ((the_target - (A*E_summed + B*H_summed))**2).mean()
            if m == None or res < m:
                m = res
                sol = A, B
    return sol

def plotPredictedXTarget(target, predicted, lim_l=0, lim_r=550, particle=""):
    plt.figure(figsize=(5, 5))
    plt.xlabel("True energy (GeV)")
    plt.ylabel("Predicted energy (GeV)")
    plt.title(particle)
    plt.scatter(target, predicted, color='g', alpha=0.5)
    plt.xlim(lim_l, lim_r)
    plt.ylim(lim_l, lim_r)
    plt.legend()
    plt.show()

def binning(nbins, label, pred, plot=False):
    out, x, y = distribute_to_bins(label, [label, pred], nb_bins=nbins, equalBins=True)
    iSize = 500 / nbins
    means = []
    rMeans = []  # normalized means
    stds = []
    rStds = []  # normalized standard deviations
    sizes = []  # number of events in the bins
    res= [] # calorimeter energy resolution

    for i in range(0, nbins):
        sizes.append(len(x[i]))
        if (plot == True):
            PredictedTarget(x[i], y[i], i * iSize, (i + 1) * iSize)
            histRelDif(x[i], y[i], nbins=150, lim=15, lim_l=i*iSize, lim_r=(i+1)*iSize)

        difference = dif(x[i], y[i])
        relDiff = rDif(x[i], y[i])
        mean = np.mean(difference)
        means.append(mean)
        rMean = np.mean(relDiff)
        rMeans.append(rMean)
        std = np.std(difference)
        stds.append(std)
        rStd = np.std(relDiff)
        rStds.append(rStd)
        eRes = std / np.mean(x[i])
        res.append(eRes)

    return x, y, means, rMeans, stds, rStds, sizes, res


def grab(direc):
    y_ele = simple_grab('Y', data=direc, label_keys='target',
                        input_keys=['ECAL', 'HCAL'])
    y_ele = y_ele[:, 1:]
    y_ele = y_ele.ravel()
    ecal_ele, hcal_ele = simple_grab('X', data=direc, label_keys=['ECAL', 'HCAL'], input_keys=['ECAL', 'HCAL'])
    s_ecal_ele = sumCal(ecal_ele)
    s_hcal_ele = sumCal(hcal_ele)
    s_ecal_ele = s_ecal_ele.ravel()
    s_hcal_ele = s_hcal_ele.ravel()

    return y_ele, s_ecal_ele, s_hcal_ele

def func(E, a, b, c):
    # equation to fit energy resolution
    return np.sqrt((a**2 / E) + b**2 + (c / E)**2)

def fit_part(func, energies, res, sigmas=None, verbose=False):
    popt, pcov = curve_fit(func, energies, res, bounds=(0,1000), sigma=sigmas)
    if (verbose):print(popt)
    return popt

def aux_stds(stds, label="", marker='.', col='black'):
    n = len(stds)
    iSize = (500-10)/n
    energies = []
    
    for i in range(n):
        x_axis = 10+(i * iSize + (i + 1) * iSize) / 2
        energies.append(x_axis)
    plt.scatter(energies, stds, marker=marker, color=col, alpha=0.5, label=label)
    return energies

def lin_bins(A, B, the_target, E_summed, H_summed, nbins=10):
    '''
    Returns bins and res arrays to be plotted.
    '''
    min_E = 10
    max_E = 500
    Elow = float(min_E)
    bins = []
    res = []
    size = []
    for Eup in np.linspace(10, 500, nbins+1):
        if not Eup or Elow == Eup: continue
        E_bin_old = np.where((the_target > Elow))
        E_bin = np.where((the_target > Elow) & (the_target <= Eup))
        content = (the_target - (A*E_summed + B*H_summed))[E_bin]
        rdif = 100.*content / the_target[E_bin]
        meandif = content.mean()
        rms2 = rdif.std()
        rms = content.std()
        size.append(len(content))
        E = (Eup + Elow) / 2.
        bins.append(E)
        res.append(rms2)
        me = content.mean()
        Elow = Eup
    return bins, res, size

def returnBinsRes(directory,div):
    GammaEscan = h5py.File(directory, "r")
    gamma_E_sum = GammaEscan['ECAL'].value.sum( axis = (1,2,3))/div
    gamma_H_sum = GammaEscan['HCAL'].value.sum( axis = (1,2,3))/div
    
    if div > 1: gamma_target = np.array(GammaEscan["energy"], dtype = "float32")
    else: 
        gamma_target = np.array(GammaEscan["target"], dtype = "float32")
        gamma_target = gamma_target[:,1]

    gamma_a, gamma_b = findAB(gamma_target, gamma_E_sum, gamma_H_sum)
    gamma_bins, gamma_res, sizes = lin_bins(gamma_a, gamma_b, gamma_target, gamma_E_sum, gamma_H_sum)
    return gamma_bins, gamma_res, sizes

def plot_eres_v1(resdatas):
    keys = ['true','pred','binning','fitline','label','style','color']
    for key in keys:
        for resdata in resdatas:
            if not key in resdata:
               print("missing key, returning")
               print(key)
               return
    plt.figure(figsize=(6, 6))
    
    for resdata in resdatas:
        resdata['x'], resdata['y'], resdata['means'], resdata['rMeans'], resdata['stds'], resdata['rStds'], resdata['sizes'], resdata['res'] = binning(resdata['binning'], resdata['true'], resdata['pred'])
        rStds = resdata['rStds']
        energies = aux_stds(rStds, label=resdata['label'], marker=resdata['style'], col=resdata['color'])
        sig_rStds = rStds / np.sqrt(2*np.asarray(resdata['sizes']))
        if resdata['fitline']:
           popt = fit_part(func, energies, rStds, sigmas= sig_rStds)
           y = func(energies, *popt)
           plt.plot(energies, y, color=resdata['color'], ls="-")
    plt.yscale('log')
    plt.xlabel("True Energy (GeV)", size=18)
    plt.ylabel(r"$\frac{\sigma({\Delta E})}{E_{true}}$", size=21)
    plt.title("Energy resolution", size=18)
    plt.xlim(0, 500)
    plt.legend(loc='best',prop={'size': 12})
