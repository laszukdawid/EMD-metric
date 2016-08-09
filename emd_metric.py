#!/usr/bin/python
# coding: UTF-8
#
# Author:  Dawid Laszuk
# Contact: laszukdawid@gmail.com
#          
# Last update: 21/02/2016
#
# Feel free to contact for any information.
#
# Introduction:
#   Metric functions are m1calc() and m2calc(). 
#   This programme performs small experiment. It decomposes signal
#   under different EMD settings and for each set it calculates metrics.
#   It assumes that EMD module is in sys.path or in the same directory.
#   All results are saved in 'results' directory.
#

import os, sys
import numpy as np
import pylab as py
import scipy.signal as ss


def m1calc(phase, freq, dt):
    """
    Calculates m1 values based on crossing over period in
    IMF's instantaneous frequencies.
    """

    nImf, N = freq.shape
    
    M = np.zeros(nImf)
    for i in range(1,nImf):

        tmp = np.zeros(N)
        for j in range(i):
            diff = freq[j] - freq[i]
            tmp[diff < 0] = 1
        M[i] = np.sum(tmp)/float(N)
    
    return M    

def m2calc(phase, freq, amp, dt):
    """
    Calculates m2 values based on Fourier transformation of IMFs' 
    phases and amplitudes as their ovelapping measure.
    """

    # Computing Fourier Transforms
    AMP = np.fft.fft(amp)
    PHI = np.fft.fft(phase)
    
    dot = lambda x,y: np.sum(np.conj(x)*y, axis=1)

    M = dot(AMP,PHI)/np.sqrt(dot(AMP, AMP)*dot(PHI,PHI))
    M = np.abs(M)
    
    return M
    
def plotInGrid(arr, title=None, saveFlag=None):
    """
    Plots results in for of a grid.
    """
     
    t = np.arange(0, arr.shape[1]/128., 1./128)
    
    py.figure()
    
    if title: py.suptitle(title)    
    imfNo = arr.shape[0]
    r = np.ceil(np.sqrt(imfNo))
    c = np.ceil(imfNo/r)
    
    for imf in range(arr.shape[0]):
        py.subplot(r, c, 1 + imf)
        py.plot(t, arr[imf])
        py.title('IMF ' + str(imf+1) )

    if saveFlag: py.savefig(title)

def filt(s, fs=128.):
    """
    Filters data to 5th of Nyquist freq.
    """
    n = 4
    Wn = float(fs/10)/(fs/2)
    b,a = ss.butter(n, Wn)
    return ss.filtfilt(b, a, s)

def smoothHilbert(S):
    """
    Smooths results based on previous and further value.
    """
    
    s1 = np.zeros(S.shape)
    s2 = np.zeros(S.shape)
    
    s1[:,::2] = S[:,::2]
    s1[:,1:-1] += 0.5*(s1[:,:-2]+s1[:,2:])
    
    s2[:,1::2] = S[:,1::2]
    s2[:,1:-1] += 0.5*(s2[:,:-2]+s2[:,2:])
    S = 0.5*(s1+s2)
    S[:,0] += S[:,0]
    S[:,-1]+= S[:,-1]
    
    return S


if __name__ == "__main__":
    
    DTYPE = np.float32
    maxImf = -1
    RESULTS = {}
    
    fs = 500.
    dt = 1./fs
    tMin, tMax = -1, 1
    
    t = np.arange(tMin, tMax, dt)
    
    #####################################
    # Chosing type of signal
    #~ sigType = 'synth'
    sigType = 'random'
    
    if sigType=='synth':
        I = 5
        rand = np.random.random
        A   = np.array([ 1., 1., 3.,  2.,  3.])
        F   = np.array([35,  25, 19,  15,   4])
        PHI = np.array([2.,  4.,  0, 3.4, 5.7])
        S = np.sum([A[i]*np.sin(2*np.pi*F[i]*t+PHI[i]) for i in range(I)], axis=0)
        
        np.random.seed(10)
        S += 0.1*np.random.normal(0, 0.01, S.size)
        
        with open('params.txt','w') as f:
            f.write(r"A & F & PHI \\"+"\n")
            info = [r"{} & {} & {} \\".format(A[i], F[i], PHI[i]) for i in range(I)]
            f.write( '\n'.join(info) )

    elif sigType == 'random':
        np.random.seed(239)
        S = np.random.normal(0, 1, t.size)
        S = filt(S, fs)

    # Plotting signal
    fig = py.figure(figsize=(12,6))
    ax = fig.add_subplot(111)
    ax.plot(t, S)
    #~ ax.set_title('Original signal')
    
    py.savefig('orgsig-'+sigType, dpi=120)
    py.close()

    ######################################
    ## Empirical Mode Decomposition
    
    import EMD
    emd = EMD.EMD()

    emd.DTYPE = DTYPE
    emd.nbsym = 2
    
    t = t.astype(DTYPE)
    S = S.astype(DTYPE)
    
    # Change directory to 'results'
    if(not 'results' in os.listdir('.')): os.mkdir('results')
    os.chdir("results")

    splineNames = []
    #~ splineNames.append('linear')
    splineNames.append('cubic')
    
    for spline in splineNames:  RESULTS[spline] = {}

    # Trimming sides
    pr = 0.1
    T = t[-1]-t[0]
    timeAnalLeft = t[0] + pr*T
    timeAnalRight = t[-1] - pr*T
    idx = np.r_[t>=timeAnalLeft] & np.r_[t<=timeAnalRight]

    # Small experiment:
    # Calculate metrics for different spline techniques
    # and different fixe_h parameter (number of proto-IMFs sifting)
    for fixe_h in np.arange(1,21,1):
        emd.FIXE_H = fixe_h
        
        for spline in splineNames:
            print spline

            emd.splineKind = spline
            IMF, EXT, ITER, imfNo = emd.emd(S, t, maxImf)
            N = imfNo
            
            tmpIMF = np.vstack([IMF[i] for i in range(imfNo)])
            filename = "{}_imfNo{}".format(spline, fixe_h)
            np.save(filename, tmpIMF)
            
            #~ imf = np.load(spline + '.npy')[:-1,idx]
            imf = tmpIMF
            H = ss.hilbert(imf)
            
            phase = np.angle(H)
            phase = smoothHilbert(phase)
            freq = np.diff(phase)/dt
            amp = np.abs(H)
            
            #~ plotInGrid(imf, title='imf_{}'.format(spline), saveFlag=1)
            #~ plotInGrid(phase, title='phase_{}'.format(spline), saveFlag=1)
            
            # Trimming sides
            imf = imf[:, idx]
            amp = amp[:, idx]
            phase = phase[:, idx]
            freq = freq[:, idx]
            
            #########################################
            # 
            # Calculating m1 value and M1 metric
            m1 = m1calc(phase, freq, dt)
            M1 = np.sum(m1)
            assert(M1>=0)
            
            # Calculating m2 value and M2, M3 metrics
            m2 = m2calc(phase, freq, amp, dt)
            M2 = np.max(m1)
            M3 = np.sum(m2)
            assert(M2>=0)
            assert(M3>=0)
            # 
            #########################################
            
            M = {'M1':M1, 'M2':M2, 'M3':M3}
            
            RESULTS[spline][fixe_h] = M
        
            print '\n'*2
            print 'fixe_h = ', fixe_h
            for name in M.keys():
                print '{} {}'.format(name, M[name])
                
                filename = "{}_imfNo{}_{}.txt".format(spline, fixe_h, name)
                with open(filename, 'w') as f:
                    f.write(str(M[name]))

            print '\n'*4

    ####################################
    # Saving results
    import copy
    
    R = copy.deepcopy(RESULTS)

    out = []

    for spline in splineNames:
        RESULTS = R[spline]
        
        out.append(spline)
        print ' ++  {}  ++ '.format(spline)
        # Print results of all computations
        LABELS = RESULTS[RESULTS.keys()[0]].keys()
        LABELS.sort()
        LABELS = [r'FIXE\_H'] + LABELS
        LABELS = LABELS + [r'SUM'] 
        labels = ' & '.join( LABELS ) + r' \\'

        # Store data in list
        out.append(labels)
        for fixe_h in RESULTS.keys():
            l = ['%i'%fixe_h]
            c = ['{:.3}'.format(RESULTS[fixe_h][label]) for label in LABELS[1:-1]]
            c = '\t&\t'.join( l + c)
            S = sum([ RESULTS[fixe_h][label] for label in LABELS[1:-1] ])
            c = c + '\t&\t{:.3}'.format(S)
            c = c + '\t'
            c = c + r'\\'
            
            #~ c = c + r' \\'
            out.append(c)
            print c
            
        # Save results to file
        with open('results.txt','w') as f:
            f.write( '\n'.join(out))
            
                
