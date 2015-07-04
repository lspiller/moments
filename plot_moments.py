import numpy as np
from root_numpy import root2array, rec2array
from rootpy.tree import Cut
from fwm import HCM, FWM

bkg_file = 'hhskim.data12-JetTauEtmiss.root'
sig_file = 'hhskim.PowPyth8_AU2CT10_VBFH125_tautauhh.mc12a.root'

basic_selection = Cut('tau1_pt > 35000 && tau2_pt > 25000 && '
                      'jet1_pt > 50000 && jet2_pt > 30000 && '
                      'MET_et > 20000')
bkg_cut = basic_selection & 'tau1_charge * tau2_charge != -1'
sig_cut = basic_selection & 'tau1_charge * tau2_charge == -1'

branches = [
    'tau1_pt', 'tau1_eta', 'tau1_phi', 'tau1_m',
    'tau2_pt', 'tau2_eta', 'tau2_phi', 'tau2_m',
    'jet1_pt', 'jet1_eta', 'jet1_phi', 'jet1_m',
    'jet2_pt', 'jet2_eta', 'jet2_phi', 'jet2_m',
]

bkg_arr = root2array(bkg_file, branches=branches, selection=str(bkg_cut))
sig_arr = root2array(sig_file, branches=branches, selection=str(sig_cut))

# convert to ndarray of uniform type
bkg_arr = rec2array(bkg_arr).reshape((bkg_arr.shape[0], 4, 4))
sig_arr = rec2array(sig_arr).reshape((sig_arr.shape[0], 4, 4))

def kinematics(arr):
    # convert array of pT, eta, phi, m
    # to array of p, px, py, pz, pT, eta, phi, m
    kin_arr = np.empty(shape=(arr.shape[0], 4, 8))
    # |p| = pT cosh eta
    kin_arr[:,:,0] = arr[:,:,0] * np.cosh(arr[:,:,1])
    # px, py, pz
    kin_arr[:,:,1] = arr[:,:,0] * np.cos(arr[:,:,2])
    kin_arr[:,:,2] = arr[:,:,0] * np.sin(arr[:,:,2])
    kin_arr[:,:,3] = arr[:,:,0] * np.sinh(arr[:,:,1])
    # pT, eta, phi, m
    kin_arr[:,:,4] = arr[:,:,0]
    kin_arr[:,:,5] = arr[:,:,1]
    kin_arr[:,:,6] = arr[:,:,2]
    kin_arr[:,:,7] = arr[:,:,3]
    return kin_arr

bkg_arr = kinematics(bkg_arr)
sig_arr = kinematics(sig_arr)

print HCM(2, bkg_arr)
print FWM(2, bkg_arr)
