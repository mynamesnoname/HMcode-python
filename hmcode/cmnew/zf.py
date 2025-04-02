import numpy as np
import camb
from hmcode_python.hmcode import cosmology


def _get_halo_collapse_redshifts(M: np.ndarray, z: float, iz: int, dc: float, g: callable,
                                 CAMB_results: camb.CAMBdata, cold=False) -> np.ndarray:
    '''
    Calculate halo collapse redshifts according to the Bullock et al. (2001) prescription
    '''
    from scipy.optimize import root_scalar
    gamma = 0.01
    a = cosmology.scalefactor_from_redshift(z)
    Om_c = CAMB_results.get_Omega(var='cdm', z=0.)
    Om_b = CAMB_results.get_Omega(var='baryon', z=0.)
    Om_nu = CAMB_results.get_Omega(var='nu', z=0.)
    Om_m = Om_c + Om_b + Om_nu
    zf = np.zeros_like(M)
    for iM, _M in enumerate(M):
        Mc = gamma * _M
        Rc = cosmology.Lagrangian_radius(Mc, Om_m)
        sigma = _get_sigmaR(Rc, iz, CAMB_results, cold=cold)
        fac = g(a) * dc / sigma
        if fac >= g(a):
            af = a  # These haloes formed 'in the future'
        else:
            af_root = lambda af: g(af) - fac
            af = root_scalar(af_root, bracket=(1e-3, 1.)).root
        zf[iM] = cosmology.redshift_from_scalefactor(af)
    return zf

def _get_sigmaR(R: np.ndarray, iz: int, CAMB_results: camb.CAMBdata, cold=False) -> np.ndarray:
    var = 'delta_nonu' if cold else 'delta_tot'
    sigmaR = CAMB_results.get_sigmaR(R, z_indices=[iz], var1=var, var2=var)[0]
    return sigmaR