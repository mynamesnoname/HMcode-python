from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import sys
sys.path.append('../hmcode')
import utility as util
sys.path.append('../CAMELSconcentration')
from CAMELSconcentration.Interpolator import CvirModel


Mvir = util.logspace(1e11, 1e13, 256)

# Predict with 'CAMELSconcentration' pack, calculate the mean of the predictions
def predict(z, sim, sn1, agn1, sn2, agn2):
    model = CvirModel(sim=sim)  # This loads + organizes the model parameters (KLLR slopes/intercepts)
    mean = model.predict(Mvir, z=z, Omega_m=0.3, sigma_8=0.85, SN1=sn1, AGN1=agn1, SN2=sn2,
                         AGN2=agn2)  # Now make the actual prediction

    c_mean_interpol = np.mean(mean, axis=0)

    return c_mean_interpol


def c_fit(M, z, iz, om, s8, sn1, agn1, sn2, agn2, sim, plot=False, plot_c=False):
    c_mean_interpol = predict(nM, z, sim, sn1, agn1, sn2, agn2)
    
    c_fitted = []
    M_mid = []
    for i in range(len(M)):
        if M[i] <= Mvir[0]:
            c_fitted.append(c_mean_interpol[0])
        if Mvir[0] <= M[i] <= Mvir[-1]:
            M_mid.append(M[i])

    model_m = CvirModel(sim=sim)  # This loads + organizes the model parameters (KLLR slopes/intercepts)
    mean_m = model_m.predict(M_mid, z=z, Omega_m=0.3, sigma_8=0.85, SN1=sn1, AGN1=agn1, SN2=sn2, AGN2=agn2)  # Now make the actual prediction

    for i in range(len(M_mid)):
        sum_m = 0
        for j in range(len(mean_m)):
            sum_m += mean_m[j][i]
        c_fitted.append(sum_m / len(mean_m))

    for i in range(len(M)):
        if M[i] >= Mvir[-1]:
            c_fitted.append(c_mean_interpol[-1])
    
    if len(c_fitted) == len(M):
        c_fitted = np.array(c_fitted)
    else:
        print("Error in c_fit")
    
    # 定义拟合函数
    #def new(M, A, C, D, E):
    #    return (A * np.log(M) ** (C / np.log(M)) + D * np.sin(np.log(M)))/((1+z)**E)

    # 定义拟合函数

    #def new(M, A, C, D, E):
    #    return (A * np.log(M) ** (C / np.log(M)) + D * np.sin(np.log(M)))/((1+z)**E)

    # 使用curve_fit进行拟合
    #popt, pcov = curve_fit(new, Mvir, c_mean_interpol, maxfev=2000)

    # 输出拟合参数
    #A_fit = popt[0]
    #C_fit = popt[1]
    #D_fit = popt[2]
    #E_fit = popt[3]

    #print("z={:}, 拟合参数A:".format(z), A_fit)

    # 绘制拟合曲线
    #c_fitted = new(M, A_fit, C_fit, D_fit, E_fit)

    #if plot and plot_c:
    #    plt.loglog(M, c_fitted, '--', color='C%d' % iz)
        # plt.loglog(M, c_Bullock01, ':', color='C%d' % iz)
    #elif plot:
    #    plt.scatter(Mvir, c_mean_interpol, s=5)
    #    plt.loglog(M, c_fitted, '--', color='C%d' % iz, label='z={:}'.format(z))
        # plt.loglog(M, c_Bullock01, ':', color='C%d'%iz)

    return c_fitted


def c_correct(M, iz, z, c_fitted, sim, om, s8, sn1, agn1, sn2, agn2, plot=False):
    M_need_to_correct = []
    index = []

    for i in range(len(M)):
        if Mvir[0] <= M[i] <= Mvir[-1]:
            M_need_to_correct.append(M[i])
            index.append(i)

    model = CvirModel(sim=sim)  # This loads + organizes the model parameters (KLLR slopes/intercepts)
    mean = model.predict(M_need_to_correct, z=z, Omega_m=0.3, sigma_8=0.85, SN1=sn1, AGN1=agn1, SN2=sn2,
                         AGN2=agn2)  # Now make the actual prediction

    c_mean_M_correct = [0] * len(M)

    for i in index:
        sum = 0
        for j in range(len(mean)):
            sum += mean[j][i - index[0]]
        c_mean_M_correct[i] = sum / len(mean)

    c = c_fitted

    for i in index:
        c[i] = 0

    c += c_mean_M_correct

    if plot:
        plt.loglog(M, c, '-', color='C%d' % iz, label='z={:}'.format(z))

    return c


def plotshow(plot_fit, plot_correct):
    if plot_fit:
        plt.plot(np.nan, color='black', ls='--', label='Cvir_fit')
        plt.plot(np.nan, color='black', ls=':', label='Cvir_Bullock')

    if plot_correct:
        plt.plot(np.nan, color='black', ls='-', label='Cvir_correct')

    plt.xlabel(r'$M_\mathrm{vir}\,[M_\odot/h]$')
    plt.ylabel('c')
    plt.legend(loc='upper right')

    if plot_correct or plot_fit:
        plt.show()
