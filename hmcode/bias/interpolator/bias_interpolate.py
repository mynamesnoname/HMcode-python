import numpy as np
from scipy.interpolate import LinearNDInterpolator, interp1d
import matplotlib.pyplot as plt
import os

avail_sims   = ['TNG']
cwd_path = os.path.dirname(__file__)
zs = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])

path = '/home/wbc/code/HMcode-python/hmcode/temp'

class bias_supression_Model(object):
    def __init__(self, sim, reds, nu):
        if sim not in avail_sims:
            raise ValueError("Requested sim, %s, is not available. Choose one of "%sim + str(avail_sims))

        self.sim = sim
        self.load_data()
        self.reds = reds
        self.nu = nu

    def load_data(self):
        for i in range(0, 9):
            if i == 0:
                temp = np.loadtxt(cwd_path+f'/../regression/{zs[i]}_bias_ratio.txt')
            else:
                temp = np.concatenate((temp, np.loadtxt(cwd_path+f'/../regression/{zs[i]}_bias_ratio.txt')), axis=0)
        self.Mvir_regression = temp[:, 0]
        self.sup_regression  = temp[:, 1]
        self.z_regression    = temp[:, 2]

    def get_Mvir_regression(self):
        return self.Mvir_regression

    def predict(self, Mvir, z, a1, a2, a_node):
        z = np.array([z]*len(Mvir))

        points = np.column_stack((self.Mvir_regression, self.z_regression))  # 组合为(N, 2)的坐标点
        interpolator = LinearNDInterpolator(points, self.sup_regression)

        if np.max(Mvir/(a_node)) < np.min(self.Mvir_regression):
            raise ValueError("a_node is too large")
        elif np.min(Mvir/(a_node)) > np.max(self.Mvir_regression):
            raise ValueError("a_node is too small")

        # 计算 S1 数组
        S_temp = interpolator(np.column_stack((Mvir, z)))
        S1 = interpolator(np.column_stack((Mvir/(a_node), z)))
        S1[(Mvir/(a_node) < np.min(self.Mvir_regression))] = S1[S1 > 0][0]
        
        # 计算 C_node 和 datas_node_re
        S_node = np.max(S_temp[(Mvir>=11) & (Mvir<=13)])
        node_index = np.where(S_temp == S_node)[0][0]
        M_node_re = Mvir[node_index] * (a_node)

        # 计算满足条件的元素数量
        mask = Mvir <= M_node_re
        node_ind_re = int(np.sum(mask))

        # 更新 S1 数组
        standard1 = 1
        S1[:node_ind_re] = (S1[:node_ind_re]-standard1)*a1+1

        start = node_ind_re
        end = len(S1)

        for i in range(start, end):
            standard2 = (1-S1[:node_ind_re][-1]) / (end-start) * (i-start) + S1[:node_ind_re][-1]
            S1[i] = (S1[i]-standard2)*a2 + 1
        
        return S1

    def predict_nu(self, z, a1, a2, a_node):
        #print(f'z={z}')
        nue = self.nu
        #print(f'nu={nue}')
        #打印nue的数据类型
        #print(f'type={type(nue)}')
        #print(f'reds={self.reds}')
        
        # 处理 z 作为单个数值的情况
        is_scalar_z = np.isscalar(z)
        z_array = np.atleast_1d(z)  # 确保 z 变为数组
        
        if is_scalar_z:
            if np.any(z_array > 4):
                return np.ones_like(z_array)
        
        reds = np.sort(self.reds)  # 确保升序
        
        if not hasattr(self, 'Mvir_regression') or not hasattr(self, 'z_regression') or not hasattr(self, 'sup_regression'):
            raise AttributeError("Missing necessary attributes in the class")
        
        Mvir = np.arange(8.5, 14, 0.1)
        
        points = np.column_stack((self.Mvir_regression, self.z_regression))
        interpolator = LinearNDInterpolator(points, self.sup_regression)
        
        if np.max(Mvir / a_node) < np.min(self.Mvir_regression):
            raise ValueError("a_node is too large")
        if np.min(Mvir / a_node) > np.max(self.Mvir_regression):
            raise ValueError("a_node is too small")
        
        def find_index(z_val, reds_arr):
            if z_val == 0:
                idx = 0
            elif z_val == 4:
                idx = len(reds_arr)-1
            else:
                idx = np.searchsorted(reds_arr, z_val) - 1
            return idx
        
        suppress_list = []
        
        for z_i in z_array:
            z_arr = np.full_like(Mvir, z_i)
            
            S_temp = interpolator(np.column_stack((Mvir, z_arr)))
            S_temp[np.isnan(S_temp)] = np.nanmin(S_temp)  # 处理 NaN
            #print(f'S_temp={S_temp}')
            
            S1 = interpolator(np.column_stack((Mvir / a_node, z_arr)))
            S1[np.isnan(S1)] = 1
            
            # S_node = np.nanmax(S_temp[Mvir <= 13])
            # node_index = np.where(S_temp == S_node)[0]
            # #print(f'S_node={S_node}, node_index={node_index}')
            
            M_node_re = 12 * a_node
            mask = Mvir <= M_node_re
            node_ind_re = int(np.sum(mask))
            
            S1[:node_ind_re] = (S1[:node_ind_re] - 1) * a1 + 1
            for i in range(node_ind_re, len(S1)):
                standard2 = (1 - S1[node_ind_re - 1]) / (len(S1) - node_ind_re) * (i - node_ind_re) + S1[node_ind_re - 1]
                S1[i] = (S1[i] - standard2) * a2 + 1
            
            z_ind = find_index(z_i, reds)
            #print(f'z_ind={z_ind}')
            if z_ind == len(reds)-1 :
                try:
                    points_i = np.loadtxt(f'{cwd_path}/../../temp/nu{reds[z_ind]}.txt')
                except Exception as e:
                    raise IOError(f"Error loading files: {e}")

                points_nu = points_i
            else:
                try:
                    points_i = np.loadtxt(f'{cwd_path}/../../temp/nu{reds[z_ind]}.txt')
                    points_i_plus_one = np.loadtxt(f'{cwd_path}/../../temp/nu{reds[z_ind + 1]}.txt')
                except Exception as e:
                    raise IOError(f"Error loading files: {e}")
                
                points_nu = points_i + (points_i_plus_one - points_i) * (z_i - zs[z_ind]) / (zs[z_ind + 1] - zs[z_ind])

            interpolator_nu = interp1d(points_nu, S1, bounds_error=False, fill_value=1.0)
                
            if isinstance(nue, (np.float64, float)):
                suppress_value = interpolator_nu(nue) if np.min(points_nu) <= nue <= np.max(points_nu) else 1
            elif isinstance(nue, np.ndarray):
                suppress_value = interpolator_nu(nue)
                suppress_value[(nue > np.max(points_nu)) | (nue < np.min(points_nu))] = 1
            else:
                raise TypeError("nu should be a float or a numpy array")
                
            suppress_list.append(suppress_value)
        
        suppress_array = np.array(suppress_list)
        #print(f'suppress_array={suppress_array}')
        return suppress_array[0] if is_scalar_z else suppress_array.reshape(z.shape)
