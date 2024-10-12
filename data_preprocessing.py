import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt
plt.set_cmap('jet')

def prepare_data(Ksteps, data_dir, state_data, ctrl_data, yobs_data, cond):
    #### Load pressure and saturation data

    hf_r = h5py.File(data_dir + state_data, 'r')
    mole = np.array(hf_r.get('Mole_frac_norm_slt')).transpose((3,2,1,0))
    sat = np.array(hf_r.get('Sg_norm_slt')).transpose((3,2,1,0))
    pres = np.array(hf_r.get('Psim_norm_slt')).transpose((3,2,1,0))
    hf_r.close()
    n_sample, steps_slt, Nx, Ny = mole.shape
    print(mole.shape)
    
    # plt.imshow(sat[0,0,:,:])
    #### Load control data
    hf_r = h5py.File(data_dir + ctrl_data)
    bhp0 = np.array(hf_r.get('Pwf_norm_slt')).transpose((2,1,0))
    rate0 = np.array(hf_r.get('Qinj_norm_slt')).transpose((2,1,0))
    hf_r.close()
    bhp = np.concatenate((bhp0,rate0),axis=1)
    
    #### Load output data
    hf_r = h5py.File(data_dir + yobs_data)
    if cond =='RC':
        Qrate_w = np.array(hf_r.get('Qpro_w_RC_norm_slt')).transpose((2,1,0))
        Qrate_g = np.array(hf_r.get('Qpro_g_RC_norm_slt')).transpose((2,1,0))
    else:
        Qrate_w = np.array(hf_r.get('Qpro_w_norm_slt')).transpose((2,1,0))
        Qrate_g = np.array(hf_r.get('Qpro_g_norm_slt')).transpose((2,1,0))
    
    BHP_inj = np.array(hf_r.get('BHPinj_norm_slt')).transpose((2,1,0))
    hf_r.close()
    yobs = np.concatenate((Qrate_w,Qrate_g,BHP_inj),axis=1)
    
    n_sample, num_well, steps_ctrl = bhp.shape
    _, num_prod, _ = bhp0.shape
    _, num_inj, _ = rate0.shape
    
    Mole_slt = []
    SAT_slt = []
    PRES_slt = []
    BHP_slt = []
    Yobs_slt = []

    indt = np.array(range(0,steps_slt-(Ksteps-1)))
    print(indt)

    
    for k in range(Ksteps):
        indt_k = indt + k
        if k ==1:
            indt_del = indt_k - indt
            indt_del = indt_del / max(indt_del)
        
        mole_t_slt = mole[:, indt_k,:, :]
        sat_t_slt = sat[:, indt_k,:, :]
        pres_t_slt = pres[:, indt_k,:, :]
        
        num_t_slt = sat_t_slt.shape[1]
        if k < Ksteps-1:
            bhp_t_slt = np.swapaxes(bhp[:,:, indt_k],1,2)
            yobs_t_slt = np.swapaxes(yobs[:,:, indt_k],1,2)
        Mole_slt.append(mole_t_slt)
        SAT_slt.append(sat_t_slt)
        PRES_slt.append(pres_t_slt)
        if k < Ksteps-1:
            BHP_slt.append(bhp_t_slt)
            Yobs_slt.append(yobs_t_slt)
    
    return Mole_slt, SAT_slt, PRES_slt, BHP_slt, Yobs_slt, num_t_slt, Nx, Ny, num_well, num_prod, num_inj



def train_split_data(Mole_slt, SAT_slt, PRES_slt, BHP_slt, Yobs_slt, num_t_slt, Nx, Ny, num_well, num_prod, num_inj, n_channels, device):
    num_all = Mole_slt[0].shape[0]
    split_ratio = int(num_all/100)
    num_run_per_case = 75
    num_run_eval = 100 - num_run_per_case # 25 cases

    mole_t_train =  np.zeros((num_run_per_case*split_ratio, num_t_slt, Nx, Ny))
    sat_t_train =  np.zeros((num_run_per_case*split_ratio, num_t_slt, Nx, Ny))
    pres_t_train = np.zeros((num_run_per_case*split_ratio, num_t_slt, Nx, Ny))
    bhp_t_train = np.zeros((num_run_per_case*split_ratio, num_t_slt, num_well))
    yobs_t_train = np.zeros((num_run_per_case*split_ratio, num_t_slt, 2*num_prod+num_inj))

    mole_t_eval = np.zeros((num_run_eval*split_ratio, num_t_slt, Nx, Ny))
    sat_t_eval = np.zeros((num_run_eval*split_ratio, num_t_slt, Nx, Ny))
    pres_t_eval = np.zeros((num_run_eval*split_ratio, num_t_slt, Nx, Ny))
    bhp_t_eval = np.zeros((num_run_eval*split_ratio, num_t_slt, num_well))
    yobs_t_eval = np.zeros((num_run_eval*split_ratio, num_t_slt, 2*num_prod+num_inj))
    
    num_train = num_run_per_case*split_ratio*num_t_slt
    shuffle_ind_train = np.random.default_rng(seed=1010).permutation(num_train)
    num_eval = num_run_eval*split_ratio*num_t_slt
    shuffle_ind_eval = np.random.default_rng(seed=1010).permutation(num_eval)
    
    STATE_train = []
    BHP_train = []
    Yobs_train = []
    STATE_eval = []
    BHP_eval = []
    Yobs_eval = []
    for i_step in range(len(SAT_slt)):

        for k in range(split_ratio):
            ind0 = k * num_run_per_case
            mole_t_train[ind0:ind0+num_run_per_case,...] = Mole_slt[i_step][k*100:k*100+num_run_per_case,...]
            sat_t_train[ind0:ind0+num_run_per_case,...] = SAT_slt[i_step][k*100:k*100+num_run_per_case,...]
            pres_t_train[ind0:ind0+num_run_per_case,...] = PRES_slt[i_step][k*100:k*100+num_run_per_case,...]            
            if i_step<len(SAT_slt)-1: 
                bhp_t_train[ind0:ind0+num_run_per_case,...] = BHP_slt[i_step][k*100: k*100+num_run_per_case,...]
                yobs_t_train[ind0:ind0+num_run_per_case,...] = Yobs_slt[i_step][k*100: k*100+num_run_per_case,...]
            # dt_train[ind0:ind0+num_run_per_case,...] =indt_d_slt[k*100: k*100+num_run_per_case, :, :]
            
            # Eval set
            ind1 = k*num_run_eval
            mole_t_eval[ind1:ind1+num_run_eval,...] = Mole_slt[i_step][k*100+num_run_per_case:k*100+100,...]
            sat_t_eval[ind1:ind1+num_run_eval,...] = SAT_slt[i_step][k*100+num_run_per_case:k*100+100,...]
            pres_t_eval[ind1:ind1+num_run_eval,...] = PRES_slt[i_step][k*100+num_run_per_case:k*100+100,...]
            if i_step<len(SAT_slt)-1:  
                bhp_t_eval[ind1:ind1+num_run_eval,...] = BHP_slt[i_step][k*100+num_run_per_case: k*100+100,...]
                yobs_t_eval[ind1:ind1+num_run_eval,...] = Yobs_slt[i_step][k*100+num_run_per_case: k*100+100,...]

            Mole_t_train = mole_t_train.reshape((num_run_per_case*split_ratio*num_t_slt, 1, Nx, Ny))
            SAT_t_train = sat_t_train.reshape((num_run_per_case*split_ratio*num_t_slt, 1, Nx, Ny))
            PRES_t_train = pres_t_train.reshape((num_run_per_case*split_ratio*num_t_slt, 1, Nx, Ny))
            if i_step<len(SAT_slt)-1: 
                BHP_t_train = bhp_t_train.reshape((num_run_per_case*split_ratio*num_t_slt, num_well))
                Yobs_t_train = yobs_t_train.reshape((num_run_per_case*split_ratio*num_t_slt, 2*num_prod+num_inj))
            # DT_train = dt_train.reshape((num_run_per_case*4*num_t_slt, 1))
            
            Mole_t_eval = mole_t_eval.reshape((num_run_eval*split_ratio*num_t_slt, 1, Nx, Ny))
            SAT_t_eval = sat_t_eval.reshape((num_run_eval*split_ratio*num_t_slt, 1, Nx, Ny))
            PRES_t_eval = pres_t_eval.reshape((num_run_eval*split_ratio*num_t_slt, 1, Nx, Ny))
            if i_step<len(SAT_slt)-1: 
                BHP_t_eval = bhp_t_eval.reshape((num_run_eval*split_ratio*num_t_slt, num_well))
                Yobs_t_eval = yobs_t_eval.reshape((num_run_eval*split_ratio*num_t_slt, 2*num_prod+num_inj))
            # DT_eval = dt_eval.reshape((num_run_eval*4*num_t_slt, 1))
            
            ### shuffle train and eval samples
            if n_channels==3:
                STATE_t_train = torch.tensor(np.concatenate((Mole_t_train, SAT_t_train, PRES_t_train),axis=1), dtype=torch.float32).to(device)
                STATE_t_eval = torch.tensor(np.concatenate((Mole_t_eval, SAT_t_eval, PRES_t_eval),axis=1), dtype=torch.float32).to(device)
            else:
                STATE_t_train = torch.tensor(np.concatenate((Mole_t_train, PRES_t_train),axis=1), dtype=torch.float32).to(device)
                STATE_t_eval = torch.tensor(np.concatenate((Mole_t_eval, PRES_t_eval),axis=1), dtype=torch.float32).to(device)
                

            STATE_t_train = STATE_t_train[shuffle_ind_train, ...]
            if i_step<len(SAT_slt)-1:
                BHP_t_train = torch.tensor(BHP_t_train[shuffle_ind_train, ...], dtype=torch.float32).to(device)
                Yobs_t_train = torch.tensor(Yobs_t_train[shuffle_ind_train, ...], dtype=torch.float32).to(device)
            # DT_train = DT_train[shuffle_ind_train, ...]
            

            STATE_t_eval = STATE_t_eval[shuffle_ind_eval, ...]
            if i_step<len(SAT_slt)-1:
                BHP_t_eval = torch.tensor(BHP_t_eval[shuffle_ind_eval, ...], dtype=torch.float32).to(device)
                Yobs_t_eval = torch.tensor(Yobs_t_eval[shuffle_ind_eval, ...], dtype=torch.float32).to(device)
            # DT_eval = DT_eval[shuffle_ind_eval, ...]
            
        STATE_train.append(STATE_t_train)
        STATE_eval.append(STATE_t_eval)
        if i_step<len(SAT_slt)-1:
            BHP_train.append(BHP_t_train)
            BHP_eval.append(BHP_t_eval)
            Yobs_train.append(Yobs_t_train)
            Yobs_eval.append(Yobs_t_eval)
    return STATE_train, BHP_train, Yobs_train, STATE_eval, BHP_eval, Yobs_eval


def save_data_to_file():
    return
    
