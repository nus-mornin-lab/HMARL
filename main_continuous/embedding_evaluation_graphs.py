

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy import stats 
import os
import time
from datetime import datetime
from scipy.stats import sem
import setting
import copy
from functools import reduce
from matplotlib.ticker import FormatStrFormatter



def run_eval(data, pre_master_loss, master_loss, iv_loss, vasso_loss, mixer_loss, datatype = 'mimic', phase = "train", train_FM = 0):
    def diff_vaso_plot(med_vaso, mort_vaso, std_vaso,col, title):
        f, ax1 = plt.subplots(1, 1, sharex='col', sharey='row', figsize = (4.5,4))
        ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax1.tick_params(axis='both', which='major', pad=0)
        step = 2
        if col == 'r':
            fillcol = 'lightsalmon'
        elif col == 'g':
            fillcol = 'palegreen'
            step = 1
        elif col == 'b':
            fillcol = 'lightblue'
        elif col == 'mediumslateblue':
            fillcol = 'lavender'
        elif col == 'darkgoldenrod':
            fillcol = 'wheat'
        ax1.plot(med_vaso, sliding_mean(mort_vaso), color=col)
        ax1.fill_between(med_vaso, sliding_mean(mort_vaso) - 1*std_vaso,  
                         sliding_mean(mort_vaso) + 1*std_vaso, color=fillcol)
        t = title
        ax1.set_title(t)
        x_r = [i/10.0 for i in range(-6,8,2)]

        y_r = [i/20.0 for i in range(0,20,step)]
        ax1.set_xticks(x_r)
        ax1.set_yticks(y_r)
        ax1.grid()

        f.text(0.6, 0.01, 'Difference in vasopressor', ha='center', fontsize=20)
        f.text(0.0, 0.5, 'Mortality', va='center', rotation='vertical', fontsize = 20)
        ax1.title.set_fontsize(25)
        plt.subplots_adjust(left=0.2, right=0.95, top=0.9, bottom=0.15)
    #     f.tight_layout()
        f.savefig(res_dir +title+'_Action_vs_mortality.png',dpi = 300)

    def diff_iv_plot(med_iv, mort_iv, std_iv,col, title):
        f, ax2 = plt.subplots(1, 1, sharex='col', sharey='row', figsize = (4.5,4))
        ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax2.tick_params(axis='both', which='major', pad=0)
        step = 2
        if col == 'r':
            fillcol = 'lightsalmon'
        elif col == 'g':
            fillcol = 'palegreen'
            step = 1
        elif col == 'b':
            fillcol = 'lightblue'
        elif col == 'mediumslateblue':
            fillcol = 'lavender'
        elif col == 'darkgoldenrod':
            fillcol = 'wheat'

        ax2.plot(med_iv, sliding_mean(mort_iv), color=col)
        ax2.fill_between(med_iv, sliding_mean(mort_iv) - 1*std_iv,  
                         sliding_mean(mort_iv) + 1*std_iv, color=fillcol)
        t = title
        ax2.set_title(t)
        x_iv = [i for i in range(-800,900,400)]
        ax2.set_xticks(x_iv)
        ax2.grid()
        f.text(0.55, 0.01, 'Difference in IV fluids', ha='center', fontsize=20)
        f.text(0.0, 0.5, 'Mortality', va='center', rotation='vertical', fontsize = 20)
        tiny_size = 15
        SMALL_SIZE = 15
        MEDIUM_SIZE = 15
        BIGGER_SIZE = 18
    #     plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    #     plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=tiny_size)    # fontsize of the tick labels
        ax2.title.set_fontsize(25)
        plt.subplots_adjust(left=0.2, right=0.95, top=0.9, bottom=0.15)
    #     f.tight_layout()
        f.savefig(res_dir +title+'_Action_vs_mortality.png',dpi = 300)    
    
    def make_df_diff_new(ai_iv, ai_vaso, df_in):
        op_vaso_med = []
        op_iv_med = []
        for i in range(len(ai_iv)):
            iv,vaso = ai_iv.iloc[i], ai_vaso.iloc[i]
            op_vaso_med.append(vaso_vals[vaso])
            op_iv_med.append(iv_vals[iv])
        iv_diff = np.array(op_iv_med) - np.array(df_in['ori_iv_fluids'])
        vaso_diff = np.array(op_vaso_med) - np.array(df_in['ori_vasopressors'])
        df_diff = pd.DataFrame()
        df_diff['mort'] = np.array(df_in['mortality_hospital'])
        df_diff['iv_diff'] = iv_diff
        df_diff['vaso_diff'] = vaso_diff
        return df_diff
    

    def plot_loss(loss, loss_type):
        if loss:
            plt.figure(figsize=(7,4))
            plt.plot(loss)
            plt.savefig(res_dir + 'loss_'+loss_type +'.png',dpi = 100)
        
    def tag_conc_rate_and_diff_mean(dt):
        for v in action_types:
            dt[v + '_conc_rate'] = dt[v + '_conc'].mean()
            dt[v + '_diff_mean'] = dt[v + '_diff'].mean()
        return dt
    
    def discre_conc_level(x):
        xx = [0.1,0.3,0.5,0.7,0.9]
        for t in xx:
            if x>= t-0.1 and x < t+0.1:
                return t
        if x == 1:
            return 0.9
        
    def discre_diff_level(x):
        xx = [-4,-3,-2,-1,0,1,2,3,4]
        for t in xx:
            if x>= t-0.5 and x < t+0.5:
                return t
        if x == 4:
            return 4
    
 

    def q_vs_outcome(data, outcome_col1 = 'mortality_hospital'):
        q_vals_phys = data['phys_Q']  
        pp = pd.Series(q_vals_phys)
        phys_df = pd.DataFrame(pp)
        phys_df['mort'] = copy.deepcopy(np.array(data[outcome_col1]))

        bin_medians = []
        mort = []
        mort_std = []
        # i = -15
        # while i <= 20:
        increment = round((q_vals_phys.quantile(0.99)- q_vals_phys.quantile(0.01))/20, 5)/2
        i = q_vals_phys.quantile([0.01,0.99]).values[0]
        while (i <= q_vals_phys.quantile([0.01,0.99]).values[1]):
            count =phys_df.loc[(phys_df.iloc[:,0]>i-increment) & (phys_df.iloc[:,0]<i+increment)]
            try:
                res = sum(count['mort'])/float(len(count))
                if len(count) >=2:
                    bin_medians.append(i)
                    mort.append(res)
                    mort_std.append(sem(count['mort']))
            except ZeroDivisionError:
                pass
            i += increment/5
        q_value = pd.DataFrame(bin_medians, columns=["q_value"])
        med = pd.DataFrame(sliding_mean(mort, window = 1), columns=["med"])
        ci = pd.DataFrame(sliding_mean(mort_std, window = 1), columns = ["std"])
        bb =pd.concat([med,ci,q_value],ignore_index=True, axis = 1)

        res_dt = pd.DataFrame(bb)
        res_dt.reset_index()

        return res_dt


    def quantitive_eval(data,res_dt, outcome_col = 'mortality_hospital'):

        q_dr_dt = pd.DataFrame()
        for mod in ['ai','phys']:    
            q_dr_dt.loc[mod,'Q'] = data[mod + '_Q'].mean()

        def find_nearest_Q(Q_mean,res_dt):
            ind = np.argmin([abs(Q_mean - i) for i in res_dt[2]])
            Q_res = res_dt.index[ind]
            return Q_res

        for mod in ['ai','phys']:
            q_dr_dt.loc[mod,'mortality'] = res_dt.loc[find_nearest_Q(q_dr_dt.loc[mod,'Q'] , res_dt),0]
            q_dr_dt.loc[mod,'std'] = res_dt.loc[find_nearest_Q(q_dr_dt.loc[mod,'Q'] , res_dt),1]


        v_cwpdis, ess = cwpdis_ess_eval(data)
        v_WIS, ess_wis = cal_WIS(data)
        v_phys = cal_phy_V(data)
        q_dr_dt.loc['ai', 'v_WIS'] = v_WIS
        q_dr_dt.loc['ai', 'v_CWPDIS'] = v_cwpdis
        q_dr_dt.loc['ai', 'effective_sample_size'] = ess
        q_dr_dt.loc['phys', 'effective_sample_size'] = ess_wis
        q_dr_dt.loc['phys', 'v_phys'] = v_phys
        q_dr_dt.to_csv(res_dir + 'qmean_and_deathreachrate.csv', encoding = 'gb18030')

        return q_dr_dt
    
    def action_concordant_rate(data):
        conc_dt = pd.DataFrame()
        for i,v in enumerate(action_types):
            phys_col = v + '_level'
            ai_col = v + '_level_ai' 
            conc_dt.loc[v, 'concordant_rate'] = str(round(np.mean(data[phys_col] == data[ai_col])*100,1)) + '%'
            
        conc_dt.loc['all', 'concordant_rate'] = str(round(np.mean(data[phys_action] == data[ai_action])*100,1)) + '%'
        
        conc_dt.to_csv(res_dir + 'action_concordant_rate.csv', encoding = 'gb18030')
        
        return conc_dt
   
    def sliding_mean(data_array, window=1):
        new_list = []
        for i in range(len(data_array)):
            indices = range(max(i - window + 1, 0),
                            min(i + window + 1, len(data_array)))
            avg = 0
            for j in indices:
                avg += data_array[j]
            avg /= float(len(indices))
            new_list.append(avg)     
        return np.array(new_list)
    def make_df_diff(op_actions, df_in):
        op_vaso_med = []
        op_iv_med = []
        for action in op_actions:
            iv,vaso = inv_action_map[action]
            op_vaso_med.append(vaso_vals[vaso])
            op_iv_med.append(iv_vals[iv])
        iv_diff = np.array(op_iv_med) - np.array(df_in['ori_iv_fluids'])
        vaso_diff = np.array(op_vaso_med) - np.array(df_in['ori_vasopressors'])
        df_diff = pd.DataFrame()
        df_diff['mort'] = np.array(df_in['mortality_hospital'])
        df_diff['iv_diff'] = iv_diff
        df_diff['vaso_diff'] = vaso_diff
        return df_diff
    
    def cwpdis_ess_eval(df):
        df = df.sort_values(by = ['stay_id','step_id'])
        data = df.copy()
        data['step_id'] = pd.to_numeric(df['step_id'], downcast='integer')
        # data.head()
        # data['concordant'] = [random.randint(0,1) for i in range(len(data))]
        data['concordant'] = data.apply(lambda x: (x[phys_action] == x[ai_action]) +0 ,axis = 1)
        
        # tag pnt
        data = data.groupby('stay_id').apply(cal_pnt)
        v_cwpdis = 0
        for t in range(1, max(data['step_id'])+1):
            tmp = data[data['step_id'] == t-1]
            if sum(tmp['pnt']) > 0:
                v_cwpdis += setting.GAMMA**t * (sum(tmp['reward']*tmp['pnt'])/sum(tmp['pnt']))

        ess = sum(data['pnt'])

        return v_cwpdis, ess
        
    def cal_pnt(dt):
        dt['conc_cumsum'] = dt['concordant'].cumsum()
        dt['pnt'] = (dt['conc_cumsum'] == (dt['step_id'] + 1))+0
        return dt
    
    def calculate_p1t(dt, gamma = 0.9):
        gamma = setting.GAMMA
        dt = dt.reset_index(drop=True)
        dt['cur_index'] = dt.index
        dt['p1_'] = dt.apply(lambda x: 1 if x['is_equal']==1 else 0, axis =1)
        dt['p1H_'] = reduce(lambda y,z: y*z, dt['p1_'].tolist())
        dt['gamma_rt'] = dt.apply(lambda x: gamma**(x['cur_index'])*x['reward'],axis =1)
        dt['sum_gamma_rt'] = sum(dt['gamma_rt'])
        dt = dt.loc[max(dt.index)]
        return dt

    def cal_WIS(data):
        df = data.copy()
        df = df.reset_index(drop = True)
        df['is_equal'] = df.apply(lambda x: (x[phys_action] == x[ai_action]) +0 ,axis = 1)
        df = df.sort_values(by = ['stay_id','step_id'])
        tmp_df = df.groupby('stay_id').apply(calculate_p1t)
        D = len(df.stay_id.unique())
        wH = sum(tmp_df['p1H_'])/D
        Ess_wis = sum(tmp_df['p1H_'])
        tmp_df['Vwis']=tmp_df.apply(lambda x: x['p1H_']/wH*x['sum_gamma_rt'] if wH*x['sum_gamma_rt'] != 0 else 0,axis =1)
        WIS = sum(tmp_df['Vwis'])/D
        return WIS, Ess_wis

    
    def cal_phy_V(df):
        df = df.copy()
        phys_vals = []
        unique_ids = df['stay_id'].unique()
        for uid in unique_ids:
            traj = df.loc[df['stay_id']==uid]
            ret = 0
            reversed_traj = traj.iloc[::-1]
            for row in reversed_traj.index:
                ret = reversed_traj.loc[row,'reward']+setting.GAMMA*ret
            if ret >30 or ret <-30:
                continue
            phys_vals.append(ret)
        return np.mean(phys_vals)
############################# main starts here #####################################################
    
    
    # prepare for data
    np.random.seed(523)
    random.seed(523)

    data = data.reset_index(drop = True)
    action_types = ['iv_fluids', 'vasopressors']
    
    phys_action = 'phys_action'
    ai_action = 'ai_action'
    
#     if phys_action not in data.columns.tolist():
    data[phys_action] = data[['iv_fluids_quantile', 'vasopressors_quantile']].apply(lambda x: int(x[0]*5 + x[1] -6),axis = 1)
    data[phys_action] = data[phys_action].astype('int64')
    data['phys_actions']=data['phys_action']    

    iv_val = data.iv_fluids[data.iv_fluids>np.min(data.iv_fluids)].quantile([0.25,0.5,0.75]).values
    vaso_val = data.vasopressors[data.vasopressors>np.min(data.vasopressors)].quantile([0.25,0.5,0.75]).values
    
    data['ai_mixer_iv_quantile'] = data['ai_action_qmix_IV'].apply(lambda x: 1 if x<=iv_val[0] 
                                                                   else 2 if x<=iv_val[1] else 3 if x<=iv_val[2] else 4)
    
    data['ai_mixer_vasso_quantile'] = data['ai_action_qmix_Vasso'].apply(lambda x: 1 if x<=vaso_val[0] 
                                                                         else 2 if x<=vaso_val[1] else 3 if x<=vaso_val[2] else 4)
    
    data['ai_single_iv_quantile'] = data['ai_action_IV_only'].apply(lambda x: 1 if x<=iv_val[0] 
                                                                    else 2 if x<=iv_val[1] else 3 if x<=iv_val[2] else 4)
    
    data['ai_single_vasso_quantile'] = data['ai_action_Vasso_only'].apply(lambda x: 1 if x<=vaso_val[0] 
                                                                          else 2 if x<=vaso_val[1] else 3 if x<=vaso_val[2] else 4)

    
    def compute_overall_ai_action(ai_action_master, ai_mixer_iv_quantile, ai_mixer_vasso_quantile,ai_single_iv_quantile,ai_single_vasso_quantile):
        if ai_action_master == 0:
            overall_action = 0
        elif ai_action_master == 1:
            overall_action = ai_single_iv_quantile*5
        elif ai_action_master == 2:
            overall_action = ai_single_vasso_quantile
        elif ai_action_master == 3:
            overall_action = ai_mixer_iv_quantile*5 + ai_mixer_vasso_quantile  
        return overall_action

    def compute_overall_phy_Q(phys_action, Q_0, Q_phys_IV_only, Q_phys_Vasso_only, Q_phys_qmix):
        phys_iv = phys_action/5
        phys_vasso = phys_action%5
        if (phys_iv == 0) and (phys_vasso == 0):
            phys_Q = Q_0
        if (phys_iv != 0) and (phys_vasso == 0):
            phys_Q = Q_phys_IV_only
        if (phys_iv == 0) and (phys_vasso != 0):
            phys_Q = Q_phys_Vasso_only
        else:
            phys_Q = Q_phys_qmix

        return phys_Q
    
    def compute_overall_ai_Q(ai_action_master, Q_0, Q_ai_IV_only, Q_ai_Vasso_only, Q_ai_qmix):
        if ai_action_master == 0:
            ai_Q = Q_0
        elif ai_action_master == 1:
            ai_Q = Q_ai_IV_only
        elif ai_action_master == 2:
            ai_Q = Q_ai_Vasso_only
        elif ai_action_master == 3:
            ai_Q = Q_ai_qmix

        return ai_Q

    data['ai_overal_continous'] = data.apply(lambda x: compute_overall_ai_action(x['ai_action_master'], 
                                                                                x['ai_mixer_iv_quantile'],
                                                                                x['ai_mixer_vasso_quantile'],
                                                                                x['ai_single_iv_quantile'],
                                                                                x['ai_single_vasso_quantile']), axis=1)

    
    data[ai_action] = data['ai_overal_continous']
    data['deeprl2_actions'] = data['ai_action']
    data['phys_Q'] = data.apply(lambda x: compute_overall_phy_Q(x['phys_action'],
                                                                x['Q_0'],
                                                                x['Q_1'],
                                                                x['Q_2'],
                                                                x['Q_3']), axis=1)    
    data['ai_Q'] = data['Q_ai_master']
    data[action_types[0] + '_level_ai'] = (data[ai_action]/5+1).apply(lambda x: int(x))
    data[action_types[1] + '_level_ai'] = (data[ai_action]%5+1).apply(lambda x: int(x))
    
    for v in action_types:
        data[v + '_diff'] = data[v + '_level'] - data[v + '_level_ai']
        data[v + '_conc'] = (data[v + '_level'] == data[v + '_level_ai']) + 0
        
    data = data.groupby('stay_id').apply(tag_conc_rate_and_diff_mean)

    for v in action_types:    
        data[v + '_diff_mean_level'] = data[v + '_diff_mean'].apply(discre_diff_level)
        data[v + '_conc_rate_level'] = data[v + '_conc_rate'].apply(discre_conc_level)

    data = data.reset_index(drop = True).copy()
    

    # Create dir for result
    res_dir = 'result/'+str(phase)+'/'+ datetime.now().strftime('%Y-%m-%d-%H-%M-%S') +'_K'+str(setting.hidden_factor)+'_ItrMaster'+str(setting.ITERATION_ROUND)+'lr_Q'+str(setting.lr_Q)+ '_batch' + str(setting.BATCH_SIZE) +'_seed' + str(setting.SEED)+ '_nn'+str(setting.nn)+'_replace' +str(setting.replace_target_iter)+'/'    
 

    if os.path.isdir(res_dir) == False:
        os.makedirs(res_dir)            

    # plots of actions_difference vs mortality
    




    interventions = data[['ori_vasopressors','ori_iv_fluids']]


    # In[8]:


    adjusted_vaso = interventions["ori_vasopressors"][interventions["ori_vasopressors"] >0]
    adjusted_iv = interventions["ori_iv_fluids"][interventions["ori_iv_fluids"]>0]


    # In[9]:


    vaso_vals = [0]
    vaso_vals.extend(adjusted_vaso.quantile([0.125,0.375,0.625,0.875]))
    iv_vals = [0]
    iv_vals.extend(adjusted_iv.quantile([0.125,0.375,0.625,0.875]))




    # In[12]:


    # Low SOFA
    df_test_orig_low = data[data['ori_sofa_24hours'] <= 5]

    # # Middling SOFA
    df_test_orig_mid = data[data['ori_sofa_24hours'] > 5]
    df_test_orig_mid = df_test_orig_mid[df_test_orig_mid['ori_sofa_24hours'] < 15]

    # # High SOFA
    df_test_orig_high = data[data['ori_sofa_24hours'] >= 15]


    # In[13]:


    # Now re-select the phys_actions, autoencode_actions, and deeprl2_actions based on the statified dataset
    deeprl2_actions_low = df_test_orig_low['deeprl2_actions'].values
    phys_actions_low = df_test_orig_low['phys_action'].values

    deeprl2_actions_mid = df_test_orig_mid['deeprl2_actions'].values
    phys_actions_mid = df_test_orig_mid['phys_action'].values

    deeprl2_actions_high = df_test_orig_high['deeprl2_actions'].values
    phys_actions_high = df_test_orig_high['phys_action'].values


    # In[14]:


    inv_action_map = {}
    count = 0
    for i in range(5):
        for j in range(5):
            inv_action_map[count] = [i,j]
            count += 1


    # In[15]:


    phys_actions_low_tuple = [None for i in range(len(phys_actions_low))]
    deeprl2_actions_low_tuple = [None for i in range(len(phys_actions_low))]

    phys_actions_mid_tuple = [None for i in range(len(phys_actions_mid))]
    deeprl2_actions_mid_tuple = [None for i in range(len(phys_actions_mid))]

    phys_actions_high_tuple = [None for i in range(len(phys_actions_high))]
    deeprl2_actions_high_tuple = [None for i in range(len(phys_actions_high))]

    for i in range(len(phys_actions_low)):
        phys_actions_low_tuple[i] = inv_action_map[phys_actions_low[i]]
        deeprl2_actions_low_tuple[i] = inv_action_map[deeprl2_actions_low[i]]

    for i in range(len(phys_actions_mid)):
        phys_actions_mid_tuple[i] = inv_action_map[phys_actions_mid[i]]
        deeprl2_actions_mid_tuple[i] = inv_action_map[deeprl2_actions_mid[i]]

    for i in range(len(phys_actions_high)):
        phys_actions_high_tuple[i] = inv_action_map[phys_actions_high[i]]
        deeprl2_actions_high_tuple[i] = inv_action_map[deeprl2_actions_high[i]]


    # In[16]:


    phys_actions_low_tuple = np.array(phys_actions_low_tuple)
    deeprl2_actions_low_tuple = np.array(deeprl2_actions_low_tuple).reshape((-1,2))

    phys_actions_mid_tuple = np.array(phys_actions_mid_tuple)
    deeprl2_actions_mid_tuple = np.array(deeprl2_actions_mid_tuple).reshape((-1,2))

    phys_actions_high_tuple = np.array(phys_actions_high_tuple)
    deeprl2_actions_high_tuple = np.array(deeprl2_actions_high_tuple).reshape((-1,2))


    # In[17]:


    phys_actions_low_iv = phys_actions_low_tuple[:,0]
    phys_actions_low_vaso = phys_actions_low_tuple[:,1]
    hist_ph1, x_edges, y_edges = np.histogram2d(phys_actions_low_iv, phys_actions_low_vaso, bins=5)

    phys_actions_mid_iv = phys_actions_mid_tuple[:,0]
    phys_actions_mid_vaso = phys_actions_mid_tuple[:,1]
    hist_ph2, _, _ = np.histogram2d(phys_actions_mid_iv, phys_actions_mid_vaso, bins=5)

    phys_actions_high_iv = phys_actions_high_tuple[:,0]
    phys_actions_high_vaso = phys_actions_high_tuple[:,1]
    hist_ph3, _, _ = np.histogram2d(phys_actions_high_iv, phys_actions_high_vaso, bins=5)


    # In[18]:


    deeprl2_actions_low_iv = deeprl2_actions_low_tuple[:,0]
    deeprl2_actions_low_vaso = deeprl2_actions_low_tuple[:,1]
    hist_drl1, _, _ = np.histogram2d(deeprl2_actions_low_iv, deeprl2_actions_low_vaso, bins=5)

    deeprl2_actions_mid_iv = deeprl2_actions_mid_tuple[:,0]
    deeprl2_actions_mid_vaso = deeprl2_actions_mid_tuple[:,1]
    hist_drl2, _, _ = np.histogram2d(deeprl2_actions_mid_iv, deeprl2_actions_mid_vaso, bins=5)

    deeprl2_actions_high_iv = deeprl2_actions_high_tuple[:,0]
    deeprl2_actions_high_vaso = deeprl2_actions_high_tuple[:,1]
    hist_drl3, _, _ = np.histogram2d(deeprl2_actions_high_iv, deeprl2_actions_high_vaso, bins=5)


    # In[19]:


    x_edges = np.arange(-0.5,5)
    y_edges = np.arange(-0.5,5)


    # In[20]:


    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16,4))
    ax1.imshow(np.flipud(hist_drl1), cmap="Blues",extent=[x_edges[0], x_edges[-1],  y_edges[0],y_edges[-1]])
    ax2.imshow(np.flipud(hist_drl2), cmap="OrRd", extent=[x_edges[0], x_edges[-1],  y_edges[0],y_edges[-1]])
    ax3.imshow(np.flipud(hist_drl3), cmap="Greens", extent=[x_edges[0], x_edges[-1],  y_edges[0],y_edges[-1]])


    # ax1.grid(color='b', linestyle='-', linewidth=1)
    # ax2.grid(color='r', linestyle='-', linewidth=1)
    # ax3.grid(color='g', linestyle='-', linewidth=1)

    # Major ticks
    ax1.set_xticks(np.arange(0, 5, 1));
    ax1.set_yticks(np.arange(0, 5, 1));
    ax2.set_xticks(np.arange(0, 5, 1));
    ax2.set_yticks(np.arange(0, 5, 1));
    ax3.set_xticks(np.arange(0, 5, 1));
    ax3.set_yticks(np.arange(0, 5, 1));


    # Labels for major ticks
    ax1.set_xticklabels(np.arange(0, 5, 1));
    ax1.set_yticklabels(np.arange(0, 5, 1));
    ax2.set_xticklabels(np.arange(0, 5, 1));
    ax2.set_yticklabels(np.arange(0, 5, 1));
    ax3.set_xticklabels(np.arange(0, 5, 1));
    ax3.set_yticklabels(np.arange(0, 5, 1));


    # Minor ticks
    ax1.set_xticks(np.arange(-.5, 5, 1), minor=True);
    ax1.set_yticks(np.arange(-.5, 5, 1), minor=True);
    ax2.set_xticks(np.arange(-.5, 5, 1), minor=True);
    ax2.set_yticks(np.arange(-.5, 5, 1), minor=True);
    ax3.set_xticks(np.arange(-.5, 5, 1), minor=True);
    ax3.set_yticks(np.arange(-.5, 5, 1), minor=True);


    # Gridlines based on minor ticks
    ax1.grid(which='minor', color='b', linestyle='-', linewidth=1)
    ax2.grid(which='minor', color='r', linestyle='-', linewidth=1)
    ax3.grid(which='minor', color='g', linestyle='-', linewidth=1)


    im1 = ax1.pcolormesh(x_edges, y_edges, hist_drl1, cmap='Blues')
    f.colorbar(im1, ax=ax1, label = "Action counts")

    im2 = ax2.pcolormesh(x_edges, y_edges, hist_drl2, cmap='Greens')
    f.colorbar(im2, ax=ax2, label = "Action counts")

    im3 = ax3.pcolormesh(x_edges, y_edges, hist_drl3, cmap='OrRd')
    f.colorbar(im3, ax=ax3, label = "Action counts")


    ax1.set_ylabel('IV fluid dose', fontsize = 18)
    ax2.set_ylabel('IV fluid dose', fontsize = 18)
    ax3.set_ylabel('IV fluid dose')
    ax1.set_xlabel('Vasopressor dose', fontsize = 18)
    ax2.set_xlabel('Vasopressor dose', fontsize = 18)
    ax3.set_xlabel('Vasopressor dose', fontsize = 18)


    ax1.set_title("RL Low SOFA policy", fontsize = 18)
    ax2.set_title("RL Mid SOFA policy", fontsize = 18)
    ax3.set_title("RL High SOFA policy", fontsize = 18)

    SMALL_SIZE = 18
    MEDIUM_SIZE = 18
    BIGGER_SIZE = 18


    plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.tight_layout()
    plt.savefig(res_dir +'RL_Action_Heatmap.png',dpi = 300)


    # In[21]:


    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16,4))
    ax1.imshow(np.flipud(hist_ph1), cmap="Blues",extent=[x_edges[0], x_edges[-1],  y_edges[0],y_edges[-1]])
    ax2.imshow(np.flipud(hist_ph2), cmap="OrRd", extent=[x_edges[0], x_edges[-1],  y_edges[0],y_edges[-1]])
    ax3.imshow(np.flipud(hist_ph3), cmap="Greens", extent=[x_edges[0], x_edges[-1],  y_edges[0],y_edges[-1]])


    # ax1.grid(color='b', linestyle='-', linewidth=1)
    # ax2.grid(color='r', linestyle='-', linewidth=1)
    # ax3.grid(color='g', linestyle='-', linewidth=1)

    # Major ticks
    ax1.set_xticks(np.arange(0, 5, 1));
    ax1.set_yticks(np.arange(0, 5, 1));
    ax2.set_xticks(np.arange(0, 5, 1));
    ax2.set_yticks(np.arange(0, 5, 1));
    ax3.set_xticks(np.arange(0, 5, 1));
    ax3.set_yticks(np.arange(0, 5, 1));


    # Labels for major ticks
    ax1.set_xticklabels(np.arange(0, 5, 1));
    ax1.set_yticklabels(np.arange(0, 5, 1));
    ax2.set_xticklabels(np.arange(0, 5, 1));
    ax2.set_yticklabels(np.arange(0, 5, 1));
    ax3.set_xticklabels(np.arange(0, 5, 1));
    ax3.set_yticklabels(np.arange(0, 5, 1));


    # Minor ticks
    ax1.set_xticks(np.arange(-.5, 5, 1), minor=True);
    ax1.set_yticks(np.arange(-.5, 5, 1), minor=True);
    ax2.set_xticks(np.arange(-.5, 5, 1), minor=True);
    ax2.set_yticks(np.arange(-.5, 5, 1), minor=True);
    ax3.set_xticks(np.arange(-.5, 5, 1), minor=True);
    ax3.set_yticks(np.arange(-.5, 5, 1), minor=True);


    # Gridlines based on minor ticks
    ax1.grid(which='minor', color='b', linestyle='-', linewidth=1)
    ax2.grid(which='minor', color='r', linestyle='-', linewidth=1)
    ax3.grid(which='minor', color='g', linestyle='-', linewidth=1)


    im1 = ax1.pcolormesh(x_edges, y_edges, hist_ph1, cmap='Blues')
    f.colorbar(im1, ax=ax1, label = "Action counts")

    im2 = ax2.pcolormesh(x_edges, y_edges, hist_ph2, cmap='Greens')
    f.colorbar(im2, ax=ax2, label = "Action counts")

    im3 = ax3.pcolormesh(x_edges, y_edges, hist_ph3, cmap='OrRd')
    f.colorbar(im3, ax=ax3, label = "Action counts")


    ax1.set_ylabel('IV fluid dose')
    ax2.set_ylabel('IV fluid dose')
    ax3.set_ylabel('IV fluid dose')
    ax1.set_xlabel('Vasopressor dose')
    ax2.set_xlabel('Vasopressor dose')
    ax3.set_xlabel('Vasopressor dose')


    ax1.set_title("Physician Low SOFA policy")
    ax2.set_title("Physician Mid SOFA policy")
    ax3.set_title("Physician High SOFA policy")
    SMALL_SIZE = 15
    MEDIUM_SIZE = 15
    BIGGER_SIZE = 18


    plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.tight_layout()
    plt.savefig(res_dir +'Phys_Action_Heatmap.png',dpi = 300)





    def make_iv_plot_data(df_diff):
        bin_medians_iv = []
        mort_iv = []
        mort_std_iv= []
        i = -800
        while i <= 900:
            count =df_diff.loc[(df_diff['iv_diff']>i-50) & (df_diff['iv_diff']<i+50)]
            try:
                res = sum(count['mort'])/float(len(count))
                if len(count) >=2:
                    bin_medians_iv.append(i)
                    mort_iv.append(res)
                    mort_std_iv.append(sem(count['mort']))
            except ZeroDivisionError:
                pass
            i += 100
        return bin_medians_iv, mort_iv, mort_std_iv


    # In[24]:



    def make_vaso_plot_data(df_diff):
        bin_medians_vaso = []
        mort_vaso= []
        mort_std_vaso= []
        i = -0.6
        while i <= 0.6:
            count =df_diff.loc[(df_diff['vaso_diff']>i-0.05) & (df_diff['vaso_diff']<i+0.05)]
            try:
                res = sum(count['mort'])/float(len(count))
                if len(count) >=2:
                    bin_medians_vaso.append(i)
                    mort_vaso.append(res)
                    mort_std_vaso.append(sem(count['mort']))
            except ZeroDivisionError:
                pass
            i += 0.1
        return bin_medians_vaso, mort_vaso, mort_std_vaso


    # In[25]:


    df_diff_low = make_df_diff(deeprl2_actions_low, df_test_orig_low)
    df_diff_mid = make_df_diff(deeprl2_actions_mid, df_test_orig_mid)
    df_diff_high = make_df_diff(deeprl2_actions_high, df_test_orig_high)


    # In[26]:


    bin_med_iv_deep_low, mort_iv_deep_low, mort_std_iv_deep_low = make_iv_plot_data(df_diff_low)
    bin_med_vaso_deep_low, mort_vaso_deep_low, mort_std_vaso_deep_low = make_vaso_plot_data(df_diff_low)

    bin_med_iv_deep_mid, mort_iv_deep_mid, mort_std_iv_deep_mid = make_iv_plot_data(df_diff_mid)
    bin_med_vaso_deep_mid, mort_vaso_deep_mid, mort_std_vaso_deep_mid = make_vaso_plot_data(df_diff_mid)

    bin_med_iv_deep_high, mort_iv_deep_high, mort_std_iv_deep_high = make_iv_plot_data(df_diff_high)
    bin_med_vaso_deep_high, mort_vaso_deep_high, mort_std_vaso_deep_high = make_vaso_plot_data(df_diff_high)


    # In[38]:
    def diff_plot(med_vaso, mort_vaso, std_vaso, med_iv, mort_iv, std_iv, col, title):
        f, ((ax1, ax2)) = plt.subplots(1, 2, sharex='col', sharey='row', figsize = (10,4))
        f.set_facecolor('white')
        ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax1.tick_params(axis='both', which='major', pad=0)
        ax2.tick_params(axis='both', which='major', pad=0)
        step = 2
        if col == 'r':
            fillcol = 'lightsalmon'
        elif col == 'g':
            fillcol = 'palegreen'
            step = 1
        elif col == 'b':
            fillcol = 'lightblue'
        ax1.plot(med_vaso, sliding_mean(mort_vaso), color=col)
        ax1.fill_between(med_vaso, sliding_mean(mort_vaso) - 1*std_vaso,  
                         sliding_mean(mort_vaso) + 1*std_vaso, color=fillcol)
        t = title + ": Vasopressors"
        ax1.set_title(t)
        x_r = [i/10.0 for i in range(-6,8,2)]

        y_r = [i/20.0 for i in range(0,20,step)]
        ax1.set_xticks(x_r)
        ax1.set_yticks(y_r)
        ax1.grid()

        ax2.plot(med_iv, sliding_mean(mort_iv), color=col)
        ax2.fill_between(med_iv, sliding_mean(mort_iv) - 1*std_iv,  
                         sliding_mean(mort_iv) + 1*std_iv, color=fillcol)
        t = title + ": IV fluids"
        ax2.set_title(t)
        x_iv = [i for i in range(-800,900,400)]
        ax2.set_xticks(x_iv)
        ax2.grid()

        f.text(0.3, 0.01, 'Difference in vasopressor dose', ha='center', fontsize=18)
        f.text(0.725, 0.01, 'Difference in IV dose', ha='center', fontsize=18)
        f.text(0.05, 0.5, 'Mortality', va='center', rotation='vertical', fontsize = 20)
        f.set_facecolor("white")
        SMALL_SIZE = 12
        MEDIUM_SIZE = 15
        BIGGER_SIZE = 18


    #     plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    #     plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels

        ax1.title.set_fontsize(20)
        ax2.title.set_fontsize(20)
        f.savefig(res_dir +title+'_Action_vs_mortality.png',dpi = 300)




    diff_plot(bin_med_vaso_deep_low, mort_vaso_deep_low, mort_std_vaso_deep_low, 
              bin_med_iv_deep_low, mort_iv_deep_low, mort_std_iv_deep_low, 'b', 'Low SOFA')


    # In[29]:


    diff_plot(bin_med_vaso_deep_mid, mort_vaso_deep_mid, mort_std_vaso_deep_mid, 
              bin_med_iv_deep_mid, mort_iv_deep_mid, mort_std_iv_deep_mid, 'g', 'Medium SOFA')


    # In[30]:


    diff_plot(bin_med_vaso_deep_high, mort_vaso_deep_high, mort_std_vaso_deep_high, 
              bin_med_iv_deep_high, mort_iv_deep_high, mort_std_iv_deep_high, 'r', 'High SOFA')


#     # In[31]:





    q_vals_phys = data['phys_Q']    
    pp = pd.Series(q_vals_phys)
    # phys_df = pd.DataFrame(pp, columns=['value'])
    phys_df = pd.DataFrame(pp)
    phys_df['mort'] = copy.deepcopy(np.array(data['mortality_hospital']))
    

    bin_medians = []
    mort = []
    mort_std = []
    
    increment = round((q_vals_phys.quantile(0.99)- q_vals_phys.quantile(0.01))/20, 5)
    half_increment = increment/2
    i = q_vals_phys.quantile([0.01,0.99]).values[0]
    while (i <= q_vals_phys.quantile([0.01,0.95]).values[1]):
        count =phys_df.loc[(phys_df.iloc[:, 0]>i-half_increment) & (phys_df.iloc[:, 0]<i+half_increment)]
        try:
            res = sum(count['mort'])/float(len(count))
            if len(count) >=2:
                bin_medians.append(i)
                mort.append(res)
                mort_std.append(sem(count['mort']))
        except ZeroDivisionError:
            pass
        i += half_increment
        
    def sliding_mean(data_array, window=2):
        new_list = []
        for i in range(len(data_array)):
            indices = range(max(i - window + 1, 0),
                            min(i + window + 1, len(data_array)))
            avg = 0
            for j in indices:
                avg += data_array[j]
            avg /= float(len(indices))
            new_list.append(avg)     
        return np.array(new_list)
    
    fig =plt.figure(figsize=(5, 4.5), facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(bin_medians, sliding_mean(mort))
    ax.fill_between(bin_medians, sliding_mean(mort) - 1*sliding_mean(mort_std),  
                     sliding_mean(mort) + 1*sliding_mean(mort_std), color='#ADD8E6')
    ax.grid()
    #     plt.xticks(range(-15,20,5))

    plt.xticks(fontsize=16)
    r = [float(i)/10 for i in range(0,11,1)]
    _ = plt.yticks(r, fontsize = 16)
    # _ = plt.title("Mortality vs Expected Return", fontsize=22)  
    _ = plt.ylabel("Mortality",fontsize=30) 
    _ = plt.xlabel("Expected Return",fontsize=30, labelpad=0) 
    fig.tight_layout()
    fig.savefig(res_dir+'q_vs_mortality.png',dpi = 300, bbox_inches='tight')

#     save original data
    df = data.copy()
    
#     ####################### Evaluate Low-level Individual Agents ####################
    # evaluate qmix agent

    data = df[(df['train_test']=='test') & (df['master_action']==3)]
    data = data.reset_index(drop=True)
    interventions = data[['ori_vasopressors','ori_iv_fluids']]
    adjusted_vaso = data["ori_vasopressors"][data["ori_vasopressors"] >0]
    adjusted_iv = data["ori_iv_fluids"][data["ori_iv_fluids"]>0]
    vaso_vals = [0]
    vaso_vals.extend(adjusted_vaso.quantile([0.125,0.375,0.625,0.875]))
    iv_vals = [0]
    iv_vals.extend(adjusted_iv.quantile([0.125,0.375,0.625,0.875]))
    deeprl2_action = data[['ai_mixer_iv_quantile','ai_mixer_vasso_quantile']]
    ai_iv = deeprl2_action.iloc[:,0]
    ai_vaso = deeprl2_action.iloc[:,1]
    df_diff = make_df_diff_new(ai_iv, ai_vaso, data)
    bin_med_iv_deep, mort_iv_deep, mort_std_iv_deep = make_iv_plot_data(df_diff)
    bin_med_vaso_deep, mort_vaso_deep, mort_std_vaso_deep = make_vaso_plot_data(df_diff)

    diff_iv_plot(bin_med_iv_deep, mort_iv_deep, mort_std_iv_deep, 'darkgoldenrod', "$M_{IV2}$")
    diff_vaso_plot(bin_med_vaso_deep, mort_vaso_deep, mort_std_vaso_deep, 'mediumslateblue', "$M_{VA2}$")


    # evaluate IV only agent

    data = df[(df['train_test']=='test') & (df['master_action']==1)]
    data = data.reset_index(drop = True)
    interventions = data[['ori_vasopressors','ori_iv_fluids']]
    adjusted_vaso = data["ori_vasopressors"][data["ori_vasopressors"] >0]
    adjusted_iv = data["ori_iv_fluids"][data["ori_iv_fluids"]>0]
    vaso_vals = [0]
    vaso_vals.extend(adjusted_vaso.quantile([0.125,0.375,0.625,0.875]))
    iv_vals = [0]
    iv_vals.extend(adjusted_iv.quantile([0.125,0.375,0.625,0.875]))
    deeprl2_action = data[['ai_single_iv_quantile', 'ai_single_vasso_quantile']]
    ai_iv = deeprl2_action.iloc[:,0]
    ai_vaso = deeprl2_action.iloc[:,1]
    df_diff = make_df_diff_new(ai_iv, ai_vaso, data)
    bin_med_iv_deep, mort_iv_deep, mort_std_iv_deep = make_iv_plot_data(df_diff)
    diff_iv_plot(bin_med_iv_deep, mort_iv_deep, mort_std_iv_deep, 'darkgoldenrod', "$M_{IV}$")

    # evaluate Vaso only agent
#     data = df[(df['train_test']=='test') & (df['master_action']==2)]
    temp = df[df['train_test']=='test']
    data = temp[temp['master_action'].isin([2,3]) ]
    data = data.reset_index(drop = True)
    interventions = data[['ori_vasopressors','ori_iv_fluids']]
    adjusted_vaso = data["ori_vasopressors"][data["ori_vasopressors"] >0]
    adjusted_iv = data["ori_iv_fluids"][data["ori_iv_fluids"]>0]
    vaso_vals = [0]
    vaso_vals.extend(adjusted_vaso.quantile([0.125,0.375,0.625,0.875]))
    iv_vals = [0]
    iv_vals.extend(adjusted_iv.quantile([0.125,0.375,0.625,0.875]))
    deeprl2_action = data[['ai_single_iv_quantile', 'ai_single_vasso_quantile']]
    ai_iv = deeprl2_action.iloc[:,0]
    ai_vaso = deeprl2_action.iloc[:,1]
    df_diff = make_df_diff_new(ai_iv, ai_vaso, data)
    bin_med_vaso_deep, mort_vaso_deep, mort_std_vaso_deep = make_vaso_plot_data(df_diff)
    diff_vaso_plot(bin_med_vaso_deep, mort_vaso_deep, mort_std_vaso_deep, 'mediumslateblue', "$M_{VA}$")    
#     ######################## Plot for Qmix action map [1-4] ############################################

        
    def plot_action_map(data, res_dir):
        data = data[data['ai_action_master']==3]

        data= data.reset_index(drop=True)
        action_types = ['iv_fluids', 'vasopressors']
        ai_action = 'ai_action'
        data['ai_action'] = data.apply(lambda x: int(x['ai_mixer_iv_quantile']*4 + x['ai_mixer_vasso_quantile'] -5),axis = 1)
        data['ai_action'] = data['ai_action'].astype('int64')


        inv_action_map1 = {}
        count = 0
        for i in range(4):
            for j in range(4):
                inv_action_map1[count] = [i,j]
                count += 1

        data['deeprl2_actions'] = data['ai_action']

        # Low SOFA
        df_test_orig_low = data[data['ori_sofa_24hours'] <= 5]
        # # Middling SOFA
        df_test_orig_mid = data[data['ori_sofa_24hours'] > 5]
        df_test_orig_mid = df_test_orig_mid[df_test_orig_mid['ori_sofa_24hours'] < 15]
        # # High SOFA
        df_test_orig_high = data[data['ori_sofa_24hours'] >= 15]

        deeprl2_actions_low = df_test_orig_low['deeprl2_actions'].values
        deeprl2_actions_mid = df_test_orig_mid['deeprl2_actions'].values
        deeprl2_actions_high = df_test_orig_high['deeprl2_actions'].values


        deeprl2_actions_low_tuple = [None for i in range(len(deeprl2_actions_low))]
        deeprl2_actions_mid_tuple = [None for i in range(len(deeprl2_actions_mid))]
        deeprl2_actions_high_tuple = [None for i in range(len(deeprl2_actions_high))]

        for i in range(len(deeprl2_actions_low)):
            deeprl2_actions_low_tuple[i] = inv_action_map1[deeprl2_actions_low[i]]

        for i in range(len(deeprl2_actions_mid)):
            deeprl2_actions_mid_tuple[i] = inv_action_map1[deeprl2_actions_mid[i]]

        for i in range(len(deeprl2_actions_high)):
            deeprl2_actions_high_tuple[i] = inv_action_map1[deeprl2_actions_high[i]]


        deeprl2_actions_low_tuple = np.array(deeprl2_actions_low_tuple).reshape((-1,2))
        deeprl2_actions_mid_tuple = np.array(deeprl2_actions_mid_tuple).reshape((-1,2))
        deeprl2_actions_high_tuple = np.array(deeprl2_actions_high_tuple).reshape((-1,2))

        deeprl2_actions_low_iv = deeprl2_actions_low_tuple[:,0]
        deeprl2_actions_low_vaso = deeprl2_actions_low_tuple[:,1]
        hist_drl1, _, _ = np.histogram2d(deeprl2_actions_low_iv, deeprl2_actions_low_vaso, bins=4)

        deeprl2_actions_mid_iv = deeprl2_actions_mid_tuple[:,0]
        deeprl2_actions_mid_vaso = deeprl2_actions_mid_tuple[:,1]
        hist_drl2, _, _ = np.histogram2d(deeprl2_actions_mid_iv, deeprl2_actions_mid_vaso, bins=4)

        deeprl2_actions_high_iv = deeprl2_actions_high_tuple[:,0]
        deeprl2_actions_high_vaso = deeprl2_actions_high_tuple[:,1]
        hist_drl3, _, _ = np.histogram2d(deeprl2_actions_high_iv, deeprl2_actions_high_vaso, bins=4)


        x_edges = np.arange(-0.5,4)
        y_edges = np.arange(-0.5,4)

        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16,4))
        ax1.imshow(np.flipud(hist_drl1), cmap="Blues",extent=[x_edges[0], x_edges[-1],  y_edges[0],y_edges[-1]])
        ax2.imshow(np.flipud(hist_drl2), cmap="OrRd", extent=[x_edges[0], x_edges[-1],  y_edges[0],y_edges[-1]])
        ax3.imshow(np.flipud(hist_drl3), cmap="Greens", extent=[x_edges[0], x_edges[-1],  y_edges[0],y_edges[-1]])
        # Major ticks
        ax1.set_xticks(np.arange(0, 4, 1));
        ax1.set_yticks(np.arange(0, 4, 1));
        ax2.set_xticks(np.arange(0, 4, 1));
        ax2.set_yticks(np.arange(0, 4, 1));
        ax3.set_xticks(np.arange(0, 4, 1));
        ax3.set_yticks(np.arange(0, 4, 1));


        # Labels for major ticks
        ax1.set_xticklabels(np.arange(1, 5, 1));
        ax1.set_yticklabels(np.arange(1, 5, 1));
        ax2.set_xticklabels(np.arange(1, 5, 1));
        ax2.set_yticklabels(np.arange(1, 5, 1));
        ax3.set_xticklabels(np.arange(1, 5, 1));
        ax3.set_yticklabels(np.arange(1, 5, 1));


        # Minor ticks
        ax1.set_xticks(np.arange(-.5, 4, 1), minor=True);
        ax1.set_yticks(np.arange(-.5, 4, 1), minor=True);
        ax2.set_xticks(np.arange(-.5, 4, 1), minor=True);
        ax2.set_yticks(np.arange(-.5, 4, 1), minor=True);
        ax3.set_xticks(np.arange(-.5, 4, 1), minor=True);
        ax3.set_yticks(np.arange(-.5, 4, 1), minor=True);


        # Gridlines based on minor ticks
        ax1.grid(which='minor', color='b', linestyle='-', linewidth=1)
        ax2.grid(which='minor', color='r', linestyle='-', linewidth=1)
        ax3.grid(which='minor', color='g', linestyle='-', linewidth=1)


        im1 = ax1.pcolormesh(x_edges, y_edges, hist_drl1, cmap='Blues')
        f.colorbar(im1, ax=ax1, label = "Action counts")

        im2 = ax2.pcolormesh(x_edges, y_edges, hist_drl2, cmap='Greens')
        f.colorbar(im2, ax=ax2, label = "Action counts")

        im3 = ax3.pcolormesh(x_edges, y_edges, hist_drl3, cmap='OrRd')
        f.colorbar(im3, ax=ax3, label = "Action counts")


        ax1.set_ylabel('IV fluid dose')
        ax2.set_ylabel('IV fluid dose')
        ax3.set_ylabel('IV fluid dose')
        ax1.set_xlabel('Vasopressor dose')
        ax2.set_xlabel('Vasopressor dose')
        ax3.set_xlabel('Vasopressor dose')


        ax1.set_title("RL Low SOFA policy")
        ax2.set_title("RL Mid SOFA policy")
        ax3.set_title("RL High SOFA policy")

        SMALL_SIZE = 15
        MEDIUM_SIZE = 15
        BIGGER_SIZE = 18


        plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
        plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.tight_layout()

        plt.savefig(res_dir+'Qmix_Action_Heatmap.png',dpi = 300)    
# ##############################################################
    
    
    data = df.copy()        
    
    # Quantitavie evaluation of the RL policy
    REWARD_FUN = setting.REWARD_FUN
    data['reward']= data.apply(eval('setting.' + REWARD_FUN) , axis = 1)
    
    res_dt = q_vs_outcome(data)
    q_dr_dt = quantitive_eval(data, res_dt)
    conc_dt = action_concordant_rate(data)
    print (q_dr_dt)
    print (conc_dt)
    plot_loss(pre_master_loss,"pretrain")
    plot_loss(master_loss,"master")
    plot_loss(vasso_loss,"vasso_only")
    plot_loss(iv_loss,"iv_only")
    plot_loss(mixer_loss,"qmix")
#     # prepare for data
    if phase in 'test_Embedding':
        data= df[df['train_test']=='test']
    else:
        data= df[df['train_test']=='train']
    plot_action_map(data, res_dir)

    print("Done!")

