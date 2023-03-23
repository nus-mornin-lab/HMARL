

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


# def run_eval(data, loss, datatype="eICU", phase = "train"):
def run_eval(data, pre_master_loss, master_loss, iv_loss, vasso_loss, mixer_loss, datatype = 'mimic', phase = "train"):
    str_quant = setting.str_quant
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
        res_dt.reset_index(drop = True)

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
            mort_temp = res_dt.loc[find_nearest_Q(q_dr_dt.loc[mod,'Q'] , res_dt),0]
            q_dr_dt.loc[mod,'mortality'] = mort_temp*100
            std_temp = res_dt.loc[find_nearest_Q(q_dr_dt.loc[mod,'Q'] , res_dt),1]
            q_dr_dt.loc[mod,'std'] = std_temp*100


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
            phys_col = v + str_quant
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

    
    action_types = ['iv_fluids', 'vasopressors']
    
    phys_action = 'phys_action'
    ai_action = 'ai_action'
    n_action = setting.action_level
    
    if phys_action not in data.columns.tolist():
        data[phys_action] = data.apply(lambda x: int(x[action_types[0]+str_quant]*n_action + x[action_types[1]+str_quant] -(n_action+1)),axis = 1)
        data[phys_action] = data[phys_action].astype('int64')
        data['phys_actions']=data['phys_action']
#     k = round(data.ai_action_qmix_IV.quantile(0.99),1)
#     j =round(data.ai_action_qmix_Vasso.quantile(0.99),1)
    
#     data.ai_action_qmix_IV = data.ai_action_qmix_IV.apply(lambda x: x if (x<=k and x>= -k) else k if x>k else -k)
#     data.ai_action_qmix_Vasso = data.ai_action_qmix_Vasso.apply(lambda x: x if (x<=j and x>= -j) else j if x>j else -j)

#     m = round(data.ai_action_IV_only.quantile(0.99),1)
#     n =round(data.ai_action_Vasso_only.quantile(0.99),1)
    
#     data.ai_action_IV_only = data.ai_action_IV_only.apply(lambda x: x if (x<=m and x>= -m) else m if x>m else -m)
#     data.ai_action_Vasso_only = data.ai_action_Vasso_only.apply(lambda x: x if (x<=n and x>= -n) else n if x>n else -n)

#     iv_val = data.iv_fluids[data.iv_fluids>np.min(data.iv_fluids)].quantile([0.25,0.5,0.75]).values
#     vaso_val = data.vasopressors[data.vasopressors>np.min(data.vasopressors)].quantile([0.25,0.5,0.75]).values
#     iv_val = data.iv_fluids[data.iv_fluids>np.min(data.iv_fluids)].quantile([0.375,0.60,0.875]).values
#     vaso_val = data.vasopressors[data.vasopressors>np.min(data.vasopressors)].quantile([0.375,0.60,0.875]).values
    
#     data['ai_mixer_iv_quantile'] = data['ai_action_qmix_IV'].apply(lambda x: 1 if x<=iv_val[0] 
#                                                                    else 2 if x<=iv_val[1] else 3 if x<=iv_val[2] else 4)
    
#     data['ai_mixer_vasso_quantile'] = data['ai_action_qmix_Vasso'].apply(lambda x: 1 if x<=vaso_val[0] 
#                                                                          else 2 if x<=vaso_val[1] else 3 if x<=vaso_val[2] else 4)
    
#     data['ai_single_iv_quantile'] = data['ai_action_IV_only'].apply(lambda x: 1 if x<=iv_val[0] 
#                                                                     else 2 if x<=iv_val[1] else 3 if x<=iv_val[2] else 4)
    
#     data['ai_single_vasso_quantile'] = data['ai_action_IV_only'].apply(lambda x: 1 if x<=vaso_val[0] 
#                                                                           else 2 if x<=vaso_val[1] else 3 if x<=vaso_val[2] else 4)

    
#     data['ai_mixer_iv_quantile'] = data['ai_action_qmix_IV'].apply(lambda x: 1 if x==0 else 3 if x==3 else 4 if x ==4 else 2)
    
#     data['ai_mixer_vasso_quantile'] = data['ai_action_qmix_Vasso'].apply(lambda x: 1 if x==0 else 3 if x==3 else 4 if x ==4 else 2)
    
#     data['ai_single_iv_quantile'] = data['ai_action_dis_IV_only'].apply(lambda x: 1 if x==0 else 3 if x==3 else 4 if x ==4 else 2)
    
#     data['ai_single_vasso_quantile'] = data['ai_action_dis_IV_only'].apply(lambda x: 1 if x==0 else 3 if x==3 else 4 if x ==4 else 2)
    
    data['ai_mixer_iv_quantile'] = data['ai_action_qmix_IV'].astype(int).apply(lambda x: 1 if x ==0 else x)
    data['ai_mixer_vasso_quantile'] = data['ai_action_qmix_Vasso'].astype(int).apply(lambda x: 1 if x ==0 else x)
    data['ai_single_iv_quantile'] = data['ai_action_dis_IV_only'].astype(int).apply(lambda x: 1 if x ==0 else x)
    data['ai_single_vasso_quantile'] = data['ai_action_dis_IV_only'].astype(int).apply(lambda x: 1 if x ==0 else x)
    
    
    def compute_overall_ai_action(ai_action_master, ai_mixer_iv_quantile, ai_mixer_vasso_quantile,ai_single_iv_quantile,ai_single_vasso_quantile):
        if ai_action_master == 0:
            overall_action = 0
        elif ai_action_master == 1:
            overall_action = (ai_single_iv_quantile)*n_action
        elif ai_action_master == 2:
            overall_action = (ai_single_vasso_quantile)
        elif ai_action_master == 3:
            overall_action = (ai_mixer_iv_quantile)*n_action + (ai_mixer_vasso_quantile)  
        return overall_action

    def compute_overall_phy_Q(phys_action, Q_0, Q_phys_IV_only, Q_phys_Vasso_only, Q_phys_qmix):
        phys_iv = int(phys_action/n_action)
        phys_vasso = int(phys_action%n_action)
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

    data['ai_overal'] = data.apply(lambda x: compute_overall_ai_action(x['ai_action_master'], 
                                                                                x['ai_mixer_iv_quantile'],
                                                                                x['ai_mixer_vasso_quantile'],
                                                                                x['ai_single_iv_quantile'],
                                                                                x['ai_single_vasso_quantile']), axis=1)
#     data['phys_Q'] = data.apply(lambda x: compute_overall_phy_Q(x['phys_action'],
#                                                                 x['Q_0'],
#                                                                 x['Q_phys_IV_only'],
#                                                                 x['Q_phys_Vasso_only'],
#                                                                 x['Q_phys_qmix']), axis=1)
    
#     data['ai_Q'] = data.apply(lambda x: compute_overall_ai_Q(x['ai_action_master'],
#                                                              x['Q_0'], 
#                                                              x['Q_ai_IV_only'],
#                                                              x['Q_ai_Vasso_only'], 
#                                                              x['Q_ai_qmix']), axis=1)   
       
#     phys_max = round(data.phys_Q.quantile(0.99),2)
#     phys_min = round(data.phys_Q.quantile(0.01),2)

#     ai_max = round(data.ai_Q.quantile(0.99),2)
#     ai_mim = round(data.ai_Q.quantile(0.01),2)

#     data.phys_Q = data.phys_Q.apply(lambda x: phys_min if x<=phys_min else x if x<=phys_max else phys_max)
#     data.ai_Q = data.ai_Q.apply(lambda x: ai_mim if x<=ai_mim else x if x<=ai_max else ai_max)

    data['ai_action'] = data['ai_overal']
    data['deeprl2_actions'] = data['ai_action']


    
    data['phys_Q'] = data.apply(lambda x: compute_overall_phy_Q(x['phys_action'],
                                                                x['Q_0'],
                                                                x['Q_1'],
                                                                x['Q_2'],
                                                                x['Q_3']), axis=1)
#     data['ai_Q'] = data.apply(lambda x: compute_overall_ai_Q(x['ai_action_master'],
#                                                              x['Q_0'], 
#                                                              x['Q_1'],
#                                                              x['Q_2'], 
#                                                              x['Q_3']), axis=1)      
    data['ai_Q'] = data['Q_ai_master']
    data[action_types[0] + '_level_ai'] = (data[ai_action]/n_action+1).apply(lambda x: int(x))
    data[action_types[1] + '_level_ai'] = (data[ai_action]%n_action+1).apply(lambda x: int(x))
    
    for v in action_types:
        data[v + '_diff'] = data[v + str_quant] - data[v + '_level_ai']
        data[v + '_conc'] = (data[v + str_quant] == data[v + '_level_ai']) + 0
        
    data = data.groupby('stay_id').apply(tag_conc_rate_and_diff_mean)

#     for v in action_types:    
#         data[v + '_diff_mean_level'] = data[v + '_diff_mean'].apply(discre_diff_level)
#         data[v + '_conc_rate_level'] = data[v + '_conc_rate'].apply(discre_conc_level)

    data = data.reset_index(drop = True).copy()
    

    # Create dir for result
    res_dir = 'result/'+str(phase)+'/'+ datetime.now().strftime('%Y-%m-%d-%H-%M-%S') +'_lrA1' + str(setting.lr_actor1) +'_lrA2' + str(setting.lr_actor2)+'_lrV' +'lr_Q'+str(setting.lr_Q)+ '_batch' + str(setting.BATCH_SIZE) +'_ItrPre' +str(setting.ITERATION_ROUND_PRETRAIN) +'_ItrIV'+ str(setting.ITERATION_ROUND_IV)+'_ItrVaso'+str(setting.ITERATION_ROUND_Vaso)+'_ItrMix'+str(setting.ITERATION_ROUND_QMIX)+'_ItrMaster'+str(setting.ITERATION_ROUND)+'_seed' + str(setting.SEED)+'_K'+str(setting.hidden_factor) + '_nn'+str(setting.nn)+'/'
 

    if os.path.isdir(res_dir) == False:
        os.makedirs(res_dir)            

    # plots of actions_difference vs mortality
    

    # In[7]:


    interventions = data[['ori_vasopressors','ori_iv_fluids']]


    # In[8]:


    adjusted_vaso = interventions["ori_vasopressors"][interventions["ori_vasopressors"] >0]
    adjusted_iv = interventions["ori_iv_fluids"][interventions["ori_iv_fluids"]>0]


    # In[9]:

    # # 7 levels
    # vaso_vals = [0]
    # vaso_vals.extend(adjusted_vaso.quantile([0.1428,0.2857,0.4286,0.5714,0.7142,0.5871]))
    # iv_vals = [0]
    # iv_vals.extend(adjusted_iv.quantile([0.1428,0.2857,0.4286,0.5714,0.7142,0.5871]))

    # # 9 levels
    # vaso_vals = [0]
    # vaso_vals.extend(adjusted_vaso.quantile([0.1111,0.2222,0.3333,0.4444,0.5556,0.6667,0.7778,0.8889]))
    # iv_vals = [0]
    # iv_vals.extend(adjusted_iv.quantile([0.1111,0.2222,0.3333,0.4444,0.5556,0.6667,0.7778,0.8889]))

    # # 11 levels
    # vaso_vals = [0]
    # vaso_vals.extend(adjusted_vaso.quantile([0.0909,0.1818,0.2727,0.3636,0.4545,0.5455,0.6364,0.7273,0.8182,0.9091]))
    # iv_vals = [0]
    # iv_vals.extend(adjusted_iv.quantile([0.0909,0.1818,0.2727,0.3636,0.4545,0.5455,0.6364,0.7273,0.8182,0.9091]))

    # # 10 levels
    vaso_vals = [0]
    vaso_vals.extend(adjusted_vaso.quantile([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]))
    iv_vals = [0]
    iv_vals.extend(adjusted_iv.quantile([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]))
    
    # # 20 levels
    # vaso_vals = [0]
    # vaso_vals.extend(adjusted_vaso.quantile([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]))
    # iv_vals = [0]
    # iv_vals.extend(adjusted_iv.quantile([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]))
    
    # # 50 levels
    # vaso_vals = [0]
    # vaso_vals.extend(adjusted_vaso.quantile([0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3, 0.32, 0.34, 0.36, 0.38, 0.4, 0.42, 0.44, 0.46, 0.48, 0.5, 0.52, 0.54, 0.56, 0.58, 0.6, 0.62, 0.64, 0.66, 0.68, 0.7, 0.72, 0.74, 0.76, 0.78, 0.8, 0.82, 0.84, 0.86, 0.88, 0.9, 0.92, 0.94, 0.96, 0.98]))
    # iv_vals = [0]
    # iv_vals.extend(adjusted_iv.quantile([0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3, 0.32, 0.34, 0.36, 0.38, 0.4, 0.42, 0.44, 0.46, 0.48, 0.5, 0.52, 0.54, 0.56, 0.58, 0.6, 0.62, 0.64, 0.66, 0.68, 0.7, 0.72, 0.74, 0.76, 0.78, 0.8, 0.82, 0.84, 0.86, 0.88, 0.9, 0.92, 0.94, 0.96, 0.98]))    

    # In[10]:


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
    phys_actions_low = df_test_orig_low['phys_actions'].values

    deeprl2_actions_mid = df_test_orig_mid['deeprl2_actions'].values
    phys_actions_mid = df_test_orig_mid['phys_actions'].values

    deeprl2_actions_high = df_test_orig_high['deeprl2_actions'].values
    phys_actions_high = df_test_orig_high['phys_actions'].values


    # In[14]:


    inv_action_map = {}
    count = 0
    for i in range(n_action):
        for j in range(n_action):
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
    deeprl2_actions_low_tuple = np.array(deeprl2_actions_low_tuple)

    phys_actions_mid_tuple = np.array(phys_actions_mid_tuple)
    deeprl2_actions_mid_tuple = np.array(deeprl2_actions_mid_tuple)

    phys_actions_high_tuple = np.array(phys_actions_high_tuple)
    deeprl2_actions_high_tuple = np.array(deeprl2_actions_high_tuple)


    # In[17]:


    phys_actions_low_iv = phys_actions_low_tuple[:,0]
    phys_actions_low_vaso = phys_actions_low_tuple[:,1]
    hist_ph1, x_edges, y_edges = np.histogram2d(phys_actions_low_iv, phys_actions_low_vaso, bins=n_action)

    phys_actions_mid_iv = phys_actions_mid_tuple[:,0]
    phys_actions_mid_vaso = phys_actions_mid_tuple[:,1]
    hist_ph2, _, _ = np.histogram2d(phys_actions_mid_iv, phys_actions_mid_vaso, bins=n_action)

    phys_actions_high_iv = phys_actions_high_tuple[:,0]
    phys_actions_high_vaso = phys_actions_high_tuple[:,1]
    hist_ph3, _, _ = np.histogram2d(phys_actions_high_iv, phys_actions_high_vaso, bins=n_action)


    # In[18]:


    deeprl2_actions_low_iv = deeprl2_actions_low_tuple[:,0]
    deeprl2_actions_low_vaso = deeprl2_actions_low_tuple[:,1]
    hist_drl1, _, _ = np.histogram2d(deeprl2_actions_low_iv, deeprl2_actions_low_vaso, bins=n_action)

    deeprl2_actions_mid_iv = deeprl2_actions_mid_tuple[:,0]
    deeprl2_actions_mid_vaso = deeprl2_actions_mid_tuple[:,1]
    hist_drl2, _, _ = np.histogram2d(deeprl2_actions_mid_iv, deeprl2_actions_mid_vaso, bins=n_action)

    deeprl2_actions_high_iv = deeprl2_actions_high_tuple[:,0]
    deeprl2_actions_high_vaso = deeprl2_actions_high_tuple[:,1]
    hist_drl3, _, _ = np.histogram2d(deeprl2_actions_high_iv, deeprl2_actions_high_vaso, bins=n_action)


    # In[19]:


    x_edges = np.arange(-0.5,n_action)
    y_edges = np.arange(-0.5,n_action)


    # In[20]:


    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16,4))
    ax1.imshow(np.flipud(hist_drl1), cmap="Blues",extent=[x_edges[0], x_edges[-1],  y_edges[0],y_edges[-1]])
    ax2.imshow(np.flipud(hist_drl2), cmap="OrRd", extent=[x_edges[0], x_edges[-1],  y_edges[0],y_edges[-1]])
    ax3.imshow(np.flipud(hist_drl3), cmap="Greens", extent=[x_edges[0], x_edges[-1],  y_edges[0],y_edges[-1]])


    # ax1.grid(color='b', linestyle='-', linewidth=1)
    # ax2.grid(color='r', linestyle='-', linewidth=1)
    # ax3.grid(color='g', linestyle='-', linewidth=1)

    # Major ticks
    ax1.set_xticks(np.arange(0, n_action, 1));
    ax1.set_yticks(np.arange(0, n_action, 1));
    ax2.set_xticks(np.arange(0, n_action, 1));
    ax2.set_yticks(np.arange(0, n_action, 1));
    ax3.set_xticks(np.arange(0, n_action, 1));
    ax3.set_yticks(np.arange(0, n_action, 1));


    # Labels for major ticks
    ax1.set_xticklabels(np.arange(0, n_action, 1));
    ax1.set_yticklabels(np.arange(0, n_action, 1));
    ax2.set_xticklabels(np.arange(0, n_action, 1));
    ax2.set_yticklabels(np.arange(0, n_action, 1));
    ax3.set_xticklabels(np.arange(0, n_action, 1));
    ax3.set_yticklabels(np.arange(0, n_action, 1));


    # Minor ticks
    ax1.set_xticks(np.arange(-.5, n_action, 1), minor=True);
    ax1.set_yticks(np.arange(-.5, n_action, 1), minor=True);
    ax2.set_xticks(np.arange(-.5, n_action, 1), minor=True);
    ax2.set_yticks(np.arange(-.5, n_action, 1), minor=True);
    ax3.set_xticks(np.arange(-.5, n_action, 1), minor=True);
    ax3.set_yticks(np.arange(-.5, n_action, 1), minor=True);


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

    
           
    
    # Quantitavie evaluation of the RL policy
    REWARD_FUN = setting.REWARD_FUN
    data['reward']= data.apply(eval('setting.' + REWARD_FUN) , axis = 1)
    
    res_dt = q_vs_outcome(data)
    q_dr_dt = quantitive_eval(data, res_dt)
    conc_dt = action_concordant_rate(data)
    print (q_dr_dt)
    print (conc_dt)

    print("Done!")
