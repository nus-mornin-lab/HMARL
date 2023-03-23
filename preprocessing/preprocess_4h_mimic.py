#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle


# In[ ]:


data_dir = './data'
output_dir = './mimic'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# In[ ]:


patients = pd.read_csv(os.path.join(data_dir,'cohort.csv'), engine='python')
demo = pd.read_csv(os.path.join(data_dir,'firstday_SIRS.csv'), engine='python')
patients = pd.merge(patients, demo[['stay_id','sirs']], on='stay_id', how='left')
# add mv_binary
mv = pd.read_csv(os.path.join(data_dir,'pivot_MV.csv'), engine='python')
patients = pd.merge(patients, mv[['stay_id','MechVent']], on='stay_id', how='left')
# add premorbid score
premorbid = pd.read_csv(os.path.join(data_dir,'elixhauser_quan_score.csv'), engine='python')
patients = pd.merge(patients, premorbid[['hadm_id','elixhauser_vanwalraven']], on='hadm_id', how='left')
patients['readmission']=0
patients['readmission']= patients['first_icu'].apply(lambda x: 1 if x>1 else 0)

patients['MechVent'].fillna(0)
patients['elixhauser_vanwalraven'].fillna(0)
# patients['sirs'].fillna(0)
patients['start_offset_actual'] = (pd.to_datetime(patients['suspected_infection_time']) - pd.Timedelta(hours=24) - pd.to_datetime(patients['intime'])).dt.total_seconds().div(60).astype(int)
patients['start_offset'] = [row['start_offset_actual'] if row['start_offset_actual'] > 0 else 0 for _, row in patients.iterrows()]
patients['end_offset'] = (pd.to_datetime(patients['suspected_infection_time']) + pd.Timedelta(hours=48) - pd.to_datetime(patients['intime'])).dt.total_seconds().div(60).astype(int)
patients['duration_time'] = patients['end_offset']-patients['start_offset']


# In[ ]:


patients = patients[patients['age'].apply(lambda x: (False if pd.isnull(x) else False if int(x)<16 else True) if '>' not in str(x) else False)]
patients['age'] = patients['age'].apply(lambda x: int(x))


# In[ ]:


patients = patients[patients['gender'].apply(lambda x: not pd.isnull(x))]
patients.shape


# In[ ]:


patients['gender'] = patients.apply(lambda x: 0 if x['gender']=='F' else 1, axis=1)
patients.gender = patients.gender.astype(np.float32)


# In[ ]:


print('patients remaining', patients.shape[0])


# In[ ]:


actions = pd.read_csv(os.path.join(data_dir,'pivot_iv_fluids.csv'), engine='python')
actions = pd.merge(actions[['stay_id','start_chartoffset','end_chartoffset','tev']], patients, on='stay_id', how='left')
actions = actions[~pd.isnull(actions['start_offset'])]
actions = actions.sort_values(by=['stay_id','start_chartoffset','end_chartoffset'])
actions = actions.reset_index(drop=True)


# In[ ]:


# split actions into several 4h
def get_actions(action_type):
    temp = actions[~pd.isnull(actions[action_type])][['stay_id','start_chartoffset',action_type,'start_offset','end_offset']].reset_index(drop=True)
    temp = temp.sort_values(by=['stay_id','start_chartoffset'])
    def get_action_result(data):
        # 4h (240 minutes) per step
        result = pd.DataFrame(list(range(int(data.start_offset.values[0])+120, int(data.end_offset.values[0]), 240)),columns=['time'])
        result['stay_id'] = data['stay_id'].values[0]
#         result = pd.merge(result, data[['stay_id','start_chartoffset',action_type]], on='stay_id', how='outer')
#         result['time_offset'] = abs(result['time']-result['start_chartoffset'])
#         result = result[result['time_offset']<=120]
#         result = result.groupby(['stay_id','time']).mean().reset_index().drop(['start_chartoffset','time_offset'],axis=1)
        return result
    new_actions = temp.groupby('stay_id').apply(get_action_result)
    return new_actions


# In[ ]:


time_4h = get_actions('tev').drop(['stay_id'],axis=1).reset_index().rename(columns={'level_1':'step_id'})


# In[ ]:


old_actions = actions.copy()
old_action_backup = old_actions.copy()


# In[ ]:


del actions
def time_actions(data):
    result = pd.DataFrame(list(range(int(data.start_offset.values[0])+120, int(data.end_offset.values[0]), 240)),columns=['time'])
    result['stay_id'] = data['stay_id'].values[0]
    return result


# In[ ]:


actions = old_actions.groupby('stay_id').apply(time_actions)
actions = actions.drop(['stay_id'],axis=1).reset_index().rename(columns={'level_1':'step_id'})
actions = pd.merge(actions, time_4h, on=['stay_id','step_id','time'], how='left')


# In[ ]:


# add iv_fluids as action
# dose_rate each 4h
old_actions = old_action_backup[['stay_id','start_chartoffset','end_chartoffset','tev']][old_action_backup['start_chartoffset']<=old_action_backup['end_offset']].reset_index(drop=True)
old_actions = old_actions.sort_values(by=['stay_id','start_chartoffset','end_chartoffset'])
old_actions = old_actions.reset_index()


# In[ ]:


def get_iv_fluids_result(data):
    # 2h (240 minutes) per step, this will not miss
    result = pd.DataFrame(list(range(int(data['start_chartoffset'].values[0]), int(data['end_chartoffset'].values[0]), 120)),columns=['drug_time'])
    result['stay_id'] = data['stay_id'].values[0]
    count_2h = result.shape[0]
    result['tev_2h'] = data['tev'].values[0]/count_2h
    return result


# In[ ]:


iv_fluid = old_actions.groupby('index').apply(get_iv_fluids_result)
iv_fluid = iv_fluid.sort_values(by=['stay_id','drug_time']).reset_index(drop=True)
iv_fluid.head(5)


# In[ ]:


iv_fluid_copy = iv_fluid.copy()


# In[ ]:


vaso = iv_fluid.copy()


# In[ ]:


# use values between time-120 and time+120. 
def get_dose_vital(vital_name, time_name='chartoffset', mode='sum'):
    temp = vaso[~pd.isnull(vaso[vital_name])][['stay_id',time_name,vital_name]].reset_index(drop=True)
    temp = temp.sort_values(by=['stay_id',time_name])
    def get_vital_result(data):
        # 4h (240 minutes) per step
        patientno = data['stay_id'].values[0]
        ptemp = temp[temp['stay_id']==patientno]
        if ptemp.shape[0] == 0:
            return pd.DataFrame([[patientno,0,0]], columns=['stay_id','time',vital_name])
        result = pd.merge(data, ptemp, on='stay_id', how='outer')
        result['time_offset'] = abs(result['time']-result[time_name])
        if mode=='nearest':
            result = result.sort_values(by=['time','time_offset'])
            result = result.groupby(['stay_id','time']).first().reset_index().drop([time_name,'time_offset'],axis=1)
        elif mode=='mean':
            result = result[result['time_offset']<=120]
            if result.shape[0]==0:
                return pd.DataFrame([[patientno,0,0]], columns=['stay_id','time',vital_name])
            result = result.groupby(['stay_id','time']).mean().reset_index()
        elif mode=='max':
            result = result[result['time_offset']<=120]
            result = result.groupby(['stay_id','time']).max().reset_index().drop([time_name,'time_offset'],axis=1)
        elif mode=='sum':
            result = result[result['time_offset']<=120]
            result = result.groupby(['stay_id','time']).sum().reset_index()
        return result
    new_vital = actions[['stay_id','time']].groupby('stay_id').apply(get_vital_result).reset_index(drop=True)
    new_actions = pd.merge(actions, new_vital, on=['stay_id','time'], how='left')
    return new_actions


# In[ ]:


actions = get_dose_vital('tev_2h', time_name='drug_time', mode='sum').reset_index(drop=True)
actions = actions.rename(columns={'tev_2h': 'iv_fluids'})
actions = actions.drop(['tev'],axis=1)
actions.head(5)


# In[ ]:


actions.shape


# In[ ]:


actions_backup = actions.copy()


# In[ ]:


abnormals = {} # min, max
abnormals['iv_fluids'] = [0,25832.2565]
def remove_abnormal(data, col):
    data[col] = data[col].apply(lambda x: np.nan if x<abnormals[col][0] or x>abnormals[col][1] else x)
remove_abnormal(actions, 'iv_fluids')
print('action rows after join patients and remove wrong time', actions.shape[0])
print('action patients number after above', actions.stay_id.unique().shape)


# In[ ]:


value_missing = actions.groupby('stay_id').apply(lambda x: x[x.apply(lambda a: pd.isnull(a['iv_fluids']) , axis=1)].shape[0]/x.shape[0])
value_missing = value_missing.reset_index().rename(columns={0:'action_missing_rate'}) # median 0.89
value_missing.to_csv(os.path.join(output_dir,'action_missing_rate.csv'), index=False)


# In[ ]:


#TODO too much fills
# use zero to fill na

actions['iv_fluids']=actions['iv_fluids'].fillna(0)
# actions = actions[~actions.apply(lambda x: pd.isnull(x['iv_fluids']), axis=1)]
actions = actions.sort_values(by=['stay_id','step_id'])
actions = actions.reset_index(drop=True)


# In[ ]:


actions['stay_id'][actions['iv_fluids']<530].nunique()


# In[ ]:


# save temperory result of actions
actions.to_csv(os.path.join(output_dir,'actions_iv_fluids_4h.csv'), index=False)
patients.to_csv(os.path.join(output_dir,'remain_patients.csv'), index=False)
# print('action patients number after above', actions.stay_id.unique().shape)


# In[ ]:


actions = pd.read_csv(os.path.join(output_dir, 'actions_iv_fluids_4h.csv'), engine='python')


# In[ ]:


# add vasopressor as action
vaso = pd.read_csv(os.path.join(data_dir, 'pivot_vaso_equivalent.csv'), engine='python')
vaso = vaso[vaso['rate_std'].apply(lambda x: x!='na')]
vaso['rate_std'] = vaso['rate_std'].apply(lambda x: float(x))

# dose_rate each 4h
old_vaso = vaso[['stay_id','start_chartoffset','end_chartoffset','rate_std']].reset_index(drop=True)
old_vaso = old_vaso.sort_values(by=['stay_id','start_chartoffset','end_chartoffset'])
old_vaso = old_vaso.reset_index()



# In[ ]:


old_vaso.columns


# In[ ]:


def get_vasopressor_result(data):
    # 2h (240 minutes) per step, this will not miss
    result = pd.DataFrame(list(range(int(data['start_chartoffset'].values[0]), int(data['end_chartoffset'].values[0]), 120)),columns=['drug_time'])
    result['stay_id'] = data['stay_id'].values[0]
    result['rate_std'] = data['rate_std'].values[0]
#     result['Duration_min'] = data['Duration_min'].values[0]
    return result


# In[ ]:


vaso = old_vaso.groupby('index').apply(get_vasopressor_result)
vaso = vaso.sort_values(by=['stay_id','drug_time']).reset_index(drop=True)
vaso.head(5)


# In[ ]:


# use values between time-120 and time+120. 
def get_dose_vital(vital_name, time_name='chartoffset', mode='max'):
    temp = vaso[~pd.isnull(vaso[vital_name])][['stay_id',time_name,vital_name]].reset_index(drop=True)
    temp = temp.sort_values(by=['stay_id',time_name])
    def get_vital_result(data):
        # 4h (240 minutes) per step
        patientno = data['stay_id'].values[0]
        ptemp = temp[temp['stay_id']==patientno]
        if ptemp.shape[0] == 0:
            return pd.DataFrame([[patientno,0,0]], columns=['stay_id','time',vital_name])
        result = pd.merge(data, ptemp, on='stay_id', how='outer')
        result['time_offset'] = abs(result['time']-result[time_name])
        if mode=='nearest':
            result = result.sort_values(by=['time','time_offset'])
            result = result.groupby(['stay_id','time']).first().reset_index().drop([time_name,'time_offset'],axis=1)
        elif mode=='mean':
            result = result[result['time_offset']<=120]
            if result.shape[0]==0:
                return pd.DataFrame([[patientno,0,0]], columns=['stay_id','time',vital_name])
            result = result.groupby(['stay_id','time']).mean().reset_index()
        elif mode=='max':
            result = result[result['time_offset']<=120]
            result = result.groupby(['stay_id','time']).max().reset_index().drop([time_name,'time_offset'],axis=1)
        elif mode=='sum':
            result = result[result['time_offset']<=120]
            result = result.groupby(['stay_id','time']).sum().reset_index()
        return result
    new_vital = actions[['stay_id','time']].groupby('stay_id').apply(get_vital_result).reset_index(drop=True)
    new_actions = pd.merge(actions, new_vital, on=['stay_id','time'], how='left')
    return new_actions


# In[ ]:


actions = get_dose_vital('rate_std', time_name='drug_time', mode='max').reset_index(drop=True)
actions = actions.rename(columns={'rate_std': 'vasopressors'})


# In[ ]:


actions.vasopressors=actions.vasopressors.fillna(0)
actions = actions.sort_values(by=['stay_id','step_id'])
actions = actions.reset_index(drop=True)


# In[ ]:


actions.to_csv(os.path.join(output_dir,'actions_iv_vaso_4h_Jan17.csv'), index=False)


# In[ ]:


actions = pd.read_csv("./mimic_v2/actions_iv_vaso_4h_Jan17.csv", index_col=False)


# In[ ]:


def get_vital(vital_name, time_name='chartoffset', mode='nearest'):
    df = pd.merge(vital, patients[['stay_id','start_offset','end_offset']], on='stay_id', how='left')
    temp = df.loc[(df["chartoffset"] >= df["start_offset"]) & (df["chartoffset"] <= df["end_offset"])]
    temp = temp[~pd.isnull(temp[vital_name])][['stay_id',time_name,vital_name]].reset_index(drop=True)
    temp = temp.sort_values(by=['stay_id',time_name])
    def get_vital_result(data):
        # 4h (240 minutes) per step
        patientno = data['stay_id'].values[0]
        ptemp = temp[temp['stay_id']==patientno]
        result = pd.merge(data, ptemp, on='stay_id', how='outer')
        result['time_offset'] = abs(result['time']-result[time_name])
        if mode=='nearest':
            result = result.sort_values(by=['time','time_offset'])
            result = result.groupby(['stay_id','time']).first().reset_index().drop([time_name,'time_offset'],axis=1)
        elif mode=='mean':
            result = result[result['time_offset']<=120]
            result = result.groupby(['stay_id','time']).mean().reset_index().drop([time_name,'time_offset'],axis=1)
        elif mode=='max':
            result = result[result['time_offset']<=120]
            result = result.groupby(['stay_id','time']).max().reset_index().drop([time_name,'time_offset'],axis=1)
        elif mode=='sum':
            result = result[result['time_offset']<=120]
            result = result.groupby(['stay_id','time']).sum().reset_index().drop([time_name,'time_offset'],axis=1)
        return result
    new_vital = actions[['stay_id','time']].groupby('stay_id').apply(get_vital_result).reset_index(drop=True)
    new_actions = pd.merge(actions, new_vital, on=['stay_id','time'], how='left')
    return new_actions


# In[ ]:


## SOFA
# del vital
vital = pd.read_csv(os.path.join(data_dir, 'pivot_sofa.csv'), engine='python')
vital.columns
for vital_name in ['sofa_24hours']:
    actions = get_vital(vital_name).reset_index(drop=True)


# In[ ]:


actions = pd.merge(actions, patients[['stay_id', 'age', 'gender', 'weight','readmission','elixhauser_vanwalraven','MechVent','sirs',
                                       'mortality_hospital','mortality_90','start_offset','end_offset']], on='stay_id', how='left')


# In[ ]:


data = actions[~pd.isnull(actions['start_offset'])]


# In[ ]:


# create action level by quartile
iv_val =data.iv_fluids[data.iv_fluids!=0].quantile([0.25,0.5,0.75]).values
iv_val[0]
# 0.25     471.103
# 0.50    1000.000
# 0.75    1972.170


# In[ ]:


vaso_val = data.vasopressors[data.vasopressors!=0].quantile([0.25,0.5,0.75]).values
# 0.25    0.0800
# 0.50    0.2000
# 0.75    0.3895


# In[ ]:


data.columns


# In[ ]:



## action stratify 5 levels for iv_fluids and 5 levels for vasopressors
iv_val =data.iv_fluids[data.iv_fluids!=0].quantile([0.25,0.5,0.75]).values
vaso_val = data.vasopressors[data.vasopressors!=0].quantile([0.25,0.5,0.75]).values
data['iv_fluids_quantile'] = data['iv_fluids'].apply(lambda x: 1 if x==0 else 2 if x<=iv_val[0] else 3 if x<=iv_val[1] else 4 if x<=iv_val[2] else 5)
data['vasopressors_quantile'] = data['vasopressors'].apply(lambda x: 1 if x==0 else 2 if x<=vaso_val[0] else 3 if x<=vaso_val[1] else 4 if x<=vaso_val[2] else 5)

data['phys_actions'] = data['iv_fluids_quantile']*5 + data['vasopressors_quantile'] -6


# In[ ]:




# Low SOFA
df_test_orig_low = data[data['sofa_24hours'] <= 5]

# # Middling SOFA
df_test_orig_mid = data[data['sofa_24hours'] > 5]
df_test_orig_mid = df_test_orig_mid[df_test_orig_mid['sofa_24hours'] < 15]

# # High SOFA
df_test_orig_high = data[data['sofa_24hours'] >= 15]


# In[13]:


# Now re-select the phys_actions, autoencode_actions, and deeprl2_actions based on the statified dataset
# deeprl2_actions_low = df_test_orig_low['deeprl2_actions'].values
phys_actions_low = df_test_orig_low['phys_actions'].values

# deeprl2_actions_mid = df_test_orig_mid['deeprl2_actions'].values
phys_actions_mid = df_test_orig_mid['phys_actions'].values

# deeprl2_actions_high = df_test_orig_high['deeprl2_actions'].values
phys_actions_high = df_test_orig_high['phys_actions'].values


# In[ ]:


inv_action_map = {}
count = 0
for i in range(5):
    for j in range(5):
        inv_action_map[count] = [i,j]
        count += 1


# In[ ]:


phys_actions_low_tuple = [None for i in range(len(phys_actions_low))]
# deeprl2_actions_low_tuple = [None for i in range(len(phys_actions_low))]

phys_actions_mid_tuple = [None for i in range(len(phys_actions_mid))]
# deeprl2_actions_mid_tuple = [None for i in range(len(phys_actions_mid))]

phys_actions_high_tuple = [None for i in range(len(phys_actions_high))]
# deeprl2_actions_high_tuple = [None for i in range(len(phys_actions_high))]

for i in range(len(phys_actions_low)):
    phys_actions_low_tuple[i] = inv_action_map[phys_actions_low[i]]
#     deeprl2_actions_low_tuple[i] = inv_action_map[deeprl2_actions_low[i]]

for i in range(len(phys_actions_mid)):
    phys_actions_mid_tuple[i] = inv_action_map[phys_actions_mid[i]]
#     deeprl2_actions_mid_tuple[i] = inv_action_map[deeprl2_actions_mid[i]]
                                                  
for i in range(len(phys_actions_high)):
    phys_actions_high_tuple[i] = inv_action_map[phys_actions_high[i]]
#     deeprl2_actions_high_tuple[i] = inv_action_map[deeprl2_actions_high[i]]


# In[ ]:


phys_actions_low_tuple = np.array(phys_actions_low_tuple)
# deeprl2_actions_low_tuple = np.array(deeprl2_actions_low_tuple)

phys_actions_mid_tuple = np.array(phys_actions_mid_tuple)
# deeprl2_actions_mid_tuple = np.array(deeprl2_actions_mid_tuple)

phys_actions_high_tuple = np.array(phys_actions_high_tuple)
# deeprl2_actions_high_tuple = np.array(deeprl2_actions_high_tuple)


# In[ ]:


phys_actions_low_iv = phys_actions_low_tuple[:,0]
phys_actions_low_vaso = phys_actions_low_tuple[:,1]
hist_ph1, x_edges, y_edges = np.histogram2d(phys_actions_low_iv, phys_actions_low_vaso, bins=5)

phys_actions_mid_iv = phys_actions_mid_tuple[:,0]
phys_actions_mid_vaso = phys_actions_mid_tuple[:,1]
hist_ph2, _, _ = np.histogram2d(phys_actions_mid_iv, phys_actions_mid_vaso, bins=5)

phys_actions_high_iv = phys_actions_high_tuple[:,0]
phys_actions_high_vaso = phys_actions_high_tuple[:,1]
hist_ph3, _, _ = np.histogram2d(phys_actions_high_iv, phys_actions_high_vaso, bins=5)

x_edges = np.arange(-0.5,5)
y_edges = np.arange(-0.5,5)

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
plt.tight_layout()


# In[ ]:


actions = pd.read_csv("./mimic_v2/actions_iv_vaso_4h_Jan17.csv", index_col=False)


# In[ ]:


## vital signs
vital = pd.read_csv(os.path.join(data_dir, 'pivot_vital.csv'), engine='python')


# In[ ]:


vital.columns


# In[ ]:


vital['sbp'] = vital.apply(lambda x: x['nibp_systolic'] if pd.isnull(x['ibp_systolic']) else x['ibp_systolic'], axis=1)
vital['dbp'] = vital.apply(lambda x: x['nibp_distolic'] if pd.isnull(x['ibp_diastolic']) else x['ibp_diastolic'], axis=1)
vital['mbp'] = vital.apply(lambda x: x['nibp_mean'] if pd.isnull(x['ibp_mean']) else x['ibp_mean'], axis=1)
vital = vital.drop(['nibp_systolic', 'nibp_distolic', 'nibp_mean', 'ibp_systolic', 'ibp_diastolic', 'ibp_mean'], axis=1)


# In[ ]:


def get_vital(vital_name, time_name='chartoffset', mode='nearest'):
    temp = vital[~pd.isnull(vital[vital_name])][['stay_id',time_name,vital_name]].reset_index(drop=True)
    temp = temp.sort_values(by=['stay_id',time_name])
    def get_vital_result(data):
        # 4h (240 minutes) per step
        patientno = data['stay_id'].values[0]
        ptemp = temp[temp['stay_id']==patientno]
        result = pd.merge(data, ptemp, on='stay_id', how='outer')
        result['time_offset'] = abs(result['time']-result[time_name])
        if mode=='nearest':
            result = result.sort_values(by=['time','time_offset'])
            result = result.groupby(['stay_id','time']).first().reset_index().drop([time_name,'time_offset'],axis=1)
        elif mode=='mean':
            result = result[result['time_offset']<=120]
            result = result.groupby(['stay_id','time']).mean().reset_index().drop([time_name,'time_offset'],axis=1)
        elif mode=='max':
            result = result[result['time_offset']<=120]
            result = result.groupby(['stay_id','time']).max().reset_index().drop([time_name,'time_offset'],axis=1)
        elif mode=='sum':
            result = result[result['time_offset']<=120]
            result = result.groupby(['stay_id','time']).sum().reset_index().drop([time_name,'time_offset'],axis=1)
        return result
    new_vital = actions[['stay_id','time']].groupby('stay_id').apply(get_vital_result).reset_index(drop=True)
    new_actions = pd.merge(actions, new_vital, on=['stay_id','time'], how='left')
    return new_actions


# In[ ]:


for vital_name in ['heartrate', 'respiratoryrate', 'spo2', 'temperature', 'sbp', 'dbp','mbp']:
    actions = get_vital(vital_name).reset_index(drop=True)


# In[ ]:


actions_backup = actions.copy()


# In[ ]:


actions= pd.DataFrame(actions_backup)


# In[ ]:


## lab test blood gas
del vital
vital = pd.read_csv(os.path.join(data_dir, 'pivot_lab_bg.csv'), engine='python')
vital.columns


# In[ ]:


def get_vital(vital_name, time_name='chartoffset', mode='nearest'):
    df = pd.merge(vital, patients[['stay_id','start_offset','end_offset']], on='stay_id', how='left')
    temp = df.loc[(df["chartoffset"] >= df["start_offset"]) & (df["chartoffset"] <= df["end_offset"])]
    temp = temp[~pd.isnull(temp[vital_name])][['stay_id',time_name,vital_name]].reset_index(drop=True)
    temp = temp.sort_values(by=['stay_id',time_name])
    def get_vital_result(data):
        # 4h (240 minutes) per step
        patientno = data['stay_id'].values[0]
        ptemp = temp[temp['stay_id']==patientno]
        result = pd.merge(data, ptemp, on='stay_id', how='outer')
        result['time_offset'] = abs(result['time']-result[time_name])
        if mode=='nearest':
            result = result.sort_values(by=['time','time_offset'])
            result = result.groupby(['stay_id','time']).first().reset_index().drop([time_name,'time_offset'],axis=1)
        elif mode=='mean':
            result = result[result['time_offset']<=120]
            result = result.groupby(['stay_id','time']).mean().reset_index().drop([time_name,'time_offset'],axis=1)
        elif mode=='max':
            result = result[result['time_offset']<=120]
            result = result.groupby(['stay_id','time']).max().reset_index().drop([time_name,'time_offset'],axis=1)
        elif mode=='sum':
            result = result[result['time_offset']<=120]
            result = result.groupby(['stay_id','time']).sum().reset_index().drop([time_name,'time_offset'],axis=1)
        return result
    new_vital = actions[['stay_id','time']].groupby('stay_id').apply(get_vital_result).reset_index(drop=True)
    new_actions = pd.merge(actions, new_vital, on=['stay_id','time'], how='left')
    return new_actions


# In[ ]:


## lab test blood gas
del vital
vital = pd.read_csv(os.path.join(data_dir, 'pivot_lab_bg.csv'), engine='python')
vital.columns


# In[ ]:


for vital_name in ['lactate', 'bicarbonate', 'pao2', 'paco2', 'pH','hemoglobin', 'baseexcess', 'chloride', 
                   'glucose','potassium', 'sodium', 'co2', 'pao2fio2ratio']:
    actions = get_vital(vital_name).reset_index(drop=True)


# In[ ]:


## lab test blood count
del vital
vital = pd.read_csv(os.path.join(data_dir, 'pivot_lab_blood_count.csv'), engine='python')

for vital_name in ['wbc', 'platelet']:
    actions = get_vital(vital_name).reset_index(drop=True)


# In[ ]:


## lab test chemistry
del vital
vital = pd.read_csv(os.path.join(data_dir, 'pivot_chemistry_new.csv'), engine='python')

for vital_name in ['bun', 'creatinine','albumin']:
    actions = get_vital(vital_name).reset_index(drop=True)


# In[ ]:


## lab test coa
del vital
vital = pd.read_csv(os.path.join(data_dir, 'pivot_lab_coa.csv'), engine='python')

for vital_name in ['ptt', 'pt','inr']:
    actions = get_vital(vital_name).reset_index(drop=True)


# In[ ]:


## lab test emzyme
del vital
vital = pd.read_csv(os.path.join(data_dir, 'pivot_lab_emzyme.csv'), engine='python')

for vital_name in ['ast', 'alt','bilirubin_total']:
    actions = get_vital(vital_name).reset_index(drop=True)


# In[ ]:


## lab test additional
del vital
vital = pd.read_csv(os.path.join(data_dir, 'pivot_lab_additional.csv'), engine='python')

for vital_name in ['magnesium', 'ionized_calcium','calcium']:
    actions = get_vital(vital_name).reset_index(drop=True)


# In[ ]:


## gcs
del vital
vital = pd.read_csv(os.path.join(data_dir, 'pivot_gcs.csv'), engine='python')
vital.columns
for vital_name in ['gcs']:
    actions = get_vital(vital_name).reset_index(drop=True)


# In[ ]:


## fio2
del vital
vital = pd.read_csv(os.path.join(data_dir, 'pivot_fio2.csv'), engine='python')
vital.columns
for vital_name in ['fio2']:
    actions = get_vital(vital_name).reset_index(drop=True)


# In[ ]:


actions.columns


# In[ ]:


actions_backup = actions.copy()


# In[ ]:


# urine output 4h

del vital
vital = pd.read_csv(os.path.join(data_dir, 'pivot_intakeoutput.csv'), engine='python')

vital = vital.sort_values(by=['stay_id', 'chartoffset'])
vital.columns


# In[ ]:


vital = vital.groupby('stay_id').apply(lambda x: x.reset_index(drop=True).reset_index().rename(columns={'index':'time_id'})).reset_index(drop=True)
vital_next = vital.copy().rename(columns={'output_total':'last_output_total','nettotal':'last_nettotal','chartoffset':'last_chartoffset'})
vital_next['time_id'] = vital['time_id']+1
vital = pd.merge(vital, vital_next, on=['stay_id','time_id'], how='left')
vital['output_hours'] = vital.apply(lambda x: x['output_total']-x['last_output_total'] if not pd.isnull(x['last_output_total']) else x['output_total'], axis=1)

for vital_name in ['output_hours']:
    actions = get_vital(vital_name, time_name='chartoffset', mode='sum').reset_index(drop=True)
    actions = actions.fillna({vital_name:0})


# In[ ]:


# output_total
for vital_name in ['output_total']:
    actions = get_vital(vital_name).reset_index(drop=True)


# In[ ]:


## SOFA
del vital
vital = pd.read_csv(os.path.join(data_dir, 'pivot_sofa.csv'), engine='python')
vital.columns
for vital_name in ['sofa_24hours']:
    actions = get_vital(vital_name).reset_index(drop=True)


# In[ ]:


actions.columns


# In[ ]:


actions_backup = pd.DataFrame(data=actions())


# In[ ]:


actions.vasopressors = actions.vasopressors.fillna(0)


# In[ ]:


## action stratify 5 levels for iv_fluids and 5 levels for vasopressors
actions['iv_fluids_level'] = actions['iv_fluids'].apply(lambda x: 1 if x==0 else 2 if x<=50 else 3 if x<=180 else 4 if x<=530 else 5)
actions['vasopressors_level'] = actions['vasopressors'].apply(lambda x: 1 if x==0 else 2 if x<=0.08 else 3 if x<=0.22 else 4 if x<=0.45 else 5)

## demography
actions = pd.merge(actions, patients[['stay_id', 'age', 'gender', 'weight','readmission','elixhauser_vanwalraven','MechVent','sirs',
                                       'mortality_hospital','mortality_90','start_offset','end_offset']], on='stay_id', how='left')


# In[ ]:


age_median = np.median(actions['age'])


# In[ ]:


actions = actions.fillna({'age':age_median})


# In[ ]:


actions = actions.rename(columns={'output_hours': 'urine_output'})


# In[ ]:


# save temp
actions.to_csv(os.path.join(output_dir,'temp_data_rl_17Jan.csv'), index=False)


# In[ ]:





# In[ ]:


actions = pd.read_csv(os.path.join(output_dir,'temp_data_rl_17Jan.csv')  )


# In[ ]:


abnormals = {} # min, max
abnormals['sbp'] = [50,300]
def remove_abnormal(data, col):
    data[col] = data[col].apply(lambda x: np.nan if x<abnormals[col][0] or x>abnormals[col][1] else x)
remove_abnormal(actions, 'sbp')


# In[ ]:


def forward_fill(data, col):
    data[col] = data.groupby(['stay_id'])[col].ffill()

def backward_fill(data, col):
    data[col] = data.groupby(['stay_id'])[col].bfill()


# In[ ]:


forward_fill(actions, 'sbp')


# In[ ]:


actions['sbp'].isna().sum()


# In[ ]:


sbp_median = np.median(actions['sbp'])


# In[ ]:


actions = actions.fillna({'sbp':sbp_median})


# In[ ]:


# shock_index = HR/SBP
actions['shock_index'] = actions[['heartrate','sbp']].apply(lambda x: x['heartrate']/x['sbp'],axis=1)


# In[ ]:


# state, action, next_state
state_cols = [ 'age', 'gender', 'weight','readmission','shock_index','elixhauser_vanwalraven','sirs','MechVent',
              'heartrate', 'respiratoryrate', 'spo2', 'temperature',
              'sbp', 'dbp', 'mbp', 'lactate', 'bicarbonate', 'pao2', 'paco2', 'pH',
              'hemoglobin', 'baseexcess', 'chloride', 'glucose', 'calcium','ionized_calcium',
              'potassium', 'sodium', 'co2', 'pao2fio2ratio', 'wbc', 'platelet', 'bun',
              'creatinine', 'ptt', 'pt', 'inr', 'ast', 'alt', 'bilirubin_total','albumin','magnesium',
              'gcs', 'fio2', 'urine_output', 'output_total', 'sofa_24hours']


# In[ ]:


for col_name in state_cols:
    forward_fill(actions, col_name)
    backward_fill(actions,col_name)


# In[ ]:


actions = actions.fillna({'elixhauser_vanwalraven':0.0})
actions = actions.fillna({'MechVent':0.0})
actions = actions.fillna({'sofa_24hours':0.0})


# In[ ]:


actions.isna().sum()


# In[ ]:


# fill values
na_fill = {}
for col_name in state_cols:
    na_fill[col_name] = actions[col_name].median()
actions = actions.fillna(na_fill)
with open(os.path.join(output_dir,'mimic_na_fill.pkl'), 'wb') as f:
    pickle.dump(na_fill, f)


# In[ ]:


actions['bloc']= actions.apply(lambda x: x['step_id']+1, axis = 1)
actions.head(3)


# In[ ]:


binary_fields = ['gender','MechVent','readmission']

norm_fields= ['age','weight','gcs','heartrate','sbp','mbp','dbp','respiratoryrate','temperature','fio2',
    'potassium','sodium','chloride','glucose','magnesium','calcium',
    'hemoglobin','wbc','platelet','ptt','pt','pH','pao2','paco2',
    'baseexcess','bicarbonate','lactate','sofa_24hours','sirs','shock_index',
    'pao2fio2ratio','elixhauser_vanwalraven', 'albumin', 'co2', 'ionized_calcium']
              
log_fields = ['iv_fluids','vasopressors','spo2','bun','creatinine','alt','ast','bilirubin_total','inr',
              'output_total','urine_output', 'bloc']


# In[ ]:


actions.to_csv("temp_data_rl_17Jan_backup.csv")


# In[ ]:


ori_actions = pd.read_csv(os.path.join(output_dir,'temp_data_rl_17Jan_backup.csv'), engine='python')
actions = pd.read_csv(os.path.join(output_dir,'temp_data_rl_17Jan_backup.csv'), engine='python')


# In[ ]:


ori_actions.columns


# In[ ]:


# state, action, next_state
state_cols_ori= ['iv_fluids','vasopressors', 'age', 'gender', 'weight','readmission','shock_index','elixhauser_vanwalraven','sirs','MechVent',
              'heartrate', 'respiratoryrate', 'spo2', 'temperature',
              'sbp', 'dbp', 'mbp', 'lactate', 'bicarbonate', 'pao2', 'paco2', 'pH',
              'hemoglobin', 'baseexcess', 'chloride', 'glucose', 'calcium','ionized_calcium',
              'potassium', 'sodium', 'co2', 'pao2fio2ratio', 'wbc', 'platelet', 'bun',
              'creatinine', 'ptt', 'pt', 'inr', 'ast', 'alt', 'bilirubin_total','albumin','magnesium',
              'gcs', 'fio2', 'urine_output', 'output_total', 'sofa_24hours','bloc']
len(state_cols_ori)


# In[ ]:


origin_rename = {i:'ori_'+i for i in state_cols_ori}
ori_actions = ori_actions.rename(columns=origin_rename)


# In[ ]:


# binary  fields
actions[binary_fields] = actions[binary_fields] - 0.5 


# In[ ]:


# normal distribution fields
for item in norm_fields:
    av = actions[item].mean()
    std = actions[item].std()
    actions[item] = (actions[item] - av) / std


# In[ ]:


# log normal fields
actions[log_fields] = np.log(0.1 + actions[log_fields])
for item in log_fields:
    av = actions[item].mean()
    std = actions[item].std()
    actions[item] = (actions[item] - av) / std


# In[ ]:


ori_actions.columns


# In[ ]:


df = pd.merge(ori_actions, actions, on=['stay_id','step_id','time','mortality_hospital','mortality_90','start_offset','end_offset','iv_fluids_level','vasopressors_level'], how='left')


# In[ ]:


# next state
# state, action, next_state
state_cols_next= ['age', 'gender', 'weight','readmission','shock_index','elixhauser_vanwalraven','sirs','MechVent',
              'heartrate', 'respiratoryrate', 'spo2', 'temperature',
              'sbp', 'dbp', 'mbp', 'lactate', 'bicarbonate', 'pao2', 'paco2', 'pH',
              'hemoglobin', 'baseexcess', 'chloride', 'glucose', 'calcium','ionized_calcium',
              'potassium', 'sodium', 'co2', 'pao2fio2ratio', 'wbc', 'platelet', 'bun',
              'creatinine', 'ptt', 'pt', 'inr', 'ast', 'alt', 'bilirubin_total','albumin','magnesium',
              'gcs', 'fio2', 'urine_output', 'output_total', 'sofa_24hours','bloc']

next_rename = {i:'next_'+i for i in state_cols_next}
next_rename['ori_lactate'] = 'next_ori_lactate'
next_rename['ori_mbp'] = 'next_ori_mbp'
next_rename['ori_sofa_24hours'] = 'next_ori_sofa_24hours'
next_data = df.copy().rename(columns=next_rename)
# next_data['step_id'] = next_data['step_id'].apply(lambda x: x-1)
for i in range(1, next_data.shape[0]):
    if next_data.loc[i,'stay_id']==df.loc[i-1,'stay_id']:
        next_data.loc[i,'step_id'] = df.loc[i-1,'step_id']
        
df = pd.merge(df, next_data[['stay_id','step_id']+list(next_rename.values())], on=['stay_id','step_id'], how='left')


# In[ ]:


actions = df.copy()
actions.head(2)


# In[ ]:


actions = actions.drop_duplicates(['stay_id','step_id'],keep= 'last')


# In[ ]:


actions['length']=actions[['stay_id', 'step_id']].groupby(['stay_id'])['step_id'].transform('count') 
actions['first_step_id']=actions[['stay_id', 'step_id']].groupby(['stay_id'])['step_id'].transform('min') 
actions['last_step_id']=actions[['stay_id', 'step_id']].groupby(['stay_id'])['step_id'].transform('max') 

actions['start'] = actions.apply(lambda x: 1 if x['step_id']==x['first_step_id'] else 0, axis=1)
actions['done'] = actions.apply(lambda x: 1 if x['step_id']==x['last_step_id'] else 0, axis=1)


# In[ ]:


actions_backup = actions.copy()


# In[ ]:


# create action level by quartile
actions.ori_iv_fluids[actions.ori_iv_fluids!=0].quantile([0.25,0.5,0.75])


# In[ ]:


actions.ori_vasopressors[actions.ori_vasopressors!=0].quantile([0.25,0.5,0.75])


# In[ ]:


## action stratify 5 levels for iv_fluids and 5 levels for vasopressors
iv_val =actions.ori_iv_fluids[actions.ori_iv_fluids!=0].quantile([0.25,0.5,0.75]).values
vaso_val = actions.ori_vasopressors[actions.ori_vasopressors!=0].quantile([0.25,0.5,0.75]).values
actions['iv_fluids_quantile'] = actions['ori_iv_fluids'].apply(lambda x: 1 if x==0 else 2 if x<=iv_val[0] else 3 if x<=iv_val[1] else 4 if x<=iv_val[2] else 5)
actions['vasopressors_quantile'] = actions['ori_vasopressors'].apply(lambda x: 1 if x==0 else 2 if x<=vaso_val[0] else 3 if x<=vaso_val[1] else 4 if x<=vaso_val[2] else 5)


# In[ ]:


# save data
actions.to_csv(os.path.join(output_dir,'data_rl_jan21.csv'), index=False)


# In[ ]:


df = pd.read_csv(os.path.join(output_dir,'data_rl_8h_jan21.csv'), engine='python')


# In[ ]:


df_unique = pd.DataFrame(df[['stay_id','mortality_hospital']][df['step_id']==0])
y = df_unique['mortality_hospital']
X = df_unique['stay_id']

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    stratify=y, 
                                                    test_size=0.25)


# In[ ]:


for idx, row in df.iterrows():
    if(df.loc[idx,'stay_id'] in train_id):
        df.loc[idx, 'train_test']="train"
    elif(df.loc[idx,'stay_id'] in test_id):
        df.loc[idx, 'train_test']="test"


# In[ ]:


df.to_csv(os.path.join(output_dir,'data_rl_4h_train_test_split.csv'), index=False)

