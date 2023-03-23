#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle


# In[2]:


df = pd.read_csv('data/data_rl_9level_pretrain_K8.csv')
data = df.drop(columns=['Unnamed: 0', 'Unnamed: 0.1',
       'iv_fluids_quantile_9level', 'vasopressors_quantile_9level',
       'phys_actions'])


# In[3]:


vaso_min = np.min(data.vasopressors)
iv_min = np.min(data.iv_fluids)
max_vaso = data.vasopressors.quantile(0.99)
data['vasopressors']=data['vasopressors'].apply(lambda x: x if x<max_vaso else max_vaso )
max_iv = data.iv_fluids.quantile(0.99)
data['iv_fluids']=data['iv_fluids'].apply(lambda x: x if x<max_iv else max_iv )


# In[4]:


# 10 levels
iv_val =data.iv_fluids[data.iv_fluids>iv_min].quantile(
       [0.1111111111111111 ,
        0.2222222222222222 ,
        0.3333333333333333 ,
        0.4444444444444444 ,
        0.5555555555555556 ,
        0.6666666666666666 ,
        0.7777777777777778 ,
        0.8888888888888888  ]).values
vaso_val = data.vasopressors[data.vasopressors>vaso_min].quantile(
       [0.1111111111111111 ,
        0.2222222222222222 ,
        0.3333333333333333 ,
        0.4444444444444444 ,
        0.5555555555555556 ,
        0.6666666666666666 ,
        0.7777777777777778 ,
        0.8888888888888888  ]).values


# In[5]:


# # 20 levels
# iv_val =data.iv_fluids[data.iv_fluids>iv_min].quantile(
#        [0.05263157894736842 ,
#         0.10526315789473684 ,
#         0.15789473684210525 ,
#         0.21052631578947367 ,
#         0.2631578947368421 ,
#         0.3157894736842105 ,
#         0.3684210526315789 ,
#         0.42105263157894735 ,
#         0.47368421052631576 ,
#         0.5263157894736842 ,
#         0.5789473684210527 ,
#         0.631578947368421 ,
#         0.6842105263157895 ,
#         0.7368421052631579 ,
#         0.7894736842105263 ,
#         0.8421052631578947 ,
#         0.8947368421052632 ,
#         0.9473684210526315 ]).values
# vaso_val = data.vasopressors[data.vasopressors>vaso_min].quantile(
#        [0.05263157894736842 ,
#         0.10526315789473684 ,
#         0.15789473684210525 ,
#         0.21052631578947367 ,
#         0.2631578947368421 ,
#         0.3157894736842105 ,
#         0.3684210526315789 ,
#         0.42105263157894735 ,
#         0.47368421052631576 ,
#         0.5263157894736842 ,
#         0.5789473684210527 ,
#         0.631578947368421 ,
#         0.6842105263157895 ,
#         0.7368421052631579 ,
#         0.7894736842105263 ,
#         0.8421052631578947 ,
#         0.8947368421052632 ,
#         0.9473684210526315 ]).values


# In[6]:


# # 50 levels
# iv_val =data.iv_fluids[data.iv_fluids>iv_min].quantile(
#        [0.02040816326530612 ,
#         0.04081632653061224 ,
#         0.061224489795918366 ,
#         0.08163265306122448 ,
#         0.10204081632653061 ,
#         0.12244897959183673 ,
#         0.14285714285714285 ,
#         0.16326530612244897 ,
#         0.1836734693877551 ,
#         0.20408163265306123 ,
#         0.22448979591836735 ,
#         0.24489795918367346 ,
#         0.2653061224489796 ,
#         0.2857142857142857 ,
#         0.30612244897959184 ,
#         0.32653061224489793 ,
#         0.3469387755102041 ,
#         0.3673469387755102 ,
#         0.3877551020408163 ,
#         0.40816326530612246 ,
#         0.42857142857142855 ,
#         0.4489795918367347 ,
#         0.46938775510204084 ,
#         0.4897959183673469 ,
#         0.5102040816326531 ,
#         0.5306122448979592 ,
#         0.5510204081632653 ,
#         0.5714285714285714 ,
#         0.5918367346938775 ,
#         0.6122448979591837 ,
#         0.6326530612244898 ,
#         0.6530612244897959 ,
#         0.673469387755102 ,
#         0.6938775510204082 ,
#         0.7142857142857143 ,
#         0.7346938775510204 ,
#         0.7551020408163265 ,
#         0.7755102040816326 ,
#         0.7959183673469388 ,
#         0.8163265306122449 ,
#         0.8367346938775511 ,
#         0.8571428571428571 ,
#         0.8775510204081632 ,
#         0.8979591836734694 ,
#         0.9183673469387755 ,
#         0.9387755102040817 ,
#         0.9591836734693877 ,
#         0.9795918367346939  ]).values
# vaso_val = data.vasopressors[data.vasopressors>vaso_min].quantile(
#        [0.02040816326530612 ,
#         0.04081632653061224 ,
#         0.061224489795918366 ,
#         0.08163265306122448 ,
#         0.10204081632653061 ,
#         0.12244897959183673 ,
#         0.14285714285714285 ,
#         0.16326530612244897 ,
#         0.1836734693877551 ,
#         0.20408163265306123 ,
#         0.22448979591836735 ,
#         0.24489795918367346 ,
#         0.2653061224489796 ,
#         0.2857142857142857 ,
#         0.30612244897959184 ,
#         0.32653061224489793 ,
#         0.3469387755102041 ,
#         0.3673469387755102 ,
#         0.3877551020408163 ,
#         0.40816326530612246 ,
#         0.42857142857142855 ,
#         0.4489795918367347 ,
#         0.46938775510204084 ,
#         0.4897959183673469 ,
#         0.5102040816326531 ,
#         0.5306122448979592 ,
#         0.5510204081632653 ,
#         0.5714285714285714 ,
#         0.5918367346938775 ,
#         0.6122448979591837 ,
#         0.6326530612244898 ,
#         0.6530612244897959 ,
#         0.673469387755102 ,
#         0.6938775510204082 ,
#         0.7142857142857143 ,
#         0.7346938775510204 ,
#         0.7551020408163265 ,
#         0.7755102040816326 ,
#         0.7959183673469388 ,
#         0.8163265306122449 ,
#         0.8367346938775511 ,
#         0.8571428571428571 ,
#         0.8775510204081632 ,
#         0.8979591836734694 ,
#         0.9183673469387755 ,
#         0.9387755102040817 ,
#         0.9591836734693877 ,
#         0.9795918367346939  ]).values


# In[7]:


# 10 levels
data['iv_fluids_quantile_10level'] = data['iv_fluids'].apply(lambda x: 1 if x<=iv_min else 2 if x<=iv_val[0] else 3 if x<=iv_val[1] else 4 if x<=iv_val[2] else 5 if x<=iv_val[3] else 6 if x<=iv_val[4] else 7 if x<=iv_val[5] else 8 if x<=iv_val[6] else 9 if x<=iv_val[7] else 10)
data['vasopressors_quantile_10level'] = data['vasopressors'].apply(lambda x: 1 if x<=vaso_min else 2 if x<=vaso_val[0] else 3 if x<=vaso_val[1] else 4 if x<=vaso_val[2] else 5 if x<=vaso_val[3] else 6 if x<=vaso_val[4] else 7 if x<=vaso_val[5] else 8 if x<=vaso_val[6] else 9 if x<=vaso_val[7] else 10)
data['phys_actions'] = data['iv_fluids_quantile_10level']*10 + data['vasopressors_quantile_10level'] -11


# In[8]:


# # 20 levels
# data['iv_fluids_quantile_20level'] = data['iv_fluids'].apply(lambda x: 1 if x<=iv_min else 2 if x<=iv_val[0] else 3 if x<=iv_val[1] else 4 if x<=iv_val[2] else 5 if x<=iv_val[3] else 6 if x<=iv_val[4] 
#                                                              else 7 if x<=iv_val[5] else 8 if x<=iv_val[6] else 9 if x<=iv_val[7] else 10 if x<=iv_val[8] 
#                                                              else 11 if x<=iv_val[9] else 12 if x<=iv_val[10] else 13 if x<=iv_val[11] else 14 if x<=iv_val[12] else 15 if x<=iv_val[13]
#                                                              else 16 if x<=iv_val[14] else 17 if x<=iv_val[15] else 18 if x<=iv_val[16] else 19 if x<=iv_val[17] else 20)
# data['vasopressors_quantile_20level'] = data['vasopressors'].apply(lambda x: 1 if x<=vaso_min else 2 if x<=vaso_val[0] else 3 if x<=vaso_val[1] else 4 if x<=vaso_val[2] else 5 if x<=vaso_val[3] else 6 if x<=vaso_val[4] 
#                                                                    else 7 if x<=vaso_val[5] else 8 if x<=vaso_val[6] else 9 if x<=vaso_val[7] else 10 if x<=vaso_val[8]
#                                                                    else 11 if x<=vaso_val[9] else 12 if x<=vaso_val[10] else 13 if x<=vaso_val[11] else 14 if x<=vaso_val[12] else 15 if x<=vaso_val[13] else 16 if x<=vaso_val[14]
#                                                                    else 17 if x<=vaso_val[15] else 18 if x<=vaso_val[16] else 19 if x<=vaso_val[17] else 20)
# data['phys_actions'] = data['iv_fluids_quantile_20level']*20 + data['vasopressors_quantile_20level'] -21


# In[9]:


# # 50 levels
# data['iv_fluids_quantile_50level'] = data['iv_fluids'].apply(lambda x: 1 if x<=iv_min else 2 if x<=iv_val[0] else 3 if x<=iv_val[1] else 4 if x<=iv_val[2] else 5 if x<=iv_val[3] else 6 if x<=iv_val[4] 
#                                                              else 7 if x<=iv_val[5] else 8 if x<=iv_val[6] else 9 if x<=iv_val[7] else 10 if x<=iv_val[8] 
#                                                              else 11 if x<=iv_val[9] else 12 if x<=iv_val[10] else 13 if x<=iv_val[11] else 14 if x<=iv_val[12] else 15 if x<=iv_val[13]
#                                                              else 16 if x<=iv_val[14] else 17 if x<=iv_val[15] else 18 if x<=iv_val[16] else 19 if x<=iv_val[17] else 20 if x<=iv_val[18]
#                                                              else 21 if x<=iv_val[19] else 22 if x<=iv_val[20] else 23 if x<=iv_val[21] else 24 if x<=iv_val[22] else 25 if x<=iv_val[23]
#                                                              else 26 if x<=iv_val[24] else 27 if x<=iv_val[25] else 28 if x<=iv_val[26] else 29 if x<=iv_val[27] else 30 if x<=iv_val[28]
#                                                              else 31 if x<=iv_val[29] else 32 if x<=iv_val[30] else 33 if x<=iv_val[31] else 34 if x<=iv_val[32] else 35 if x<=iv_val[33]
#                                                              else 36 if x<=iv_val[34] else 37 if x<=iv_val[35] else 38 if x<=iv_val[36] else 39 if x<=iv_val[37] else 40 if x<=iv_val[38]
#                                                              else 41 if x<=iv_val[39] else 42 if x<=iv_val[40] else 43 if x<=iv_val[41] else 44 if x<=iv_val[42] else 45 if x<=iv_val[43]
#                                                              else 46 if x<=iv_val[44] else 47 if x<=iv_val[45] else 48 if x<=iv_val[46] else 49 if x<=iv_val[47] else 50)
# data['vasopressors_quantile_50level'] = data['vasopressors'].apply(lambda x: 1 if x<=vaso_min else 2 if x<=vaso_val[0] else 3 if x<=vaso_val[1] else 4 if x<=vaso_val[2] else 5 if x<=vaso_val[3] else 6 if x<=vaso_val[4] 
#                                                              else 7 if x<=vaso_val[5] else 8 if x<=vaso_val[6] else 9 if x<=vaso_val[7] else 10 if x<=vaso_val[8] 
#                                                              else 11 if x<=vaso_val[9] else 12 if x<=vaso_val[10] else 13 if x<=vaso_val[11] else 14 if x<=vaso_val[12] else 15 if x<=vaso_val[13]
#                                                              else 16 if x<=vaso_val[14] else 17 if x<=vaso_val[15] else 18 if x<=vaso_val[16] else 19 if x<=vaso_val[17] else 20 if x<=vaso_val[18]
#                                                              else 21 if x<=vaso_val[19] else 22 if x<=vaso_val[20] else 23 if x<=vaso_val[21] else 24 if x<=vaso_val[22] else 25 if x<=vaso_val[23]
#                                                              else 26 if x<=vaso_val[24] else 27 if x<=vaso_val[25] else 28 if x<=vaso_val[26] else 29 if x<=vaso_val[27] else 30 if x<=vaso_val[28]
#                                                              else 31 if x<=vaso_val[29] else 32 if x<=vaso_val[30] else 33 if x<=vaso_val[31] else 34 if x<=vaso_val[32] else 35 if x<=vaso_val[33]
#                                                              else 36 if x<=vaso_val[34] else 37 if x<=vaso_val[35] else 38 if x<=vaso_val[36] else 39 if x<=vaso_val[37] else 40 if x<=vaso_val[38]
#                                                              else 41 if x<=vaso_val[39] else 42 if x<=vaso_val[40] else 43 if x<=vaso_val[41] else 44 if x<=vaso_val[42] else 45 if x<=vaso_val[43]
#                                                              else 46 if x<=vaso_val[44] else 47 if x<=vaso_val[45] else 48 if x<=vaso_val[46] else 49 if x<=vaso_val[47] else 50)
# data['phys_actions'] = data['iv_fluids_quantile_50level']*50 + data['vasopressors_quantile_50level'] -51


# In[10]:


data.to_csv('data/data_rl_10level_pretrain_K8.csv', index=False)


# In[ ]:




