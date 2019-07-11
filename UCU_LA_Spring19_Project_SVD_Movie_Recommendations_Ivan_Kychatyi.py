
# coding: utf-8

# In[34]:


import numpy as np
import pandas as pd
import math
import time


# ##### Load the full dataset

# In[3]:


data = pd.read_csv('data/ml-100k/u.data',sep='\t', header=None, names=['userId', 'itemId', 'rating','timestamp'])


# In[9]:


movie_info = pd.read_csv('data/ml-100k/u.item',sep='|', 
                         header=None,
                         index_col = False,
                         names=['itemId', 'title'],
                         usecols = [0,1],
                         encoding = "ISO-8859-1")


# In[13]:


df = pd.merge(data, movie_info, left_on='itemId', right_on='itemId')


# ##### Load the train/test splitted dataset

# In[35]:


#load the data split
train_1 = pd.read_csv('data/ml-100k/ua.base',sep='\t', header=None, names=['userId', 'itemId', 'rating','timestamp'])
test_1 = pd.read_csv('data/ml-100k/ua.test',sep='\t', header=None, names=['userId', 'itemId', 'rating','timestamp'])
train_2 = pd.read_csv('data/ml-100k/ub.base',sep='\t', header=None, names=['userId', 'itemId', 'rating','timestamp'])
test_2 = pd.read_csv('data/ml-100k/ub.test',sep='\t', header=None, names=['userId', 'itemId', 'rating','timestamp'])


# In[55]:


#check how many ratings we have in the train and test splits
train_ratings = train_1.userId.count()
test_ratings = test_1.userId.count()
train_frac = train_ratings/(train_ratings+test_ratings)*100
print(f'Train/test split is: {train_frac, 100-train_frac}')


# In[50]:


#look at the composition of the train set for a specific user
test_1[test_1.userId==4]


# In[51]:


#look at the composition of the test set for a specific user
train_1[train_1.userId==4]


# In[73]:


#convert the rating dataframe into the Rating train and test matrix
R_train1 = pd.pivot_table(train_1, values='rating', index='userId', columns='itemId')
R_test1 = pd.pivot_table(test_1, values='rating', index='userId', columns='itemId')


# In[101]:


#Check the structure of the rating matrix R
R_train1.head()


# In[75]:


R_train1.shape


# In[76]:


R_test1.shape


# In[99]:


#Cerate the implicit feedback matrix
F = R_train1.copy()

for i in F.index:
    for j in F.columns:
        if np.isnan(F.loc[i,j]) == True:
            F.loc[i,j] = 0
        elif np.isnan(F.loc[i,j]) == False:
            F.loc[i,j] = 1
        else: continue


# In[100]:


F.head()


# In[77]:


print(np.mean(np.nanmean(R_train1, axis=0)))
print(np.mean(np.nanmean(R_train1, axis=1)))


# In[91]:


#Calculate the global mean (bias)
count = 0
r_sum = 0

for i in R_train1.index:
    for j in R_train1.columns:
        if np.isnan(R_train1.loc[i,j]) == False:
            count += 1
            r_sum += R_train1.loc[i,j]
        else: continue

mu = r_sum/count


# In[98]:


#Print the global mean
print(f'The global mean/bias for our matrix is: {mu}')


# In[276]:


def SVDplusplus(R, F, mu, factors, steps, lrn_rate, regular, lrn_rate_bias, regular_bias):
    
    start = time.time()
    #get the dimentions of R
    n = len(R.index)
    m = len(R.columns)
    
    #initialize the user, item and global bias
    Bu = pd.DataFrame(np.random.rand(n), index=R.index) 
    Bi = pd.DataFrame(np.random.rand(m), index=R.columns)
    mu = mu
    
    #initialize the user-factor, item-factor and feedback-factor matrices
    P = pd.DataFrame(np.random.rand(n,factors), index=R.index)
    Q = pd.DataFrame(np.random.rand(m,factors), index=R.columns)
    Y = pd.DataFrame(np.random.rand(m,factors), index=R.columns)
    
    #precalculate implicit feedback metrics
    y_norm = pd.DataFrame(pow(np.sum(F, axis=1), -1/2), index=F.index)
    
    loss_total = []
    
    for s in range(steps):
        print(f'Iteration {s} started')
        for i in R.index:
            for j in R.columns:
                if np.isnan(R.loc[i,j]) == False:
                    
                    #print our vars for debugging purposes
                    print(f'''User: {i}, Item: {j}, Rating: {R.loc[i,j]},
                          mu: {mu}  \n,
                          Bu: {Bu.loc[i]}  \n,
                          Bi: {Bi.loc[j]} \n,
                          Q_j: {Q.loc[j]} \n,
                          P_i: {P.loc[i]} \n,
                          Y_j: {Y.loc[j]} \n,
                          y_norm_i: {y_norm.loc[i]} \n,
                          Y_sum: {np.sum(Y[F.loc[i]==1], axis=0)} \n''')
                    
                    
                    
                    eij = (R.loc[i,j] 
                           - mu 
                           - Bu.loc[i] 
                           - Bi.loc[j] 
                           - np.dot(Q.loc[j], P.loc[i] + float(y_norm.loc[i])*np.sum(Y[F.loc[i]==1], axis=0)))
                    
                    Bu.loc[i] = Bu.loc[i] + lrn_rate_bias*(eij - regular_bias*Bu.loc[i])
                    Bi.loc[j] = Bi.loc[j] + lrn_rate_bias*(eij - regular_bias*Bi.loc[j])
                    P.loc[i] = P.loc[i] + lrn_rate*(eij*Q.loc[j] - regular*P.loc[i])
                    Q.loc[j] = Q.loc[j] + lrn_rate*(eij*(P.loc[i] + (y_norm.loc[i])*np.sum(Y[F.loc[i]==1], axis=0))                                                    - regular*Q.loc[j])
                    Y.loc[j] = Y.loc[j] + lrn_rate*(eij*y_norm.loc[i]*Q.loc[j] - regular*Y.loc[j])
                    
                    #print our vars for debugging purposes
                    print(f'''y_norm*Y_sum: \n {float(y_norm.loc[i])*np.sum(Y[F.loc[i]==1], axis=0)} \n,
                          P_i + y_norm*Y_sum: \n {P.loc[i] + float(y_norm.loc[i])*np.sum(Y[F.loc[i]==1], axis=0)} \n,
                          dot product: {np.dot(Q.loc[j], P.loc[i].array + float(y_norm.loc[i])*np.sum(Y[F.loc[i]==1], axis=0))}\n,
                          eij: {eij} \n,
                          Bu: {Bu.loc[i]} \n,
                          Bi: {Bi.loc[j]} \n,
                          Q_j: {Q.loc[j]} \n,
                          P_i: {P.loc[i]} \n,
                          Y_j: {Y.loc[j]} \n,
                          Bu nan {np.sum(np.isnan(Bu))},
                          Bi nan {np.sum(np.isnan(Bi))},
                          P nan {np.sum(np.sum(np.isnan(P)))},
                          Q nan {np.sum(np.sum(np.isnan(Q)))},
                          Y nan {np.sum(np.sum(np.isnan(Y)))}''')
        
        #calculae the loss
        loss_iter = 0
        for i in R.index:
            for j in R.columns:
                if np.isnan(R.loc[i,j]) == False:
                    p_nrm = np.linalg.norm(P.loc[i])**2
                    q_nrm = np.linalg.norm(Q.loc[j])**2
                    y_nrm = np.linalg.norm(Y[F.loc[i]==1], axis=1)**2
                    loss_iter += (R.loc[i,j] 
                                  - mu 
                                  - Bu.loc[i] 
                                  - Bi.loc[j] 
                                  - np.dot(Q.loc[j], P.loc[i] + y_norm.loc[i]*np.sum(Y[F.loc[i]==1], axis=0)))**2 \
                                  + regular*(p_nrm + q_nrm + np.sum(y_nrm))\
                                  + regular_bias*(Bu.loc[i]**2 + Bi.loc[j]**2)
        loss_total.append(loss_iter)
        print(f'Iteration: {s} \t Loss: {loss_total[-1]} \t Time passed: {time.time()-start}')            
    
    return P, Q, Y, Bu, Bi, loss_total


# In[277]:


P, Q, Y, Bu, Bi, loss = SVDplusplus(R_train1, F, mu, 15, 2, 0.02, 0.1, 0.01, 0.007)


# In[269]:


F.loc[4] + F.loc[3]


# In[162]:


#precalculate y_norm for rating prediction
y_norm = pd.DataFrame(pow(np.sum(F, axis=1), -1/2), index=F.index)


# In[237]:


np.sum(np.sum(np.isnan(P)))


# In[235]:


Bi = pd.DataFrame(np.random.rand(len(R_train1.columns)), index=R_train1.columns)
Bu = pd.DataFrame(np.random.rand(len(R_train1.index)), index=R_train1.index)
print(Bu.loc[943])
print (Bi.loc[4])
print(Bu.loc[943]+Bi.loc[4])


# In[187]:


#Let's evaluate the performance with RMSE and MAE metrics

counter = 0
se = 0
ae = 0

for i in R_test1.index:
    for j in R_test1.columns:
        if np.isnan(R_test1.loc[i,j]) == False:
            if j in Q.index == True:
                r_pred = (mu 
                          + Bu.loc[i]
                          + Bi.loc[j]
                          + np.dot(Q.loc[j], P.loc[i] + y_norm.loc[i]*np.sum(Y[F.loc[i]==1], axis=0)))
                se += (R_test1.loc[i,j] - r_pred)**2
                ae += np.abs(R_test1.loc[i,j] - r_pred)
                counter += 1
            elif j in Q.index == False:
                r_pred = mu + Bi[j]
                se += (R_test1.loc[i,j] - r_pred)**2
                ae += np.abs(R_test1.loc[i,j] - r_pred)
                counter += 1
            else: continue
    
rmse = (se/counter)**(1/2)
mae = ae/counter

print(f'The model RMSE: {rmse} \t The model MAE: {mae}')

