# -*- coding: utf-8 -*-
"""
Created on Sat Mar  1 14:35:55 2025

"""

import policy as p
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 16
# %% make histogram for distribitions for comparison with the authors implementaion 
nDraw = 50000
numBin = 200
alp = 0.5
a = p.A()
b = p.B()
c = p.C()

a.draw(nDraw)
b.draw(nDraw)
c.draw(nDraw)

fig, ax = plt.subplots()
ax.hist(a.rewards,bins=numBin,alpha=alp,density=True)
ax.hist(b.rewards,bins=numBin,alpha=alp,density=True)
ax.hist(c.rewards,bins=numBin,alpha=alp,density=True)
plt.xlabel('Reward')
plt.ylabel('Frequency')
plt.legend(['A','B','C'])
plt.ylim([0, 0.5])
plt.xlim([0, 20])
plt.title('Reward Distributions')
plt.grid(visible=True)
plt.show()
fig.savefig('ABC Distributions.jpg')

# set the number of itteration to run 
n_iter = 5000 # set to 5000

# set up budget for the 5 optimal cases
budgets = list(range(1000,4001,250))
n_optimal = 5

# %% batch proc for test case 1
quantile = 0.5
# set up the environment as a lambda function
env = lambda: p.env1(n_optimal)

# lambda funcitons for the policy algorithms 
qsar = lambda: p.qsar(n_optimal,quantile)
quniform = lambda: p.q_uniform(n_optimal,quantile)
qsr = lambda: p.qsr(n_optimal,quantile)
qbe = lambda: p.qbe(n_optimal,quantile) 

# set up the batch simulaitons with the lambda functions
pErr_qsar = p.batchSim(budgets, n_iter, qsar,env)
pErr_quniform = p.batchSim(budgets, n_iter, quniform,env)
pErr_qsr = p.batchSim(budgets, n_iter, qsr,env)
pErr_qbe = p.batchSim(budgets, n_iter, qbe,env)


# plot the results
p.plotBudgetVErr(pErr_qsr,pErr_qsar,pErr_qbe, pErr_quniform, n_iter,
                 ['Q-SR','Q-SAR','Q-BS','Q-Uniform'],
                 'BAI with ' + str(quantile) + ' quantile (m=' + str(n_optimal) + ')',
                 'case1')

#%% batch proc for test case 2
quantile = 0.8
# set up the environment as a lambda function
env = lambda: p.env2(n_optimal)

# lambda funcitons for the policy algorithms 
qsar = lambda: p.qsar(n_optimal,quantile)
sar = lambda: p.sar(n_optimal)
qsr = lambda: p.qsr(n_optimal,quantile)
sr = lambda: p.sr(n_optimal)

# set up the batch simulaitons with the lambda functions
pErr_qsar = p.batchSim(budgets, n_iter, qsar,env)
pErr_sar = p.batchSim(budgets, n_iter, sar,env)
pErr_qsr = p.batchSim(budgets, n_iter, qsr,env)
pErr_sr = p.batchSim(budgets, n_iter, sr,env)

# plot the results
p.plotBudgetVErr(pErr_qsr,pErr_qsar, pErr_sr,pErr_sar, n_iter,
                 ['Q-SR','Q-SAR','SR','SAR'],
                 'BAI with ' + str(quantile) + ' quantile (m=' + str(n_optimal) + ')',
                 'case2')

#%% set budget for single optimal cases
n_optimal = 1
budgets = list(range(500,2501,250))
#%% batch proc for test case 3
quantile = 0.5
# set up the environment as a lambda function
env = lambda: p.env1(n_optimal)

# lambda funcitons for the policy algorithms 
qsar = lambda: p.qsar(n_optimal,quantile)
quniform = lambda: p.q_uniform(n_optimal,quantile)
qsr = lambda: p.qsr(n_optimal,quantile)
qbe = lambda: p.qbe(n_optimal,quantile)

# set up the batch simulaitons with the lambda functions
pErr_qsar = p.batchSim(budgets, n_iter, qsar,env)
pErr_quniform = p.batchSim(budgets, n_iter, quniform,env)
pErr_qsr = p.batchSim(budgets, n_iter, qsr,env)
pErr_qbe = p.batchSim(budgets, n_iter, qbe,env)

# plot the results
p.plotBudgetVErr(pErr_qsr,pErr_qsar,pErr_qbe,pErr_quniform, n_iter,
                 ['Q-SR','Q-SAR','Q-BS','Q-Uniform'],
                 'BAI with ' + str(quantile) + ' quantile (m=' + str(n_optimal) + ')',
                 'case3')

# %% batch proc for test case 4
quantile = 0.8
# set up the environment as a lambda function
env = lambda: p.env2(n_optimal)

# lambda funcitons for the policy algorithms 
qsar = lambda: p.qsar(n_optimal,quantile)
sar = lambda: p.sar(n_optimal)
qsr = lambda: p.qsr(n_optimal,quantile)
sr = lambda: p.sr(n_optimal)

# set up the batch simulaitons with the lambda functions
pErr_qsar = p.batchSim(budgets, n_iter, qsar,env)
pErr_sar = p.batchSim(budgets, n_iter, sar,env)
pErr_qsr = p.batchSim(budgets, n_iter, qsr,env)
pErr_sr = p.batchSim(budgets, n_iter, sr,env)

# plot the results
p.plotBudgetVErr(pErr_qsr,pErr_qsar, pErr_sr,pErr_sar, n_iter,
                 ['Q-SR','Q-SAR','SR','SAR'],
                 'BAI with ' + str(quantile) + ' quantile (m=' + str(n_optimal) + ')',
                 'case4')
#############################################

