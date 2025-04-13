# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 19:53:51 2025

"""
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 16
#%% set up the arms and their distributions 

class exp_dist:
    def __init__(self,mean,label): # initialize the class
        self.mean = mean
        self.rewards = np.array([])
        self.label = label
        
    def pdf(self,x): 
        lm = 1/self.mean
        return lm*np.exp(-lm*x)
    
    def draw(self,n): # draw from the distribution 
        lm = 1/self.mean
        reward = -np.log(1-np.random.uniform(size=n))/lm
        self.rewards = np.concatenate((self.rewards,reward))

class abs_gauss_dist:
    
    def __init__(self,mean,std_dev,label): # initilize the class
        self.mean = mean
        self.std_dev = std_dev
        self.rewards = np.array([])
        self.label = label
        
    def pdf(self,x):
        return (1/(self.std_dev*np.sqrt(2*np.pi)))*np.exp(-(x-self.mean)**2/(2*self.std_dev**2)) \
        + (1/(self.std_dev*np.sqrt(2*np.pi)))*np.exp(-(x+self.mean)**2/(2*self.std_dev**2))
        
    def draw(self,n): # draw from the distribution
        reward = np.abs(np.random.normal(self.mean,self.std_dev,size=n))
        self.rewards = np.concatenate((self.rewards,reward))

#%% set up algorithms

class qsar: 

    def __init__(self,n_optimal,quantile): # initilize the class
        self.quantile = quantile
        self.n_optimal = n_optimal
        self.type = 'SAR'
        
    def getQuantiles(self,arms): # compute the quantiles
        self.active_quantiles = []
        for arm in arms:
            self.active_quantiles.append(np.quantile(arm.rewards, self.quantile))
        
    def run(self,arms,accepted): # set up the run algorithm
        # get the quantiles and sort the arms
        self.getQuantiles(arms)
        order = np.flip(np.argsort(self.active_quantiles))
        sorted_quantiles = sorted(self.active_quantiles,reverse=True)
        
        # compute the gaps 
        delta_best = sorted_quantiles[0] - sorted_quantiles[self.n_optimal+1]
        delta_worst = sorted_quantiles[self.n_optimal]-sorted_quantiles[-1]
        
        # compare the gaps and select or reject arms
        if delta_best > delta_worst:
            accepted.append(arms[int(order[0])])
            del arms[int(order[0])]
            self.n_optimal -= 1
        else:
            del arms[int(order[-1])]
        return arms,accepted

class sar: 

    def __init__(self,n_optimal): # initialize the class
        self.n_optimal = n_optimal
        self.type = 'SAR'
        
    def getMeans(self,arms): # calculate the means
        self.active_means = []
        for arm in arms:
            self.active_means.append(np.mean(arm.rewards))
        
    def run(self,arms,accepted):
        # get the means and sort the arms
        self.getMeans(arms)
        order = np.flip(np.argsort(self.active_means))
        sorted_means = sorted(self.active_means,reverse=True)
        
        # compute the gaps
        delta_best = sorted_means[0] - sorted_means[self.n_optimal+1]
        delta_worst = sorted_means[self.n_optimal]-sorted_means[-1]
        
        # compare the gaps and select or reject arms
        if delta_best > delta_worst:
            accepted.append(arms[int(order[0])])
            del arms[int(order[0])]
            self.n_optimal -= 1
        else:
            del arms[int(order[-1])]
        return arms,accepted

# %% set up the successive rejects policies
class qsr: 

    def __init__(self,n_optimal,quantile): # initialize policy
        self.quantile = quantile
        self.n_optimal = n_optimal
        self.type = 'SR'
        
    def getQuantiles(self,arms): # compute quantiles
        self.active_quantiles = []
        for arm in arms:
            self.active_quantiles.append(np.quantile(arm.rewards, self.quantile))
        
    def run(self,arms,accepted): 
        # compute the quantiles and sort the arms
        self.getQuantiles(arms)
        order = np.flip(np.argsort(self.active_quantiles))
        
        # reject the arms if the number of arms left is number of optimal move the optimal arms to the accepted
        if len(arms) > self.n_optimal+1:
            del arms[int(order[-1])]
        else:
            del arms[int(order[-1])]
            for arm in arms:
                accepted.append(arm)
        return arms,accepted

class sr: 

    def __init__(self,n_optimal):
        self.n_optimal = n_optimal
        self.type = 'SR'
        
    def getMeans(self,arms):
        self.active_means = []
        for arm in arms:
            self.active_means.append(np.mean(arm.rewards))
            
    def run(self,arms,accepted):
        # get means and sort
        self.getMeans(arms)
        order = np.flip(np.argsort(self.active_means))
        
        # reject the arms if the number of arms left is number of optimal move the optimal arms to the accepted
        if len(arms) > self.n_optimal+1:
            del arms[int(order[-1])]
        else:
            del arms[int(order[-1])]
            for arm in arms:
                accepted.append(arm)
        return arms,accepted

# %% setup batch eliminaiton and uniform quantile policies
class qbe: # quantile batch elimination
    def __init__(self,n_optimal,quantile): # initialize policy
        self.quantile = quantile
        self.n_optimal = n_optimal
        self.type = 'BE'
        
    def getQuantiles(self,arms): # compute quantiles
        self.active_quantiles = []
        for arm in arms:
            self.active_quantiles.append(np.quantile(arm.rewards, self.quantile))
        
    def run(self,arms,accepted): 
        # compute the quantiles and sort the arms
        self.getQuantiles(arms)
        order = np.flip(np.argsort(self.active_quantiles))
        
        # reject the arms if the number of arms left is number of optimal move the optimal arms to the accepted
        if len(arms) > self.n_optimal+1:
            del arms[int(order[-1])]
        else:
            del arms[int(order[-1])]
            for arm in arms:
                accepted.append(arm)
        return arms,accepted
        
class q_uniform:
    def __init__(self,n_optimal,quantile):
        self.n_optimal = n_optimal
        self.quantile = quantile
        self.type = 'UNI'
        
    def getQuantiles(self,arms):
        self.active_quantiles = []
        for arm in arms:
            self.active_quantiles.append(np.quantile(arm.rewards, self.quantile))
        
    def run(self,arms,accepted):
        # sort the arms 
        self.getQuantiles(arms)
        order = np.flip(np.argsort(self.active_quantiles))
        
        # select the top n_optimal quantiles 
        for iOrder in order[0:self.n_optimal]:
            accepted.append(arms[iOrder])
            
        return arms,accepted
        
    
        
#%% set up simulation
class sim:

    def __init__(self,budget,arms,algo):
        self.budget = budget
        self.arms = arms
        self.n_arms = len(arms)
        self.np0 = 0

        if (algo.type == 'SR') | (algo.type == 'BE'):
            self.n_rounds = self.n_arms - algo.n_optimal + 1
            self.log_k = algo.n_optimal/(algo.n_optimal+1)
            for i in range(1,self.n_rounds):
                self.log_k += 1/(self.n_arms+1-i)
        else:
            self.n_rounds = self.n_arms
            self.log_k = 1/2
            for i in range(2,self.n_arms+1):
                self.log_k += 1/i
            
        self.active = arms
        self.n_active = len(arms)
        self.algo = algo
        self.accept = []
        
    def getBudget(self,rnd):
        
        if (self.algo.type == 'SAR') | (self.algo.type == 'SR'): # calculate the budget for the accept and reject algorithms
            n_p = np.ceil((1/self.log_k)*(self.budget-self.n_arms)/(self.n_arms+1-rnd))
            budget = n_p - self.np0
            self.np0 = n_p
        
            return budget
        elif self.algo.type == 'BE': # budget calculation for  batch elimination
            L = self.n_rounds-1
            xl = 1
            sum_xl = 0
            for l in range(1,L+1):
                sum_xl += xl*(L-l)
            budget = np.floor(self.budget/(L*self.n_arms-sum_xl))
            # print('H = ' + str(budget))
            return budget  
        else: # otherwise just divide the buy the number of rounds and active arms
            return np.floor(self.budget/self.n_rounds/self.n_active)
    
    def run(self):
        
        # if uniform skip the rounds and just sample all of the arms equally
        if self.algo.type == 'UNI':
            # get uniform samples
            sample = int(np.floor(self.budget/self.n_arms))
            for arm in self.active:
                arm.draw(sample)
            [self.active,self.accept]= self.algo.run(self.active,self.accept)
            return self.accept
            
            
        # for each round
        for rnd in range(1,self.n_rounds):
            sample = int(self.getBudget(rnd))
            
            # sample all of the active arms
            for arm in self.active:
                arm.draw(sample)
                
            # update the active and accpted values
            [self.active,self.accept]= self.algo.run(self.active,self.accept)
            self.n_active = len(self.active)
            
        # # debug
        # labels = []
        # for i in self.accept:
        #     labels.append(i.label)
        # print(labels)
        
        
        return self.accept
#%% Environment 1
def A():
    return abs_gauss_dist(0,2,'A')
def B():
    return abs_gauss_dist(3.5,2,'B')
def C():
    return exp_dist(4.0,'C')
def env1(n_optimal):
    # environment optimal for 0.5 quantile
    arms = []
    optimal = []
    for i in range(15):
        # A arms 
        arms.append(A())
    for i in range(n_optimal):
        # B arms
        arms.append(B())
        optimal.append('B')
    for i in range(5):
        # C arms
        arms.append(C())
    
    return arms,optimal
        
def env2(n_optimal):
    # environment optimal for 0.8 quantile
    arms = []
    optimal = []
    for i in range(15):
        # A arms 
        arms.append(A())
    for i in range(5):
        # B arms
        arms.append(B())
    for i in range(n_optimal):
        # C arms
        arms.append(C())
        optimal.append('C')

    return arms,optimal 
    
def batchSim(budgets,n_iter,algo,env):
    pErr = {}
    classStr = str(type(algo()))
    startStr =  classStr.find('.')+1
    endStr = classStr.rfind('\'')
    thisClass = classStr[startStr:endStr].upper()
    
    for thisBudget in budgets:
        count = 0
    # for n iterations 
        for it in range(n_iter+1):
            arms,optimal = env()
    
            this_sim = sim(thisBudget,arms,algo())
            
            # run for policy algorithm
            accepted = this_sim.run()
            accept_label = []
            for item in accepted:
                accept_label.append(item.label)
            
            # count successes
            if optimal == accept_label:
                count += 1
                
            if (it % 1000 == 0) & (it !=0):
                accuracy = round(count/it*100,2)
                print(f"Done with {thisClass} Budget: {thisBudget} round [{it}/{n_iter}] [Acc:{accuracy}%]")
        # get probability of error for the given budget
        pErr[thisBudget] = 1-count/n_iter
    return pErr


def plotBudgetVErr(pErr_1,pErr_2,pErr_3,pErr_4,n_iter,leg,title,name): # plot the outputs
    fig, ax = plt.subplots() # make figure
    # plot the input pErrors
    ax.plot(pErr_1.keys(),pErr_1.values(),marker='x',linewidth=2)
    ax.plot(pErr_2.keys(),pErr_2.values(),marker='o',linewidth=2)
    ax.plot(pErr_3.keys(),pErr_3.values(),marker='v',linewidth=2)
    ax.plot(pErr_4.keys(),pErr_4.values(),marker='s',linewidth=2)
    # add the labels
    plt.title(title)
    plt.xlabel('Budget')
    plt.ylabel('Probability of Error')
    plt.ylim([-.1,1.1])
    plt.xlim([min(pErr_1.keys()), max(pErr_1.keys())])
    plt.legend(leg)
    plt.grid(visible=True)
    plt.show()
    # save the output
    fig.savefig('ProbErr_nIt_'+ str(n_iter)+name)