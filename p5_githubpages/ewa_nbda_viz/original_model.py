#!/usr/bin/env python
from functools import partial
import csv
from collections import Counter
import math
import sys
import networkx as nx
import random
import numpy as np
from os.path import isfile,join
from multiprocessing import Pool,Lock,Value,Manager
import pandas as pd
from os import listdir,cpu_count,remove
import time
from scipy.special import logsumexp
#replace with file of params you'd like to sim
from paramlist_baseline_production import *

#define the class behavior
class behavior:
    def __init__(self,name,payoff,base_rate):
        self.name = name #string name of behavior
        self.payoff = payoff #float payoff of behavior
        self.base_rate = base_rate #baseline learning parameter (lambda_0) from NBDA

all_behaviors = dict((name, behavior(name,behavior_payoffs[i], base_rate)) for i,name in enumerate(behavior_names))

class agent:
    '''
    instances of agents are created and attached to nodes in the network
    agent in node x can be referenced with G.nodes[x]["data"]
    '''
    def __init__(self,ID,params_list):
        self.id = ID #keeps agents identifiable across simulations
        self.knowledge = {} #name, e_mat, i_mat, s_mat, p_mat,solve_count
        self.ind_memory = [] #records agent's own productions for the duration of the memory window
        self.temp_social_memory = [] #records observed productions for the duration of the timestep
        self.long_social_memory = [] #records lists of lists, each sublist from a single timestep within the memory window
        self.naive = True # binary variable that flips to False once any behavior has been learned by the agent
        self.produced_b = False #binary variable that flips to True once the novel behavior has been performed.
        self.time_acquired_b = -1 #for debugging, this will change to the timestep agents acquired once it happens
        self.time_produced_b = -1 #for debugging, this will change to the timestep agents produced novel first once it happens
        self.total_productions_b = 0 #this be used for the transmission weight (total_prod/timesteps knowledgable) during inferential NBDA

        #EWA parameters
        self.sigma = params_list[0] #social information bias
        self.rho = params_list[1] #recent payoff bias
        self.chi = params_list[2] #conformity of social influence exponent
        self.alpha = params_list[3] #inverse exploration parameter (conservatism)

    def observe(self,behavior):
        if behavior in self.knowledge:
            self.temp_social_memory.append(behavior)
            #print("observation by {}: {}".format(self.id, self.knowledge[behavior]))

    def prune_ind_memory(self):
        while len(self.ind_memory) > memory_window:
            self.ind_memory.pop(0)
        #assert len(self.ind_memory) <= memory_window, print("Individual memory has exceeded memory window.",len(self.ind_memory))

    def prune_social_memory(self):
        while len(self.long_social_memory) > memory_window:
            self.long_social_memory.pop(0)
        #assert len(self.long_social_memory) <= memory_window, print("Social memory has exceeded memory window.",len(self.long_social_memory))

    def consolidate_social_memory(self):
        self.long_social_memory.append(self.temp_social_memory)

    def reset_temp_social_memory(self):
        self.temp_social_memory = []

    def E_mat_update(self, produced_behavior):
        reward = all_behaviors[produced_behavior].payoff
        #print("payoff {}".format(reward))
        for known_behavior in self.knowledge.keys():
            if known_behavior==produced_behavior:
                new_A_kt = (1 - self.rho)*self.knowledge[known_behavior]["e_mat"][-1] + self.rho*reward
                #assert new_A_kt >= self.knowledge[known_behavior]["e_mat"], print("Akit update of produced behavior bad{}".format(new_A_kt))
            else:
                new_A_kt = (1 - self.rho)*self.knowledge[known_behavior]["e_mat"][-1] #null payoff for non-produced behavior
                #assert new_A_kt <= self.knowledge[known_behavior]["e_mat"], print("Akit update of un-produced behavior bad {}".format(new_A_kt))
            self.knowledge[known_behavior]["e_mat"] += [new_A_kt]

    def I_mat_update(self):
        if not self.naive:
            #print("Agent{} I_mat_update(). old I_mat: {}".format(self.id, self.I_mat))
            E_mat = np.array([behavior["e_mat"][-1] for behavior in self.knowledge.values()])
            alpha_by_E_mat = np.multiply(E_mat, self.alpha)
            summed_E_mat = logsumexp(alpha_by_E_mat)
            #assert np.sum(new_I_mat)<=1, print(new_I_mat, np.sum(new_I_mat))
            for count, behavior in enumerate(self.knowledge.values()):
                behavior["i_mat"] += [alpha_by_E_mat[count] - summed_E_mat]


    def S_mat_update(self):
        if not self.naive:
            social_memory = [item for subl in self.long_social_memory for item in subl]
            #assert len(social_memory) <= (N * memory_window), print("memory {} exceeds memory window {}".format(len(social_memory),N*memory_window))
            if len(social_memory)>0:
                denom=0
                for behavior in self.knowledge.keys():
                    denom += social_memory.count(behavior)**self.chi
                for behavior in self.knowledge.keys():
                    new_S_kt = (social_memory.count(behavior)**self.chi) / denom
                    #assert new_S_kt<=1, print("S matrix value exceeds 1", new_S_kt)
                    self.knowledge[behavior]["s_mat"] += [new_S_kt]

            #if no social memory present
            else:
                for count,behavior in enumerate(self.knowledge.values()):
                    new_S_kt = 0
                    behavior["s_mat"] += [new_S_kt]


    def P_mat_update(self):
        if not self.naive:
            #print("Agent{} P_mat_update(). Old P_mat{}".format(self.id, self.P_mat))
            social_memory = np.array([x["s_mat"][-1] for x in self.knowledge.values()])
            #print("social memory {}".format(social_memory))
            if np.sum(social_memory)>0:
                for behavior in self.knowledge.values():
                    behavior["p_mat"] += [(1 - self.sigma)*np.exp(behavior["i_mat"][-1]) + self.sigma*behavior["s_mat"][-1]]

            else:
                for behavior in self.knowledge.values():
                    behavior["p_mat"] += [np.exp(behavior["i_mat"][-1])]


    def produceBehavior(self, timestep, end_flag):
        #print("Agent{} produceBehavior(). P_mat is {}".format(self.id, self.P_mat))
        behavior_choices = [behavior for behavior in self.knowledge.keys()]
        #print(behavior_choices)
        p_mat = np.array([x["p_mat"][-1] for x in self.knowledge.values()])
        #assert np.sum(p_mat)==1, "pmat not summing to 1: {}".format(np.sum(p_mat))
        #prevent negative values from floating point error
        #p_mat = np.clip(p_mat, 0, 1)

        #choose production with probability from p_mat
        try:
            production = np.random.choice(behavior_choices, p=p_mat)
        except Exception as e:
            print("Error producing behavior:", e)
            print(sum(p_mat))

        else:
            if production=="b" and end_flag==-1:
                self.total_productions_b +=1
                if self.produced_b==False:
                    self.time_produced_b=timestep
                    self.produced_b=True
            self.ind_memory.append(production) #adds production to memory
            self.E_mat_update(production) #update attraction score for next production

        return production

    def acquire_behavior(self, G, timestep):
        neighbors = [node for node in G.neighbors(self.id)]
        for behavior.name in all_behaviors:
            if behavior.name not in self.knowledge.keys():
                acquisition_prob, transmission_weight = lambda_t(behavior.name,neighbors,G)
                #assert 0 <= acquisition_prob <=1, "resulting acquision prob from NBDA must be between 0 and 1"
                if random.random() < acquisition_prob:
                    self.naive=False
                    self.time_acquired_b=timestep
                    self.knowledge[behavior.name] = {"e_mat": [0],"i_mat": [],"s_mat":[],"p_mat":[]}
                    self.I_mat_update()
                    self.S_mat_update()
                    self.P_mat_update()


def generate_network(graph_type,params_list):
    if graph_type == "complete":
        G = nx.complete_graph(N)
    elif graph_type == "random_regular":
        G = nx.random_regular_graph(6, N, seed=None)
    elif graph_type == "random_erdos":
        G = nx.gnp_random_graph(N, 0.26087, seed=None, directed=False) # khat=p(n-1)
        while not nx.is_connected(G):
            G = nx.gnp_random_graph(N, 0.26087, seed=None, directed=False)
    elif graph_type == "random_barabasi":
        G = nx.barabasi_albert_graph(N, 3, seed=None) #khat=2m
    elif graph_type == "random_small_world":
        G = nx.connected_watts_strogatz_graph(N, int(N/4), 0, tries=200, seed=None)
    elif graph_type == "custom_adj_list":
        G = nx.read_adjlist(custom_adj_list_filename)
    else:
        print("Incorrect graph name!")

    random_agent = random.randint(0, N-1) #agent randomly selected as seed agent (see below)

    for x in list(G.nodes()):
        G.nodes[x]['data'] = agent(x,params_list) #give instance of agent object as 'data' attribute in each node

        if random.random() <= initial_knowledgable_prop:
            G.nodes[x]['data'].naive=False

            if production_only:
                G.nodes[x]['data'].knowledge["a"] = {"e_mat": [all_behaviors["a"].payoff],"i_mat": [],"s_mat": [],"p_mat": []}
                G.nodes[x]['data'].knowledge["b"] = {"e_mat": [0],"i_mat": [],"s_mat": [],"p_mat": []}
                G.nodes[x]['data'].time_acquired_b = 0

            else:
                if full_initial_weights:
                    G.nodes[x]['data'].knowledge["a"] = {"e_mat": [all_behaviors["a"].payoff],"i_mat": [],"s_mat":[],"p_mat":[]}
                else:
                    G.nodes[x]['data'].knowledge["a"] = {"e_mat": [0],"i_mat": [],"s_mat":[],"p_mat":[]}

                #seed one individual as knowledgable of the novel behavior
                if x==random_agent:
                    G.nodes[x]['data'].knowledge["b"] = {"e_mat": [0],"i_mat": [],"s_mat":[],"p_mat":[]}
                    G.nodes[x]['data'].time_acquired_b = 0

            G.nodes[x]['data'].I_mat_update()
            G.nodes[x]['data'].S_mat_update()
            G.nodes[x]['data'].P_mat_update()
    return G

##the following functions are used in the NBDA generative model
def NBDA(G,timestep):
    for agent in list(G.nodes()):
        G.nodes[agent]["data"].acquire_behavior(G, timestep+1)

#loops through neighbors and produces either a binary or proportion of all behaviors produced by each neighbor which are a given behavior. This value is then summed for all neighbors of the focal individual.
def w_jbt(behavior, neighbors, G):

    if trans_weight=="binary":
        weight_list = [1 for neighbor in neighbors if behavior in G.nodes[neighbor]["data"].knowledge.keys()]
    elif trans_weight=="proportional":
        weight_list = [G.nodes[neighbor]["data"].ind_memory.count(behavior)/memory_window for neighbor in neighbors if behavior in G.nodes[neighbor]["data"].knowledge.keys()]
    else:
        print("trans_weight not valid")

    return weight_list


#this is the function T((a_i,z(t))) (Firth 2020)
def transmission_function(behavior,neighbors,G):
    weight_list = w_jbt(behavior, neighbors, G)

    transmission_weight=sum(weight_list)

    #Hoppitt_Laland 2013 General form, s is unbounded
    #lambda(t) = baseline rate function*(rate of acquisition * sum(a_ij*z_j(t)) + A)
    transmission_func = s_param * transmission_weight

    return transmission_func, transmission_weight

#returns the second half of the NBDA equation relating to individual learning. Simplest case is a value of 1 or 0, but could take ind level variables
def A_param():
    if asocial_learning:
        asocial_learning_func=1
    else:
        asocial_learning_func=0

    return asocial_learning_func

#calculates the acquisition rate of a behavior given the productions of the neighbors
#all neighbors are considered as equivalently associated, but this could be changed by adding a term
def lambda_t(behavior, neighbors, G):

    transmission_func, transmission_weight = transmission_function(behavior, neighbors, G)

    asocial_learning_func = A_param()

    #lambda(t) = baseline rate function*(transmission_func + ind_learning_parameters)
    acq_rate = all_behaviors[behavior].base_rate * (transmission_func + asocial_learning_func)

    #convert from rate to probability within the last timestep that an agent has acquired the behavior
    acq_prob = 1-math.exp( -acq_rate )

    return acq_prob, transmission_weight

def production_event(G, producer, timestep, end_flag):
    #print("\n***new production_event***\nproducer is agent {:d} ".format(producer))
    beh_production = G.nodes[producer]["data"].produceBehavior(timestep, end_flag)
    #print("\n***new production_event***\nproducer is agent {:d}, produces {}".format(producer, beh))

    neighbors = [node for node in G.neighbors(producer)]
    for neighbor in neighbors:
        G.nodes[neighbor]["data"].observe(beh_production)
    return beh_production

def create_csv(file_path, file_path_acq_prod, file_path_agent_mat_val):
    #create variable names for behaviors to be included in simulation
    behavior_list = ["behavior_{}".format(behavior.name) for behavior in all_behaviors.values()]
    #labels=["E_mat","I_mat","S_mat","P_mat"]
    #agent_matrix_list = ["behavior_{}_{}".format(behavior.name,label) for behavior in all_behaviors.values() for label in labels]

    #writes header for main data
    with open(file_path,"w") as fout:
        fout.write("sim, graph_type, pop_size, memory_window, NBDA_basehazard, NBDA_s_param, NBDA_transmission_weight, asocial_learning, EWA_sigma, EWA_rho, EWA_chi, EWA_alpha, timestep, num_know_novel, num_produced_b, {}\n".format(",".join(behavior_list)))

    #writes header for main data
    with open(file_path_acq_prod,"w") as fout:
        fout.write("sim, graph_type, pop_size, memory_window, NBDA_basehazard, NBDA_s_param, NBDA_transmission_weight, asocial_learning, EWA_sigma, EWA_rho, EWA_chi, EWA_alpha, agent, timestep_acquisition_b, timestep_production_b, total_productions_b\n")

    #writes header for main data
    with open(file_path_agent_mat_val,"w") as fout:
        fout.write("sim, agent, timestep, B_e, B_i, B_s, B_p\n")


def write_csv(file_path,sim_num,params_list,timestep,num_know_novel,num_produced_b,beh_freqs):
    behavior_counts = [str(beh) for beh in beh_freqs.values()]
    #print(behavior_counts)
    count_string = ",".join(behavior_counts)

    #writes header for main data
    with open(file_path,"a+") as fout:
        fout.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
        sim_num,
        graph_type,
        N,
        memory_window,
        base_rate,
        s_param,
        trans_weight,
        asocial_learning,
        params_list[0],
        params_list[1],
        params_list[2],
        params_list[3],
        timestep,
        num_know_novel,
        num_produced_b,
        count_string))

def write_diffusion_csv(file_path,sim_num,params_list,G):
    for agent in range(N):
        #writes header for main data
        with open(file_path,"a+") as fout:
            fout.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
            sim_num,
            graph_type,
            N,
            memory_window,
            base_rate,
            s_param,
            trans_weight,
            asocial_learning,
            params_list[0],
            params_list[1],
            params_list[2],
            params_list[3],
            G.nodes[agent]["data"].id,
            G.nodes[agent]["data"].time_acquired_b,
            G.nodes[agent]["data"].time_produced_b,
            G.nodes[agent]["data"].total_productions_b))

def write_agent_mat_val_csv(file_path, sim_num, G, timestep):
    for agent in range(N):
        E = np.round(G.nodes[agent]["data"].knowledge["b"]["e_mat"],4)
        I = np.round(G.nodes[agent]["data"].knowledge["b"]["i_mat"],4)
        S = np.round(G.nodes[agent]["data"].knowledge["b"]["s_mat"],4)
        P = np.round(G.nodes[agent]["data"].knowledge["b"]["p_mat"],4)
        length=len(E)
        timesteps = list(range(G.nodes[agent]["data"].time_acquired_b, timestep+1))
        sim = [sim_num for i in range(length+1)]
        ag = [agent for i in range(length+1)]
        rows = zip(sim,ag,timesteps,E,I,S,P)
        with open(file_path,"a+") as fout:
            writer = csv.writer(fout)
            writer.writerows(rows)

def agent_values(agent, behavior,G):
    #if the agent knows a behavior, return associated matrix values, otherwise return zeroes
    if behavior in G.nodes[agent]["data"].knowledge.keys():
        return [G.nodes[agent]["data"].knowledge[behavior]["e_mat"],
        G.nodes[agent]["data"].knowledge[behavior]["i_mat"],
        G.nodes[agent]["data"].knowledge[behavior]["s_mat"],
        G.nodes[agent]["data"].knowledge[behavior]["p_mat"]]
    else:
        return [0,0,0,0]

def simulation(lock,master_sim_num,params_list):
    global s_param
    s_param = params_list[4]
    global trans_weight
    trans_weight = params_list[5]
    global memory_window
    memory_window = params_list[6]
    global graph_type
    graph_type = params_list[7]
    global asocial_learning
    asocial_learning = params_list[8]


    #get unique simulation number for this round, increment global master_sim_num
    with lock:
        sim_num = master_sim_num.value
        master_sim_num.value += 1

    #create csv file with headers
    file_path = "../model_outputs/csvs_raw/timestep_data/{}/sim_{}.csv".format(condition_name, sim_num)
    file_path_acq_prod = "../model_outputs/csvs_raw/acq_prod_data/{}/sim_{}.csv".format(condition_name, sim_num)
    file_path_agent_mat_val = "../model_outputs/csvs_raw/agent_mat_values/{}/sim_{}.csv".format(condition_name, sim_num)

    #if not isfile(file_path):
    create_csv(file_path,file_path_acq_prod,file_path_agent_mat_val)

    print("simulation{} EWA(sigma{} rho{} chi{} alpha{}) NBDA(s_param{}, trans_weight{}, soc_memory_window{}".format(sim_num,params_list[0],params_list[1],params_list[2],params_list[3],params_list[4],params_list[5],params_list[6]))

    np.random.seed() #reset seed for random num generation

    G = generate_network(graph_type,params_list) #create network populated with agents

    nx.write_edgelist(G, "../model_outputs/csvs_raw/adjlists_data/{}/adjlist_sim_{}.txt".format(condition_name,sim_num))

    timestep=0
    end_flag=-1

    while True:

        know_novel = [agent for agent in range(N) if "b" in G.nodes[agent]["data"].knowledge.keys()]
        num_know_novel = len(know_novel)

        #create counter object to keep track of frequencies of behaviors for this timestep
        beh_freqs = Counter()
        for behavior in all_behaviors.values():
            beh_freqs[behavior.name] += 0

        #create list of all knowledgable agents
        knowledgable = [agent for agent in range(N) if G.nodes[agent]["data"].naive == False ]

        #knowledgable individuals produce given number of behaviors beh_per_TS
        for individual in knowledgable:
            for interactions in range(beh_per_TS):
                produced_behavior = production_event(G, individual, timestep, end_flag)
                beh_freqs[produced_behavior] += 1

        num_produced_b = len([agent for agent in range(N) if G.nodes[agent]["data"].produced_b == True ])


        # write data and end sim when at full diffusion
        if num_produced_b==N and end_flag==-1:
            final_timestep = timestep+5
            end_flag = final_timestep // 5 * 5 #allow to go 50 more timesteps past full production

        #write data
        if timestep%1==0:
            write_csv(file_path,sim_num,params_list,timestep,num_know_novel,num_produced_b,beh_freqs)
            if timestep==end_flag:
                write_diffusion_csv(file_path_acq_prod, sim_num,params_list, G)
                write_agent_mat_val_csv(file_path_agent_mat_val, sim_num, G, timestep)
                break

        #housekeeping for next timestep
        for agent in range(N):
            G.nodes[agent]["data"].prune_ind_memory()
            G.nodes[agent]["data"].consolidate_social_memory()
            G.nodes[agent]["data"].prune_social_memory()
            G.nodes[agent]["data"].reset_temp_social_memory()
            G.nodes[agent]["data"].I_mat_update()
            G.nodes[agent]["data"].S_mat_update()
            G.nodes[agent]["data"].P_mat_update()

        #run NBDA calculations
        NBDA(G,timestep)
        timestep+=1

if __name__=="__main__":

    #clear out old adjlist
    adjlist_path = "../model_outputs/csvs_raw/adjlists_data/{}".format(condition_name)
    for f in listdir(adjlist_path):
        if ".txt" in f:
            remove(join(adjlist_path,f))

    #create manager to handle common memory for multicore processing
    with Manager() as manager:

        master_sim_num = manager.Value('i', 0)
        # creating a Lock object to handle multiple processes accessing master sim #
        lock = manager.Lock()

        #create massive list of other parameter values
        params_list = [(sigma,rho,chi,alpha,s_param,trans_weight,memory_window,graph_type, asocial_learning) \
        for sigma in sigma_values \
        for rho in rho_values \
        for chi in chi_values \
        for alpha in alpha_values \
        for s_param in s_param_values \
        for trans_weight in transmission_weight_values \
        for memory_window in memory_window_values \
        for graph_type in graph_types \
        for asocial_learning in asocial_learning_values \
        for replicate in range(replicates)]

        random.shuffle(params_list)

        print("{} simulations to go".format(len(params_list)))
        time.sleep(3)
        func = partial(simulation, lock, master_sim_num)
        with Pool(processes=cpu_count()-1) as pool:
            pool.map(func,params_list)

    #once sims finish collect timestep data and put into 1 csv
    #where raw data is put
    directory_path="../model_outputs/csvs_raw/timestep_data/{}".format(condition_name)
    #where concatenated dataframes will be put after all sims finish running
    new_directory_path="../model_outputs/csvs_concat"
    #concating dataframes
    df_list = [pd.read_csv(join(directory_path,f)) for f in listdir(directory_path) if ".csv" in f]
    df_concat = pd.concat(df_list)
    #set final filename here
    df_concat.to_csv(join(new_directory_path,file_name), index = False)
    #remove raw data files
    for f in listdir(directory_path):
        if ".csv" in f:
            remove(join(directory_path,f))

    #once sims finish collect data and put into 1 csv
    #where raw data is put
    directory_path="../model_outputs/csvs_raw/acq_prod_data/{}".format(condition_name)
    #where concatenated dataframes will be put after all sims finish running
    new_directory_path="../model_outputs/csvs_concat"
    #concating dataframes
    df_list = [pd.read_csv(join(directory_path,f)) for f in listdir(directory_path) if ".csv" in f]
    df_concat = pd.concat(df_list)
    #set final filename here
    df_concat.to_csv(join(new_directory_path,file_name_acq_prod), index = False)
    #remove raw data files
    for f in listdir(directory_path):
        if ".csv" in f:
            remove(join(directory_path,f))

    #once sims finish collect data and put into 1 csv
    #where raw data is put
    directory_path="../model_outputs/csvs_raw/agent_mat_values/{}".format(condition_name)
    #where concatenated dataframes will be put after all sims finish running
    new_directory_path="../model_outputs/csvs_concat"
    #concating dataframes
    df_list = [pd.read_csv(join(directory_path,f)) for f in listdir(directory_path) if ".csv" in f]
    df_concat = pd.concat(df_list)
    #set final filename here
    df_concat.to_csv(join(new_directory_path,file_name_agent_mat_val), index = False)
    #remove raw data files
    for f in listdir(directory_path):
        if ".csv" in f:
            remove(join(directory_path,f))
