# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 15:35:11 2020

@author: user
"""
#%%
import torch
import torch.functional as F
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader, Dataset
import copy
from sklearn.metrics import accuracy_score
import numpy as np
import time 

#ourself libs
from model_initiation import model_init
from data_preprocess import data_init, data_init_with_shadow
from FL_base import global_train_once
from FL_base import fedavg
from FL_base import test

from FL_base import FL_Train, FL_Retrain
from Fed_Unlearn_base import unlearning, unlearning_without_cali, federated_learning_unlearning
from membership_inference import train_attack_model, attack

"""Step 0. Initialize Federated Unlearning parameters"""
class Arguments():
    def __init__(self):
        #Federated Learning Settings
        self.N_total_client = 100
        self.N_client = 25
        self.data_name = 'adult'# purchase, cifar10, mnist, adult

        self.global_epoch = 20
        self.local_epoch = 10

        #Model Training Settings
        self.local_batch_size = 64
        self.local_lr = 0.005
        
        self.test_batch_size = 64
        self.seed = 1
        self.save_all_model = True
        self.cuda_state = torch.cuda.is_available()
        self.use_gpu = True
        self.train_with_test = False
        
        
        #Federated Unlearning Settings
        self.unlearn_interval= 2#Used to control how many rounds the model parameters are saved.1 represents the parameter saved once per round  N_itv in our paper.
        self.forget_client_idx = 2 #If want to forget, change None to the client index
        
                                #If this parameter is set to False, only the global model after the final training is completed is output
        self.if_retrain = True#If set to True, the global model is retrained using the FL-Retrain function, and data corresponding to the user for the forget_client_IDx number is discarded.
        
        self.if_unlearning = False#If set to False, the global_train_once function will not skip users that need to be forgotten;If set to True, global_train_once skips the forgotten user during training
        
        self.forget_local_epoch_ratio = 0.5 #When a user is selected to be forgotten, other users need to train several rounds of on-line training in their respective data sets to obtain the general direction of model convergence in order to provide the general direction of model convergence.
                                            #forget_local_epoch_ratio*local_epoch Is the number of rounds of local training when we need to get the convergence direction of each local model
        # self.mia_oldGM = False

def Federated_Unlearning():
    """Step 1.Set the parameters for Federated Unlearning"""
    FL_params = Arguments()
    torch.manual_seed(FL_params.seed)
    #kwargs for data loader 
    print(60*'=')
    print("Step1. Federated Learning Settings \n We use dataset: "+FL_params.data_name+(" for our Federated Unlearning experiment.\n"))


    """Step 2. construct the necessary user private data set required for federated learning, as well as a common test set"""
    print(60*'=')
    print("Step2. Client data loaded, testing data loaded!!!\n       Initial Model loaded!!!")
    #加载数据   
    init_global_model = model_init(FL_params.data_name)
    client_all_loaders, test_loader = data_init(FL_params)

    selected_clients=np.random.choice(range(FL_params.N_total_client),size=FL_params.N_client, replace=False)
    client_loaders = list()
    for idx in selected_clients:
        client_loaders.append(client_all_loaders[idx])
    # client_all_loaders = client_loaders[selected_clients]
    # client_loaders, test_loader, shadow_client_loaders, shadow_test_loader = data_init_with_shadow(FL_params)
    """
    This section of the code gets the initialization model init Global Model
    User data loader for FL training Client_loaders and test data loader Test_loader
    User data loader for covert FL training, Shadow_client_loaders, and test data loader Shadowl_test_loader
    """

    """Step 3. Select a client's data to forget，1.Federated Learning, 2.Unlearning(FedEraser), and 3.(Accumulating)Unlearing without calibration"""
    print(60*'=')
    print("Step3. Fedearated Learning and Unlearning Training...")
    # FedAvg, FedEraser, FedAccum,
    old_GMs, unlearn_GMs, uncali_unlearn_GMs, _ = federated_learning_unlearning(init_global_model,
                                                                                            client_loaders, 
                                                                                            test_loader, 
                                                                                            FL_params)

    if(FL_params.if_retrain == True):
        
        t1 = time.time()
        #FedRetrain
        retrain_GMs = FL_Retrain(init_global_model, client_loaders, test_loader, FL_params)
        t2 = time.time()
        print("Time using = {} seconds".format(round(t2-t1),3))

    #Evaluation
    fedavg_test_loss, fedavg_test_acc = test(old_GMs[-1], test_loader)
    federaser_test_loss, federaser_test_acc = test(unlearn_GMs[-1], test_loader)
    fedaccum_test_loss, fedaccum_test_acc = test(uncali_unlearn_GMs[-1], test_loader)
    fedretrain_test_loss, fedretrain_test_acc = test(retrain_GMs[-1], test_loader)

    target_loader = client_loaders[FL_params.forget_client_idx]
    fedavg_target_loss, fedavg_target_acc = test(old_GMs[-1], target_loader)
    federaser_target_loss, federaser_target_acc = test(unlearn_GMs[-1], target_loader)
    fedaccum_target_loss, fedaccum_target_acc = test(uncali_unlearn_GMs[-1], target_loader)
    fedretrain_target_loss, fedretrain_target_acc = test(retrain_GMs[-1], target_loader)

    print(5*"*"+"  Result Summary  "+5*"*")
    print("[FedEraser] Test set: Average loss = {:.8f}, Average acc = {:.4f}".format(federaser_test_loss, federaser_test_acc))
    print("[FedAccum] Test set: Average loss = {:.8f}, Average acc = {:.4f}".format(fedaccum_test_loss, fedaccum_test_acc))
    print("[FedRetrain] Test set: Average loss = {:.8f}, Average acc = {:.4f}".format(fedretrain_test_loss, fedretrain_test_acc))
    print("[FedAvg] Test set: Average loss = {:.8f}, Average acc = {:.4f}".format(fedavg_test_loss, fedavg_test_acc))
    print("\n")
    print("[FedEraser] Target set: Average loss = {:.8f}, Average acc = {:.4f}".format(federaser_target_loss, federaser_target_acc))
    print("[FedAccum] Target set: Average loss = {:.8f}, Average acc = {:.4f}".format(fedaccum_target_loss, fedaccum_target_acc))
    print("[FedRetrain] Target set: Average loss = {:.8f}, Average acc = {:.4f}".format(fedretrain_target_loss, fedretrain_target_acc))
    print("[FedAvg] Target set: Average loss = {:.8f}, Average acc = {:.4f}".format(fedavg_target_loss, fedavg_target_acc))

    """Step 4  The member inference attack model is built based on the output of the Target Global Model on client_loaders and test_loaders.In this case, we only do the MIA attack on the model at the end of the training"""
    
    """MIA:Based on the output of oldGM model, MIA attack model was built, and then the attack model was used to attack unlearn GM. If the attack accuracy significantly decreased, it indicated that our unlearn method was indeed effective to remove the user's information"""
    print(60*'=')
    print("Step4. Membership Inference Attack aganist GM...")

    T_epoch = -1
    # MIA setting:Target model == Shadow Model
    old_GM = old_GMs[T_epoch]
    attack_model = train_attack_model(old_GM, client_loaders, test_loader, FL_params)


    print("\nEpoch  = {}".format(T_epoch))
    print("Attacking against FL Standard  ")
    target_model = old_GMs[T_epoch]
    (PRE_old, REC_old, F1_old) = attack(target_model, attack_model, client_loaders, test_loader, FL_params)

    if(FL_params.if_retrain == True):
        print("Attacking against FL Retrain  ")
        target_model = retrain_GMs[T_epoch]
        (PRE_retrain, REC_retrain, F1_retrain) = attack(target_model, attack_model, client_loaders, test_loader, FL_params)

    print("Attacking against FL Unlearn  ")
    target_model = unlearn_GMs[T_epoch]
    (PRE_unlearn, REC_unlearn, F1_unlearn) = attack(target_model, attack_model, client_loaders, test_loader, FL_params)

    print("Attacking against FL Unlearn without calibration  ")
    target_model = uncali_unlearn_GMs[T_epoch]
    (PRE_uncali_unlearn, REC_uncali_unlearn, F1_uncali_unlearn) = attack(target_model, attack_model, client_loaders, test_loader, FL_params)


if __name__=='__main__':
    Federated_Unlearning()














































