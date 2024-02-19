import argparse
import torch
import numpy as np
import time

#ourself libs
from model_initiation import model_init
from data_preprocess import data_init
from FL_base import test

from FL_base import FL_Train, FL_Retrain
from Fed_Unlearn_base import unlearning, unlearning_without_cali, federated_learning_unlearning
from membership_inference import train_attack_model, attack


"""Step 0. Initialize Federated Unlearning parameters"""


class Arguments():
    def __init__(self):
        # Federated Learning Settings
        self.N_total_client = 100
        self.N_client = 20
        self.data_name = 'cifar10'  # purchase, cifar10, mnist, adult

        self.global_epoch = 20
        self.local_epoch = 10

        # Model Training Settings
        self.local_batch_size = 64
        self.local_lr = 0.005

        self.test_batch_size = 64
        self.seed = 1
        self.save_all_model = True
        self.cuda_state = torch.cuda.is_available()
        self.use_gpu = True
        self.train_with_test = False

        # Federated Unlearning Settings
        self.unlearn_interval = 2  # Used to control how many rounds the model parameters are saved.1 represents the parameter saved once per round  N_itv in our paper.
        self.forget_client_idx = 2  # If want to forget, change None to the client index

        # If this parameter is set to False, only the global model after the final training is completed is output
        self.if_retrain = True  # If set to True, the global model is retrained using the FL-Retrain function, and data corresponding to the user for the forget_client_IDx number is discarded.

        self.if_unlearning = False  # If set to False, the global_train_once function will not skip users that need to be forgotten;If set to True, global_train_once skips the forgotten user during training

        self.forget_local_epoch_ratio = 0.5  # When a user is selected to be forgotten, other users need to train several rounds of on-line training in their respective data sets to obtain the general direction of model convergence in order to provide the general direction of model convergence.
        # forget_local_epoch_ratio*local_epoch Is the number of rounds of local training when we need to get the convergence direction of each local model
        # self.mia_oldGM = False

def args_parser():
    parser = argparse.ArgumentParser()

    # experiment arguments
    parser.add_argument('--data_name', default='mnist',
                        type=str, help="name of dataset")

    # federated learning arguments (Notation for the arguments followed from paper)
    parser.add_argument('--N_total_client', default=100,
                        type=int, help="number of total clients: N")
    parser.add_argument('--N_client', default=20,
                        type=int, help="number of participating clients: K")
    parser.add_argument('--global_epoch', default=20,
                        type=int, help="number of rounds of global training (before unlearning): E_global")
    parser.add_argument('--local_epoch', default=10,
                        type=int, help="number of local epochs (before unlearning): E_local")

    # model training arguments
    parser.add_argument('--local_batch_size', default=64,
                        type=int, help="local batch size")
    parser.add_argument('--local_lr', default=0.005,
                        type=float, help="learning rate of local model training")
    parser.add_argument('--test_batch_size', default=64,
                        type=int, help="test batch size")
    parser.add_argument('--seed', default=1,
                        type=int, help="random seed")
    parser.add_argument('--save_all_model', default=True,
                        type=bool, help="save all models or not")
    parser.add_argument('--cuda_state', default=torch.cuda.is_available(),
                        type=bool, help="cuda state")
    parser.add_argument('--use_gpu', default=True,
                        type=bool, help="use gpu or not")
    parser.add_argument('--train_with_test', default=False,
                        type=bool, help="train with test data or not")

    # federated unlearning arguments
    parser.add_argument('--if_retrain', default=True,
                        type=bool, help="if retrain the global model using the FL-Retrain function; "
                                        "if set to True, the global model is retrained using the FL-Retrain function, and data corresponding to the user for the forget_client_idx number is discarded;"
                                        "if set to False, only the global model after the final training is completed is output")
    parser.add_argument('--if_unlearning', default=False,
                        type=bool, help="if unlearning the client to forget; "
                                        "if set to True, the global_train_once function skips the forgotten user during training;"
                                        "if set to False, the global_train_once function will not skip users that need to be forgotten")
    parser.add_argument('--forget_client_idx', default=None,
                        type=int, help="client index to forget; if None, no client is forgotten; if want to forget, change None to the client index")
    parser.add_argument('--unlearn_interval', default=2,
                        type=int, help="retaining interval of FedEraser, used to control how many rounds the model parameters are saved")
    parser.add_argument('--forget_local_epoch_ratio', default=0.5,
                        type=float, help="when a user is selected to be forgotten, other users need to train several rounds in their respective datasets to obtain the general direction of model convergence in order to provide the general direction of model convergence;"
                                         "forget_local_epoch_ratio*local_epoch is the number of rounds of local training when we need to get the convergence direction of each local model")
    args = parser.parse_args()
    return args


def Federated_Unlearning():
    """Step 1.Set the parameters for Federated Unlearning"""
    FL_params = args_parser()
    torch.manual_seed(FL_params.seed)
    # kwargs for data loader
    print(60 * '=')
    print("Step1. Federated Learning Settings \n We use dataset: " + FL_params.data_name + (
        " for our Federated Unlearning experiment.\n"))

    """Step 2. construct the necessary user private data set required for federated learning, as well as a common test set"""
    print(60 * '=')
    print("Step2. Client data loaded, testing data loaded!!!\n       Initial Model loaded!!!")
    # 加载数据
    init_global_model = model_init(FL_params.data_name)
    client_all_loaders, test_loader = data_init(FL_params)

    selected_clients = np.random.choice(range(FL_params.N_total_client), size=FL_params.N_client, replace=False)
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
    print(60 * '=')
    print("Step3. Fedearated Learning and Unlearning Training...")
    # FedAvg, FedEraser, FedAccum,
    old_GMs, unlearn_GMs, uncali_unlearn_GMs, _ = federated_learning_unlearning(init_global_model,
                                                                                client_loaders,
                                                                                test_loader,
                                                                                FL_params)

    if (FL_params.if_retrain == True):
        t1 = time.time()
        # FedRetrain
        retrain_GMs = FL_Retrain(init_global_model, client_loaders, test_loader, FL_params)
        t2 = time.time()
        print("Time using = {} seconds".format(round(t2 - t1), 3))

    # Evaluation
    fedavg_test_loss, fedavg_test_acc = test(old_GMs[-1], test_loader)
    federaser_test_loss, federaser_test_acc = test(unlearn_GMs[-1], test_loader)
    fedaccum_test_loss, fedaccum_test_acc = test(uncali_unlearn_GMs[-1], test_loader)
    fedretrain_test_loss, fedretrain_test_acc = test(retrain_GMs[-1], test_loader)

    target_loader = client_loaders[FL_params.forget_client_idx]
    fedavg_target_loss, fedavg_target_acc = test(old_GMs[-1], target_loader)
    federaser_target_loss, federaser_target_acc = test(unlearn_GMs[-1], target_loader)
    fedaccum_target_loss, fedaccum_target_acc = test(uncali_unlearn_GMs[-1], target_loader)
    fedretrain_target_loss, fedretrain_target_acc = test(retrain_GMs[-1], target_loader)

    print(5 * "*" + "  Result Summary  " + 5 * "*")
    print("[FedEraser] Test set: Average loss = {:.8f}, Average acc = {:.4f}".format(federaser_test_loss,
                                                                                     federaser_test_acc))
    print("[FedAccum] Test set: Average loss = {:.8f}, Average acc = {:.4f}".format(fedaccum_test_loss,
                                                                                    fedaccum_test_acc))
    print("[FedRetrain] Test set: Average loss = {:.8f}, Average acc = {:.4f}".format(fedretrain_test_loss,
                                                                                      fedretrain_test_acc))
    print("[FedAvg] Test set: Average loss = {:.8f}, Average acc = {:.4f}".format(fedavg_test_loss, fedavg_test_acc))
    print("\n")
    print("[FedEraser] Target set: Average loss = {:.8f}, Average acc = {:.4f}".format(federaser_target_loss,
                                                                                       federaser_target_acc))
    print("[FedAccum] Target set: Average loss = {:.8f}, Average acc = {:.4f}".format(fedaccum_target_loss,
                                                                                      fedaccum_target_acc))
    print("[FedRetrain] Target set: Average loss = {:.8f}, Average acc = {:.4f}".format(fedretrain_target_loss,
                                                                                        fedretrain_target_acc))
    print("[FedAvg] Target set: Average loss = {:.8f}, Average acc = {:.4f}".format(fedavg_target_loss,
                                                                                    fedavg_target_acc))

    """Step 4  The member inference attack model is built based on the output of the Target Global Model on client_loaders and test_loaders.In this case, we only do the MIA attack on the model at the end of the training"""

    """MIA:Based on the output of oldGM model, MIA attack model was built, and then the attack model was used to attack unlearn GM. If the attack accuracy significantly decreased, it indicated that our unlearn method was indeed effective to remove the user's information"""
    print(60 * '=')
    print("Step4. Membership Inference Attack aganist GM...")

    T_epoch = -1
    # MIA setting:Target model == Shadow Model
    old_GM = old_GMs[T_epoch]
    attack_model = train_attack_model(old_GM, client_loaders, test_loader, FL_params)

    print("\nEpoch  = {}".format(T_epoch))
    print("Attacking against FL Standard  ")
    target_model = old_GMs[T_epoch]
    (PRE_old, REC_old, F1_old) = attack(target_model, attack_model, client_loaders, test_loader, FL_params)

    if (FL_params.if_retrain == True):
        print("Attacking against FL Retrain  ")
        target_model = retrain_GMs[T_epoch]
        (PRE_retrain, REC_retrain, F1_retrain) = attack(target_model, attack_model, client_loaders, test_loader,
                                                        FL_params)

    print("Attacking against FL Unlearn  ")
    target_model = unlearn_GMs[T_epoch]
    (PRE_unlearn, REC_unlearn, F1_unlearn) = attack(target_model, attack_model, client_loaders, test_loader, FL_params)

    print("Attacking against FL Unlearn without calibration  ")
    target_model = uncali_unlearn_GMs[T_epoch]
    (PRE_uncali_unlearn, REC_uncali_unlearn, F1_uncali_unlearn) = attack(target_model, attack_model, client_loaders,
                                                                         test_loader, FL_params)


if __name__ == '__main__':
    Federated_Unlearning()
