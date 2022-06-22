# -*- coding: utf-8 -*-
"""
learning aided gradient descent for MISO beamforming
"""
import numpy as np
from numpy import random
import torch
from timeit import default_timer as timer
import math
import copy
from copy import deepcopy
import time
import matplotlib.pyplot as plt
import os
from torch.utils.tensorboard import SummaryWriter

fold = 'D:\WMMSE\TB_online'
folder1 = 'Meta_WSR'
folder2 = 'WMMSE_WSR'
folder3 = 'Meta/WMMSE'
# folder4 = 'Ave_Meta/WMMSE'
# folder5 = 'Line_1'
# folder6 = 'Ave_WMMSE_WSR'
# folder7 = 'Ave_Meta_WSR'
folder8 = 'num_meta_iterations'
# folder9 = 'Ave_meta_iterations'
folder10 = 'num_WMMSE_iterations'
# folder11 = 'Ave_WMMSE_iterations'
writer1 = SummaryWriter(log_dir=os.path.join(fold, folder1), flush_secs=20)
writer2 = SummaryWriter(log_dir=os.path.join(fold, folder2), flush_secs=20)
writer3 = SummaryWriter(log_dir=os.path.join(fold, folder3), flush_secs=20)
# writer4 = SummaryWriter(log_dir=os.path.join(fold,folder4), flush_secs=20)
# writer5 = SummaryWriter(log_dir=os.path.join(fold,folder5), flush_secs=20)
# writer6 = SummaryWriter(log_dir=os.path.join(fold,folder6), flush_secs=20)
# writer7 = SummaryWriter(log_dir=os.path.join(fold,folder7), flush_secs=20)
writer8 = SummaryWriter(log_dir=os.path.join(fold, folder8), flush_secs=20)
# writer9 = SummaryWriter(log_dir=os.path.join(fold,folder9), flush_secs=20)
writer10 = SummaryWriter(log_dir=os.path.join(fold, folder10), flush_secs=20)
# writer11 = SummaryWriter(log_dir=os.path.join(fold,folder11), flush_secs=20)


USE_CUDA = False
if torch.cuda.is_available():
    USE_CUDA = True
    print("GPU_USE : ", USE_CUDA)
gpu_count = torch.cuda.device_count()
print("GPU_Count : ",gpu_count)


# Set variables
noise_power = 0.1
SNR = 30
nr_of_BS_antennas = 4
nr_of_users = 4

signal_power = noise_power*10**(SNR/10)
total_power = noise_power + signal_power  # power constraint in the weighted sum rate maximization problem - eq. (3) in our paper
lr_rate_v= 1 *10e-3

WMMSE_Meta = 0 #Meta for 0 or WMMSE_Meta for 1
WMMSE_Meta_iters = 0
External_iteration = 500
Update_steps=1
Internal_iteration = 1
Epo=1


judge_flag = 1*10e-6 #2*10e-5
# mu_fix=0.04

optimizer_lr_V = lr_rate_v

hidden_size_V=20
layer=2
epoch=1
nr_of_training = 100 # used for training
nr_of_testing = 0 # used for testing


test_rate=0.9
# noise=1
def Adam():
    return torch.optim.Adam()



selected_users = [0,1,2,3,4,5,6,7] # array of scheduled users. Note that we schedule all the users.
#scheduled_users = [0,1,2,3,4,5,6,7]
scheduled_users = selected_users
epsilon = 0.0001 # used to end the iterations of the WMMSE algorithm in Shi et al. when the number of iterations is not fixed (note that the stopping criterion has precendence over the fixed number of iterations)
power_tolerance = 0.0001 # used to end the bisection search in the WMMSE algorithm in Shi et al.
nr_of_iterations = 10 # for WMMSE algorithm in Shi et al.


M_1=torch.eye(nr_of_users)
M_2=torch.zeros(nr_of_users,nr_of_users)
M_Re=torch.cat((M_1,M_2),dim=0)
M_1=torch.eye(nr_of_users)
M_2=torch.zeros(nr_of_users,nr_of_users)
M_Im=torch.cat((M_2,M_1),dim=0)
user_weights = np.ones(nr_of_users)
user_weights_for_regular_WMMSE = np.ones(nr_of_users)

# run WMMSE algorithm in Shi et al.
def run_WMMSE(transmitter_precoder, epsilon, channel, selected_users, total_power, noise_power, user_weights, max_nr_of_iterations,
              log=True):
    channel=channel.numpy()
    nr_of_users = np.size(channel, 0)
    nr_of_BS_antennas = np.size(channel, 1)
    WSR = []  # to check if the WSR (our cost function) increases at each iteration of the WMMSE
    break_condition = epsilon + 1  # break condition to stop the WMMSE iterations and exit the while
    receiver_precoder = np.zeros(nr_of_users) + 1j * np.zeros(
        nr_of_users)  # receiver_precoder is "u" in the paper of Shi et al. (it's a an array of complex scalars)
    mse_weights = np.ones(
        nr_of_users)  # mse_weights is "w" in the paper of Shi et al. (it's a an array of real scalars)
    # transmitter_precoder = np.zeros((nr_of_users, nr_of_BS_antennas)) + 1j * np.zeros((nr_of_users,
    #                                                                                    nr_of_BS_antennas))  # transmitter_precoder is "v" in the paper of Shi et al. (it's a complex matrix)

    new_receiver_precoder = np.zeros(nr_of_users) + 1j * np.zeros(nr_of_users)  # for the first iteration
    new_mse_weights = np.zeros(nr_of_users)  # for the first iteration
    new_transmitter_precoder = np.zeros((nr_of_users, nr_of_BS_antennas)) + 1j * np.zeros(
        (nr_of_users, nr_of_BS_antennas))  # for the first iteration

    # Initialization of transmitter precoder (V)
    # for user_index in range(nr_of_users):
    #     if user_index in selected_users:
    #         transmitter_precoder[user_index, :] = channel[user_index, :]
    # transmitter_precoder = transmitter_precoder / np.linalg.norm(transmitter_precoder) * np.sqrt(total_power)

    # Store the WSR obtained with the initialized trasmitter precoder
    WSR.append(compute_weighted_sum_rate_WMMSE(user_weights, channel, transmitter_precoder, noise_power, selected_users))

    # Compute the initial power of the transmitter precoder
    initial_power = 0
    for user_index in range(nr_of_users):
        if user_index in selected_users:
            initial_power = initial_power + (compute_norm_of_complex_array(transmitter_precoder[user_index, :])) ** 2
    if log == True:
        print("Power of the initialized transmitter precoder:", initial_power)

    nr_of_iteration_counter = 0  # to keep track of the number of iteration of the WMMSE

    while break_condition >= epsilon and nr_of_iteration_counter <= max_nr_of_iterations:

        nr_of_iteration_counter = nr_of_iteration_counter + 1
        if log == True:
            print("WMMSE ITERATION: ", nr_of_iteration_counter)

        # Optimize receiver precoder(u) - eq. (5) in the paper of Shi et al.
        for user_index_1 in range(nr_of_users):
            if user_index_1 in selected_users:
                user_interference = 0.0
                for user_index_2 in range(nr_of_users):
                    if user_index_2 in selected_users:
                        user_interference = user_interference + (np.absolute(
                            np.matmul(np.conj(channel[user_index_1, :]), transmitter_precoder[user_index_2, :]))) ** 2

                new_receiver_precoder[user_index_1] = np.matmul(np.conj(channel[user_index_1, :]),
                                                                transmitter_precoder[user_index_1, :]) / (
                                                                  noise_power + user_interference)

        # Optimize mse_weights(w)- eq. (13) in the paper of Shi et al.
        for user_index_1 in range(nr_of_users):
            if user_index_1 in selected_users:

                user_interference = 0  # it includes the channel of all selected users
                inter_user_interference = 0  # it includes the channel of all selected users apart from the current one

                for user_index_2 in range(nr_of_users):
                    if user_index_2 in selected_users:
                        user_interference = user_interference + (np.absolute(
                            np.matmul(np.conj(channel[user_index_1, :]), transmitter_precoder[user_index_2, :]))) ** 2
                for user_index_2 in range(nr_of_users):
                    if user_index_2 != user_index_1 and user_index_2 in selected_users:
                        inter_user_interference = inter_user_interference + (np.absolute(
                            np.matmul(np.conj(channel[user_index_1, :]), transmitter_precoder[user_index_2, :]))) ** 2

                new_mse_weights[user_index_1] = (noise_power + user_interference) / (
                            noise_power + inter_user_interference)

        A = np.zeros((nr_of_BS_antennas, nr_of_BS_antennas)) + 1j * np.zeros((nr_of_BS_antennas, nr_of_BS_antennas))
        for user_index in range(nr_of_users):
            if user_index in selected_users:
                # hh should be an hermitian matrix of size (nr_of_BS_antennas X nr_of_BS_antennas)
                hh = np.matmul(np.reshape(channel[user_index, :], (nr_of_BS_antennas, 1)),
                               np.conj(np.transpose(np.reshape(channel[user_index, :], (nr_of_BS_antennas, 1)))))
                A = A + (new_mse_weights[user_index] * user_weights[user_index] * (np.absolute(new_receiver_precoder[user_index])) ** 2) * hh

        Sigma_diag_elements_true, U = np.linalg.eigh(A)
        Sigma_diag_elements = copy.deepcopy(np.real(Sigma_diag_elements_true))
        Lambda = np.zeros((nr_of_BS_antennas, nr_of_BS_antennas)) + 1j * np.zeros(
            (nr_of_BS_antennas, nr_of_BS_antennas))

        for user_index in range(nr_of_users):
            if user_index in selected_users:
                hh = np.matmul(np.reshape(channel[user_index, :], (nr_of_BS_antennas, 1)),
                               np.conj(np.transpose(np.reshape(channel[user_index, :], (nr_of_BS_antennas, 1)))))
                Lambda = Lambda + ((user_weights[user_index]) ** 2) * ((new_mse_weights[user_index]) ** 2) * (
                            (np.absolute(new_receiver_precoder[user_index])) ** 2) * hh

        Phi = np.matmul(np.matmul(np.conj(np.transpose(U)), Lambda), U)
        Phi_diag_elements_true = np.diag(Phi)
        Phi_diag_elements = copy.deepcopy(Phi_diag_elements_true)
        Phi_diag_elements = np.real(Phi_diag_elements)

        for i in range(len(Phi_diag_elements)):
            if Phi_diag_elements[i] < np.finfo(float).eps:
                Phi_diag_elements[i] = np.finfo(float).eps
            if (Sigma_diag_elements[i]) < np.finfo(float).eps:
                Sigma_diag_elements[i] = 0

        # Check if mu = 0 is a solution (eq.s (15) and (16) of in the paper of Shi et al.)(mu is the Lagrange multiplier)
        power = 0  # the power of transmitter precoder (i.e. sum of the squared norm)
        for user_index in range(nr_of_users):
            if user_index in selected_users:
                if np.linalg.det(A) != 0:
                    w = np.matmul(np.linalg.inv(A), np.reshape(channel[user_index, :], (nr_of_BS_antennas, 1))) * \
                        user_weights[user_index] * new_mse_weights[user_index] * (new_receiver_precoder[user_index])
                    power = power + (compute_norm_of_complex_array(w)) ** 2

        # If mu = 0 is a solution, then mu_star = 0
        if np.linalg.det(A) != 0 and power <= total_power:
            mu_star = 0
        # If mu = 0 is not a solution then we search for the "optimal" mu by bisection
        else:
            power_distance = []  # list to store the distance from total_power in the bisection algorithm
            mu_low = np.sqrt(1 / total_power * np.sum(Phi_diag_elements))
            mu_high = 0
            # low_point = compute_P(Phi_diag_elements, Sigma_diag_elements, mu_low)
            # high_point = compute_P(Phi_diag_elements, Sigma_diag_elements, mu_high)

            obtained_power = total_power + 2 * power_tolerance  # initialization of the obtained power such that we enter the while

            # Bisection search
            while np.absolute(total_power - obtained_power) > power_tolerance:
                mu_new = (mu_high + mu_low) / 2
                obtained_power = compute_P(Phi_diag_elements, Sigma_diag_elements,
                                           mu_new)  # eq. (18) in the paper of Shi et al.
                power_distance.append(np.absolute(total_power - obtained_power))
                if obtained_power > total_power:
                    mu_high = mu_new
                if obtained_power < total_power:
                    mu_low = mu_new
            mu_star = mu_new
            if log == True:
                print("first value:", power_distance[0])
                # plt.title("Distance from the target value in bisection (it should decrease)")
                # plt.plot(power_distance)
                # plt.show()
                #Equation (5c)
        for user_index in range(nr_of_users):
            if user_index in selected_users:
                new_transmitter_precoder[user_index, :] = np.matmul(
                    np.linalg.inv(A + mu_star * np.eye(nr_of_BS_antennas)), channel[user_index, :]) * user_weights[
                                                              user_index] * new_mse_weights[user_index] * (
                                                          new_receiver_precoder[user_index])

                # To select only the weights of the selected users to check the break condition
        mse_weights_selected_users = []
        new_mse_weights_selected_users = []
        for user_index in range(nr_of_users):
            if user_index in selected_users:
                mse_weights_selected_users.append(mse_weights[user_index])
                new_mse_weights_selected_users.append(new_mse_weights[user_index])

        mse_weights = deepcopy(new_mse_weights)
        transmitter_precoder = deepcopy(new_transmitter_precoder)
        receiver_precoder = deepcopy(new_receiver_precoder)

        WSR.append(compute_weighted_sum_rate_WMMSE(user_weights, channel, transmitter_precoder, noise_power, selected_users))

        break_condition = np.absolute(
            np.log2(np.prod(new_mse_weights_selected_users)) - np.log2(np.prod(mse_weights_selected_users)))

    if log == True:
        plt.title("Change of the WSR at each iteration of the WMMSE (it should increase)")
        plt.plot(WSR, 'bo')
        plt.show()
    # print(WSR)
    # print(WSR[-1])
    return transmitter_precoder, receiver_precoder, mse_weights, WSR[-1] ,nr_of_iteration_counter


#Compute WSR
def compute_weighted_sum_rate_WMMSE(user_weights, channel, precoder, noise_power, selected_users):
    result = 0
    nr_of_users = np.size(channel, 0)

    for user_index in range(nr_of_users):
        if user_index in selected_users:
            user_sinr = compute_sinr(channel, precoder, noise_power, user_index, selected_users)
            result = result + user_weights[user_index] * np.log2(1 + user_sinr)

    return result

# Randomly initialize the channel
def compute_channel(nr_of_BS_antennas, nr_of_users, total_power):
    channel_WMMSE = np.zeros((nr_of_users, nr_of_BS_antennas)) + 1j * np.zeros((nr_of_users, nr_of_BS_antennas))
    for i in range(nr_of_users):
        result_real = np.sqrt(0.5) * np.random.normal(size=(nr_of_BS_antennas, 1))
        result_imag = np.sqrt(0.5) * np.random.normal(size=(nr_of_BS_antennas, 1))
        channel_WMMSE[i, :] = np.reshape(result_real, (1, nr_of_BS_antennas)) + 1j * np.reshape(result_imag,(1, nr_of_BS_antennas))
    return channel_WMMSE

#Compute SINR
def compute_sinr(channel, precoder, noise_power, user_id, selected_users):
    nr_of_users = np.size(channel, 0)
    numerator = (np.absolute(np.matmul(np.conj(channel[user_id, :]), precoder[user_id, :]))) ** 2

    inter_user_interference = 0
    for user_index in range(nr_of_users):
        if user_index != user_id and user_index in selected_users:
            inter_user_interference = inter_user_interference + (
                np.absolute(np.matmul(np.conj(channel[user_id, :]), precoder[user_index, :]))) ** 2
    denominator = noise_power + inter_user_interference

    result = numerator / denominator
    return result

#Compute WSR
def compute_weighted_sum_rate(user_weights, channel, precoder_in, noise_power, selected_users):
    result = 0
    nr_of_users = np.size(channel, 0)
    precoder=precoder_in.detach()
    #transmitter_precoder=precoder.mm(M_Re)+1j*precoder.mm(M_Im)

    transmitter_precoder = precoder[:, 0:nr_of_BS_antennas]+1j*precoder[:, nr_of_BS_antennas:nr_of_BS_antennas*2]

    transmitter_precoder=transmitter_precoder.detach().numpy()
    channel=channel.numpy()
    for user_index in range(nr_of_users):
        if user_index in selected_users:
            user_sinr = compute_sinr(channel, transmitter_precoder, noise_power, user_index, selected_users)
            result = result + user_weights[user_index] * np.log2(1 + user_sinr)
            # user_sinr = compute_sinr(channel, receiver_precoder, noise_power, user_index, selected_users)
            # result = result + user_weights[user_index] * np.log2(1 + user_sinr)
    return result

#Compute SINR
def compute_sinr_V(channel, transmitter_precoder_in, noise_power, user_id, selected_users):
    #print('######################')
    #print(channel)
    nr_of_users = np.size(channel, 0)
    h_i=torch.conj(channel[user_id, :])
    h_re=h_i.real.float()
    h_re=h_re.reshape(1,nr_of_BS_antennas)
    h_im=h_i.imag.float()
    h_im=h_im.reshape(1,nr_of_BS_antennas)
    h_i=h_i.reshape(1,nr_of_BS_antennas)
    # V_re=transmitter_precoder_in.mm(M_Re)
    # V_im=transmitter_precoder_in.mm(M_Im)
    V_re=transmitter_precoder_in[:,0:nr_of_BS_antennas]
    V_im = transmitter_precoder_in[:, nr_of_BS_antennas:nr_of_BS_antennas*2]
    # print('###################')
    # print(transmitter_precoder_in)
    # print('###################')
    # print(V_re)
    # print('###################')
    # print(V_im)
    v_re1=V_re[user_id,:]
    v_im1=V_im[user_id,:]
    v_re1=v_re1.reshape(nr_of_BS_antennas,1)
    v_im1=v_im1.reshape(nr_of_BS_antennas,1)
    hv_re=h_re.mm(v_re1)-h_im.mm(v_im1)
    hv_im=h_re.mm(v_im1)+h_im.mm(v_re1)
    hv_i=hv_re.pow(2)+hv_im.pow(2)
    numerator = hv_i
    inter_user_interference = 0
    for user_index in range(nr_of_users):
        if user_index != user_id and user_index in selected_users:
            v_re2=V_re[user_index,:]
            v_im2=V_im[user_index,:]
            v_re2=v_re2.reshape(nr_of_BS_antennas,1)
            v_im2=v_im2.reshape(nr_of_BS_antennas,1)
            hvj_re=h_re.mm(v_re2)-h_im.mm(v_im2)
            hvj_im=h_im.mm(v_re2)+h_re.mm(v_im2)
            hv_j=hvj_re.pow(2)+hvj_im.pow(2)       
            #inter_user_interference = inter_user_interference + hv_j+noise_power
            inter_user_interference = inter_user_interference + hv_j
    denominator = noise_power + inter_user_interference
    result = numerator / denominator
    return result

#Compute loss
def Compute_Loss_V(user_weights, channel, precoder_in, noise_power, selected_users):
    result = 0
    nr_of_users = np.size(channel, 0)
    transmitter_precoder=precoder_in
    # print('################################')
    # print(transmitter_precoder)
    # print(M_Re)
    # print(M_Im)

    # V_re=transmitter_precoder.mm(M_Re)
    # V_im=transmitter_precoder.mm(M_Im)
    # Trace_V=torch.trace(V_re.t().mm(V_re)+V_im.t().mm(V_im))
    Trace_V = torch.trace(transmitter_precoder.t().mm(transmitter_precoder))


    # receiver_precoder=precoder.mm(M_Re)+1j*precoder.mm(M_Im)
    # receiver_precoder=receiver_precoder.t()
    # receiver_precoder=receiver_precoder
    channel=channel
    for user_index in range(nr_of_users):
        if user_index in selected_users:
            user_sinr = compute_sinr_V(channel, transmitter_precoder, noise_power, user_index, selected_users)
            result = result + user_weights[user_index] * torch.log2(1 + user_sinr)
    #result=result+mu*(total_power-Trace_V)
    #result = result + mu * (total_power - Trace_V) - mu * (total_power - Trace_V)**2
    # if total_power < Trace_V:
    #     result = result - 0.001 * (Trace_V - total_power)
    return -result


def compute_norm_of_complex_array(x):
    result = np.sqrt(np.sum((np.absolute(x)) ** 2))
    return result

#Compute power
def compute_P(Phi_diag_elements, Sigma_diag_elements, mu):
    nr_of_BS_antennas = Phi_diag_elements.size
    mu_array = mu * np.ones(Phi_diag_elements.size)
    result = np.divide(Phi_diag_elements, (Sigma_diag_elements + mu_array) ** 2)
    result = np.sum(result)
    return result

# Initialize the transmitter precoder
def initia_transmitter_precoder(channel_realization):
    # channel_realization = np.zeros((nr_of_users, nr_of_BS_antennas)) + 1j * np.zeros((nr_of_users, nr_of_BS_antennas))
    # for i in range(nr_of_users):
    #     result_real = np.sqrt(0.5) * np.random.normal(size=(nr_of_BS_antennas, 1))
    #     result_imag = np.sqrt(0.5) * np.random.normal(size=(nr_of_BS_antennas, 1))
    #     channel_realization[i, :] = np.reshape(result_real, (1, nr_of_BS_antennas)) + 1j * np.reshape(result_imag,(1, nr_of_BS_antennas))
    transmitter_precoder_init = np.zeros((nr_of_users, nr_of_BS_antennas)) + 1j * np.zeros((nr_of_users,nr_of_BS_antennas)) 
    for user_index in range(nr_of_users):
        if user_index in selected_users:
            transmitter_precoder_init[user_index, :] = channel_realization[user_index, :]
    transmitter_precoder_initialize = transmitter_precoder_init / np.linalg.norm(transmitter_precoder_init) * np.sqrt(total_power)
    
    transmitter_precoder_init=torch.from_numpy(transmitter_precoder_initialize)
    transmitter_precoder_complex=transmitter_precoder_init
    transmitter_precoder_Re=transmitter_precoder_complex.real
    transmitter_precoder_Im=transmitter_precoder_complex.imag
    transmitter_precoder=torch.cat((transmitter_precoder_Re,transmitter_precoder_Im),dim=1)
    return transmitter_precoder, transmitter_precoder_initialize


#Dimensions of variables
DIM_u1=nr_of_users*2
DIM_u2=1
DIM_w1=nr_of_users
DIM_w2=1
DIM_V1=nr_of_BS_antennas*2
DIM_V2=nr_of_users


#network settings
input_size_u=DIM_u1
output_size_u=DIM_u1
batchsize_u=DIM_u2

input_size_w=DIM_w2
output_size_w=DIM_w2
batchsize_w=DIM_w1

input_size_V=DIM_V1
output_size_V=DIM_V1
batchsize_V=DIM_V2


#Build LAGD netowrk
class LAGD_Optimizee_V(torch.nn.Module):
    def __init__(self ):
        super(LAGD_Optimizee_V,self).__init__()
        self.fnn1=torch.nn.Linear(input_size_V,hidden_size_V)
        # self.fnn2 = torch.nn.Linear(hidden_size_V, hidden_size_V)
        self.out=torch.nn.Linear(hidden_size_V,output_size_V)
    def forward(self,gradient, state):
            gradient=gradient.unsqueeze(0)
            if state is None:
                state=(torch.zeros(layer,batchsize_V,hidden_size_V),
                torch.zeros(layer,batchsize_w,hidden_size_V))
            if USE_CUDA:
                state = (torch.zeros(layer, batchsize_V, hidden_size_V).cuda(),
                         torch.zeros(layer, batchsize_V, hidden_size_V).cuda())
            x = self.fnn1(gradient)
            # x = self.fnn2(x)
            update=self.out(x)
            update=update.squeeze(0)
            return update, state



optimizee_V=LAGD_Optimizee_V()
if USE_CUDA:
    optimizee_V = optimizee_V.cuda()
#adam_global_optimizer_V = torch.optim.Adam(optimizee_V.parameters(),lr = optimizer_lr_V)#update optimizee with adam


print(optimizee_V)
WSR_record=[]
WSR_WMMSE_record=[]
Loss_record=[]
meta_list=[]
WMMSE_list=[]
epoch_record_loss=0

Meta_temp=np.zeros(External_iteration + 1)
Meta_D_WMMSE=0
sum_M_D_W = 0
sum_WSR_WMMSE = 0
sum_WSR_Meta = 0
sum_meta_iterations = 0
sum_WMMSE_iterations = 0

max_WMMSE = 0
min_WMMSE = 50

max_meta = 0
min_meta = 50

#Start the training/testing process
for batch_step in range(epoch):
    epoch_record_meta=0
    epoch_record_WMMSE=0
    WSR_list_per_sample=torch.zeros(nr_of_training,External_iteration)
    WMMSE_list_per_sample=torch.zeros(nr_of_training,External_iteration)
    Loss_v_list_per_sample=torch.zeros(nr_of_training,External_iteration)
    Loss_w_list_per_sample=torch.zeros(nr_of_training,External_iteration)

    for ex_step in range(nr_of_training):
        optimizee_V = LAGD_Optimizee_V()
        if USE_CUDA:
            optimizee_V = optimizee_V.cuda()
        adam_global_optimizer_V = torch.optim.Adam(optimizee_V.parameters(),
                                                   lr=optimizer_lr_V)  # update optimizee with adam

        # initialization
        # mu_fix=0
        # epoch_record_meta=0
        # epoch_record_WMMSE=0
        channel_realization = compute_channel(nr_of_BS_antennas, nr_of_users, total_power)
        channel_realization = torch.from_numpy(channel_realization)
        channel = channel_realization
        # norm_channel=torch.norm(abs(channel))
        transmitter_precoder_init, transmitter_precoder_initialize=initia_transmitter_precoder(channel_realization)
        transmitter_precoder=transmitter_precoder_init
        mse_weight=torch.rand(DIM_w1,DIM_w2)
        mse_weight_init=mse_weight
        receiver_precoder1=torch.rand(DIM_u2,nr_of_users, dtype=torch.float)
        receiver_precoder2=torch.zeros(DIM_u2,nr_of_users, dtype=torch.float)
        receiver_precoder=torch.cat((receiver_precoder1,receiver_precoder2),dim=1)
        receiver_precoder_init=receiver_precoder
        #mu=compute_mu(channel,mse_weight,receiver_precoder)
        # mu=0.1
        # WSR=compute_weighted_sum_rate(user_weights, channel, transmitter_precoder, noise_power, selected_users)
        # print(mu)
        mu=0
        LossAccumulated_u=0
        LossAccumulated_w=0
        LossAccumulated_V=0
        #WSR_external=0
        print('\n=======> Epoch', batch_step+1, 'Training sample: {}'.format(ex_step+1))


        # optimizee_V = LSTM_Optimizee_V()
        # if USE_CUDA:
        #     optimizee_V = optimizee_V.cuda()
        # adam_global_optimizer_V = torch.optim.Adam(optimizee_V.parameters(),lr=optimizer_lr_V)  # update optimizee with adam

        learning_rate_flag = 0
        run_WMMSE_flag = 0
        start = timer()
        WSR_max=0
        #print('before step:',transmitter_precoder_init[0][0])
        for in_step in range(External_iteration):

            transmitter_precoder_init, transmitter_precoder_initialize = initia_transmitter_precoder(
                channel_realization)

            if run_WMMSE_flag == 0:
                V_WMMSE, u_WMMSE, w_WMMSE, WSR_WMMSE_one_sample, WMMSE_step = run_WMMSE(
                        transmitter_precoder_initialize,
                        epsilon,
                        channel_realization,
                        scheduled_users,
                        total_power, noise_power,
                        user_weights_for_regular_WMMSE,
                        WMMSE_Meta_iters - 1, log=False)
                V_WMMSE_plus_Init = V_WMMSE
                V_WMMSE_plus_Re = torch.tensor(V_WMMSE_plus_Init.real)
                V_WMMSE_plus_Im = torch.tensor(V_WMMSE_plus_Init.imag)
                V_WMMSE_plus = torch.cat((V_WMMSE_plus_Re, V_WMMSE_plus_Im), dim=1)
                V_WMMSE_plus = torch.Tensor(V_WMMSE_plus.float())
                transmitter_precoder_init_WMMSE = V_WMMSE_plus

                V_WMMSE, u_WMMSE, w_WMMSE, WSR_WMMSE_one_sample,WMMSE_step = run_WMMSE(transmitter_precoder_initialize, epsilon,
                                                                            channel_realization, scheduled_users,
                                                                            total_power, noise_power,
                                                                            user_weights_for_regular_WMMSE,
                                                                            nr_of_iterations - 1, log=False)
                V_WMMSE_plus_Init = V_WMMSE


                V_WMMSE_plus_Re = torch.tensor(V_WMMSE_plus_Init.real)
                V_WMMSE_plus_Im = torch.tensor(V_WMMSE_plus_Init.imag)
                V_WMMSE_plus = torch.cat((V_WMMSE_plus_Re, V_WMMSE_plus_Im), dim=1)

                V_WMMSE_plus = torch.Tensor(V_WMMSE_plus.float())
                Trace_V_WMMSE = torch.trace(V_WMMSE_plus.t().mm(V_WMMSE_plus))
                Tr_WMMSE = int(Trace_V_WMMSE.detach().numpy())
                run_WMMSE_flag = 1

            if WMMSE_Meta == 1:
                transmitter_precoder_init = transmitter_precoder_init_WMMSE

            transmitter_precoder_internal = transmitter_precoder_init
            transmitter_precoder_internal = transmitter_precoder_internal.float()

            state = None
            transmitter_precoder_internal.requires_grad = True

            L = Compute_Loss_V(user_weights, channel, transmitter_precoder_internal, noise_power, selected_users)
            L.backward(retain_graph=True)

            if USE_CUDA:
                transmitter_precoder_update, state = optimizee_V(
                    transmitter_precoder_internal.grad.clone().detach().cuda(), state=None)
            else:
                transmitter_precoder_update, state = optimizee_V(transmitter_precoder_internal.grad.clone().detach(),
                                                               state=None)
            transmitter_precoder_update = transmitter_precoder_update.cpu()

            transmitter_precoder = transmitter_precoder_internal + transmitter_precoder_update

            Trace_V = torch.trace(transmitter_precoder.t().mm(transmitter_precoder))
            Tr = int(Trace_V.detach().numpy())
            normV = torch.norm(transmitter_precoder)
            WW = math.sqrt(total_power) / (normV)
            if Tr > total_power:
                transmitter_precoder = transmitter_precoder * WW

            Trace_V = torch.trace(transmitter_precoder.t().mm(transmitter_precoder))
            Tr = int(Trace_V.detach().numpy())

            loss_V=Compute_Loss_V(user_weights, channel, transmitter_precoder, noise_power, selected_users)
            LossAccumulated_V = LossAccumulated_V+loss_V

            WSR=compute_weighted_sum_rate(user_weights, channel, transmitter_precoder, noise_power, selected_users)


            if WSR > WSR_max:
                WSR_max = WSR

            Meta_temp[in_step + 1] = WSR

            if (in_step+1)% Update_steps == 0 and (ex_step+1) <= (nr_of_training - nr_of_testing):
                # if ex_step<(test_rate*nr_of_training):

                adam_global_optimizer_V.zero_grad()
                Average_loss_V=LossAccumulated_V/Update_steps
                Average_loss_V.backward(retain_graph=True)
                adam_global_optimizer_V.step()
                time = timer() - start
                WSR=compute_weighted_sum_rate(user_weights, channel, transmitter_precoder, noise_power, selected_users)
                if WSR > WSR_max:
                    WSR_max = WSR
                # Tr=int(Trace_V.detach().numpy())
                LossAccumulated_V=0
                #if (in_step + 1) % (Update_steps) == 0:
                if (in_step + 1) % (Update_steps*10) == 0:

                    print('->  step :' ,in_step+1,'WSR=','%.2f'%WSR,'WMMSE =','%.2f'%WSR_WMMSE_one_sample,'time=','%.0f'%time,'Trace',Tr,'Trace_WMMSE',Tr_WMMSE)

                
            if (in_step+1)==External_iteration:

                Trace_V = torch.trace(transmitter_precoder.t().mm(transmitter_precoder))
                Tr = int(Trace_V.detach().numpy())
                normV = torch.norm(transmitter_precoder)
                WW = math.sqrt(total_power) / (normV)
                if Tr > total_power:
                    transmitter_precoder = transmitter_precoder * WW

                Trace_V = torch.trace(transmitter_precoder.t().mm(transmitter_precoder))
                Tr = int(Trace_V.detach().numpy())
                WSR = compute_weighted_sum_rate(user_weights, channel, transmitter_precoder, noise_power,
                                                selected_users)

                print('->  step :' ,in_step+1,'WSR=','%.2f'%WSR,'WMMSE','%.2f'%WSR_WMMSE_one_sample,'time=','%.0f'%time,'Trace',Trace_V.detach().numpy())

                # print('->  step :' ,in_step+1,'WSR=','%.2f'%WSR_max,'WMMSE','%.2f'%WSR_WMMSE_one_sample,'time=','%.0f'%time,'Trace',Trace_V.detach().numpy())


        if WSR_WMMSE_one_sample > max_WMMSE :
            max_WMMSE = WSR_WMMSE_one_sample

        if WSR_WMMSE_one_sample < min_WMMSE :
            min_WMMSE = WSR_WMMSE_one_sample

        if WSR_max > max_meta :
            max_meta = WSR_max

        if WSR_max < min_meta :
            min_meta = WSR_max

        writer1.add_scalar('WSR/Random samples', WSR_max, ex_step)
        writer2.add_scalar('WSR/Random samples',WSR_WMMSE_one_sample, ex_step)
        writer3.add_scalar('Rate/Random samples', WSR_max/WSR_WMMSE_one_sample, ex_step)
        writer8.add_scalar('Num_iterations/Random samples', in_step, ex_step)
        writer10.add_scalar('Num_iterations/Random samples', WMMSE_step, ex_step)
        sum_M_D_W += WSR_max / WSR_WMMSE_one_sample
        sum_WSR_WMMSE += WSR_WMMSE_one_sample
        sum_WSR_Meta += WSR
        sum_meta_iterations += in_step
        sum_WMMSE_iterations += WMMSE_step
        if (ex_step+1)%10==0:
            print('epoch_mean_WSR',epoch_record_meta/ex_step)
            print('epoch_mean_WMMSE',epoch_record_WMMSE/ex_step)

        if (ex_step+1)==(nr_of_training-nr_of_testing):
            meta_list.append(epoch_record_meta/nr_of_training)
            print('epoch_mean_WSR',epoch_record_meta/nr_of_training)
            print('epoch_mean_WMMSE',epoch_record_WMMSE/nr_of_training)
            WMMSE_list.append(epoch_record_loss/nr_of_training)
            #print(optimizee_V.lstm.all_weights[0][0][0])

    nr_of_training = float(nr_of_training)
    #print(sum_WSR_Meta)
    ave_sum_M_D_W = sum_M_D_W / nr_of_training
    ave_sum_WSR_WMMSE = sum_WSR_WMMSE / nr_of_training
    ave_sum_WSR_Meta = float(sum_WSR_Meta / nr_of_training)
    ave_meta_iterations = sum_meta_iterations / nr_of_training
    ave_WMMSE_iterations = sum_WMMSE_iterations / nr_of_training

    print('->  Average! step :' ,in_step+1,'ave_sum_M_D_W=','%.2f'%ave_sum_M_D_W,'ave_sum_WSR_WMMSE=','%.2f'%ave_sum_WSR_WMMSE,
          'ave_sum_WSR_Meta=','%.2f'%ave_sum_WSR_Meta,'ave_meta_iterations=','%.0f'%ave_meta_iterations,'ave_WMMSE_iterations=','%.0f'%ave_WMMSE_iterations,'min_WMMSE=','%.2f'%min_WMMSE,'max_WMMSE=','%.2f'%max_WMMSE,'min_meta=','%.2f'%min_meta,'min_WMMSE=','%.2f'%max_meta)
