from dataServer import dataserver_thread
import numpy as np
import scipy.io as scio
from scipy.signal import resample
import time
from algorithmInterface import algorithmthread
from threading import Event
from multiprocessing import Process, Manager
from EyeDataServer import OnlineEyeDataRecver
import random
from algorithmInterface import algorithmthread
import json
import mne


if __name__ == '__main__':
    choice = 0
#-------脑电处理的模板导入-----------#
    # load data
    filepath = './data/'
    # 获得空间滤波器
    w = np.load(filepath+'W.npy')

    # 获得模板信号
    template = np.load(filepath+'Template.npy')

    # # 获得LDA分类器
    # data = scio.loadmat(filepath+'LDA_classifier.mat')
    # ldaW = data['LDA_classifier']

    flagstop = False
    n_chan = 9
    srate = 1000                      # 采样频率
    time_buffer = 10 # second          # 数据buffer
    epochlength = int(srate*0.64)     # 数据长度
    delay = int(srate*0.14)           # 延迟时间
    addloop = 1                       # 轮次
    savedata = np.zeros((40, 21, 500))
    count = 0

#---------眼动设置-----------------#
    eye_fs = 100
    packet_length = 0.04
    eye_datalength = int(eye_fs*0.5)
    eye_epochlength = int(eye_fs*0.5)

#--------------子线程--------------#

    presettingfile = open('Sti_merges.json')
    stim_position= json.load(presettingfile)

    dataRunner = algorithmthread(w, template, addloop, eye_datalength)

    datapath = 'E:/wyl/BE测试实验/0823/online.cnt'
    raw = mne.io.read_raw_cnt(datapath, eog=['HEO', 'VEO'], emg=['EMG'], ecg=['EKG'], preload=True, verbose=False)
    # raw.pick_channels(['P7', 'P5', 'P3', 'P1', 'PZ',
    #             'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ',
    #             'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2'])
    Numcorrect = 0
    num = 255
    mapping_ssvep = dict()
    for idx_command in range(1, num+1):
        mapping_ssvep[str(idx_command)] = idx_command
    events_ssvep, events_ids_ssvep = mne.events_from_annotations(raw, event_id=mapping_ssvep)
    event = events_ssvep[:, 0] + 1
    testdata, _ = raw[:]
    event_type = events_ssvep[:, 2]
    truelabel = np.mod(event_type-1, 6)
    Ntrials = event.shape[0]
    epoch_length = int(0.64*1000)
    delay = int(0.14*1000)
    N_correct = 0
    for trial_i in range(Ntrials):
        triggerpos = int(event[trial_i])
        cutdata = testdata[:, triggerpos + 1:]
        epochdata = cutdata[:, delay:epoch_length]
        # epochdata = resample(epochdata, 125, axis=1)
        dataRunner.recvData(epochdata, trial_i+1)
        dataRunner.run()
        if dataRunner.result == truelabel[trial_i]:
                Numcorrect += 1

    # print(Numcorrect/36)
    print(dataRunner.results)


