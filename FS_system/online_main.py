from dataServer import dataserver_thread
import numpy as np
from EyeDataServer import OnlineEyeDataRecver
from algorithmInterface import algorithmthread
from threading import Event
import time
import os
import atexit
import logging
import datetime

if __name__ == '__main__':
    if os.path.exists('log_tmp') == False:
        os.mkdir('log_tmp')
    time_now = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    user_name = 'xj'
    logging.basicConfig(level=logging.DEBUG,
                        format = '%(asctime)s.%(msecs)03d %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt = '%H:%M:%S',
                        filename = './log_tmp/{}_{}.log'.format(user_name, time_now),
                        filemode = 'w'
                        )
    logger_obj = logging.getLogger('Online_log')
    #-------脑电处理的模板导入-----------#
    # load data
    filepath = './data/'
    # 获得空间滤波器
    w = np.load(filepath+'W.npy')
    # 获得模板信号
    template = np.load(filepath+'Template.npy')
    P = np.load(filepath+'P.npy')

    #---------眼动设置-----------------#
    savedata = np.zeros((21, 500, 100))
    i = 0
    flagstop = False
    n_chan = 21
    srate = 1000                      # 采样频率
    brain_time_buffer = 10            # second         # 数据buffer
    epochlength = int(srate*0.44)     # 数据长度
    delay = int(srate*0.14)           # 延迟时间
    addloop = 1# 轮次
    trigger_flag = 0

    #---------眼动设置-----------------#
    eye_fs = 100
    packet_length = 0.04
    eye_time_buffer = 10 # 1
    eye_datalength = int(eye_fs*0.44)
    #--------------子线程--------------#
    #
    '''
    处理线程默认模式：
        mode-脑电分析算法：tdca（还可选’trca‘）
        control_mode-控制模式：eye（还可选brian（眼动分区再脑控）, brain_eye(脑眼并行控制））
    '''
    datalocker = Event()
    dataRunner = algorithmthread(w, template, datalocker,addloop, eye_datalength, P, Nfb=2, l=3, mode='tdca',control_mode = 'eye')
    dataRunner.Daemon = True
    dataRunner.start()

    '''眼动线程'''
    thread_eyedata_sever = OnlineEyeDataRecver(eye_fs,packet_length, time_buffer=eye_time_buffer)
    thread_eyedata_sever.daemon = True
    thread_eyedata_sever.wait_connect()
    thread_eyedata_sever.start()

    '''脑电线程'''
    thread_data_server = dataserver_thread(time_buffer=brain_time_buffer)
    thread_data_server.Daemon = True
    notconnect = thread_data_server.connect_tcp()
    if notconnect:
        raise TypeError("Can't connect recoder, Please open the hostport")
    else:
        thread_data_server.start_acq()
        thread_data_server.start()
        print('Data server connected')

    #--------------主线程--------------#
    B_label_flag = 0
    E_label_flag = 0
    T_flag = 0
    decide_time = []
    while not flagstop:
        if thread_data_server.stop_flag == 1:
            break
        data, eventline = thread_data_server.get_buffer_data()
        Eyedata, E_eventline = thread_eyedata_sever.get_buffer_data()
        B_triggerPos = np.nonzero(eventline)[0]
        E_triggerPos = np.nonzero(E_eventline)[0]
        # print('B_label:{}, E_lable:{}'.format(B_triggerPos, E_triggerPos))

        if B_triggerPos.shape[0] >= 1:
            B_currentTriggerPos = B_triggerPos[-1]
        if E_triggerPos.shape[0]>=1:
            E_currentTriggerPos = E_triggerPos[-1]

        if B_triggerPos.shape[0] >= 1 and E_triggerPos.shape[0]>=1:

            if T_flag == 0 and E_eventline[E_currentTriggerPos] != E_label_flag:
                t1 = time.perf_counter()
                T_flag = 1

            if data[:,B_currentTriggerPos+1:].shape[1]>=epochlength and Eyedata[:,E_currentTriggerPos+1:].shape[1]>=eye_datalength:
                # if eventline[B_currentTriggerPos] == E_eventline[E_currentTriggerPos]:
                    if eventline[B_currentTriggerPos] != B_label_flag and E_eventline[E_currentTriggerPos] != E_label_flag:        # 相邻的两个标签不同才决策
                        print('B_label:{}, E_lable:{}'.format(eventline[B_currentTriggerPos], E_eventline[E_currentTriggerPos]))
                        B_cutdata = data[:, B_currentTriggerPos+1:]
                        B_epochdata = B_cutdata[:,delay:epochlength]
                        E_cutdata = Eyedata[:, E_currentTriggerPos+1:]
                        E_epochdata = E_cutdata[:,0:eye_datalength]
                        # savedata[:, :, i] = epochdata
                        # i += 1
                        if datalocker.is_set() == True:
                                datalocker.clear()
                        '''脑电'''
                        dataRunner.recvData(B_epochdata,eventline[B_currentTriggerPos])
                        '''眼动'''
                        dataRunner.RecvEyeData(E_epochdata)
                        datalocker.wait()
                        t2 = time.perf_counter()
                        B_label_flag = eventline[B_currentTriggerPos]
                        E_label_flag = E_eventline[E_currentTriggerPos]

                        # 处理时间计时的
                        T = t2-t1
                        T_flag = 0
                        decide_time.append(T)
                        logger_obj.info('输出的指令:{}'.format(dataRunner.Send_command))
                        logger_obj.info('刺激开始时间: {}, 决策输出时间: {}, 指令输出耗时:{}ms'.format(t1, t2,np.mean(t2-t1)*1000))
                        print('决策时间:{0}'.format(T))

    print('平均决策时间:{0}'.format(np.mean(decide_time)))
    logger_obj.info('整个实验的平均指令输出耗时:{}ms'.format(np.mean(decide_time)*1000))
    # np.save('savedata.npy', savedata)




