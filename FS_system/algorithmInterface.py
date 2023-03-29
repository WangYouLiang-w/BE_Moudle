# -- coding: utf-8 --**
import numpy as np
from numpy import ndarray
import scipy.signal as signal
import scipy.io as scio
import socket
import time
from scipy.stats import pearsonr
import math
import json
from threading import Thread
import threading

class Kalman_filter:
    def __init__(self, length, fs,CD_ip_address,CD_port):
        self.data_length = length
        self.Q = np.load('Q.npy')#np.zeros((4,4))
        self.A = np.array([[1,0,1/fs,0],[0,1,0,1/fs],[0,0,1,0],[0,0,0,1]])
        self.P = np.eye(4,dtype=int)
        self.X = np.array([[0.0],[0.0],[0],[0]])
        self.H = np.array([[1,0,0,0],[0,1,0,0]])
        self.R = np.load('R.npy')#np.zeros((2,2))
        self.err = np.load('err.npy')#np.zeros((1,2))
        self.K = np.zeros((4,2))
        self.t = 1/fs
        #----
        self.W_screen = 1/2560         #注视点的归一化范围
        self.H_screen = 1/1440
        self.local_Setting = {}
        self.UScreen_position = []
        self.MScreen_position = []
        self.stim_position_U = []
        self.stim_position_M = []
        self.max_gap_length=1
        self.fs = fs
        self.distance = []
        self.local_position = {'1':[],'2':[],'3':[],'4':[],'5':[],'6':[]}
        self.gaze_points = np.zeros((1,6))
        self.ip_address = CD_ip_address
        self.port = CD_port
        self.stim_num = 0


    def Connect_CD(self):
        '''与中控建立TCP连接'''
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        for i in range(5):
            try:
                time.sleep(1.5)
                self.client_socket.connect((self.ip_address, self.port))
                print('CD Connect Successfully.')
                buff_size_send = self.client_socket.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
                buff_size_recv = self.client_socket.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
                print('Current eye recv buffer size is {} bytes, send buff size is {} bytes.'.format(buff_size_recv, buff_size_send))
                break
            except:
                print('没有和中控建立有效连接')
                self.client_socket.close()


    def recv_local_Json(self):
        '''
        界面配置文件
        :param stim_position:{}  刺激块在屏幕中的像素坐标位置
        :return:self.stim_position_U:[list] 上屏划分区域后归一化后的坐标位置
                self.stim_position_M:[list] 下屏划分区域后归一化后的坐标位置
        '''
        # 接收界面配置Json文件
        recv_msg = self.client_socket.recv(4096)
        recv_msg = str(bytes(recv_msg), "utf8")
        if recv_msg != 'no':
            self.local_Setting = {}
            self.UScreen_position = []
            self.MScreen_position = []

            self.stim_position_U = []
            self.stim_position_M = []
            self.local_position = {'1':[],'2':[],'3':[],'4':[],'5':[],'6':[]}
            print(recv_msg)
            self.local_Setting = json.loads(recv_msg)   # 将Json转成字典形式
            # 获取当前配置文件中包含每个屏的信息
            # print(self.local_Setting)
            for key in self.local_Setting :
                if key[0] == 'U':
                    self.UScreen_position.append(key)
                elif key[0] == 'M':
                    self.MScreen_position.append(key)

            if len(self.UScreen_position) == 0 and len(self.MScreen_position) == 0:
                self.stim_num = 0
            else:
                self.stim_num = 1

            # 获取每个屏幕上刺激块的位置
            for num_U in range(len(self.UScreen_position)):  # 1,2  ["",X,Y,Size]
                U_GazePoint = [0.5+self.local_Setting[self.UScreen_position[num_U]][2]*self.W_screen,
                                                1-self.local_Setting[self.UScreen_position[num_U]][3]*self.H_screen]
                self.stim_position_U.append(U_GazePoint)
                # 区域1
                # print(self.UScreen_position[num_U])
                if self.local_Setting[self.UScreen_position[num_U]][0] == '1':
                    self.local_position['1'].append(U_GazePoint)
                # 区域2
                if self.local_Setting[self.UScreen_position[num_U]][0] == '2':
                    self.local_position['2'].append(U_GazePoint)
                # 区域3
                if self.local_Setting[self.UScreen_position[num_U]][0] == '3':
                    self.local_position['3'].append(U_GazePoint)

            for num_M in range(len(self.MScreen_position)):  # 1,2  ["",X,Y,Size]
                M_GazePoint = [1.5+self.local_Setting[self.MScreen_position[num_M]][2]*self.W_screen,
                                                1-self.local_Setting[self.MScreen_position[num_M]][3]*self.H_screen]
                self.stim_position_M.append(M_GazePoint)
                # 区域4
                if self.local_Setting[self.MScreen_position[num_M]][0] == '4':
                    self.local_position['4'].append(M_GazePoint)
                # 区域5
                if self.local_Setting[self.MScreen_position[num_M]][0] == '5':
                    self.local_position['5'].append(M_GazePoint)
                # 区域6
                if self.local_Setting[self.MScreen_position[num_M]][0] == '6':
                    self.local_position['6'].append(M_GazePoint)

    
    def liner_insert(self,fixations):
        '''
        线性插值
        :param fixations:list
        :return:fixations:list
        '''
        # 小于interval_num 需要内插
        interval_num = (self.fs*self.max_gap_length)/1000
        fix_len = len(fixations)
        a = np.nonzero(fixations)
        a = np.array(a)
        # FT定义为开始 TF定义为结束
        start_idx, end_idx = [], []
        for ii in range(a[0,0],fix_len-2):
            if (fixations[ii] != 0) & \
                (fixations[ii+1] == 0) :
                start_idx.append(ii)
            if (fixations[ii] == 0) & \
                (fixations[ii+1] != 0):
                end_idx.append(ii)
        for start, end in zip(start_idx, end_idx):
            nan_len = end-start
            if nan_len>interval_num:
                px = [fixations[start], fixations[end+1]]
                interx = ((px[1]-px[0])*np.arange(nan_len+1)/float(nan_len+1)+px[0]).tolist()
                for ii in range(1, len(interx)):
                    fixations[start+ii] = interx[ii]
        return fixations


    def get_eyedata(self,data):
        '''
        @ 获取眼动数据
        :param data: data (nparray).shape(4,n)
                     n:样本点数
        :return: eye_data (nparray).shape(n,3)
                 (x,y)
                 screen_flag:0 or 1 (0:上屏,1:下屏)
        '''
        eye_data = np.zeros((data.shape[1],3))
        for num in range(data.shape[1]):
            # print(data)
            eye_data[num,2] = data[3,num]/10000
            # print(eye_data[num,2])
            if eye_data[num,2] == 1.0:        # 上屏点
                gaze_right_eye = data[0, num]/10000
                gaze_left_eye = data[1, num]/10000

            elif eye_data[num,2] == 2.0:
                gaze_right_eye = data[0, num]/10000 + 1
                gaze_left_eye = data[1, num]/10000 + 1

            else:
                gaze_right_eye = 0
                gaze_left_eye = 0
            eye_data[num,0:2] = [gaze_right_eye,gaze_left_eye]
        # 线性插值
        # eye_data[:,0] = self.liner_insert(eye_data[:,0])
        # eye_data[:,1] = self.liner_insert(eye_data[:,1])
        return eye_data

    
    def filter(self,Eyedata):
        '''
        @ 对数据进行卡尔曼滤波
        :param data: Eyedata [nparray].shape(4,n)
        :return:filter_data  [nparray].shape(n,3):[:,(x,y,flag)]
        '''
        data = self.get_eyedata(Eyedata)
        # data = Eyedata
        xy_data = data[:,0:2]
        pre_data = self.X
        P = self.P
        K = self.K
        # for i in range(xy_data.shape[0]):
        #     pre_datas = xy_data[i,:] - self.err     # 弥补上偏差
        #     measure_data = np.mat(pre_datas).T # 测量值
        #     P = np.dot(np.dot(self.A,P),self.A.T) + self.Q
        #     a = np.dot(np.dot(self.H,P),self.H.T) + self.R
        #     b = np.dot(P,self.H.T)
        #     K = np.dot(b,np.linalg.inv(a))
        #     filter_data = pre_data + np.dot(K,measure_data-np.dot(self.H,pre_data))
        #     pre_data = filter_data
        #     data[i,0:2] = np.array(filter_data[0:2,0].T)
        #     P = np.dot(np.eye(4)-np.dot(K,self.H),P)
        return data


    def normal_model(self,stim_pos,Eyedata):
        '''
        @构建卡尔曼滤波后数据距离分布模型,计算目标位置的概率
        :param data: stim_pos:[list]:[[x1,y1],[x2,y2],....,[xn,yn]]
        :param data: Eyedata (nparray).shape(4,n)
        :return: [P1(x),P2(x),...,Pn(x)]
                 n: 当前决策区域内的刺激块个数
        '''
        P = []
        filter_data  = self.filter(Eyedata)
        # a = np.nonzero(filter_data[:,0])
        x = 0
        x_count = 0
        y = 0
        y_count = 0
        print(len(filter_data[:,0]))
        # 去除零元素对注视中心的计算
        for i in range(len(filter_data[:,0])):
            if filter_data[i,0] != 0:
                x = x + filter_data[i,0]/self.W_screen
                x_count += 1

        for j in range(len(filter_data[:,1])):
            if filter_data[j,1] != 0:
                y = y + filter_data[j,1]/self.H_screen
                y_count += 1
        x_mean = x/x_count
        y_mean = y/y_count

        if len(stim_pos) != 0:
            for pos in range(len(stim_pos)):
                x = stim_pos[pos][0]/self.W_screen
                y = stim_pos[pos][1]/self.H_screen
                dis_xy = math.sqrt((x-x_mean)*(x-x_mean)+(y-y_mean)*(y-y_mean))
                P.append(dis_xy)
            print('P:{}'.format(min(P)))
        else:
            P = None
        return P


class EyeOnlineProcess(Kalman_filter):
    def __init__(self,datalength,ip_address,CD_port):
        Kalman_filter.__init__(self, length=datalength, fs=100,CD_ip_address=ip_address,CD_port=CD_port)
        self.SuitPoint = 0
        self.block_W = 555/2
        self.block_H = 405/2

    def EyeTrace_Position(self,eyedatas,LocalNum):
        '''
        获取眼动追踪的区域
        '''
        gaze_right_left_eyes = eyedatas
        # print("眼动追踪区域:{},模板数量:{}".format(LocalNum,self.template_num))
        if ((gaze_right_left_eyes[0] > 0.05 and gaze_right_left_eyes[0] < 0.95) and (gaze_right_left_eyes[1] > 0.05 and gaze_right_left_eyes[1] < 0.95)):
            self.gaze_points[0][int(LocalNum)-1] = self.gaze_points[0][int(LocalNum)-1] + 1


    def EyeDecideApply(self,data,index):
        '''
        眼动决策结果
        '''
        eye_decide_result = 0
        stim_position = self.local_position[str(index+1)]
        P = self.normal_model(stim_position,data)
        if P != None:
            if min(P) > 150:
                eye_decide_result = 0
            else:
                eye_decide_result = (P.index(min(P)) + 1)
        else:
            eye_decide_result = 0
        # if P != None:
        #     if min(P) > 200:
        #         eye_decide_result = 0
        #     else:
        #         if index <= 4:
        #             eye_decide_result = (P.index(min(P)) + 1) + index * 6
        #         else:
        #             eye_decide_result = (P.index(min(P)) + 1) + 28
        # else:
        #     eye_decide_result = 0

        return  eye_decide_result


    def ResultEyeDecide(self,data):
        '''
        发送眼动决策结果
        '''
        eye_decide_result = 0
        self.gaze_points = np.zeros((1,6))
        EyeData = self.filter(data)
        for num in range(data.shape[1]):
            screen_flag = data[3,num]/10000
            # gaze_right_eye = EyeData[num, 0]
            # gaze_left_eye = EyeData[num, 1]
            gaze_right_eye = data[0, num]/10000
            gaze_left_eye = data[1, num]/10000
            gaze_right_left_eyes = [gaze_right_eye,gaze_left_eye]
            # if screen_flag == 2.0:
            #     gaze_right_left_eyes = [gaze_right_eye-1,gaze_left_eye-1]
            if screen_flag == 1.0:
                # 上屏眼动数据有效
                if ((gaze_right_left_eyes[0] > 0 and gaze_right_left_eyes[0] < 0.5)):
                    self.EyeTrace_Position(gaze_right_left_eyes,'1')
                if ((gaze_right_left_eyes[0] > 0.5 and gaze_right_left_eyes[0] < 1) and (gaze_right_left_eyes[1] > 0 and gaze_right_left_eyes[1] < 0.5)):
                    self.EyeTrace_Position(gaze_right_left_eyes,'2')
                if ((gaze_right_left_eyes[0] > 0.5 and gaze_right_left_eyes[0] < 1) and (gaze_right_left_eyes[1] > 0.5 and gaze_right_left_eyes[1] < 1.0)):
                    self.EyeTrace_Position(gaze_right_left_eyes,'3')
            if screen_flag == 2.0:
                # 下屏眼动数据有效
                if ((gaze_right_left_eyes[0] > 0 and gaze_right_left_eyes[0] < 0.7)
                        and (gaze_right_left_eyes[1] > 0.3 and gaze_right_left_eyes[1] < 0.7)):
                    self.EyeTrace_Position(gaze_right_left_eyes,'5')
                elif ((gaze_right_left_eyes[0] > 0.8 and gaze_right_left_eyes[0] < 1)
                      and (gaze_right_left_eyes[1] > 0 and gaze_right_left_eyes[1] < 1)):
                    self.EyeTrace_Position(gaze_right_left_eyes,'6')
                elif ((gaze_right_left_eyes[0] > 0 and gaze_right_left_eyes[0] < 0.8)
                      and ((gaze_right_left_eyes[1] > 0 and gaze_right_left_eyes[1] < 0.3)or(gaze_right_left_eyes[1] > 0.7 and gaze_right_left_eyes[1] < 1))):
                    self.EyeTrace_Position(gaze_right_left_eyes,'4')

        gaze_points_max = np.max(self.gaze_points)
        if gaze_points_max != 0:
            index = np.argmax(self.gaze_points)
            eye_decide_result = self.EyeDecideApply(data,index)
        else:
            eye_decide_result = 0
            index = -1

        if eye_decide_result == 0:
            # 异步作用
            index = -1

        return eye_decide_result,index



# offer a common test interface to any algorithm.
class preprocess():

    def __init__(self, filterModelName='./data/'+'IIR_filterModel.mat'):
        self.filterModelName = filterModelName
        #fusion coefficient
        self.a = (np.array([i for i in range(1, 11)])).reshape(1, 10) ** (-1.25) + 0.25

    def loadfilterModel(self):
        '''
        Read the matlab .mat format filter model file, and reshape to python dict format.
        To use, please put the filter model file in the same path with this file.
        '''
        filterpara = {}
        IIR_filterModel_dict = scio.loadmat(self.filterModelName)
        filterModel_b = IIR_filterModel_dict['IIR_filterModel'][0, 0]['f_b'][0]
        filterModel_a = IIR_filterModel_dict['IIR_filterModel'][0, 0]['f_a'][0]
        N_sub = filterModel_a.shape[0]
        for sub_i in range(N_sub):
            filterpara[''.join(('f_b', str(sub_i)))] = filterModel_b[sub_i][0]
            filterpara[''.join(('f_a', str(sub_i)))] = filterModel_a[sub_i][0]
        return filterpara

    def timefilter(self, notchdata, sub_i, fs=None):
        self.filterpara = self.loadfilterModel()
        fiteredData = np.zeros((notchdata.shape[0], notchdata.shape[1]))
        f_b = self.filterpara[''.join(('f_b', str(sub_i)))]
        f_a = self.filterpara[''.join(('f_a', str(sub_i)))]
        fiteredData= signal.filtfilt(f_b, f_a, notchdata, axis=1, padlen=3*(max(len(f_b), len(f_a))-1))
        return fiteredData


    def notch_filter_python(self,raweeg,fs):
        """
        数据进行预处理
        @param raweeg: (n_chans,samples,blocks)
        @param fs: 采样率
        @return: 滤波后的数据data2
        """
        wo = 50/(fs/2)
        b,a = signal.iirnotch(wo,45)
        notch_data = signal.filtfilt(b, a, raweeg, axis=1, padlen=3*(max(len(b), len(a))-1))
        return notch_data


    def timefilter_python(self, raweeg, idx_fb, fs, axis):
        """
        数据进行预处理
        @param raweeg: (n_chans,samples,blocks)
        @param fs: 采样率
        @param idx_fb:子带索引
        @return: 滤波后的数据data2
        """
        passband = [28.0, 60.0, 68.0]
        stopband = [22.0, 54.0, 62.0]
        wp = [2*passband[idx_fb]/fs, 2*90/fs]
        ws = [2*stopband[idx_fb]/fs, 2*100/fs]
        if fs == 250:
            gpass = 3
            gstop = 40
        elif fs == 1000:
            gpass = 3
            gstop = 10

        N, wn = signal.cheb1ord(wp, ws, gpass, gstop)
        sos_system = signal.cheby1(N, rp=0.5, Wn=wn, btype='bandpass', output='sos')
        filterdata = signal.sosfiltfilt(sos_system, raweeg, axis=axis)

        return filterdata


def classfierdecide(filtdata, W, template):
    R = np.zeros((template.shape[0]))
    currentTestData = filtdata[:, :].T
    data1 = np.dot(np.ascontiguousarray(currentTestData), np.ascontiguousarray(W.T))
    for target in range(template.shape[0]):
        currentTemp = template[target, :, :]
        # R[target] = np.corrcoef(data1.reshape(-1), np.dot(np.ascontiguousarray(currentTemp.T), np.ascontiguousarray(W[:, :].T)).reshape(-1))[0][1]
        R[target] = compute_corr2(data1, np.dot(currentTemp.T, W[:, :].T))
    return R

def compute_corr2(X1, X2):
    '''
    Pearson correlation coefficients of two matrices

    Parameters
    ----------
    X1 : ndarray(XX, XX)
        matrice.
    X2 : ndarray(XX, XX)
        matrice.

    Returns
    -------
    r : float
        Pearson correlation coefficient.

    '''
    X11 = np.sqrt((X1 ** 2).sum())
    X22 = np.sqrt((X2 ** 2).sum())
    r = ((X1 * X2).sum()) / (X11 * X22)

    return r

def Corr(X, Y):
    X = np.reshape(X, (-1))
    Y = np.reshape(Y, (-1))
    rho = pearsonr(X, Y)[0]
    return rho

class TRCA(preprocess):
    
    def __init__(self, TRCA_spatialFilter, template, Nsub=2):
        preprocess.__init__(self, filterModelName='./data/'+'IIR_filterModel.mat')
        self.W = TRCA_spatialFilter     # W为TRCA空间滤波器，ndarray (Nsub, Nchannels, Ntarget)
        self.template = template        # template为模板，ndarray (Nsub, Ntarget, Nchannels, Ntimes)
        self.Nsub = Nsub
        self.Nc = template.shape[2]
        self.Ntimes = template.shape[3]
        self.Ntarget = template.shape[1]

    
    def test_algorithm(self, rawdata):
        R = np.zeros((self.Nsub,self.template.shape[1]))
        self.filtdata = np.zeros((self.Nsub, self.Nc, self.Ntimes))
        for sub_i in range(self.Nsub):
            self.filtdata[sub_i, :, :] = self.timefilter_python(rawdata, sub_i, fs=1000, axis=1)
            R[sub_i, :] = classfierdecide(self.filtdata[sub_i, :, :], self.W[sub_i, :, :], self.template[sub_i, :, :, :])
        self.result = np.matmul(self.a[0, 0:self.Nsub], R)
        return self.result


class TDCA(preprocess):
    def __init__(self, W, Template, P, Nfb, l, fs=1000, ip_address='192.168.1.10', port=40007):
        preprocess.__init__(self, filterModelName='./data/'+'IIR_filterModel.mat')
        self.W = W
        self.Template = Template
        self.P = P
        self.Nfb = Nfb
        self.fs = fs
        self.l = l
        self.Np = self.P.shape[-1]
        self.fb_coefs = (np.array([i for i in range(1, self.Nfb + 1)])).reshape(1, self.Nfb) ** (-1.25) + 0.25
        self.Nf = self.P.shape[0]
        self.brain_sock_client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.brain_CentralAddr = (ip_address, port)

    def aug_2(self, X: ndarray, P: ndarray, training: bool=True):
        '''
        获得二阶增广数据
        '''
        n_channels, n_points = X.shape
        aug_X = np.zeros(((self.l+1)*n_channels, self.Np))
        if training:
            for i in range(self.l+1):
                aug_X[i*n_channels:(i+1)*n_channels, :] = X[..., i:i+self.Np]
        else:
            for i in range(self.l+1):
                aug_X[i*n_channels:(i+1)*n_channels, :self.Np-i] = X[..., i:self.Np]
        aug_Xp = aug_X@P
        aug_X = np.concatenate([aug_X, aug_Xp], axis=-1)
        return aug_X

    def test_algorithm(self, testdata, templatenum=None):
        if templatenum is not None:
            self.Template = self.Template[:, :templatenum, :, :]
        rho = np.zeros((self.Nfb, self.Nf))
        for fb_i in range(self.Nfb):
            for class_i in range(self.Nf):
                filter_test = self.timefilter_python(testdata, fb_i, self.fs, axis=-1)
                # a = self.aug_2(filter_test, self.P[class_i, :, :], training=False)
                # print(a.shape)
                test_model = np.matmul(self.W[fb_i, :], self.aug_2(filter_test, self.P[class_i, :, :], training=False))
                rho[fb_i, class_i] = Corr(test_model, np.matmul(self.W[fb_i, :], self.Template[fb_i, class_i, :, :]))

        r = np.matmul(self.fb_coefs, rho)
        return r

    def send_result(self, result):
        self.brain_sock_client.sendto(bytes(str(result), 'utf8'), self.brain_CentralAddr)


class algorithmthread(Thread):
    def __init__(self,w,template,datalocker,addloop,datalength, P, Nfb, l, mode='tdca',control_mode = 'eye'):
        Thread.__init__(self)
        self.sendresult = 0
        self.datalocker = datalocker
        self.control_model = control_mode
        if mode == 'trca':
            self.classfier = TRCA(w,template, Nsub=2)
        else:
            self.classfier = TDCA(w, template, P, Nfb, l)
        self.addloop = addloop
        #-----------脑电部分---------#
        self.testdata = np.array([0])
        self.resultCount = np.zeros((template.shape[1],addloop))
        self.currentPtr = 0
        self.resultPtr = 0
        self.addloop = addloop
        #-----------------------#
        self.sock_client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.controlCenterAddr = ('192.168.1.70', 7810)
        self.sock_Feedback = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.FeedbackAddr = ('192.168.1.70', 7820)
        self.controlCenterAddr1 = ('192.168.1.70', 7788)
        self.result = 0
        self.all_count = 0
        self.right_count = 0
        self.target = 0
        self.Msg = 'S'
        #--------眼动部分---------#
        self.EyeTracker = EyeOnlineProcess(datalength,'192.168.1.70', 9998)
        self.eye_testdata = np.array([0])
        self.labels = 0
        self.results = np.zeros((3,36))
        self.result_count = 0
        self.eye_result = 0
        self.local_indx = 0
        self.Send_command = 'IDLE'


    def run(self):
        '''与中控建立连接'''
        self.EyeTracker.Connect_CD()
        while True:
            # 接收中控的界面配置
            self.EyeTracker.client_socket.send(bytes(self.Msg,'utf-8'))
            self.EyeTracker.recv_local_Json()
            if self.testdata.shape[0] == 1 or self.EyeTracker.stim_num == 0:
                self.datalocker.set()
                continue
            else:
                r = self.classfier.test_algorithm(self.testdata)
                self.appendResult(r)
                self.SenderRsult()                         # 发送决策结果
                self.clearTestdata()                       # 清除数据缓存
                self.datalocker.set()

    def get_brain_local(self,eye_result):
        brain_local = int((eye_result+1)/2)
        return  brain_local


    def SenderRsult(self):
        '''发送决策结果'''
        # eye_result：1-6  err:0, local_indx:0-5 err:-1
        eye_result, local_indx = self.EyeResultDecide()
        # brain_result：0-5
        brain_result = self.BrainResultDecide()
        if self.control_model == 'brain':
            # send_result = self.fina_result(local_indx,brain_result)
            eye_result = self.fina_result(local_indx,eye_result)
            brain_local = self.get_brain_local(eye_result)
            send_result = (brain_local-1)*2+brain_result+1
            print("local:{0}, eye:{1},brain_local:{2} brain:{3}, send_com:{4}".format(local_indx,eye_result,brain_local,brain_result,send_result))
            if local_indx != -1:
                #脑眼决策值
                if local_indx == 3:
                    self.sendCommand(int((send_result)),self.controlCenterAddr)
                self.sendFeedback(int((send_result)),self.FeedbackAddr)
                setting_pos = list(self.EyeTracker.local_position.keys())
                self.Send_command = self.EyeTracker.local_position[setting_pos[int(brain_result-1)]][1]
            else:
                print("brain_control err！")
                self.Send_command = 'IDLE'
                self.sendFeedback(int(40),self.FeedbackAddr)

        if self.control_model == 'eye':
            send_result = self.fina_result(local_indx,eye_result)
            print("local:{0}, eye:{1}, brain:{2}, send_com:{3}".format(local_indx,eye_result,brain_result,send_result))
            if eye_result != 0:
                #眼动决策值
                # if local_indx == 3:
                #     self.sendCommand(int((send_result+10-1)),self.controlCenterAddr)
                #     self.sendFeedback(int((send_result+10)), self.FeedbackAddr)
                # else:
                self.sendCommand(int((send_result-1)), self.controlCenterAddr)
                self.sendFeedback(int((send_result)),self.FeedbackAddr)
            else:
                print("eye_control err!")
                self.sendFeedback(int(40),self.FeedbackAddr)

        if self.control_model == 'brain_eye':
            send_result = self.fina_result(local_indx,eye_result)
            print("local:{0}, eye:{1}, brain:{2}, send_com:{3}".format(local_indx,eye_result,brain_result,send_result))
            if eye_result != 0 and eye_result-1 == brain_result:
                #脑眼并行决策值
                self.sendCommand(int((send_result-1)),self.controlCenterAddr)
            else:
                print("brain_eye_control err!")


    def recvData(self,rawdata,label):
        '''接收待处理的脑电数据'''
        self.testdata = np.copy(rawdata)
        self.target = np.mod(label-1, 6)    #标签对应的脑电结果
        self.labels = label-1              # 标签值 0-17


    def RecvEyeData(self,rawdata):
        '''获取待处理的眼动数据'''
        self.eye_testdata = rawdata


    def appendResult(self,data):
        '''叠加轮次'''
        self.resultCount[:,np.mod(self.currentPtr, self.addloop)] = data    # 存放的是相关系数 叠加轮次是5轮 0-4
        self.currentPtr += 1


    def BrainResultDecide(self):
        '''获取脑电决策结果'''
        decide = np.sum(self.resultCount,axis=1,keepdims=False)
        brain_decide_result = np.argmax(decide)
        return brain_decide_result


    def EyeResultDecide(self):
        '''获取眼动决策值'''
        eye_result, local_indx = self.EyeTracker.ResultEyeDecide(self.eye_testdata)
        return eye_result, local_indx


    def fina_result(self,local_index,result):
        '''
        映射屏幕中刺激的数目
        @local_index：区域编号：0-5
        @result：眼动决策值或脑电决策值：0-5
        :return 最终发送结果
        '''
        stim_num = 0
        if local_index != -1 and local_index != 0:
            for i in range(local_index):
                stim_num = stim_num + len(self.EyeTracker.local_position[str(i+1)])
        return result + stim_num


    def sendCommand(self, command,addr):
        msg = bytes(str(command), "utf8")
        self.sock_client.sendto(msg,addr)


    def sendFeedback(self, command,addr):
        msg = bytes(str(command), "utf8")
        self.sock_Feedback.sendto(msg,addr)


    def clearTestdata(self):
        self.testdata = np.array([0])
        self.eye_testdata = np.array([0])









