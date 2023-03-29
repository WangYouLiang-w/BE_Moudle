import numpy as np
import scipy.signal as signal
import scipy.io as scio
import socket
import time
import heapq
import json
import threading
from scipy.stats import norm
import math
from scipy import stats





class Kalman_filter:
    def __init__(self, length, fs):
        self.data_length = length
        self.Q = np.zeros((4,4))
        self.A = np.array([[1,0,1/fs,0],[0,1,0,1/fs],[0,0,1,0],[0,0,0,1]])
        self.P = np.eye(4,dtype=int)
        self.X = np.array([[0.0],[0.0],[0],[0]])

        self.H = np.array([[1,0,0,0],[0,1,0,0]])
        self.R = np.zeros((2,2))
        self.err = np.zeros((1,2))

        self.K = np.zeros((4,2))
        self.t = 1/fs
        #----
        self.W_screen = 1/2560         #注视点的归一化范围
        self.H_screen = 1/1440
        self.UI_Setting = {}
        self.UScreen_position = []
        self.MScreen_position = []

        self.stim_position_U = []
        self.stim_position_M = []

        self.max_gap_length=75
        self.fs = fs
        self.distance = []



    def recv_UI_Json(self,stim_position):
        '''
        界面配置文件
        :param stim_position:{}  刺激块在屏幕中的像素坐标位置
        :return:self.stim_position_U:[list] 上屏刺激块归一化后的坐标位置
                self.stim_position_M:[list] 下屏刺激块归一化后的坐标位置
        '''
        # 接收界面配置Json文件
        self.UI_Setting = stim_position

        # 获取当前配置文件中包含每个屏的信息
        for key in self.UI_Setting :
            if key[0] == 'U':
                self.UScreen_position.append(key)
            elif key[0] == 'M':
                self.MScreen_position.append(key)

        # 获取每个屏幕上刺激块的位置
        for num_U in range(len(self.UScreen_position)):  # 1,2  ["",X,Y,Size]
            self.stim_position_U.append([0.5+self.UI_Setting[self.UScreen_position[num_U]][2]*self.W_screen,
                                            1-self.UI_Setting[self.UScreen_position[num_U]][3]*self.H_screen])

        for num_M in range(len(self.MScreen_position)):  # 1,2  ["",X,Y,Size]
            self.stim_position_M.append([0.5+self.UI_Setting[self.MScreen_position[num_M]][2]*self.W_screen,
                                            0-self.UI_Setting[self.MScreen_position[num_M]][3]*self.H_screen])

    
    def liner_insert(self,fixations):
        '''
        线性插值
        :param fixations:list
        :return:fixations:list
        '''
        # 小于interval_num 需要内插
        interval_num = (self.fs*self.max_gap_length)/1000
        fix_len = len(fixations)
        # FT定义为开始 TF定义为结束
        start_idx, end_idx = [], []
        for ii in range(fix_len-2):
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
            gaze_right_eye = data[0, num]/10000
            gaze_left_eye = data[1, num]/10000
            eye_data[num,0:2] = [gaze_right_eye,gaze_left_eye]
            eye_data[num,3] = data[3,num]/10000
        # 线性插值
        eye_data[:,0] = self.liner_insert(eye_data[:,0])
        eye_data[:,1] = self.liner_insert(eye_data[:,1])
        return eye_data

    
    def filter(self,Eyedata):
        '''
        @ 对数据进行卡尔曼滤波
        :param data: Eyedata [nparray].shape(4,n)
        :return:filter_data  [nparray].shape(n,3):[:,(x,y,flag)]
        '''
        data = self.get_eyedata(Eyedata)
        xy_data = data[:,0:2]
        pre_data = self.X
        P = self.P
        K = self.K
        for i in range(xy_data.shape[0]):
            pre_datas = xy_data[i,:] - self.err     # 弥补上偏差
            measure_data = np.mat(pre_datas[i,:]).T # 测量值
            P = np.dot(np.dot(self.A,P),self.A.T)               # + self.Q
            a = np.dot(np.dot(self.H,P),self.H.T) + self.R
            b = np.dot(P,self.H.T)
            K = np.dot(b,np.linalg.inv(a))
            filter_data = pre_data + np.dot(K,measure_data-np.dot(self.H,pre_data))
            pre_data = filter_data
            data[i,0:2] = np.array(filter_data[0:2,0].T)
            P = np.dot(np.eye(4)-np.dot(K,self.H),P)

        return data



    def normal_model(self,stim_pos,Eyedata):
        '''
        @构建卡尔曼滤波后数据的正态分布模型,计算目标位置的概率
        :param data: stim_pos:[list]:[[x1,y1],[x2,y2],....,[xn,yn]]
        :param data: Eyedata (nparray).shape(4,n)
        :return: [P1(x),P2(x),...,Pn(x)]
                 n: 当前决策区域内的刺激块个数
        '''
        filter_data  = self.filter(Eyedata)
        x_mean = np.mean(filter_data[:,0]/self.W_screen,axis=0)
        y_mean = np.mean(filter_data[:,1]/self.H_screen,axis=0)

        for pos in len(stim_pos): 
            x = stim_pos[pos][0]/self.W_screen
            y = stim_pos[pos][1]/self.H_screen
            dis_xy = math.sqrt((x-x_mean)*(x-x_mean)+(y-y_mean)*(y-y_mean))
            self.distance[pos] = dis_xy


    
class EyeOnlineProcess(Kalman_filter):

    def __init__(self,datalength):
        Kalman_filter.__init__(self, length=datalength, fs=100)
        self.SuitPoint = 0
        # self.block_size = 180/2   # 块的大小
        self.block_W = 555/2
        self.block_H = 405/2
        self.eyetrack_threshold = int(datalength*0.35)   # 所占比例0.4
    

    def local_result(self,data):
        '''计算眼动决策结果'''
        local_result = 0
        # self.position_stim = self.stim_position_U + self.stim_position_M
        self.gaze_points_U = np.zeros((1,len(self.stim_position_U)))
        self.gaze_points_M = np.zeros((1,len(self.stim_position_M)))

        if len(self.stim_position_M) == 0:
            Is_M = False
        else:
            Is_M = True

        if len(self.stim_position_U) == 0:
            Is_U = False
        else:
            Is_U = True

        for num in range(data.shape[1]):
            gaze_right_eye = data[0, num]/10000
            gaze_left_eye = data[1, num]/10000
            screen_flag = data[3,num]/10000
            gaze_right_left_eyes = (gaze_right_eye,gaze_left_eye)
            tic = time.time()

            if screen_flag == 2.0:
                if ((gaze_right_left_eyes[0] > 0.05 and gaze_right_left_eyes[0] < 0.95) and (gaze_right_left_eyes[1] > 0.05 and gaze_right_left_eyes[1] < 0.95)):
                    self.SuitPoint = self.SuitPoint + 1
                    m = 0
                    for position in self.stim_position_M:
                        if (gaze_right_left_eyes[0] > position[0]- self.block_W*self.W_screen and gaze_right_left_eyes[0] < position[0]+ self.block_W*self.W_screen) and (gaze_right_left_eyes[1] > position[1]- self.block_H*self.H_screen and gaze_right_left_eyes[1] < position[1]+ self.block_H*self.H_screen):
                            self.gaze_points_M[0][m] = self.gaze_points_M[0][m] + 1
                        m = m + 1

            elif screen_flag == 1.0:
                # 上屏眼动数据有效
                if ((gaze_right_left_eyes[0] > 0.05 and gaze_right_left_eyes[0] < 0.95) and (gaze_right_left_eyes[1] > 0.05 and gaze_right_left_eyes[1] < 0.95)):
                    self.SuitPoint = self.SuitPoint + 1
                    u = 0
                    for position in self.stim_position_U:
                        if (gaze_right_left_eyes[0] > position[0]- self.block_W*self.W_screen and gaze_right_left_eyes[0] < position[0]+ self.block_W*self.W_screen) and (gaze_right_left_eyes[1] > position[1]- self.block_H*self.H_screen and gaze_right_left_eyes[1] < position[1]+ self.block_H*self.H_screen):
                            self.gaze_points_U[0][u] = self.gaze_points_U[0][u] + 1
                        u = u + 1

            toc = time.time()
            # print("数据长度：{}，注视位置：{},屏幕位置：{} 时间：{}".format(data.shape[1],gaze_right_left_eyes,screen_flag,toc-tic))

        self.SuitPoint = 0
        if Is_M and Is_U:
            if max(self.gaze_points_U[0])>max(self.gaze_points_M[0]):
                if max(self.gaze_points_U[0]) > self.eyetrack_threshold:
                    local_result = np.argmax(self.gaze_points_U[0])+1
                else:
                    local_result = 0
            else:
                if max(self.gaze_points_M[0]) > self.eyetrack_threshold:
                    local_result = np.argmax(self.gaze_points_M[0])+1+len(self.stim_position_U)
                else:
                    local_result = 0
            # print("上下屏都有的眼动决策结果{}".format(eye_decide_result))

        if Is_U and Is_M == False:
            if np.max(self.gaze_points_U[0]) > self.eyetrack_threshold:
                local_result = np.argmax(self.gaze_points_U[0])+1
                # print("上屏眼动决策结果{},最大值{}".format(eye_decide_result,np.max(self.gaze_points_U[0])))
            else:
                local_result = 0
            # print("上屏眼动决策结果{},阈值为{}".format(eye_decide_result,self.eyetrack_threshold))
            # print("上屏眼动决策结果数组{}".format(self.gaze_points_U[0]))

        if Is_M and Is_U == False:
            if np.max(self.gaze_points_M[0]) > self.eyetrack_threshold:
                local_result = np.argmax(self.gaze_points_M[0])+1
                # print("下屏眼动决策结果{},最大值{}".format(eye_decide_result,np.max(self.gaze_points_M[0])))
            else:
                local_result = 0
            # print("下屏眼动决策结果数组{}".format(self.gaze_points_M[0]))


        return local_result



    def ResultEyeDecide(self,data):
        '''计算眼动决策结果'''
        eye_decide_result = 0
        # self.position_stim = self.stim_position_U + self.stim_position_M
        self.gaze_points_U = np.zeros((1,len(self.stim_position_U)))
        self.gaze_points_M = np.zeros((1,len(self.stim_position_M)))

        if len(self.stim_position_M) == 0:
            Is_M = False
        else:
            Is_M = True

        if len(self.stim_position_U) == 0:
            Is_U = False
        else:
            Is_U = True

        if Is_U and Is_M == False:
            P = self.normal_model(self.stim_position_U,data)
        
        if Is_M and Is_U == False:
            P = self.normal_model(self.stim_position_M,data)

        if Is_M and Is_U:
            P = self.normal_model(self.stim_position_U + self.stim_position_M,data)

        if min(P) > 100:
            eye_decide_result = 0
        else:
            eye_decide_result, = min(P)

        # if screen_flag == 2.0:


        return eye_decide_result


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


    def timefilter_python(self, raweeg, idx_fb, fs):
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
        filterdata = signal.sosfiltfilt(sos_system, raweeg, axis=1)

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

class TRCA(preprocess):
    
    def __init__(self, TRCA_spatialFilter, template, Nsub=2):
        preprocess.__init__(self, filterModelName='./data/'+'IIR_filterModel.mat')
        self.W = TRCA_spatialFilter     # W为TRCA空间滤波器，ndarray (Nsub, Nchannels, Ntarget)
        self.template = template        # template为模板，ndarray (Nsub, Ntarget, Nchannels, Ntimes)
        self.Nsub = Nsub
        self.Nc = template.shape[2]
        self.Ntimes = template.shape[3]
        self.Ntarget = template.shape[1]

    
    def transform(self, rawdata):
        self.filtdata = np.zeros((self.Nsub, self.Nc, self.Ntimes))
        # rawdata = self.notch_filter_python(rawdata, fs=1000)
        for sub_i in range(self.Nsub):
            self.filtdata[sub_i, :, :] = self.timefilter_python(rawdata, sub_i, fs=1000)
            # self.filtdata[sub_i, :, :] = self.timefilter(rawdata, sub_i, fs=1000)

    def apply(self):
        R = np.zeros((self.Nsub,self.template.shape[1]))
        for sub_i in range(self.Nsub):
            R[sub_i, :] = classfierdecide(self.filtdata[sub_i, :, :], self.W[sub_i, :, :], self.template[sub_i, :, :, :])
        self.result = np.matmul(self.a[0, 0:self.Nsub], R)

class algorithmthread():

    def __init__(self,w,template,addloop,datalength):
        self.sendresult = 0
        self.classfier = TRCA(w,template, Nsub=2)
        self.addloop = addloop
      
        #-----------脑电部分---------#
        self.testdata = np.array([0])
        self.resultCount = np.zeros((template.shape[1],addloop))
        self.currentPtr = 0
        self.resultPtr = 0
        self.addloop = addloop

        #-----------------------#
        self.sock_client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.controlCenterAddr = ('169.254.26.10', 7810)                     # ('127.0.0.1',8847)
        self.controlCenterAddr1 = ('169.254.26.10', 7788)
        self.result = 0
        self.all_count = 0
        self.right_count = 0
        self.target = 0

        #--------眼动部分---------#
        self.EyeTracker = EyeOnlineProcess(datalength)
        self.eye_testdata = np.array([0])
        self.labels = 0
        self.results = np.zeros((3,36))
        self.result_count = 0


    def recvData(self,rawdata,label):
        '''接收待处理的脑电数据'''
        self.testdata = np.copy(rawdata)
        self.target = np.mod(label-1, 6)    #标签对应的脑电结果
        self.labels = label-1              # 标签值 0-17


    def RecvEyeData(self,rawdata):
        '''获取待处理的眼动数据'''
        self.eye_testdata = rawdata

    def run(self):
        self.classfier.transform(self.testdata)    # 滤波处理
        self.classfier.apply()                     # 获取相关系数
        self.appendResult(self.classfier.result)
        self.SenderRsult()                         # 发送决策结果
        # print(self.BrainResultDecide())
        self.clearTestdata()                       # 清除数据缓存


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
        eye_result = self.EyeTracker.ResultEyeDecide(self.eye_testdata)
        return eye_result


    def SenderRsult(self):
        '''发送决策结果'''
        # 脑电的决策结果：0-3
        # 眼动的决策结果：1-8  如果决策错误 0
        # brain_result = self.BrainResultDecide()
        eye_result = self.EyeResultDecide()
        brain_result = self.BrainResultDecide()
        print("脑电:{0}, 眼动:{1}".format(brain_result,eye_result))

        '''计算正确率'''
        if eye_result != 0:
            msg = bytes(str(int((eye_result-1)*4 + brain_result + 1)), "utf8")
            self.sock_client.sendto(msg, self.controlCenterAddr)
            self.right_count = self.right_count + 1
            self.all_count = self.all_count + 1
        else:
            print("决策错误！")
            self.sock_client.sendto(bytes(str(40), "utf8"),self.controlCenterAddr)
            self.all_count = self.all_count + 1

    def sendCommand(self, command):
        msg = bytes(str(command), "utf8")
        print('the result is :{}'.format(msg))
        self.sock_client.sendto(msg, self.controlCenterAddr)


    def clearTestdata(self):
        self.testdata = np.array([0])
        self.eye_testdata = np.array([0])



