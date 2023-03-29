import numpy as np
from threading import Thread, currentThread
import scipy.io as scio
import json
import copy
import matplotlib
from matplotlib import pyplot as plt
from scipy.stats import norm
import math
from scipy import stats
from algorithmInterface import Kalman_filter
def liner_insert(fixations):
    '''
    线性插值
    :param fixations:list
    :return:fixations:list
    '''
  # 小于interval_num 需要内插
    interval_num = (100*10)/1000
    fix_len = len(fixations)
    # FT定义为开始 TF定义为结束
    start_idx, end_idx = [], []
    a = np.nonzero(fixations)
    a = np.array(a)
    for ii in range(a[0,0],fix_len-2):
        if (fixations[ii] != 0) & \
            (fixations[ii+1] == 0) :
            start_idx.append(ii)
        if (fixations[ii] == 0) & \
            (fixations[ii+1] != 0):
            end_idx.append(ii)
    for start, end in zip(start_idx, end_idx):
        # if start > end:
        #     start ,end = end, start
        nan_len = end-start
        if nan_len >= interval_num:
            px = [fixations[start], fixations[end+1]]
            interx = ((px[1]-px[0])*np.arange(nan_len+1)/float(nan_len+1)+px[0]).tolist()
            for ii in range(1, len(interx)):
                fixations[start+ii] = interx[ii]

    return fixations


def cut_data(data, event_data, length):
    '''
    截取眼动数据
    :param data: data[] ,event_data:[]
    :return: epoch:{}
    '''
    epoch = {}
    uniquetrigger = np.unique(event_data[:,0])    # 不同的标签值
    triggertype = data[:,3]                       # 标签导的值
    for tirggernum in uniquetrigger:
        currentpos = np.argwhere(triggertype == tirggernum)
        epoch_data = np.zeros((length*len(currentpos),2))
        for trailnum in range(len(currentpos)):
            eye_data = data[currentpos[trailnum,0]+1:currentpos[trailnum,0]+length+1,0:2]
            eye_data[:,0] = liner_insert(eye_data[:,0])
            eye_data[:,1] = liner_insert(eye_data[:,1])
            epoch_data[trailnum*length:length*(trailnum+1),:]= eye_data
            #线性插值
        epoch[str(int(tirggernum))] = epoch_data
    return epoch


class Kalman_filter:
    def __init__(self, epoch, event_data, length, fs):
        self.data = epoch
        self.event_data = event_data  # 14*2
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

        self.filter_datas = copy.deepcopy(self.data)
        self.pre_datas = copy.deepcopy(self.data)
        self.data_err = copy.deepcopy(self.data)
        self.distance_xy = {'1':[],'2':[],'3':[],'4':[],'5':[]}       # 正态模型的标准差

        #----
        self.W_screen = 1/2560         #注视点的归一化范围
        self.H_screen = 1/1440
        self.UI_Setting = {}
        self.UScreen_position = []
        self.MScreen_position = []

        self.stim_position_U = []
        self.stim_position_M = []

        self.S = {'1':[],'2':[],'3':[],'4':[],'5':[]}       # 正态模型的标准差
        self.U = {'1':[],'2':[],'3':[],'4':[],'5':[]}       # 正态模型的均值

        self.max_gap_length=75
        self.fs = fs


    def recv_UI_Json(self,stim_position):
            '''接收界面配置Json文件'''
            self.UI_Setting = stim_position

            '''获取当前配置文件中包含每个屏的信息'''
            for key in self.UI_Setting :
                if key[0] == 'U':
                    self.UScreen_position.append(key)
                elif key[0] == 'M':
                    self.MScreen_position.append(key)
            '''获取每个屏幕上刺激块的位置'''
            for num_U in range(len(self.UScreen_position)):  # 1,2  ["",X,Y,Size]
                self.stim_position_U.append([0.5+self.UI_Setting[self.UScreen_position[num_U]][2]*self.W_screen,
                                             1-self.UI_Setting[self.UScreen_position[num_U]][3]*self.H_screen])

            for num_M in range(len(self.MScreen_position)):  # 1,2  ["",X,Y,Size]
                self.stim_position_M.append([0.5+self.UI_Setting[self.MScreen_position[num_M]][2]*self.W_screen,
                                             0-self.UI_Setting[self.MScreen_position[num_M]][3]*self.H_screen])


    def filter(self):
        '''
        @ 对数据进行卡尔曼滤波
        :return:filter_data
        '''
        for key in self.data.keys():
            pre_data = self.X
            P = self.P
            K = self.K

            for i in range(self.data[key].shape[0]):
                # pre_index = self.normal_model1(self.data[key][i,0],self.data[key][i,1])
                # pre_data_xy = np.mat(self.U[str(pre_index)][0] + [0,0])
                # pre_data = pre_data_xy.T
                # c = list((self.data[key][i,:] - self.err)[0,:])
                # pre_data = np.mat(c+[c[0]-self.U[key][0][0],c[1]-self.U[key][0][1]]).T
                self.pre_datas[key][i,:] = self.data[key][i,:] - self.err
                measure_data = np.mat(self.pre_datas[key][i,:]).T # 测量值
                P = np.dot(np.dot(self.A,P),self.A.T) + self.Q
                a = np.dot(np.dot(self.H,P),self.H.T) + self.R
                b = np.dot(P,self.H.T)
                K = np.dot(b,np.linalg.inv(a))
                filter_data = pre_data + np.dot(K,measure_data-np.dot(self.H,pre_data))
                pre_data = filter_data

                self.filter_datas[key][i,:] = np.array(filter_data[0:2,0].T)
                P = np.dot(np.eye(4)-np.dot(K,self.H),P)


    def xy_bias_estimate(self):
        '''
        @ 获取系统的过程误差
        :return: Q
        '''
        epoch_datas =  np.array([[] for i in range(2)]).T  # 创建二维空列表  多行，两列
        x_data_var = []
        y_data_var = []
        x_data_bias = []
        y_data_bias = []
        for key in self.data.keys():
            # epoch_datas = np.vstack((epoch_datas,self.data[key]))
            epoch_datas = self.data[key]
            x_data_var.append(np.std(epoch_datas[:,0]))
            y_data_var.append(np.std(epoch_datas[:,1]))

            for i in range(epoch_datas.shape[0]-1):
                x_data_bias.append((epoch_datas[i+1,0]-epoch_datas[i,0])/self.t)
                y_data_bias.append((epoch_datas[i+1,1]-epoch_datas[i,1])/self.t)

        # self.Q[0,0] = np.mean(x_data_var)
        # self.Q[1,1] = np.mean(x_data_var)
        # self.Q[2,2] = np.std(x_data_bias)
        # self.Q[3,3] = np.std(y_data_bias)
        self.Q[0,0] = 0.0001
        self.Q[1,1] = 0.0001
        self.Q[2,2] = 0.0001
        self.Q[3,3] = 0.0001
        c = 1
        np.save('Q',self.Q)


    def var_com(self,mean,data,axis=0):
        '''
        计算方差
        :param mean:
        :param data:
        :return: var
        '''
        sum = 0
        data_length = data.shape[axis]
        for i in range(data_length):
            sum = sum + (data[i] - mean)*(data[i] - mean)

        return sum/data_length


    def measurement_bias_estimate(self):
        '''
        @获取测量误差
        :return: R
        '''
        eye_data = copy.deepcopy(self.data)
        x_var = 0
        y_var = 0
        err_arr_x = []
        err_arr_y = []
        if len(self.stim_position_U) != 0: # 上屏标定
            i = 0
            for key in eye_data.keys():
                x_s =self.var_com(self.stim_position_U[i][0],eye_data[key][:,0],axis=0)
                y_s =self.var_com(self.stim_position_U[i][1],eye_data[key][:,1],axis=0)
                eye_data[key][:,0] = eye_data[key][:,0]- self.stim_position_U[i][0]
                eye_data[key][:,1] = eye_data[key][:,1]- self.stim_position_U[i][1]
                err_arr_x.append(np.mean(eye_data[key][:,0],axis=0))
                err_arr_y.append(np.mean(eye_data[key][:,1],axis=0))
                x_var = x_var + x_s
                y_var = y_var + y_s
                self.S[key].append([x_s,y_s])
                self.U[key].append(self.stim_position_U[i])
                i = i+1
        elif len(self.stim_position_M) != 0: # 下屏标定
            i = 0
            for key in eye_data.keys():
                x_s =self.var_com(self.stim_position_M[i][0],eye_data[key][:,0],axis=0)
                y_s =self.var_com(self.stim_position_M[i][1],eye_data[key][:,1],axis=0)
                eye_data[key][:,0] = eye_data[key][:,0]- self.stim_position_M[i][0]
                eye_data[key][:,1] = eye_data[key][:,1]- self.stim_position_M[i][1]
                err_arr_x.append(np.mean(eye_data[key][:,0],axis=0))
                err_arr_y.append(np.mean(eye_data[key][:,1],axis=0))

                x_var = x_var + x_s
                y_var = y_var + y_s
                self.S[key].append([x_s,y_s])
                self.U[key].append(self.stim_position_M[i])
                i = i+1
        # epoch_datas =  np.array([[] for i in range(2)]).T  # 创建二维空列表  多行，两列
        # for key in eye_data.keys():
        #     epoch_datas = np.vstack((epoch_datas,eye_data[key]))
        #
        # R_left = epoch_datas[:,0]
        # R_right = epoch_datas[:,1]
        self.R[0,0] = x_var/5
        self.R[1,1] = y_var/5
        self.err[0,0] = np.mean(err_arr_x)
        self.err[0,1] = np.mean(err_arr_y)
        np.save('R',self.R)
        np.save('err',self.err)


    def normal_distribution(self,x,mean,var):
        '''
        @计算标准正太分布模型概率
        :return:P
        '''
        z = (x-mean)/var
        if x > mean:   # 如果参数x大于x_u
            P = 1 - norm.cdf(x,mean,var)
        else:
            P = norm.cdf(x,mean,var)
        return P


    def fx_fuction(self,mean_x,std_x,mean_y,std_y):
        x = np.linspace(mean_x-100,mean_x+100,num=1000)
        y = np.linspace(mean_y-100,mean_y+100,num=1000)
        fx = 1 / (std_x * pow(2 * math.pi, 0.5)) * np.exp(-((x - mean_x) ** 2) / (2 * std_x ** 2))
        fy = 1 / (std_y * pow(2 * math.pi, 0.5)) * np.exp(-((y - mean_y) ** 2) / (2 * std_y ** 2))
        # 多条曲线在同一张图上进行对比
        # plt.plot(x, fx,label = '均值 = {}, 标准差={}'.format(mean_x,std_x))  # 绘制概率密度函数图像
        # plt.plot(y,fy,label = '均值 = {}, 标准差={}'.format(mean_y,std_y))
        # plt.legend() # 显示标签 label
        # plt.xlabel("数值")
        # plt.ylabel('数值的概率')
        # plt.title('服从正太分布的概率密度图')
        # plt.show()  # 显示图像




    def normal_model(self):
        '''
        @构建卡尔曼滤波后数据的正态分布模型,计算目标位置的概率
        :return: [P1(x),P2(x),...,Pn(x)]
        '''
        # P = []
        # filter_data  = self.filter(Eyedata)
        # x_mean = np.mean(filter_data[:,0]/self.W_screen,axis=0)
        # y_mean = np.mean(filter_data[:,1]/self.H_screen,axis=0)
        #
        # for pos in len(stim_pos):
        #     x = stim_pos[pos][0]/self.W_screen
        #     y = stim_pos[pos][1]/self.H_screen
        #     dis_xy = math.sqrt((x-x_mean)*(x-x_mean)+(y-y_mean)*(y-y_mean))
        #     P[pos] = dis_xy
        P = {'1':[],'2':[],'3':[],'4':[],'5':[]}
        i = 0
        for key in self.filter_datas.keys():
            # self.data_err[key][0:50,0] = (self.filter_datas[key][0:50,0] - self.U[key][0][0])*2560
            # self.data_err[key][0:50,1] = (self.filter_datas[key][0:50,1] - self.U[key][0][1])*1440
            x_mean = np.mean(self.filter_datas[key][0:50,0]*2560,axis=0)
            x_std = np.std(self.filter_datas[key][0:50,0]*2560,axis=0)
            y_mean = np.mean(self.filter_datas[key][0:50,1]*1440,axis=0)
            y_std = np.std(self.filter_datas[key][0:50,1]*1440,axis=0)
            # for j in range(50):
            #     distance = self.data_err[key][j,0]*self.data_err[key][j,0] + self.data_err[key][j,1]*self.data_err[key][j,1]
            #     self.distance_xy[key].append(math.sqrt(distance))

            # dis_mean = np.mean(self.distance_xy[key])
            # dis_var = np.var(self.distance_xy[key])
            self.fx_fuction(x_mean,x_std,y_mean,y_std)
            for pos in self.U.keys():
                x = self.U[pos][0][0]*2560
                y = self.U[pos][0][1]*1440
                dis_xy = math.sqrt((x-x_mean)*(x-x_mean)+(y-y_mean)*(y-y_mean))
                # if dis_xy > dis_mean:
                #     p_dis = stats.norm(dis_mean,dis_var).cdf(dis_mean-(dis_xy-dis_mean))
                # else:
                #     p_dis = stats.norm(dis_mean,dis_var).cdf(dis_xy)
                # px = self.normal_distribution(self.U[pos][0][0]*2560,x_mean,x_var)
                # py = self.normal_distribution(self.U[pos][0][1]*1440,y_mean,y_var)
                P[key].append(dis_xy)
            i = i + 1

        pass
        # P = {'1':[],'2':[],'3':[],'4':[],'5':[]}
        # i = 0
        # for key in self.filter_datas.keys():
        #     self.data_err[key][0:50,0] = (self.filter_datas[key][0:50,0] - self.U[key][0][0])*2560
        #     self.data_err[key][0:50,1] = (self.filter_datas[key][0:50,1] - self.U[key][0][1])*1440
        #     x_mean = np.mean(self.filter_datas[key][0:50,0]*2560,axis=0)
        #     y_mean = np.mean(self.filter_datas[key][0:50,1]*1440,axis=0)
        #     for j in range(50):
        #         distance = self.data_err[key][j,0]*self.data_err[key][j,0] + self.data_err[key][j,1]*self.data_err[key][j,1]
        #         self.distance_xy[key].append(math.sqrt(distance))
        #
        #     dis_mean = np.mean(self.distance_xy[key])
        #     dis_var = np.var(self.distance_xy[key])
        #
        #     for pos in self.U.keys():
        #         x = self.U[pos][0][0]*2560
        #         y = self.U[pos][0][1]*1440
        #         dis_xy = math.sqrt((x-x_mean)*(x-x_mean)+(y-y_mean)*(y-y_mean))
        #         if dis_xy > dis_mean:
        #             p_dis = stats.norm(dis_mean,dis_var).cdf(dis_mean-(dis_xy-dis_mean))
        #         else:
        #             p_dis = stats.norm(dis_mean,dis_var).cdf(dis_xy)
        #         # px = self.normal_distribution(self.U[pos][0][0]*2560,x_mean,x_var)
        #         # py = self.normal_distribution(self.U[pos][0][1]*1440,y_mean,y_var)
        #         P[key].append(p_dis)
        #     i = i + 1



    def draw_figure(self,samplenum,pos):
        X = list(range(0,samplenum))
        base_x = (np.ones((1,samplenum))*self.U[pos][0][0])[0]
        base_y = (np.ones((1,samplenum))*self.U[pos][0][1])[0]

        left_filter = self.filter_datas[pos][:,0].T
        left_pre = self.pre_datas[pos][:,0].T
        left_data = self.data[pos][:,0].T
        right_filter = self.filter_datas[pos][:,1].T
        right_data = self.data[pos][:,1].T
        right_pre = self.pre_datas[pos][:,1].T
        #设置图片的大小
        matplotlib.rc('figure', figsize = (14, 7)) #单位为厘米
        #设置字体的大小
        matplotlib.rc('font', size = 14) #size为字体的大小
        #是否显示背景网格
        matplotlib.rc('axes', grid = False)
        #grid：取为Flase为不显示背景网格，True为显示
        #背景颜色
        matplotlib.rc('axes', facecolor = 'white')
        #数据及线属性
        plt.figure(1)
        plt.subplot(211)
        plt.plot(X, left_filter,'r',label='filter_data')
        plt.plot(X, left_data,'g',label='raw_data')
        plt.plot(X,base_x,'b',label='base')
        plt.plot(X,left_pre,'y',label='remove_err_data')
        #标题设置
        plt.legend()
        plt.title('x')
        plt.xlabel('T')
        plt.ylabel('data')

        plt.subplot(212)
        plt.plot(X, right_filter,'r',label='filter_data')
        plt.plot(X, right_data,'g',label='raw_data')
        plt.plot(X,base_y,'b',label='base')
        plt.plot(X,right_pre,'y',label='remove_err_data')
        plt.legend()
        plt.title('y')
        plt.xlabel('T')
        plt.ylabel('data')
        plt.show()



if __name__ == '__main__':
    #---------读取眼动数据------------#
    fs = 100
    length = int(3.0*fs)
    FilePath = './eye_data/S1024_1.mat'
    EYE = scio.loadmat(FilePath)

    Cal_Settingfile = open('EYE_Offline_Example.json')
    Cal_Position = json.load(Cal_Settingfile)

    eye_data = EYE['eye_data']
    event_data = EYE['event_data']
    epoch = cut_data(eye_data,event_data,length)
    kalman = Kalman_filter(epoch,event_data,length,fs)
    kalman.recv_UI_Json(Cal_Position)
    kalman.xy_bias_estimate()
    kalman.measurement_bias_estimate()
    kalman.filter()
    kalman.normal_model()
    kalman.draw_figure(1800,'1')

    Q = np.load('Q.npy')
    pass






