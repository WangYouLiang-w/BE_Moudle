import numpy as np
from scipy import linalg as sLA
from numpy import ndarray
from scipy.stats import pearsonr
from scipy import signal
import mne
import scipy.io as scio

def ITR(n, p, t):
    if p == 1:
       itr = np.log2(n) * 60 / t
    else:
        itr = (np.log2(n) + p*np.log2(p) + (1-p)*np.log2((1-p)/(n-1))) * 60 / t

    return itr

def Corr(X, Y):
    X = np.reshape(X, (-1))
    Y = np.reshape(Y, (-1))
    rho = pearsonr(X, Y)[0]
    return rho

def load_data(datapath, fs, stimlength):
    '''
    read .cnt to .mat
    :param datapath:
    :return:
    '''
    raw = mne.io.read_raw_cnt(datapath, eog=['HEO', 'VEO'], emg=['EMG'], ecg=['EKG'], preload=True, verbose=False)
    # raw.resample(250)
    # raw.pick_channels(['P7', 'P5', 'P3', 'P1', 'PZ',
    #         'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ',
    #         'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2'])
    num = 255
    mapping_ssvep = dict()
    for idx_command in range(1, num+1):
        mapping_ssvep[str(idx_command)] = idx_command
    events_ssvep, events_ids_ssvep = mne.events_from_annotations(raw, event_id=mapping_ssvep)
    data, times = raw[:]
    fs = fs
    stimlength = stimlength
    latency = 0.14
    Nc = 21
    Ntimes = int(stimlength*fs)

    triggertype = events_ssvep[:, 2]
    triggerpos = events_ssvep[:, 0]
    triggernum = events_ssvep.shape[0]
    uniquetrigger = np.unique(triggertype)
    Ntarget = uniquetrigger.shape[0]
    Nblocks = int(triggernum/Ntarget)

    epochdata = np.zeros((Nc, Ntimes, Ntarget, Nblocks))
    for trigger_i in range(uniquetrigger.shape[0]):
        currenttrigger = uniquetrigger[trigger_i]
        currenttriggerpos = triggerpos[np.where(triggertype==currenttrigger)]
        for j in range(currenttriggerpos.shape[0]):
            epochdata[:, :, uniquetrigger[trigger_i]-1, j] = data[:, int(currenttriggerpos[j]+latency*fs+2):int(currenttriggerpos[j]+stimlength*fs+latency*fs+2)]
    # if fs == 1000:
    #     epochdata = signal.resample(epochdata, 125, axis=1)
    return epochdata

class Preprocess():

    def __init__(self, filterModelName='./data/'+'IIR_filterModel.mat'):
        self.filterModelName = filterModelName

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

    def timefilter(self, notchdata, sub_i):
        self.filterpara = self.loadfilterModel()
        fiteredData = np.zeros((notchdata.shape[0], notchdata.shape[1]))
        f_b = self.filterpara[''.join(('f_b', str(sub_i)))]
        f_a = self.filterpara[''.join(('f_a', str(sub_i)))]
        fiteredData= signal.filtfilt(f_b, f_a, notchdata, axis=-1, padlen=3*(max(len(f_b), len(f_a))-1))
        return fiteredData

    def timefilter_python(self, raweeg, fs, idx_fb):
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
        filterdata = signal.sosfiltfilt(sos_system, raweeg, axis=-1)

        return filterdata

class TDCA(Preprocess):

    def __init__(self, freqs, phases=None, fs=1000, Np=500, Nh=3, l=5, Nfb=3):
        Preprocess.__init__(self, filterModelName='./data/'+'IIR_filterModel.mat')
        self.freqs = freqs
        self.phases = phases
        self.fs = fs
        self.Np = Np
        self.Nh = Nh
        self.l = l
        self.Nfb = Nfb
        self.fb_coefs = (np.array([i for i in range(1, self.Nfb + 1)])).reshape(1, self.Nfb) ** (-1.25) + 0.25

    def generate_cca_references(self):
        if isinstance(self.freqs, int) or isinstance(self.freqs, float):
            freqs = [self.freqs]
        freqs = np.array(self.freqs)[:, np.newaxis]
        if self.phases is None:
            phases = 0
        if isinstance(self.phases, int) or isinstance(self.phases, float):
            phases = [self.phases]
        phases = np.array(self.phases)[:, np.newaxis]
        t = np.linspace(0, (self.Np-1)/self.fs, self.Np)

        Yf = []
        for i in range(self.Nh):
            Yf.append(np.stack([
                np.sin(2*np.pi*(i+1)*freqs*t + np.pi*phases),
                np.cos(2*np.pi*(i+1)*freqs*t + np.pi*phases)], axis=1))
        Yf = np.concatenate(Yf, axis=1)
        return Yf

    def generate_P(self):
        '''
        正交分解
        '''
        Yf = self.generate_cca_references()
        Nf = Yf.shape[0]
        Np = Yf.shape[2]
        self.P = np.zeros((Nf, Np, Np))
        for class_i in range(Nf):
            Q, R = sLA.qr(Yf[class_i, :, :].T, mode='economic')
            self.P[class_i, :, :] = Q@Q.T
        return self.P

    def aug_2(self, X: ndarray, P: ndarray, training: bool=True):
        '''
        获得二阶增广数据
            :param X: ndarray(Nchans, m). m>n_points(training)
            :param P: ndarray(Nsamples, Nsamples), orthogonal projection matrix
            :param training: mode(train or test)
        :return:
            aug_X: ndarray((l+1)Nchans, Nsamples), further secondary augmented data
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

    def tdca(self, train_data):
        '''
        TDCA
            :param train_data: ndarray(Nf, Nt, Nc, m). Training dataset.
        :return:
            w: ndarray (1, (l+1)Nc), the tdca w
            class_center: ndarray (Nf, (l+1)Nc, 2Np), template
        '''

        # basic information
        Nf = train_data.shape[0]
        Nt = train_data.shape[1]
        Nc = train_data.shape[2]
        Np = self.P.shape[-1]

        # secondary augmented data
        X_aug_2 = np.zeros((Nf, Nt, (self.l+1)*Nc, 2*Np))

        for class_i in range(Nf):
            for trial_i in range(Nt):
                X_aug_2[class_i, trial_i, :, :] = self.aug_2(train_data[class_i, trial_i, :, :], self.P[class_i, :, :])

        # between-class difference Hb -> scatter matrix Sb
        class_center = X_aug_2.mean(axis=1)  # (Nf, (l+1)Nc, 2Np)
        total_center = class_center.mean(axis=0)   # ((l+1)Nc, 2Np)
        Hb = class_center - total_center  # (Nf, (l+1)Nc, 2Np)
        Sb = np.einsum('fcp,fhp->ch', Hb, Hb)  # ((l+1)Nc, (l+1)Nc)
        Sb /= Nf

        # within-class difference Hw -> scatter matrix Sw
        Hw = X_aug_2 - class_center[:, None, ...]  # (Ne, Nt, (l+1)Nc, 2Np)
        Sw = np.einsum('ftcp,fthp->ch', Hw, Hw)   # ((l+1)Nc, (l+1)Nc)
        Sw /= (Nf*Nt)

        e_val, e_vec = sLA.eig(sLA.solve(Sw, Sb))
        w_index = np.argmax(e_val)
        w = e_vec[:, [w_index]].T
        return w, class_center

    def filterbank_train(self, train_data):
        '''
        滤波器组
        :param train_data: ndarray(Nf, Nt, Nc, m). Training dataset.
        :return:
        '''
        Nc = train_data.shape[2]
        Nf = train_data.shape[0]
        Np = self.P.shape[-1]
        W = np.zeros((self.Nfb, (self.l+1)*Nc))
        Template = np.zeros((self.Nfb, Nf, (self.l+1)*Nc, 2*Np))
        for fb_i in range(self.Nfb):
            filterdata = self.timefilter_python(train_data, self.fs, fb_i)
            W[fb_i, :], Template[fb_i, :, :, :] = self.tdca(filterdata)
        return W, Template

    def test_tdca(self, testdata, w, template):
        '''
        Test tdca
            :param testdata: ndarray(Nc, Np), test eeg data
            :param w: ndarray(1, (l+1)Nc), tdca W
            :param template: ndarray(Nf, (l+1)Nc, 2Np), Template
            :return:
        '''

        Nf = self.P.shape[0]
        rho = np.zeros((self.Nfb, Nf))
        for fb_i in range(self.Nfb):
            for class_i in range(Nf):
                filter_test = self.timefilter_python(testdata, self.fs, fb_i)
                test_model = np.matmul(w[fb_i, :], self.aug_2(filter_test, self.P[class_i, :, :], training=False))
                rho[fb_i, class_i] = Corr(test_model, np.matmul(w[fb_i, :], template[fb_i, class_i, :, :]))

        r = np.matmul(self.fb_coefs, rho)
        result = np.argmax(r)
        return r, result

def offline_test(rawdata, Test_TDCA, Np):
    Nblocks = rawdata.shape[3]
    trainblock = list(range(Nblocks))
    original_trainblock = list(range(Nblocks))
    ACCs = list()
    for block_i in range(Nblocks):
        N_correct = 0
        trainblock.remove(block_i)
        traindata = np.transpose(rawdata[:, :, :, trainblock], (2, 3, 0, 1))
        testdata = rawdata[:, :Np, :, block_i]
        Ntrials = testdata.shape[-1]
        w, template = Test_TDCA.filterbank_train(traindata)
        for trial_i in range(Ntrials):
            _, result = Test_TDCA.test_tdca(testdata[:, :, trial_i], w, template)
            if result == trial_i:
                N_correct += 1
        acc = N_correct/Ntrials
        ACCs.append(acc)
        trainblock = list(np.copy(original_trainblock))
    return ACCs


if __name__ == "__main__":
    fs = 1000
    stimlength = 1
    validlength = 0.36
    Np = int(validlength*fs)
    l = 3
    Nfb = 2
    Nh = 3

    freqs = [31.0, 32.0, 33.0, 34.0,35.0,36.0]
    phases = list(np.array([0, 1, 0, 1, 0, 1])*np.pi)

    datapath = 'E:/wyl/BE测试实验/1025/offline.cnt'
    rawdata = load_data(datapath, fs, stimlength)

    Test_TDCA = TDCA(freqs, phases, fs, Np, Nh, l, Nfb)
    Test_TDCA.generate_P()

    # offline test
    ACCs = offline_test(rawdata, Test_TDCA, Np)
    AVe_acc = np.array(ACCs).mean()
    print('Ave_ACC: {0}'.format(AVe_acc))

    # save W, Template and P
    traindata = np.transpose(rawdata, (2, 3, 0, 1))
    wn, templates = Test_TDCA.filterbank_train(traindata)
    print('Template of shape:{0}'.format((templates.shape)))
    print('W of shape:{0}'.format((wn.shape)))
    np.save('./data/Template.npy', templates)
    np.save('./data/W.npy', wn)
    np.save('./data/P.npy', Test_TDCA.P)
    print('保存完成！')




