import numpy as np
from scipy import signal
import mne
import scipy.io as scio
import heapq

def ITR(n, p, t):
    if p == 1:
       itr = np.log2(n) * 60 / t
    else:
        itr = (np.log2(n) + p*np.log2(p) + (1-p)*np.log2((1-p)/(n-1))) * 60 / t

    return itr

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

def load_data(datapath, stimlength):
    '''
    read .cnt to .mat
    :param datapath:
    :return:
    '''
    raw = mne.io.read_raw_cnt(datapath, eog=['HEO', 'VEO'], emg=['EMG'], ecg=['EKG'], preload=True, verbose=False)
    # raw.resample(250)
    num = 255
    mapping_ssvep = dict()
    for idx_command in range(1, num+1):
        mapping_ssvep[str(idx_command)] = idx_command
    events_ssvep, events_ids_ssvep = mne.events_from_annotations(raw, event_id=mapping_ssvep)
    data, times = raw[:]
    fs = 1000
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
            epochdata[:, :, uniquetrigger[trigger_i]-1, j] = data[:, int(currenttriggerpos[j]+latency*fs+2):int(currenttriggerpos[j]+latency*fs+stimlength*fs+2)]
    # if fs == 1000:
    # epochdata = signal.resample(epochdata, 125, axis=1)

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
        fiteredData= signal.filtfilt(f_b, f_a, notchdata, axis=1, padlen=3*(max(len(f_b), len(f_a))-1))
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
        filterdata = signal.sosfiltfilt(sos_system, raweeg, axis=1)

        return filterdata

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

class TRCA(Preprocess):

    def __init__(self, fs, num_fbs, num_targs):
        Preprocess.__init__(self, filterModelName='./data/'+'IIR_filterModel.mat')
        self.fs = fs
        self.num_fbs = num_fbs
        self.num_targs = num_targs
        self.fb_coefs = (np.array([i for i in range(1, num_fbs + 1)])).reshape(1, num_fbs) ** (-1.25) + 0.25

    def trca(self, X):
        '''
        TRCA spacial filter
        :param X: train eeg data, ndarray(channel, samples, trials)
        :return: V[:, 0]: TRCA spacial filter, ndarray(channel, )
        '''

        # zero means
        values_mean = X.mean(axis=1, keepdims=True)
        X = (X - values_mean)  # /Xin_std

        n_chans = X.shape[0]
        n_trial = X.shape[2]
        S = np.zeros((n_chans, n_chans))

        for trial_i in range(n_trial):
            for trial_j in range(n_trial):
                x_i = X[:, :, trial_i]
                x_j = X[:, :, trial_j]
                S = S + np.dot(x_i, x_j.T)
        X1 = X.reshape([n_chans, -1])
        X1 = X1 - np.mean(X1, axis=1).reshape((n_chans, 1))
        Q = np.dot(X1, X1.T)

        # TRCA eigenvalue algorithm
        [W, V] = np.linalg.eig(np.linalg.solve(Q, S))

        return V[:, 0]

    def train_trca(self, traineeg):
        '''
        Obtain the trca spacial filter and template
        :param traineeg: train eeg data, ndarray(channel, samples, trials, blocks)
        :return:
        templates: eeg templates, ndarray(num_targ, num_fbs, channel, samples)
        wn: trca spacial filter, ndarray(num_fbs, num_targ, channel)
        '''

        num_chans = traineeg.shape[0]
        num_samples = traineeg.shape[1]
        num_targs = traineeg.shape[2]
        num_blocks = traineeg.shape[3]
        templates = np.zeros((self.num_fbs, num_targs, num_chans, num_samples))
        wn = np.zeros((self.num_fbs, num_targs, num_chans))

        for targ_i in range(num_targs):
            eeg_tmp = traineeg[:, :, targ_i, :]
            # eeg_tmp = self.notch_filter_python(eeg_tmp, self.fs)
            for fb_i in range(self.num_fbs):
                eeg_tmp = self.timefilter_python(eeg_tmp, self.fs, fb_i)
                # eeg_tmp = self.timefilter(eeg_tmp, fb_i)
                templates[fb_i, targ_i, :, :] = np.mean(eeg_tmp, axis=2)
                wn[fb_i, targ_i, :] = self.trca(eeg_tmp)[None, :]

        return templates, wn

    def test_trca(self, testeeg, templates, wn):
        '''
        classify one single test trial

        Parameters
        ----------
        testeeg : ndarray(channel, samples)
            one single test trial.
        templates : ndarray(num_targ, num_fb, channel, samples)
            all targets' template.
        wn : ndarray(num_fb, num_targ, channel)
            all targets' trca w.

        Returns
        -------
        rho : ndarray(1, num_targs)
            correlation coefficient.
        results : int--> result
            classify result.

        '''
        # testeeg = testeeg - np.mean(testeeg, axis=1, keepdims=True)
        r = np.zeros((self.num_fbs, self.num_targs))
        # testeeg = self.notch_filter_python(testeeg, self.fs)
        for fb_i in range(self.num_fbs):
            testdata = self.timefilter_python(testeeg, self.fs, fb_i)
            # testdata = self.timefilter(testeeg, fb_i)
            for targ_i in range(self.num_targs):
                template = templates[fb_i, targ_i, :, :]
                # w = wn[fb_i, targ_i, :]
                w = wn[fb_i, :, :]
                r[fb_i, targ_i] = compute_corr2(testdata.T.dot(w.T), template.T.dot(w.T))
        rho = np.matmul(self.fb_coefs, r)
        results = np.argmax(rho)
        return rho, results


def test_acc(Test_TRCA, eeg, num_targs):
    '''
    test classifical algorithm's performance

    Parameters
    ----------
    Test_TRCA:
    eeg : ndarray(channel, samples, trials, blocks)
        eeg data.
    fs : int .i.g 250Hz
        the sample rate.
    num_fbs : int
        filter number.
    num_targs : int
        Nf frequency.

    Returns
    -------
    ACCs : list
        accuracy corresponding to every trial.

    '''
    num_blocks = eeg.shape[3]
    trainblock = list(range(num_blocks))
    original_trainblock = list(range(num_blocks))
    ACCs = list()
    ITRs = list()
    for block_i in range(num_blocks):
        N_correct = 0
        trainblock.remove(block_i)
        traineeg = eeg[:, :, :, trainblock]
        testeeg = eeg[:, :, :, block_i]

        templates, wn = Test_TRCA.train_trca(traineeg)
        num_trials = testeeg.shape[2]
        for trial_i in range(num_trials):
            rho, result = Test_TRCA.test_trca(testeeg[:, :, trial_i], templates, wn)
            # print('The trial {0} : {1}'.format(trial_i, result))
            if result == trial_i:
                N_correct += 1
        acc = N_correct / num_trials
        itr = ITR(num_targs, acc, 1.5)
        ACCs.append(acc)
        ITRs.append(itr)
        trainblock = list(np.copy(original_trainblock))

    return ACCs, ITRs


def test_acc_over_trial(Test_TRCA, eeg, num_targs, overlap_trials):
    '''
    test classifical algorithm's performance

    Parameters
    ----------
    Test_TRCA:
    eeg : ndarray(channel, samples, trials, blocks)
        eeg data.
    fs : int .i.g 250Hz
        the sample rate.
    num_fbs : int
        filter number.
    num_targs : int
        Nf frequency.

    Returns
    -------
    ACCs : list
        accuracy corresponding to every trial.

    '''
    num_blocks = 6
    blocks = 30
    trainblock = list(range(blocks))
    original_trainblock = list(range(blocks))
    count = 0
    right_trials_count = 0
    total_trial_num = (eeg.shape[3]*eeg.shape[2]/5)*(6-overlap_trials)
    biggest_coefs_mat = []
    label_mat = list()
    ACCs = list()
    ITRs = list()
    for block_i in range(num_blocks-1):
        N_correct = 0
        removetrials = range(block_i*num_blocks, (block_i+1)*num_blocks)
        trainblock.remove(removetrials)
        traineeg = eeg[:, :, :, trainblock]
        testeeg = eeg[:, :, :, removetrials]

        templates, wn = Test_TRCA.train_trca(traineeg)

        for target_itr in range(testeeg.shape[2]):
            coef_mat = np.zeros((testeeg.shape[2], overlap_trials))
            for test_iter in range(testeeg.shape[3]):
                data_epoch = testeeg[:, :, target_itr, test_iter]
                trial_temp = np.mod(test_iter+1, overlap_trials)
                rho, result = Test_TRCA.test_trca(data_epoch, templates, wn)
                coef_mat[:, trial_temp] = rho

                if test_iter >= overlap_trials-1:
                    decision_vector = np.sum(coef_mat, axis=1)
                    count += 1
                    if np.argmax(decision_vector) == target_itr:
                        right_trials_count += 1
                        label_mat.append(1)
                    else:
                        label_mat.append(0)
                    biggest_desion_value = np.array(heapq.nlargest(2, decision_vector))
                    biggest_coefs_mat.append(biggest_desion_value)

        offline_acc = right_trials_count/total_trial_num*100
        return offline_acc


def simu_online(Test_TRCA, datapath, W, Template):

    raw = mne.io.read_raw_cnt(datapath, eog=['HEO', 'VEO'], emg=['EMG'], ecg=['EKG'], preload=True, verbose=False)
    # raw.pick_channels(['P7', 'P5', 'P3', 'P1', 'PZ',
    #             'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ',
    #             'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2'])
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
        # epochdata = signal.resample(epochdata, 125, axis=1)
        rho, result = Test_TRCA.test_trca(epochdata, Template, W)
        print('the trial  {0}, result: {1}'.format(trial_i, result))
        if result == truelabel[trial_i]:
            N_correct += 1
    acc = N_correct/Ntrials

    return acc


if __name__ == "__main__":
    # fs = 250
    fs = 1000
    num_fbs = 2
    num_targs = 2
    stimlength=0.5
    Test_TRCA = TRCA(fs, num_fbs, num_targs)

    datapath1 = 'E:/wyl/BE测试实验/1120/offline.cnt'
    # datapath2 = 'E:/wyl/BE测试实验/0820/offline3.cnt'
    eeg1 = load_data(datapath1, stimlength)
    # eeg2 = load_data(datapath2, stimlength)
    # eeg = np.concatenate((eeg1, eeg2), axis=3)

    ACCs, ITRs = test_acc(Test_TRCA, eeg1, num_targs)
    Ave_ACC, Ave_ITR = np.array(ACCs).mean(), np.array(ITRs).mean()
    print('Ave_ACC: {0}, Ave_ITR: {1}'.format(Ave_ACC, Ave_ITR))

    templates, wn = Test_TRCA.train_trca(eeg1)
    np.save('./data/Template.npy', templates)
    np.save('./data/W.npy', wn)

    # testdatapath = 'E:/wyl/BE测试实验/08233/online1.cnt'
    # Test_acc = simu_online(Test_TRCA, testdatapath,  wn, templates)
    # print('The simulation online ACC is {}'.format(Test_acc))

