import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LinearRegression

from PandaCANDecoder.signal import Signal
from PandaCANDecoder.message import Message
from PandaCANDecoder.utils import convert_signal, validate_tokenization_method, validate_signedness_method, validate_byte_order
from PandaCANDecoder.plotter import plot_probability_chart, plot_bit_table, plot_ts_signal, plot_ts_signal_multi, plot_ts_signal_comparison


class Decoder:
    '''
    Utilities for decoding CAN data
    '''
    def __init__(self, csv_file_path=None):
        '''
        Decoder constructor

            Parameters:
                csv_file_path (str): File path to csv log
        '''
        self.csv_file_path = csv_file_path
        self.time_series_msg_dir = './time_series_msgs' # TODO: make static
        self.msgs = []

        # get data
        # remove data on panda saftey check buses 128, 129, and 130
        self.all_data = pd.read_csv(self.csv_file_path)
        self.all_data.drop(self.all_data[
                (self.all_data['Bus'] == 128) |
                (self.all_data['Bus'] == 129) |
                (self.all_data['Bus'] == 130)].index, inplace=True)

    def get_message(self, msg_id):
        '''
        Get message object from id

            Parameters:
                msg_id (str): In hex
        '''
        for msg in self.msgs:
            if msg.msg_id == msg_id:
                return msg

    def get_signal(self, msg_id, signal_id):
        '''
        Get signal object from id

            Parameters:
                msg_id (str): In hex
                signal_id (str)
        '''
        msg = self.get_message(msg_id)
        return msg.get_signal(signal_id)

    def generate_msgs(self):
        '''
        Generates PandaCANDecoder.Message() objects for each unique message in CAN file
        '''
        unique_pairings = self.all_data.drop_duplicates(subset=['MessageID', 'PandaNum', 'Bus', 'MessageLength'])
        unique_msg_ids = unique_pairings['MessageID'].drop_duplicates().to_list()

        for msg_id in tqdm(unique_msg_ids, desc="Generating messages".ljust(30)):
            msg_data = unique_pairings[unique_pairings['MessageID'] == msg_id]

            panda_buses = {}
            unique_pandas = sorted(msg_data['PandaNum'].drop_duplicates().to_list())
            for panda in unique_pandas:
                buses = sorted(msg_data[msg_data['PandaNum'] == panda]['Bus'].to_list())
                panda_buses[panda] = buses

            msg_length = msg_data['MessageLength'].iloc[0]

            self.msgs.append(Message(msg_id=msg_id, panda_buses=panda_buses, msg_length=msg_length))

    def print_msgs(self):
        '''
        Prints messages available on each Panda and CAN bus combination.
        '''
        if not self.msgs:
            print("WARNING: No messages have been generated. Use Decoder.generate_msgs() to do so.")
            return

        print("-----------------------------------")
        print("Message ID: {Panda Number: [Buses]}")
        print("-----------------------------------")
        for msg in self.msgs:
            print(msg)

    def generate_msg_ts_data(self, rewrite=False):
        '''
        Saves time series messages as uint64 numpy array to ./time_series_msgs/{msg_id}.npy

            Parameters:
                rewite (bool): If True, will overwrite existing .npy file
        '''
        if not self.msgs:
            print("WARNING: No messages have been generated. Use Decoder.generate_msgs() to do so.")
            return

        # save time series data
        os.makedirs(self.time_series_msg_dir, exist_ok=True)
        for msg in tqdm(self.msgs, desc="Generating message data".ljust(30)):
            msg.generate_ts_data(self.all_data, rewrite)

    def calculate_signals(self, tokenization_method, signedness_method, alpha1=0.01, alpha2=0.5, gamma1=0.2):
        '''
        Calculate predicted signals for all messages
        '''
        for msg in tqdm(self.msgs, desc="Generating signals".ljust(30)):
            self.calculate_signal(msg, tokenization_method, signedness_method, alpha1, alpha2, gamma1)

    def calculate_signal(self, msg, tokenization_method, signedness_method, alpha1=0.01, alpha2=0.5, gamma1=0.2):
        '''
        Calculate signal using multiple methods

        To calculate signal tokenization (boundaries)
        (1) TANG method: "Unsupervised Time Series Extraction from Controller
                Area Network Payloads" by Nolan et. al.
        (2) Conditional Bit Flip method: "CAN-D: A Modular Four-Step Pipeline
                for Comprehensively Decoding Controller Area Network Data" by
                Verma et. al.

        To calculate signedness
        (3) Most Significant Bits method: "CAN-D: A Modular Four-Step Pipeline
                for Comprehensively Decoding Controller Area Network Data" by
                Verma et. al.

            Parameters:
                msg (PandaCANDecoder.Message())
                tokenization_method (str): (1)='tang', (2)='conditional_bit_flip'
                signedness_method (str): (3)='msb_classifier'
                alpha1 (float): hyperparameter for (1) and (2)
                alpha2 (float): hyperparameter for (2)
                gamma1 (float): hyperparameter for (3)
        '''
        # validate methods
        validate_tokenization_method(tokenization_method)
        validate_signedness_method(signedness_method)

        msg.calculate_probability_vectors()

        signals_be = self._tokenize_msg(msg, 'be', tokenization_method, signedness_method, alpha1, alpha2, gamma1)
        signals_le = self._tokenize_msg(msg, 'le', tokenization_method, signedness_method, alpha1, alpha2, gamma1)

        msg.signals = signals_be + signals_le

    def _method_condition(self, method, i, bf_probability, conditional_bf_probability, alpha1, alpha2):
        '''
        Return true if bit index i is at the boundary of a signal

            Parameters:
                method (str)
                i (int): bit index
                bf_probability (np.array): bit flip probability
                conditional_bf_probability (np.array): bit flip probability conditional on previous bit flip
                alpha1 (float)
                alpha2 (float)
        '''
        if method == 'tang':
            return (bf_probability[i] > bf_probability[i+1]+alpha1)
        elif method == 'conditional_bit_flip':
            return (((conditional_bf_probability[i] < alpha1) or (conditional_bf_probability[i+1]-conditional_bf_probability[i] > alpha2)) and bf_probability[i] != 0)


    def _tokenize_msg(self, msg, byte_order, tokenization_method, signedness_method, alpha1, alpha2, gamma):
        '''
        Tokenize a message into signals and calculating signedness

            Parameters:
                byte_order (str)
                see self.calculate_signal()
        '''
        validate_byte_order(byte_order)

        if byte_order == 'be':
            range_start = 0
            range_end = msg.msg_length*8
            bf_probability = msg.bf_probability_be
            conditional_bf_probability = msg.conditional_bf_probability_be
        elif byte_order == 'le':
            range_start = 64 - msg.msg_length*8
            range_end = 64
            bf_probability = msg.bf_probability_le
            conditional_bf_probability = msg.conditional_bf_probability_le

        start_bit = range_start
        length = 1
        signals = []
        signal_name_idx = 0

        # tokenize and create signals
        for i in range(range_start, range_end):
            # At end or message, end of signal determined by method, or end of constant signal
            if (i == range_end-1) or \
                    self._method_condition(tokenization_method, i, bf_probability, conditional_bf_probability, alpha1, alpha2) or \
                    (bf_probability[i] == 0 and bf_probability[i+1] != 0):
                start_bit, _ = convert_signal(start_bit, length, 'inorder', byte_order)
                signal_name = f"SIG_{(byte_order).upper()}_{signal_name_idx}"
                if bf_probability[i] == 0:
                    signal = Signal(start_bit, length, byte_order=byte_order, classification='constant', msg=msg, name=signal_name)
                else:
                    signal = Signal(start_bit, length, byte_order=byte_order, classification='ts', msg=msg, name=signal_name)
                signals.append(signal)
                start_bit = i+1
                length = 1
                signal_name_idx +=1
            else:
                length += 1

        # get signal signedness
        for signal in signals:
            signal.signedness = self._calculate_signedness('msb_classifier', signal, gamma)

        return signals

    def _calculate_signedness(self, method, signal, gamma):
        '''
        Calculate the signedess of a signal. Return string

            Parameters:
                method (str)
                signal (PandaCANDecoder.signal.Signal())
                gamma (float)
        '''
        msg = signal.msg

        if signal.length==1:
            return 'unsigned'

        # get signal data
        ts_signal_data_raw = signal.ts_data_raw

        if method == 'msb_classifier':
            # convert to binary
            ts_bin = (((ts_signal_data_raw[:,None] & (1 << np.arange(signal.length, dtype='uint64')[::-1]))) > 0).astype(int)

            # conditions
            prob_1_0 = np.count_nonzero((ts_bin[:,0]==1) & (ts_bin[:,1]==0))/msg.msg_quantity
            prob_0_1 = np.count_nonzero((ts_bin[:,0]==0) & (ts_bin[:,1]==1))/msg.msg_quantity

            bool_0_0 = (ts_bin[:,0]==0) & (ts_bin[:,1]==0)
            bool_1_1 = (ts_bin[:,0]==1) & (ts_bin[:,1]==1)

            prob_0_0_and_1_1 = np.count_nonzero(bool_0_0[:-1] & bool_1_1[1:])/msg.msg_quantity

            if prob_1_0 + prob_0_1 == 0:
                return 'signed'
            if prob_0_0_and_1_1 == 0:
                return 'unsigned'
            if prob_1_0 + prob_0_1 < gamma:
                return 'signed'
            return 'unsigned'

    def plot_message_from_id(self, msg_id):
        '''
        Plot signal boundaries on bit flip probability bar chart and bit table

            Parameters:
                msg_id (str): In hex
        '''
        msg = self.get_message(msg_id)
        self.plot_message(msg)

    def plot_message(self, msg):
        '''
        Plot signal boundaries on bit flip probability bar chart and bit table

            Parameters:
                msg (PandaCANDecoder.message.Message())
        '''
        fig, ax = plt.subplots(2, 2, figsize=(15,6), gridspec_kw={'width_ratios': [3, 1]} , constrained_layout=True)
        fig.set_constrained_layout_pads(hspace=.1, wspace=0)
        fig.suptitle(f'Message ID: {msg.msg_id}', fontsize=16)
        plot_probability_chart(ax[0][0], msg, 'be')
        plot_probability_chart(ax[1][0], msg, 'le')
        plot_bit_table(ax[0][1], msg, 'be')
        plot_bit_table(ax[1][1], msg, 'le')
        plt.show()

    def plot_signal_from_id(self, msg_id, signal_name):
        '''
        Plot time series signal value

            Parameters:
                msg_id (str): In hex
                signal_name (str)
        '''
        signal = self.get_signal(msg_id, signal_name)
        self.plot_signal(signal)

    def plot_signal(self, signal):
        '''
        Plot time series signal value

            Parameters:
                signal (PandaCANDecoder.signal.Signal())
        '''
        fig, ax = plt.subplots(1, 1, figsize=(10,5), constrained_layout=True)
        plot_ts_signal(ax, signal)
        plt.show()

    def plot_signal_multi(self, signals, autoscale=False):
        '''
        Plot multiple time series signals

            Parameters:
                signal (list[PandaCANDecoder.signal.Signal()])
                autoscale (bool): If true stack signals and autoscale them to be visible
        '''
        fig, ax = plt.subplots(1, 1, figsize=(10,5), constrained_layout=True)
        plot_ts_signal_multi(ax, signals, autoscale)
        plt.show()

    def find_signal_match(self, signal_b, thresh=0.5, print_sig=False, plot=False, start_time = 0.0, end_time = 10**10):
        '''
        Loop through all signals and find matches to input signal

            Parameters:
                signal_b (PandaCANDecoder.signal.Signal())
                thresh (float): print and plot signals with R^2 scores about this threshold
        '''
        for msg in self.msgs:
            for signal_a in msg.signals:
                if signal_a == signal_b:
                    continue
                model, r_sq = self.signal_match(signal_a, signal_b, start_time, end_time)
                if r_sq > thresh:
                    if print_sig:
                        print(signal_a, " | R^2: ", r_sq)
                    if plot:
                        self.plot_signal_match(signal_a, signal_b, model, r_sq)

    def signal_match(self, signal_a, signal_b, start_time, end_time):
        '''
        Get linear regression match between two signals. Return linear regression model and R^2 score

            Parameters:
                signal_a (PandaCANDecoder.signal.Signal())
                signal_b (PandaCANDecoder.signal.Signal())
        '''
        x_time_idx = np.where((signal_a.ts_data_timestamps > start_time) & (signal_a.ts_data_timestamps < end_time))
        y_time_idx = np.where((signal_b.ts_data_timestamps > start_time) & (signal_b.ts_data_timestamps < end_time))
        x = signal_a.ts_data[x_time_idx]
        y = signal_b.ts_data[y_time_idx]

        # correct for different sample rates
        # TODO: should match timestamps between samples however this works well enough
        if x.shape[0] > y.shape[0]:
            idx = np.round(np.linspace(0, len(x) - 1, y.shape[0])).astype(int)
            x = x[idx]
        else:
            idx = np.round(np.linspace(0, len(y) - 1, x.shape[0])).astype(int)
            y = y[idx]

        x = x.reshape((-1, 1))
        model = LinearRegression()
        model.fit(x, y)
        r_sq = model.score(x, y)
        return model, r_sq

    def plot_signal_match(self, signal_a, signal_b, model, r_sq):
        '''
        Plot two signals with linear regression applied

            Parameters:
                signal_a (PandaCANDecoder.signal.Signal())
                signal_b (PandaCANDecoder.signal.Signal())
                model (sklearn.linear_model.LinearRegression())
        '''
        b = model.intercept_
        m = model.coef_[0]

        fig, ax = plt.subplots(1, 1, figsize=(10,5), constrained_layout=True)
        plot_ts_signal_comparison(ax, signal_a, signal_b, m, b, r_sq)
        plt.show()
