import os

import numpy as np
from PandaCANDecoder.utils import convert_signal


class Signal:
    '''
    CAN signal
    '''
    time_series_msg_dir = './time_series_msgs' #TODO: move somewhere

    byte_order_dict = {'be':'0', 'le':'1'}
    byte_order_dict_inv = {'0':'be', '1':'le'}
    signedness_dict = {'unsigned':'+', 'signed':'-'}
    signedness_dict_inv = {'+':'unsigned', '-':'signed'}

    def __init__(self, start_bit, length, byte_order='be', signedness='unsigned', factor=1, offset=0, classification=None, msg=None, name="", min=0, max=0, unit=""):
        '''
        Signal constructor
        '''
        self.start_bit = start_bit
        self.length = length # bits
        self.byte_order = byte_order
        self.signedness = signedness
        self.factor = factor
        self.offset = offset

        self.name = name
        self.unit = unit

        self.classification = classification # constant, ts, counter, checksum

        self.msg = msg

        self.min = min
        self.max = max # 2**self.length # this is hacky, change this

    def __repr__(self):
        return f"Message: {self.msg.msg_id}, Signal: {self.name}"

    @property
    def dbc_str(self):
        return f"SG_ {self.name} : {self.start_bit}|{self.length}@{self.byte_order_dict[self.byte_order]}{self.signedness_dict[self.signedness]} ({self.factor},{self.offset}) [{self.min}|{self.max}] \"{self.unit}\" XXX"

    @property
    def ts_data_timestamps(self):
        return self.msg.ts_data[:,0]  * 10**-9

    @property
    def ts_data_raw(self):
        # trim to signal
        # correct for byte order
        ts_msg = self.msg.ts_data[:,1]

        if self.byte_order == 'be':
            ts_msg = ts_msg << np.uint64(64 - self.msg.msg_length*8)
        elif self.byte_order == 'le':
            ts_msg = ts_msg.byteswap(inplace=False)
            ts_msg = ts_msg >> np.uint64(64 - self.msg.msg_length*8)

        start_idx, _ = convert_signal(self.start_bit, self.length, self.byte_order, 'inorder')

        bit_shift = 64 - (start_idx + self.length)
        mask = np.array((1 << self.length) - 1, dtype='uint64')
        ts_msg_shifted = ts_msg >> np.uint64(bit_shift)
        ts_msg_masked_shifted = ts_msg_shifted & mask

        return ts_msg_masked_shifted

    @property
    def ts_data(self):
        # correct for signedness
        ts_signal_data_raw = self.ts_data_raw

        if self.signedness == 'signed':
            sign_change_bit_shift = 64 - self.length
            ts_signal_data = (ts_signal_data_raw<<sign_change_bit_shift).astype(np.int64)>>sign_change_bit_shift
        else:
            ts_signal_data = ts_signal_data_raw

        # correct for scale and offset
        ts_signal_data = ts_signal_data * self.factor + self.offset

        return ts_signal_data

    def first_derivative_ts_data(self, sampling_rate=1):
        # return d(ts_data)/dt and time stamps

        ts_data_filtered = self.ts_data[0::sampling_rate]
        time_data_filtered = self.ts_data_timestamps[0::sampling_rate]

        return np.gradient(ts_data_filtered, time_data_filtered), time_data_filtered
