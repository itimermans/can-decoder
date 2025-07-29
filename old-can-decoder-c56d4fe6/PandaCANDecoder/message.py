import os
import numpy as np

class Message:
    '''
    CAN message

        Attributes:
            msg_id (string): hex value
            pandas (dict{int: list[int]})
            buses (list[int])
            msg_length (int): bytes
            msg_quantity (int)
            tang (np.array[int])
            msg_ts_data (np.array[uint64])
            signal_boundaries (list[int])
    '''
    time_series_msg_dir = './time_series_msgs' # TODO: change

    def __init__(self, msg_id, panda_buses, msg_length):
        self.msg_id = msg_id
        self.panda_buses = panda_buses
        self.msg_length = msg_length # bytes
        self.msg_quantity = None

        self.bf_probability_be = None
        self.bf_probability_le = None

        self.conditional_bf_probability_be = None
        self.conditional_bf_probability_le = None

        self.name = ""
        self.signals = None

        self.ts_data = None

    def __repr__(self):
        '''
        Override print string
        '''
        return f"Message {self.msg_id}: {self.panda_buses}"

    @property
    def dbc_str(self):
        return f"BO_ {int(self.msg_id, 16)} {self.name} : {self.msg_length} XXX"

    def get_signal(self, signal_id):
        for signal in self.signals:
            if signal.name == signal_id:
                return signal

    def generate_ts_data(self, all_data, rewrite):
        '''
        Generate time series message data and set to self.ts_data. Save data to
        numpy file for easy recall if it does not exist

            Parameters:
                all_data (Pandas.dataframe)
        '''
        data_path = os.path.join(self.time_series_msg_dir, self.msg_id + '.npy')

        # if data file does not exist
        if not os.path.exists(data_path) or rewrite:
            panda = list(self.panda_buses.keys())[0]
            bus = self.panda_buses[panda][0]

            time_series_msg = all_data[
                    (all_data['PandaNum'] == panda) &
                    (all_data['Bus'] == bus) &
                    (all_data['MessageID'] == self.msg_id)][['TimeStampNs', 'Message']]
            time_series_msg['Message'] = time_series_msg['Message'].apply(int, base=16)

            np.save(data_path, time_series_msg.to_numpy(dtype='uint64'))

        self.ts_data = np.load(data_path)

    def _calculate_bf_probability(self, ts_msg, byte_order):
        '''
        Caclulated probability that each bit within a message flipped during the log.
        Return bit flip probability vector and binary transition matrix

            Parameters:
                ts_msg (np.array): time series data corrected for byte order
                byte_order (str): ['be', 'le']
        '''
        # B(t_i) XOR B(t_i+1)
        transition_matrix_raw = ts_msg[:-1] ^ ts_msg[1:]
        # previously 8 was msg.msg_length
        transition_matrix_bin = (((transition_matrix_raw[:,None] & (1 << np.arange(self.msg_length*8, dtype='uint64')[::-1]))) > 0).astype(int)
        if byte_order == 'be':
            transition_matrix_bin = np.pad(transition_matrix_bin, ((0,0), (0, 64-transition_matrix_bin.shape[1])))
        elif byte_order == 'le':
            transition_matrix_bin = np.pad(transition_matrix_bin, ((0,0), (64-transition_matrix_bin.shape[1], 0)))

        # probability of bit flip
        # [P(F_i), P(F_i+1), ...]
        if self.msg_quantity > 1:
            bf_probability = (transition_matrix_bin.sum(axis=0) / (self.msg_quantity-1))
        else:
            bf_probability = transition_matrix_bin.sum(axis=0) / 1

        return bf_probability, transition_matrix_bin

    def _calculate_conditional_bf_probability(self, bf_probability, transition_matrix_bin):
        '''
        Calculate the probability that bit will flip condition on the previous bit flipping.
        Return probability vector

            Parameters:
                bf_probability (np.array)
                transition_matrix_bin (np.array)
        '''
        # F_i AND F_i+1
        transition_matrix_bin_shifted = np.pad(transition_matrix_bin, ((0,0),(0,1)))[:,1:]
        transition_matrix_and = ((transition_matrix_bin+transition_matrix_bin_shifted) > 1).astype(int)

        # [P(F_i AND F_i+1), P(F_i+1 AND F_i+2), ..., 0]
        if self.msg_quantity > 1:
            and_bf_probability = (transition_matrix_and.sum(axis=0) / (self.msg_quantity-1))
        else:
            and_bf_probability = transition_matrix_bin.sum(axis=0) / 1

        # [P(F_i+1|F_i), P(F_i+2|F_i+1), ..., 0]
        # correct for divide by zero error so x/0 = 0
        conditional_bf_probability = np.divide(and_bf_probability, bf_probability, out=np.zeros_like(and_bf_probability), where=bf_probability!=0)

        return conditional_bf_probability

    def calculate_probability_vectors(self):
        '''
        Calculate probability vectors for messages:
            - bit flip probability: P(F_i)
            - bit flip probability conditional on flip of previous bit: P(F_i+1|F_i)
        '''

        if self.ts_data is None:
            print(f"WARNING: Message data does not exist. Used \
                    Decoder.generate_msg_ts_data() to generate this data")
            return

        ts_msg_be = self.ts_data[:,1]
        ts_msg_le = ts_msg_be.byteswap() >> (64-(self.msg_length*8))

        self.msg_quantity = ts_msg_be.shape[0]

        self.bf_probability_be, transition_matrix_bin_be = self._calculate_bf_probability(ts_msg_be, 'be')
        self.bf_probability_le, transition_matrix_bin_le = self._calculate_bf_probability(ts_msg_le, 'le')

        self.conditional_bf_probability_be = self._calculate_conditional_bf_probability(self.bf_probability_be, transition_matrix_bin_be)
        self.conditional_bf_probability_le = self._calculate_conditional_bf_probability(self.bf_probability_le, transition_matrix_bin_le)
