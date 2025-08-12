import os

import numpy as np
import pandas as pd

from can_decoder.message import Message
from can_decoder.plotter import plot_probability_chart, plot_ts_signal
from can_decoder.signal import Signal
from can_decoder.utils import (
    convert_signal,
    validate_byte_filters,
    validate_byte_order,
    validate_signedness_method,
    validate_tokenization_method,
    determine_length
)


class Decoder:
    """
    Main Decoder
    """

    def __init__(self, csv_file_path=None):
        """
        Decoder constructor

            Parameters:
                csv_file_path (str): File path to csv log
        """
        self.csv_file_path = csv_file_path
        self.time_series_msg_dir = "./time_series_msgs"  # TODO: make static
        self.msgs = []

        # Load data
        if csv_file_path:
            self.all_data = pd.read_csv(self.csv_file_path)

            # CSV sanitation...this can be a very complex function

            # 1. Byte columns defined as nullable integers
            # Determine which byte columns exist (from '1' up to the max found)
            byte_cols = []
            for i in range(1, 65):
                col_name = str(i)
                if col_name in self.all_data.columns:
                    byte_cols.append(col_name)
            for col in byte_cols:
                self.all_data[col] = self.all_data[col].astype(pd.UInt8Dtype())

    def get_message(self, msg_id, byte_filter_values=None):
        """
        Get message object from id

            Parameters:
                msg_id (str): In hex
                byte_filter_values (None or list of ints): Values for byte filter columns
        """
        matches = [msg for msg in self.msgs if msg.msg_id == msg_id]
        if byte_filter_values is not None:
            # Filter by byte_filter_values list
            for msg in matches:
                if msg.msg_byte_filter is not None and list(msg.msg_byte_filter.values()) == byte_filter_values:
                    return msg
            raise ValueError(f"No message found with msg_id={msg_id} and byte_filter_values={byte_filter_values}")
        else:
            if len(matches) == 1:
                return matches[0]
            elif len(matches) == 0:
                raise ValueError(f"No message found with msg_id={msg_id}")
            else:
                print(f"Multiple messages found with msg_id={msg_id}:")
                for msg in matches:
                    print(f"  msg_byte_filter={msg.msg_byte_filter}")
                raise ValueError(
                    f"Multiple messages found with msg_id={msg_id}. "
                    "Specify byte_filter_values to disambiguate."
                )

    def get_signal(self, msg_id, byte_filter_values=None, signal_id=None):
        """
        Get signal object from id

            Parameters:
                msg_id (str): In hex
                signal_id (str)
        """
        msg = self.get_message(msg_id, byte_filter_values)
        if signal_id is None:
            raise Warning(f"signal_id is None. Available signals: {[s.name for s in msg.signals]}")
        return msg.get_signal(signal_id)

    def generate_msgs(self, byte_filters=None):
        """
        Generates PandaCANDecoder.Message() objects for each unique message in CAN file
        """

        #unique_pairings = self.all_data.drop_duplicates(subset=["arb_id"])
        self.unique_msg_ids = self.all_data.drop_duplicates(subset=["arb_id"])["arb_id"].to_list()

        if byte_filters is not None:
            # Validate byte_filters format
            # TODO: Decide what to do if error in validation
            byte_filters = validate_byte_filters(byte_filters)

        # for msg_id in tqdm(unique_msg_ids, desc="Generating messages".ljust(30)):
        for msg_id in self.unique_msg_ids:

            msg_data = self.all_data[self.all_data["arb_id"] == msg_id]

            if byte_filters and (msg_id in byte_filters):
                # Generate sub-messages with byte filter
                msg_byte_filter = byte_filters[msg_id]

                
                unique_msg_pairings = msg_data.drop_duplicates(subset = msg_byte_filter)

                for _, row in unique_msg_pairings.iterrows():
                    sub_msg_data = msg_data[
                        (msg_data[msg_byte_filter] == row[msg_byte_filter]).all(axis=1)
                    ]

                    # sub_msg_length = determine_length(sub_msg_data) # TODO: Apply and refine
                    sub_msg_length = sub_msg_data["length"].iloc[0]

                    # Create a dict mapping column names to their values in the current row
                    msg_byte_filter_dict = {col: row[col] for col in msg_byte_filter}

                    self.msgs.append(
                        Message(
                            msg_id=msg_id,
                            msg_length=sub_msg_length,
                            msg_byte_filter=msg_byte_filter_dict,
                        )
                    )


            else:
                msg_length = msg_data["length"].iloc[0]

                self.msgs.append(
                Message(
                    msg_id=msg_id,
                    msg_length=msg_length,
                    msg_byte_filter=None,
                )
            )

        print(f"Generated {len(self.msgs)} messages")

    def generate_msg_ts_data(self, rewrite=False):
        """
        Saves time series messages as pandas DataFrame
            Parameters:
                rewite (bool): If True, will overwrite existing .npy file
        """
        if not self.msgs:
            print(
                "WARNING: No messages have been generated. Use Decoder.generate_msgs() "
                "to do so."
            )
            return
        # save time series data
        os.makedirs(self.time_series_msg_dir, exist_ok=True)
        for msg in self.msgs:
            msg.generate_ts_data(self.all_data, rewrite)

    def calculate_signals(
        self,
        tokenization_method,
        signedness_method,
        alpha1=0.01,
        alpha2=0.5,
        gamma1=0.2,
    ):
        """
        Calculate predicted signals for all messages
        """
        for msg in self.msgs:
            self.calculate_signal(
                msg, tokenization_method, signedness_method, alpha1, alpha2, gamma1
            )

    def calculate_signal(
        self,
        msg,
        tokenization_method,
        signedness_method,
        alpha1=0.01,
        alpha2=0.5,
        gamma1=0.2,
    ):
        """
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
        """
        # validate methods
        validate_tokenization_method(tokenization_method)
        validate_signedness_method(signedness_method)

        msg.calculate_probability_vectors()

        signals_be = self._tokenize_msg(
            msg, "be", tokenization_method, signedness_method, alpha1, alpha2, gamma1
        )
        signals_le = self._tokenize_msg(
            msg, "le", tokenization_method, signedness_method, alpha1, alpha2, gamma1
        )

        msg.signals = signals_be + signals_le
        # msg.signals = signals_be

    def _method_condition(
        self, method, i, bf_probability, conditional_bf_probability, alpha1, alpha2
    ):
        """
        Return true if bit index i is at the boundary of a signal

            Parameters:
                method (str)
                i (int): bit index
                bf_probability (np.array): bit flip probability
                conditional_bf_probability (np.array): bit flip probability
                conditional on previous bit flip
                alpha1 (float)
                alpha2 (float)
        """
        if method == "tang":
            return bf_probability[i] > bf_probability[i + 1] + alpha1
        elif method == "conditional_bit_flip":
            return (
                (conditional_bf_probability[i] < alpha1)
                or (
                    conditional_bf_probability[i + 1] - conditional_bf_probability[i]
                    > alpha2
                )
            ) and bf_probability[i] != 0

    def _tokenize_msg(
        self,
        msg,
        byte_order,
        tokenization_method,
        signedness_method,
        alpha1,
        alpha2,
        gamma,
    ):
        """
        Tokenize a message into signals and calculating signedness

            Parameters:
                byte_order (str)
                see self.calculate_signal()
        """
        validate_byte_order(byte_order)

        range_start = 0
        range_end = msg.msg_length * 8
        if byte_order == "be":
            bf_probability = msg.bf_probability_be
            conditional_bf_probability = msg.conditional_bf_probability_be
        elif byte_order == "le":
            bf_probability = msg.bf_probability_le
            conditional_bf_probability = msg.conditional_bf_probability_le

        start_bit = range_start
        length = 1
        signals = []
        signal_name_idx = 0

        # tokenize and create signals
        for i in range(range_start, range_end):
            # At end or message, end of signal determined by method,
            # or end of constant signal
            if (
                (i == range_end - 1)
                or self._method_condition(
                    tokenization_method,
                    i,
                    bf_probability,
                    conditional_bf_probability,
                    alpha1,
                    alpha2,
                )
                or (bf_probability[i] == 0 and bf_probability[i + 1] != 0)
            ):

                start_bit, _ = convert_signal(
                    start_bit,
                    length,
                    "inorder",
                    byte_order,
                    msg_size_bytes=msg.msg_length,
                )
                signal_name = f"SIG_{(byte_order).upper()}_{signal_name_idx}"
                if bf_probability[i] == 0:
                    signal = Signal(
                        start_bit,
                        length,
                        byte_order=byte_order,
                        classification="constant",
                        msg=msg,
                        name=signal_name,
                    )
                else:
                    signal = Signal(
                        start_bit,
                        length,
                        byte_order=byte_order,
                        classification="ts",
                        msg=msg,
                        name=signal_name,
                    )
                signals.append(signal)
                start_bit = i + 1
                length = 1
                signal_name_idx += 1
            else:
                length += 1

        # get signal signedness
        for signal in signals:
            signal.signedness = self._calculate_signedness(
                "msb_classifier", signal, gamma
            )

        return signals

    def _calculate_signedness(self, method, signal, gamma):
        """
        Calculate the signedness of a signal. Return string

            Parameters:
                method (str)
                signal (Signal)
                gamma (float)
        """

        if signal.length == 1:
            return "unsigned"

        ts_signal_data_raw = signal.ts_data_raw

        if method == "msb_classifier":
            # Convert to binary matrix: shape (n_samples, signal.length)
            n = len(ts_signal_data_raw)
            signal_length = signal.length

            # Each row is a list of bits, MSB first
            ts_bin = np.array(
                [
                    [
                        (int(val) >> (signal_length - 1 - bit)) & 1
                        for bit in range(signal_length)
                    ]
                    for val in ts_signal_data_raw
                ],
                dtype=int,
            )

            # Probabilities and conditions as before
            prob_1_0 = np.count_nonzero((ts_bin[:, 0] == 1) & (ts_bin[:, 1] == 0)) / n
            prob_0_1 = np.count_nonzero((ts_bin[:, 0] == 0) & (ts_bin[:, 1] == 1)) / n

            bool_0_0 = (ts_bin[:, 0] == 0) & (ts_bin[:, 1] == 0)
            bool_1_1 = (ts_bin[:, 0] == 1) & (ts_bin[:, 1] == 1)

            # For transitions, exclude last sample for bool_0_0 and first for bool_1_1
            prob_0_0_and_1_1 = np.count_nonzero(bool_0_0[:-1] & bool_1_1[1:]) / n

            if prob_1_0 + prob_0_1 == 0:
                return "signed"
            if prob_0_0_and_1_1 == 0:
                return "unsigned"
            if prob_1_0 + prob_0_1 < gamma:
                return "signed"
            return "unsigned"

        # Different version for max 8 bytes signals
        # def _calculate_signedness(self, method, signal, gamma):
        #     """
        #     Calculate the signedness of a signal. Return string

        #         Parameters:
        #             method (str)
        #             signal (Signal)
        #             gamma (float)
        #     """

        #     if signal.length == 1:
        #         return "unsigned"

        #     ts_signal_data_raw = signal.ts_data_raw  # np.uint64 array

        #     if method == "msb_classifier":
        #         n = len(ts_signal_data_raw)
        #         signal_length = signal.length

        #         # Vectorized binary conversion: shape (n_samples, signal_length)
        #         bit_shifts = np.arange(signal_length - 1, -1, -1, dtype=np.uint64)
        #         ts_bin = ((ts_signal_data_raw[:, None] >> bit_shifts) & 1).astype(int)

        #         prob_1_0 = np.count_nonzero((ts_bin[:, 0] == 1)
        # & (ts_bin[:, 1] == 0)) / n
        #         prob_0_1 = np.count_nonzero((ts_bin[:, 0] == 0)
        # & (ts_bin[:, 1] == 1)) / n

        #         bool_0_0 = (ts_bin[:, 0] == 0) & (ts_bin[:, 1] == 0)
        #         bool_1_1 = (ts_bin[:, 0] == 1) & (ts_bin[:, 1] == 1)

        #         prob_0_0_and_1_1 = np.count_nonzero(bool_0_0[:-1] & bool_1_1[1:]) / n

        #         if prob_1_0 + prob_0_1 == 0:
        #             return "signed"
        #         if prob_0_0_and_1_1 == 0:
        #             return "unsigned"
        #         if prob_1_0 + prob_0_1 < gamma:
        #             return "signed"
        #         return "unsigned"

    def plot_message_from_id(self, msg_id, byte_filter_values=None, byte_order="be"):
        """
        Plot a message by its ID.
        """
        msg = self.get_message(msg_id, byte_filter_values=byte_filter_values)
        if msg is None:
            print(f"Message {msg_id} not found.")
            return

        fig = plot_probability_chart(msg, byte_order=byte_order)
        fig.show()

    def plot_signal_from_id(self, msg_id, byte_filter_values=None, signal_name=None):
        """
        Plot a signal by its message ID and signal name.
        """
        signal = self.get_signal(msg_id, byte_filter_values=byte_filter_values, signal_name=signal_name)
        if signal is None:
            print(f"Signal {signal_name} in message {msg_id} not found.")
            return
        fig = plot_ts_signal(signal)
        fig.show()
