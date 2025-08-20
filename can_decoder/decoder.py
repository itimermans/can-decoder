import os

import numpy as np
import pandas as pd

from can_decoder.message import Message
from can_decoder.plotter import plot_probability_chart, plot_ts_signal, plot_ts_signals
from can_decoder.signal import Signal
from can_decoder.utils import (
    convert_signal,
    validate_byte_filters,
    validate_byte_order,
    validate_signedness_method,
    validate_tokenization_method,
)


class Decoder:
    """
    Main Decoder
    """

    def __init__(self, df=None, csv_file_path=None):
        """
        Decoder constructor

            Parameters:
                csv_file_path (str): File path to csv log
        """
        self.csv_file_path = csv_file_path
        self.time_series_msg_dir = "./time_series_msgs"  # TODO: make static
        self.msgs = []

        if df is not None:
            # If df is provided, use it directly
            self.all_data = df
        else:
            # Load data from CSV file
            if csv_file_path:
                self.all_data = pd.read_csv(self.csv_file_path)

                # CSV sanitation...this can be a very complex function

        # Throw error if no data loaded
        if (
            not hasattr(self, "all_data")
            or self.all_data is None
            or self.all_data.empty
        ):
            raise ValueError(
                "No data loaded. Provide a valid DataFrame or CSV file path."
            )

        # Common df sanitation
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
                byte_filter_values (None or list of ints): Values for
                byte filter columns
        """
        matches = [msg for msg in self.msgs if msg.msg_id == msg_id]
        if byte_filter_values is not None:
            # Filter by byte_filter_values list
            for msg in matches:
                if (
                    msg.msg_byte_filter is not None
                    and list(msg.msg_byte_filter.values()) == byte_filter_values
                ):
                    return msg
            raise ValueError(
                (
                    f"No message found with msg_id={msg_id} "
                    f"and byte_filter_values={byte_filter_values}"
                )
            )
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
            raise Warning(
                f"signal_id is None. Available signals: {[s.name for s in msg.signals]}"
            )
        return msg.get_signal(signal_id)

    def generate_msgs(self, byte_filters=None):
        """
        Generates PandaCANDecoder.Message() objects for each unique message in CAN file
        """

        # unique_pairings = self.all_data.drop_duplicates(subset=["arb_id"])
        self.unique_msg_ids = self.all_data.drop_duplicates(subset=["arb_id"])[
            "arb_id"
        ].to_list()

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

                unique_msg_pairings = msg_data.drop_duplicates(subset=msg_byte_filter)

                for _, row in unique_msg_pairings.iterrows():
                    sub_msg_data = msg_data[
                        (msg_data[msg_byte_filter] == row[msg_byte_filter]).all(axis=1)
                    ]

                    # sub_msg_length = determine_length(sub_msg_data) # TODO: Apply
                    # and refine
                    sub_msg_length = sub_msg_data["length"].iloc[0]

                    # Create a dict mapping column names to their values in the
                    # current row
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
        msg.tokenization_method_used = tokenization_method

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
                signal_name = (
                    f"S_{(msg.msg_id).upper()}_{(byte_order).upper()}_{signal_name_idx}"
                )
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

        conditional_bit_flip = msg.tokenization_method_used == "conditional_bit_flip"

        fig = plot_probability_chart(
            msg, byte_order=byte_order, conditional_bit_flip=conditional_bit_flip
        )
        fig.show()

    def plot_message_ts_signals(
        self, msg_id, byte_filter_values=None, return_fig=False, normalized=False
    ):
        msg = self.get_message(msg_id, byte_filter_values=byte_filter_values)
        if msg is None:
            print(f"Message {msg_id} not found.")
            return

        fig = plot_ts_signals(
            [signal for signal in msg.signals if signal.classification == "ts"],
            normalized=normalized,
        )
        if return_fig:
            return fig
        fig.show()

    def plot_signals(self, signals, return_fig=False, normalized=False):
        """
        Plot a signal.
        """
        if not signals:
            print("Signals not found.")
            return

        # Make sure it is a list, even with one signal
        if not isinstance(signals, list):
            signals = [signals]

        fig = plot_ts_signals(signals, normalized=normalized)
        if return_fig:
            return fig
        fig.show()

    def plot_signal_from_id(
        self, msg_id, byte_filter_values=None, signal_id=None, return_fig=False
    ):
        """
        Plot a signal by its message ID and signal name.
        """
        signal = self.get_signal(
            msg_id, byte_filter_values=byte_filter_values, signal_id=signal_id
        )
        if signal is None:
            print(f"Signal {signal_id} in message {msg_id} not found.")
            return

        fig = plot_ts_signal(signal)
        if return_fig:
            return fig
        fig.show()

    def signal_match(self, signal_a, signal_b, start_time=None, end_time=None):
        """
        Assume signal_a is the low update rate known signal,
        and signal_b is the high update rate unknown signal we want to check signal_a
        against.
        Steps:
        1. Filter both signals by start_time and end_time.
        2. Interpolate high-rate signal_b to the timestamps of low-rate signal_a.
        3. Perform linear regression between signal_a and interpolated signal_b.
        Returns:
            result_dict: dict with keys 'x_time', 'x', 'y_interp', 'regression_coef',
            'regression_intercept', 'r_value', 'p_value', 'std_err'
        """
        # Step 1: Filter signal_a (low-rate)
        a_time = signal_a.ts_data_timestamps
        a_vals = signal_a.ts_data
        if start_time is not None:
            mask_a = a_time >= start_time
        else:
            mask_a = np.ones_like(a_time, dtype=bool)
        if end_time is not None:
            mask_a &= a_time <= end_time
        x_time = a_time[mask_a]
        x = a_vals[mask_a]

        # Step 1: Filter signal_b (high-rate)
        b_time = signal_b.ts_data_timestamps
        b_vals = signal_b.ts_data
        if start_time is not None:
            mask_b = b_time >= start_time
        else:
            mask_b = np.ones_like(b_time, dtype=bool)
        if end_time is not None:
            mask_b &= b_time <= end_time
        y_time = b_time[mask_b]
        y = b_vals[mask_b]

        # Step 2: Interpolate high-rate signal_b to low-rate timestamps
        # Use numpy.interp for 1D interpolation
        # y_interp will have same length as x_time
        if len(y_time) == 0 or len(x_time) == 0:
            print("No data in selected time range.")
            return None
        y_interp = np.interp(x_time, y_time, y)

        # Step 3: Linear regression
        # Use scipy.stats.linregress
        try:
            from scipy.stats import linregress
        except ImportError:
            print("scipy is required for linear regression. Please install scipy.")
            return None
        try:
            regression = linregress(x, y_interp)
        except Exception as e:
            import warnings
            warnings.warn(f"Linear regression failed for signal_a {signal_a.name} ({signal_a.msg.msg_id}) and signal_b {signal_b.name} ({signal_b.msg.msg_id}): \n{e}")
            return None

        # # Step 3 Alternative
        #         # Step 3: Linear regression using sklearn
        # x_reshaped = x.reshape(-1, 1)
        # y_interp_reshaped = y_interp.reshape(-1, 1)
        # reg = LinearRegression()
        # reg.fit(x_reshaped, y_interp_reshaped)
        # regression = type('RegressionResult', (object,), {})()
        # regression.slope = reg.coef_[0][0]
        # regression.intercept = reg.intercept_[0]
        # # Calculate r_value
        # y_pred = reg.predict(x_reshaped)
        # ss_res = np.sum((y_interp_reshaped - y_pred) ** 2)
        # ss_tot = np.sum((y_interp_reshaped - np.mean(y_interp_reshaped)) ** 2)
        # regression.rvalue = np.sqrt(1 - ss_res / ss_tot) if ss_tot != 0 else 0
        # regression.pvalue = None  # Not available in sklearn
        # regression.stderr = None  # Not available in sklearn
        # regression.score = reg.score(x_reshaped, y_interp_reshaped)  # Coefficient of
        # determination (R^2)

        # Step 4: Return results
        # result_dict = {
        #     'regression_coef': regression.slope,
        #     'regression_intercept': regression.intercept,
        #     'r_value': regression.rvalue,
        #     'p_value': regression.pvalue,
        #     'std_err': regression.stderr,
        #     'score': regression.rvalue ** 2  # Coefficient of determination (R^2)
        # }
        return regression, regression.rvalue**2

    def find_signal_match(
        self,
        signal,
        thresh=0.5,
        start_time=None,
        end_time=None,
        messages=None,
        only_ts=False,
    ):
        """
        Loop through all signals of all other messages (or the ones specified in
        messages) and
        find a matching signal for the given signal within the specified time range.
        """
        if messages is None:
            messages = self.msgs
        else:
            # Make sure its a list
            # TODO: More validation
            if not isinstance(messages, list):
                messages = [messages]

        candidates = []

        for msg in messages:
            # TODO: Might not have signals
            for other_signal in msg.signals:
                if (other_signal == signal) or (
                    only_ts and other_signal.classification != "ts"
                ):
                    continue
                regression, r_sq = self.signal_match(
                    signal, other_signal, start_time, end_time
                )
                if r_sq is not None and r_sq >= thresh:
                    print(
                        (
                            f"{signal.name:<20} --> {other_signal.name:<20} "
                            f"in {msg.msg_id:<10} , r^2={r_sq:.6f}"
                        )
                    )
                    candidates.append((other_signal, regression, r_sq))
                    # Sort candidates by r^2 value
                    candidates.sort(key=lambda x: x[2], reverse=True)

        return candidates

    def plot_signal_matches(self, signal, candidates, return_fig=False):

        signals = [signal] + [c[0] for c in candidates]

        fig = self.plot_signals(signals, return_fig=True, normalized=True)

        fig.data[0].line.color = "black"  # Original signal in black
        fig.data[0].mode = "lines+markers"  # Thicker line for original signal
        fig.data[0].marker.symbol = "cross"
        fig.data[0].yaxis = "y2"
        fig.update_layout(
            yaxis2=dict(
                title="Matched Signals",
                overlaying="y",
                side="right",
                showgrid=False,
            )
        )

        # Add R2 to names
        for i, c in enumerate(candidates):
            fig.data[i + 1].name = f"{c[0].name}<br>(R²={c[2]:.6f})"

        if return_fig:
            return fig
        fig.show()
