import os

import numpy as np
import pandas as pd

from can_decoder.plotter import plot_probability_chart


class Message:
    """
    CAN message

        Attributes:
            msg_id (string): hex value
            msg_length (int): bytes
            msg_quantity (int)
            tang (np.array[int])
            msg_ts_data (np.array[uint64])
            signal_boundaries (list[int])
    """

    time_series_msg_dir = "./time_series_msgs"

    def __init__(self, msg_id, msg_length, msg_byte_filter=None):
        self.msg_id = msg_id
        self.msg_length = msg_length  # bytes
        self.msg_quantity = None

        # For Diagnostics
        self.msg_byte_filter = msg_byte_filter

        self.bf_probability_be = None
        self.bf_probability_le = None

        self.conditional_bf_probability_be = None
        self.conditional_bf_probability_le = None

        self.transition_matrix_bin_be = None
        self.transition_matrix_bin_le = None

        self.tokenization_method_used = None

        self.name = ""
        self.signals = None

        self.ts_data = None

    def __repr__(self):
        """
        Override print string
        """
        if self.msg_byte_filter is None:
            str = f"Message {self.msg_id}: Length {self.msg_length} bytes"
        else:
            filter_str = "|".join(f"B{k}:{v}" for k, v in self.msg_byte_filter.items())
            str = (
                f"Message {self.msg_id} - Filter [{filter_str}] : "
                f"Length {self.msg_length} bytes"
            )
        return str

    def get_signal(self, signal_id):
        for signal in self.signals:
            if signal.name == signal_id:
                return signal

    def generate_ts_data(self, all_data, rewrite):
        """
        Generate time series message data and set to self.ts_data. Save data to
        csv file for easy recall if it does not exist

            Parameters:
                all_data (Pandas.dataframe)
        """
        # Data path
        # TODO: Add byte filter in data path to avoid errors
        data_path = os.path.join(self.time_series_msg_dir, self.msg_id + ".csv")

        # if data file does not exist
        if not os.path.exists(data_path) or rewrite:

            # Filter all_data for this message ID
            filtered_data = all_data[all_data["arb_id"] == self.msg_id]

            # If msg_byte_filter is provided, filter further by byte values
            if self.msg_byte_filter is not None:
                for byte_idx, byte_val in self.msg_byte_filter.items():
                    # byte_idx is expected to be int or str representing the column
                    filtered_data = filtered_data[
                        filtered_data[str(byte_idx)] == byte_val
                    ]

            # Drop "length" column if present
            if "length" in filtered_data.columns:
                filtered_data = filtered_data.drop(columns=["length"])

            # Rename "timestamp" to "abs_time" if present
            if "timestamp" in filtered_data.columns:
                filtered_data = filtered_data.rename(columns={"timestamp": "abs_time"})
            # For each time series keep only columns with data
            # TODO: Hard boundaries between active and inactive columns: All cells
            # should be numeric in the last active column and all NaN in the next,
            # otherwise throw in an error
            byte_cols = [col for col in filtered_data.columns if col.isdigit()]
            byte_cols_present = [
                col for col in byte_cols if filtered_data[col].notna().any()
            ]

            ts_dataframe = filtered_data[["abs_time"] + byte_cols_present]

            # Save to CSV
            ts_dataframe.to_csv(data_path, index=False)

        # Load
        self.ts_data = pd.read_csv(data_path)

    def _calculate_bf_probability(self, ts_msg_df, byte_order):
        """
        Calculate probability that each bit within a message flipped during the log.
        Return bit flip probability vector and binary transition matrix

        MODIFIED: Now works with DataFrame structure (one column per byte),
        supporting arbitrary message lengths. Output format unchanged.

            Parameters:
                ts_msg_df (pd.DataFrame): DataFrame with one column per
                byte (str: 1,2,...,N)
                byte_order (str): ['be', 'le']
        """
        # Get number of bytes from DataFrame columns
        byte_columns = [col for col in ts_msg_df.columns if col.isdigit()]
        byte_columns = sorted(byte_columns, key=lambda x: int(x))
        n_bytes = len(byte_columns)
        n_bits = n_bytes * 8
        n_rows = len(ts_msg_df)

        # Build a (n_rows, n_bits) array of all bits for each message
        bits = np.zeros((n_rows, n_bits), dtype=np.uint8)
        for i, col in enumerate(byte_columns):
            byte_vals = ts_msg_df[col].values.astype(np.uint8)
            for b in range(8):
                if byte_order == "be":
                    bit_idx = i * 8 + b
                else:  # little-endian: LSB is leftmost bit
                    bit_idx = (n_bytes - 1 - i) * 8 + b
                bits[:, bit_idx] = (byte_vals >> (7 - b)) & 1

        # Compute bit flips: XOR between consecutive rows
        transition_matrix_bin = (bits[:-1] ^ bits[1:]).astype(int)

        # probability of bit flip per bit
        if self.msg_quantity > 1:
            bf_probability = transition_matrix_bin.sum(axis=0) / (self.msg_quantity - 1)
        else:
            bf_probability = transition_matrix_bin.sum(axis=0) / 1

        return bf_probability, transition_matrix_bin

    def _calculate_conditional_bf_probability(
        self, bf_probability, transition_matrix_bin
    ):
        """
        Calculate the probability that bit will flip condition on the previous
        bit flipping.
        Return probability vector

        MODIFIED: No change needed for DataFrame input, as transition_matrix_bin
        is now always (n-1, n_bits)

            Parameters:
                bf_probability (np.array)
                transition_matrix_bin (np.array)
        """
        # F_i AND F_i+1 (for each bit position)
        transition_matrix_bin_shifted = np.pad(transition_matrix_bin, ((0, 0), (0, 1)))[
            :, 1:
        ]
        transition_matrix_and = (
            (transition_matrix_bin + transition_matrix_bin_shifted) > 1
        ).astype(int)

        if self.msg_quantity > 1:
            and_bf_probability = transition_matrix_and.sum(axis=0) / (
                self.msg_quantity - 1
            )
        else:
            and_bf_probability = transition_matrix_bin.sum(axis=0) / 1

        # correct for divide by zero error so x/0 = 0
        conditional_bf_probability = np.divide(
            and_bf_probability,
            bf_probability,
            out=np.zeros_like(and_bf_probability),
            where=bf_probability != 0,
        )

        return conditional_bf_probability

    def calculate_probability_vectors(self):
        """
        Calculate probability vectors for messages:
            - bit flip probability: P(F_i)
            - bit flip probability conditional on flip of previous bit: P(F_i+1|F_i)

        MODIFIED: Now works with DataFrame structure for arbitrary message length.
        """
        if self.ts_data is None:
            print(
                "WARNING: Message data does not exist. Use "
                "Decoder.generate_msg_ts_data() to generate this data"
            )
            return

        # Use all byte columns (str: 1,2,...,N)
        byte_columns = [col for col in self.ts_data.columns if col.isdigit()]
        byte_columns = sorted(byte_columns, key=lambda x: int(x))
        ts_msg_df = self.ts_data[byte_columns]

        self.msg_quantity = ts_msg_df.shape[0]

        self.bf_probability_be, self.transition_matrix_bin_be = (
            self._calculate_bf_probability(ts_msg_df, "be")
        )
        self.bf_probability_le, self.transition_matrix_bin_le = (
            self._calculate_bf_probability(ts_msg_df, "le")
        )

        self.conditional_bf_probability_be = self._calculate_conditional_bf_probability(
            self.bf_probability_be, self.transition_matrix_bin_be
        )
        self.conditional_bf_probability_le = self._calculate_conditional_bf_probability(
            self.bf_probability_le, self.transition_matrix_bin_le
        )

    def plot(self, byte_order="be"):
        fig = plot_probability_chart(
            self,
            byte_order=byte_order,
            conditional_bit_flip=(
                self.tokenization_method_used == "conditional_bit_flip"
            ),
        )
        fig.show()
