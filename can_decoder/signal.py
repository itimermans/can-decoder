import numpy as np

from can_decoder.utils import convert_signal


class Signal:
    """
    CAN signal
    """

    time_series_msg_dir = "./time_series_msgs"  # TODO: move somewhere

    byte_order_dict = {"be": "0", "le": "1"}
    byte_order_dict_inv = {"0": "be", "1": "le"}
    signedness_dict = {"unsigned": "+", "signed": "-"}
    signedness_dict_inv = {"+": "unsigned", "-": "signed"}

    def __init__(
        self,
        start_bit,
        length,
        byte_order="be",
        signedness="unsigned",
        factor=1,
        offset=0,
        classification=None,
        msg=None,
        name="",
        min=0,
        max=0,
        unit="",
    ):
        """
        Signal constructor
        """
        self.start_bit = start_bit
        self.length = length  # bits
        self.byte_order = byte_order
        self.signedness = signedness
        self.factor = factor
        self.offset = offset

        self.name = name
        self.unit = unit

        self.classification = classification  # constant, ts, counter, checksum

        self.msg = msg

        self.min = min
        self.max = max  # 2**self.length # this is hacky, change this

    def __repr__(self):
        return f"Message: {self.msg.msg_id}, Signal: {self.name}"

    @property
    def dbc_str(self):
        return (
            f"SG_ {self.name} : {self.start_bit}|{self.length}@"
            f"{self.byte_order_dict[self.byte_order]}"
            f"{self.signedness_dict[self.signedness]} "
            f"({self.factor},{self.offset}) "
            f'[{self.min}|{self.max}] "{self.unit}" XXX'
        )

    @property
    def ts_data_timestamps(self):
        return self.msg.ts_data["abs_time"]

    @property
    def ts_data_raw(self):
        # self.msg.ts_data: DataFrame with columns: 'abs_time', 'byte0', ..., 'byteN'
        import numpy as np

        df = self.msg.ts_data
        msg_length = self.msg.msg_length  # in bytes
        byte_cols = [str(i) for i in range(1, msg_length + 1)]
        byte_array = df[byte_cols].to_numpy(dtype=np.uint8)

        # Assemble each row as a Python int for arbitrary bit width
        vals = []
        if self.byte_order == "be":
            for row in byte_array:
                val = 0
                for i in range(msg_length):
                    shift = 8 * (msg_length - 1 - i)
                    val |= int(row[i]) << shift
                vals.append(val)
        elif self.byte_order == "le":
            for row in byte_array:
                val = 0
                for i in range(msg_length):
                    shift = 8 * i
                    val |= int(row[i]) << shift
                vals.append(val)
        else:
            raise ValueError(f"Unknown byte order: {self.byte_order}")

        # Extract the signal bits
        start_idx, _ = convert_signal(
            self.start_bit,
            self.length,
            self.byte_order,
            "inorder",
            msg_size_bytes=msg_length,
        )
        # TODO: Fix this crap. When self.byte_order == "be" vals goes as np.int64,
        # but as int when "le". bit_shift is np.int64 so creates an
        # error for the le case
        bit_shift = int((msg_length * 8) - (start_idx + self.length))
        mask = (1 << self.length) - 1
        vals_shifted = [int(v) >> bit_shift for v in vals]
        vals_masked = [int(v) & mask for v in vals_shifted]
        return np.array(vals_masked, dtype=object)

    # Different version for max 8 bytes signals
    # def ts_data_raw(self):
    # # self.msg.ts_data: DataFrame with columns: 'abs_time', 'byte0', ..., 'byteN'
    # import numpy as np

    # df = self.msg.ts_data
    # msg_length = self.msg.msg_length  # in bytes
    # byte_cols = [str(i) for i in range(msg_length)]
    # byte_array = df[byte_cols].to_numpy(dtype=np.uint8)

    # # Flatten bytes into a single integer per row, according to byte order
    # if self.byte_order == "be":
    #     # Big endian: byte0 is most significant
    #     vals = np.zeros(len(byte_array), dtype=np.uint64)
    #     for i in range(msg_length):
    #         shift = 8 * (msg_length - 1 - i)
    #         vals |= byte_array[:, i].astype(np.uint64) << shift
    # elif self.byte_order == "le":
    #     # Little endian: byte0 is least significant
    #     vals = np.zeros(len(byte_array), dtype=np.uint64)
    #     for i in range(msg_length):
    #         shift = 8 * i
    #         vals |= byte_array[:, i].astype(np.uint64) << shift
    # else:
    #     raise ValueError(f"Unknown byte order: {self.byte_order}")

    # # Extract the signal bits
    # start_idx, _ = convert_signal(
    #     self.start_bit,
    #     self.length,
    #     self.byte_order,
    #     "inorder",
    #     msg_size_bytes=msg_length,
    # )
    # bit_shift = (msg_length * 8) - (start_idx + self.length)
    # mask = np.array((1 << self.length) - 1, dtype="uint64")
    # vals_shifted = vals >> np.uint64(bit_shift)
    # vals_masked = vals_shifted & mask
    # return vals_masked

    @property
    def ts_data(self):
        ts_signal_data_raw = self.ts_data_raw

        # Correct for signedness for arbitrary-width integers
        if self.signedness == "signed":
            sign_bit = 1 << (self.length - 1)
            mask = (1 << self.length) - 1

            def sign_extend(val):
                val = int(val) & mask
                return val - (1 << self.length) if (val & sign_bit) else val

            ts_signal_data = np.array(
                [sign_extend(v) for v in ts_signal_data_raw], dtype=object
            )
        else:
            ts_signal_data = ts_signal_data_raw

        # correct for scale and offset
        ts_signal_data = np.array(
            [v * self.factor + self.offset for v in ts_signal_data], dtype=float
        )

        return ts_signal_data

    # Different version for max 8 bytes signals
    # def ts_data(self):
    #     ts_signal_data_raw = self.ts_data_raw  # np.uint64 array

    #     # Correct for signedness using numpy vectorized operations
    #     if self.signedness == "signed":
    #         sign_change_bit_shift = 64 - self.length
    #         ts_signal_data = (ts_signal_data_raw << sign_change_bit_shift)
    # .astype(np.int64) >> sign_change_bit_shift
    #     else:
    #         ts_signal_data = ts_signal_data_raw

    #     # correct for scale and offset
    #     ts_signal_data = ts_signal_data * self.factor + self.offset

    #     return ts_signal_data

    def first_derivative_ts_data(self, sampling_rate=1):
        # return d(ts_data)/dt and time stamps

        ts_data_filtered = self.ts_data[0::sampling_rate]
        time_data_filtered = self.ts_data_timestamps[0::sampling_rate]

        return np.gradient(ts_data_filtered, time_data_filtered), time_data_filtered
