import numpy as np


def convert_signal(start_bit, length, in_format, out_format, msg_size_bytes):
    """
    Convert signal defined by start_bit and length between
    formats: ['inorder', 'be', 'le']
    Supports variable message lengths from 1 to 64 bytes.

        Parameters:
            start_bit (int): Starting bit position
            length (int): Signal length in bits
            in_format (str): ['inorder', 'be', 'le']
            out_format (str): ['inorder', 'be', 'le']
            msg_size_bytes (int): Message size in bytes (1-64), default 8
    """

    # Calculate total bits based on message size
    total_bits = msg_size_bytes * 8

    # Create reference arrays for the actual message size
    inorder_ref = np.arange(0, total_bits)

    # Big endian: reverse bits within each byte
    be_ref = inorder_ref.reshape((msg_size_bytes, 8))[:, ::-1].reshape(
        total_bits,
    )

    # Little endian: complete reverse
    le_ref = inorder_ref[::-1]

    # # Validate bit positions are within bounds
    # if start_bit >= total_bits:
    #     raise ValueError(
    #         f"start_bit {start_bit} exceeds message size {total_bits} bits"
    #     )
    # if start_bit + length > total_bits:
    #     raise ValueError(
    #         f"Signal extends beyond message size: start_bit({start_bit})
    # + length({length}) > {total_bits}"
    #     )

    if in_format == "inorder":
        if out_format == "be":
            return be_ref[start_bit], length
        elif out_format == "le":
            return le_ref[start_bit + length - 1], length
        elif out_format == "inorder":
            return start_bit, length

    elif in_format == "be":
        if out_format == "inorder":
            return inorder_ref[np.where(be_ref == start_bit)][0], length
        elif out_format == "le":
            # Convert be -> inorder -> le
            inorder_bit = inorder_ref[np.where(be_ref == start_bit)][0]
            return le_ref[inorder_bit + length - 1], length
        elif out_format == "be":
            return start_bit, length

    elif in_format == "le":
        if out_format == "inorder":
            return inorder_ref[np.where(le_ref == start_bit)[0] - length + 1][0], length
        elif out_format == "be":
            # Convert le -> inorder -> be
            inorder_bit = inorder_ref[np.where(le_ref == start_bit)[0] - length + 1][0]
            return be_ref[inorder_bit], length
        elif out_format == "le":
            return start_bit, length

    raise ValueError(f"Unsupported conversion from {in_format} to {out_format}")


def validate_byte_order(byte_order):
    """
    Validate byte order is in ['be', 'le']
    """
    valid_byte_orders = ["be", "le"]
    if byte_order not in valid_byte_orders:
        raise ValueError(f"byte_order value must be one of {valid_byte_orders}")


def validate_tokenization_method(tokenization_method):
    """
    Validate tokenization method is in ['tang', 'conditional_bit_flip']
    """
    valid_tokenization_method = ["tang", "conditional_bit_flip"]
    if tokenization_method not in valid_tokenization_method:
        raise ValueError(
            f"tokenization_method value must be one of {valid_tokenization_method}"
        )


def validate_signedness_method(signedness_method):
    """
    Validate signedness method is in ['msb_classifier']
    """
    valid_signedness_methods = ["msb_classifier"]
    if signedness_method not in valid_signedness_methods:
        raise ValueError(
            f"tokenization_method value must be one of {valid_signedness_methods}"
        )


def validate_byte_filters(byte_filters):
    # TODO: byte filters validator
    return byte_filters


def determine_length(data):
    byte_cols = []
    for i in range(1, 65):
        col_name = str(i)
        if col_name in data.columns:
            byte_cols.append(col_name)
    # Find the first column (from "1" to "64") where all values are NA
    length = 0
    for col in byte_cols:
        if data[col].notna().any():
            length += 1
        else:
            break
    return length
