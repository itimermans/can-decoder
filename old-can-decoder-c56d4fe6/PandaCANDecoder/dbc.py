from PandaCANDecoder.signal import Signal
from PandaCANDecoder.message import Message

import re

def dbc_str_to_signal(dbc_str, msg):
    '''
    Convert dbc signal string into PandaCANDecoder.signal.Signal() object

    Parameters:
        dbc_str (str)
        msg (PandaCANDecoder.message.Message()): Message object associated with signal
    '''

    dbc_str = dbc_str.replace("SG_", "")
    name = re.search("\w+", dbc_str).group(0)
    start_bit = int(re.search("\d+(?=|)", dbc_str).group(0))
    length = int(re.search("\d+(?=@)", dbc_str).group(0))
    byte_order = re.search("(?<=@)[0,1]", dbc_str).group(0)
    byte_order = Signal.byte_order_dict_inv[byte_order]
    signedness = re.search("(?<=@[0,1])[+,-]", dbc_str).group(0)
    signedness = Signal.signedness_dict_inv[signedness]
    factor = float(re.search("(?<=\()[-]?\d+\.?\d*", dbc_str).group(0))
    offset = float(re.search("[-]?\d+\.?\d*(?=\))", dbc_str).group(0))
    min = int(re.search("(?<=\[)\d+", dbc_str).group(0))
    max = int(re.search("\d+(?=\])", dbc_str).group(0))
    # TODO: unit does not work if there are special characters
    unit = re.search("(?<=\")\w*(?=\")", dbc_str).group(0)

    signal = Signal(start_bit=start_bit, length=length, byte_order=byte_order, signedness=signedness, factor=factor, offset=offset, classification=None, msg=msg, name=name, min=min, max=max, unit=unit)

    return signal
