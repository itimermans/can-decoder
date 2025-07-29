import numpy as np

def convert_signal(start_bit, length, in_format, out_format):
    '''
    Convert signal defined by start_bit and length between formats: ['inorder', 'be', 'le']

        Parameters:
            length (int)
            in_format (str): ['inorder', 'be', 'le']
            out_format (str): ['inorder', 'be', 'le']
    '''
    # TODO: validate in_format and out_format
    inorder_ref = np.arange(0,64)
    be_ref = inorder_ref.reshape((8,8))[:, ::-1].reshape(64,)
    le_ref = inorder_ref[::-1]

    if in_format == 'inorder':
        if out_format == 'be':
            return be_ref[start_bit], length
        elif out_format == 'le':
            return le_ref[start_bit+length-1], length
    elif in_format == 'be':
        if out_format == 'inorder':
            return inorder_ref[np.where(be_ref == start_bit)][0], length
    elif in_format == 'le':
        if out_format == 'inorder':
            return inorder_ref[np.where(le_ref == start_bit)[0]-length+1][0], length

def validate_byte_order(byte_order):
    '''
    Validate byte order is in ['be', 'le']
    '''
    valid_byte_orders = ['be', 'le']
    if byte_order not in valid_byte_orders:
        raise ValueError(f"byte_order value must be one of {valid_byte_orders}")

def validate_tokenization_method(tokenization_method):
    '''
    Validate tokenization method is in ['tang', 'conditional_bit_flip']
    '''
    valid_tokenization_method = ['tang', 'conditional_bit_flip']
    if tokenization_method not in valid_tokenization_method:
        raise ValueError(f"tokenization_method value must be one of {valid_byte_orders}")

def validate_signedness_method(signedness_method):
    '''
    Validate signedness method is in ['msb_classifier']
    '''
    valid_signedness_methods = ['msb_classifier']
    if signedness_method not in valid_signedness_methods:
        raise ValueError(f"tokenization_method value must be one of {valid_signedness_methods}")
