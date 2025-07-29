import numpy as np
import math
from PandaCANDecoder.utils import convert_signal


colors = {'ts': ['#79A8DD', '#DD79DA', '#DDAE79', '#79DD7C'], 'constant': ['darkgrey'], 'unused': ['dimgrey']}
colors_sat = {'ts': ['#1a85ff', '#ff1af7', '#ff941a', '#1aff21']}
hatch_styles = {'signed': '.....', 'unsigned': '', 'unused': '///'}


def plot_probability_chart(ax, msg, byte_order):
    '''
    Plot bit flip probability chart with signal boundaries for a specified message

        Parameters:
            ax (matplotlib.axes.Axes)
            msg (PandaCANDecoder.Message)
            byte_order (str)
    '''

    # get x values based on byte-order
    x_values_inorder = np.arange(0,64)
    if byte_order == 'be':
        x_values_byte_order = x_values_inorder.reshape((8,8))[:, ::-1].reshape(64,)
    elif byte_order == 'le':
        x_values_byte_order = x_values_inorder[::-1]

    # set up axes
    ax.set_xticks(x_values_inorder)
    ax.set_xticklabels(x_values_byte_order)
    ax.tick_params(axis='x', labelsize=7)
    ax.set_xlim(-0.5, 63.5)
    ax.set_ylim(0, 1.1)

    # draw lines and labels
    ax.axhline(y=1, color='k', linestyle='-', linewidth=1)
    for xc in range(0, 64, 8):
        ax.axvline(x=xc-.5, color='k', linestyle=':', linewidth=1.3)
        if byte_order == 'be':
            ax.text(xc, 1.02, f"Byte {xc//8}", fontsize=8)
        elif byte_order == 'le':
            ax.text(xc, 1.02, f"Byte {7-(xc//8)}", fontsize=8)

    # label graph and axes
    if byte_order == 'be':
        title = "Big-Endian"
    elif byte_order == 'le':
        title = "Little-Endian"
    ax.set_title(f"{title}: Bit Flip Probabilities")
    ax.set_xlabel("Bit Number")
    ax.set_ylabel("Bit Flip Probability")

    # plot signals
    color_idx = 0
    for signal in msg.signals:
        if signal.byte_order == byte_order:
            idx, _ = convert_signal(signal.start_bit, signal.length, in_format=signal.byte_order, out_format='inorder')
            for _ in range(signal.length):
                ax.bar(idx, 1, width=1,
                        color=colors[signal.classification][color_idx%len(colors[signal.classification])],
                        hatch=hatch_styles[signal.signedness], label=signal.name)
                idx += 1
            color_idx += 1 * (signal.classification == 'ts')

    # plot probabilities
    if byte_order == 'be':
        bf_probability = msg.bf_probability_be
    elif byte_order == 'le':
        bf_probability = msg.bf_probability_le
    ax.bar(x_values_inorder, bf_probability, width=1, color='k', edgecolor='w')

    # signal legend
    handles, labels = ax.get_legend_handles_labels()
    legend_dict = {k:v for k,v in zip(labels, handles)}
    ax.legend(legend_dict.values(), legend_dict.keys(), loc='upper center', ncol=10, bbox_to_anchor=(0.5, -0.19), prop={'size': 7})


def plot_bit_table(ax, msg, byte_order):
    '''
    Plot bit table with signal boundaries for a specified message

    Parameters:
        ax (matplotlib.axes.Axes)
        msg (PandaCANDecoder.Message)
        byte_order (str)
    '''

    # set up table axes
    ax.set_xlim(0,8)
    ax.set_xticks(np.arange(8))
    ax.set_xticklabels([])
    ax.set_ylim(-1,7)
    ax.set_yticks(np.arange(8))
    ax.set_yticks(np.arange(7, -1, -1)-0.5, minor=True)
    ax.set_yticklabels(np.arange(8), minor=True)
    ax.set_yticklabels([])

    # add grid and text lables
    ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1)
    for i in range(8):
        for j in range(8):
            ax.text(i+0.37, j-0.75, f"{(7-i)+(7-j)*8}")

    # label graph and axes
    if byte_order == 'be':
        title = "Big-Endian"
    elif byte_order == 'le':
        title = "Little-Endian"
    ax.set_ylabel("Byte")
    ax.set_title(f"{title}: Bit Numbers")

    # plot signals
    color_idx = 0
    for signal in msg.signals:
        if signal.byte_order == byte_order:
            idx, _ = convert_signal(signal.start_bit, signal.length, in_format=signal.byte_order, out_format='inorder')
            for _ in range(signal.length):
                if byte_order == 'be':
                    r_grid = 7-(idx//8)
                elif byte_order == 'le':
                    r_grid = idx//8
                c_grid = idx%8
                ax.axhspan(ymin=r_grid-1, ymax=r_grid, xmin=c_grid/8, xmax=(c_grid+1)/8,
                        facecolor=colors[signal.classification][color_idx%len(colors[signal.classification])],
                        hatch=hatch_styles[signal.signedness])
                idx += 1
            color_idx += 1 * (signal.classification == 'ts')

def plot_ts_signal(ax, signal):
    '''
    Plot time series signal values

    Parameters:
        ax (matplotlib.axes.Axes)
        signal_values (np.array[])
    '''
    # set title and plot
    # TODO: axis tables, etc
    ax.set_title(f"Signal: Time Series Data")
    ax.set_xlabel("Seconds")
    ax.set_ylabel(f"Signal Value {signal.unit}")
    ax.plot(signal.ts_data_timestamps, signal.ts_data, linewidth=1.5, color=colors_sat['ts'][0], label=signal)
    ax.legend()

def plot_ts_signal_multi(ax, signals, autoscale):
    ax.set_title("Signal Comparison: Time Series Data")
    ax.set_xlabel("Seconds")
    ax.set_ylabel("Signal Value")
    if autoscale:
        ax.get_yaxis().set_visible(False)
        prev_max = 0
    for i, signal in enumerate(signals):
        m = 1
        b = 0
        if autoscale:
            max = np.amax(signal.ts_data)
            min = np.amin(signal.ts_data)
            m = 1/(max-min)
            b = prev_max - m*min
            prev_max = b + m*max + .2
        ax.plot(signal.ts_data_timestamps, m*signal.ts_data+b, linewidth=1, color=colors_sat['ts'][i%4], label=signal)
    ax.legend()

def plot_ts_signal_comparison(ax, signal_a, signal_b, m, b, r_sq):
    '''
    Plot signal comparison

        Parameters:
            signal_a (PandaCANDecoder.signal.Signal())
            signal_b (PandaCANDecoder.signal.Signal())
            m (float): scale factor
            b (float): offset factor
    '''
    ax.set_xlabel("Seconds")
    ax.set_ylabel("Signal Value (Rescaled)")
    ax.set_title(f"Signal Match: Time Series Data\nm={m:.2f}, b={b:.2f}, R^2={r_sq:.4f}")
    ax.plot(signal_b.ts_data_timestamps, signal_b.ts_data, linewidth=1.5, color=colors_sat['ts'][2], label=signal_b)
    ax.plot(signal_a.ts_data_timestamps, m*signal_a.ts_data+b, linewidth=1.5, color=colors_sat['ts'][0], label=signal_a)
    ax.legend()
