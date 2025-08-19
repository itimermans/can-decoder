import numpy as np
import plotly.graph_objects as go

from can_decoder.utils import convert_signal


def plot_probability_chart(msg, byte_order):

    n_bits = msg.msg_length * 8
    x_values_inorder = np.arange(0, n_bits)
    if byte_order == "be":
        x_values_byte_order = x_values_inorder.reshape((msg.msg_length, 8))[
            :, ::-1
        ].reshape(
            n_bits,
        )
        bf_probability = msg.bf_probability_be
    elif byte_order == "le":
        x_values_byte_order = x_values_inorder[::-1]
        bf_probability = msg.bf_probability_le
    else:
        raise ValueError("byte_order must be 'be' or 'le'")

    fig = go.Figure()

    # Plot bit flip probabilities
    fig.add_trace(
        go.Bar(
            x=x_values_inorder,
            y=bf_probability,
            marker_color="black",
            name="Bit Flip Probability",
            opacity=0.7,
        )
    )

    # Add vertical lines ("walls") to separate bytes
    for i in range(1, msg.msg_length):
        fig.add_shape(
            type="line",
            x0=i * 8 - 0.5,
            x1=i * 8 - 0.5,
            y0=0,
            y1=max(bf_probability) * 1.05,
            line=dict(color="gray", width=1, dash="dash"),
        )

    # Add vertical rectangles for byte boundaries and annotate byte numbers
    for xc in range(0, n_bits, 8):
        fig.add_vrect(
            x0=xc - 0.5,
            x1=xc + 7.5,
            fillcolor=None,
            opacity=0.0,
            line_width=1,
            line_color="black",
            annotation_text=(
                f"Byte {xc//8}"
                if byte_order == "be"
                else f"Byte {msg.msg_length-1-(xc//8)}"
            ),
            annotation_position="outside top",
            annotation=dict(font_size=10, showarrow=False),
        )

    # Highlight signal regions using vrects, using convert_signal for correct alignment
    colors = [
        "#79A8DD",
        "#DD79DA",
        "#DDAE79",
        "#79DD7C",
        "#79A8DD",
        "#DD79DA",
        "#DDAE79",
        "#79DD7C",
        "#79A8DD",
        "#DD79DA",
        "#DDAE79",
        "#79DD7C",
    ]
    color_idx = 0
    for signal in msg.signals:
        if signal.byte_order == byte_order and signal.classification == "ts":
            # Use convert_signal to get correct start index in 'inorder' format
            idx, _ = convert_signal(
                signal.start_bit,
                signal.length,
                in_format=signal.byte_order,
                out_format="inorder",
                msg_size_bytes=msg.msg_length,
            )
            x0 = idx
            x1 = idx + signal.length - 1
            fig.add_vrect(
                x0=x0 - 0.5,
                x1=x1 + 0.5,
                fillcolor=colors[color_idx % len(colors)],
                opacity=0.2,
                layer="below",
                line_width=0,
                annotation_text=signal.name,
                annotation_position="top left",
            )
            color_idx += 1

    title = "Big-Endian" if byte_order == "be" else "Little-Endian"
    fig.update_layout(
        title=f"{title}: Bit Flip Probabilities",
        xaxis_title="Bit Number",
        yaxis_title="Bit Flip Probability",
        xaxis=dict(
            tickmode="array",
            tickvals=x_values_inorder,
            ticktext=x_values_byte_order,
            tickfont=dict(size=8),
        ),
        yaxis=dict(range=[0, 1.1]),
        legend=dict(
            orientation="h", y=-0.2, x=0.5, xanchor="center", font=dict(size=8)
        ),
    )
    return fig


def plot_ts_signal(signal):
    fig = go.Figure(
        data=[
            go.Scatter(
                x=signal.ts_data_timestamps,
                y=signal.ts_data,
                mode="lines",
                name=signal.name,
            ),
        ]
    )
    return fig

def plot_ts_signals(signals, normalized=False):
    fig = go.Figure(
        data = [
            go.Scatter(
                x=signal.ts_data_timestamps,
                y=signal.ts_data_raw_normalized if normalized else signal.ts_data,
                mode="lines",
                name=signal.name,
            )
            for signal in signals
        ]
    )
    return fig