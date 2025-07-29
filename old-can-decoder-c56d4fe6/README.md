# PandaCANDecoder
A python library for logging and decoding CAN messages using [Comma.ai Pandas](https://github.com/commaai/panda)

### Documentation
[`/docs`](./docs):
- [Installation](./docs/installation.md)
- [Hardware](./docs/hardware.md)
- [Logging](./docs/logging.md)
- [Decoding](./docs/decoding.md)


### Examples
[`/examples`](./examples):
- [Logging](./examples/log_example.py)
- [Decoding](./examples/decode_example.ipynb)

### Capabilities
**Logging**
- Read high speed data from all CAN buses using multiple Pandas
- Send diagnostic PIDs
- Record to CSV file
- Use Comma.ai `pandacan` library for sending/receiving data efficiently

**Decoding**
- Get unique messages across all CAN buses
- Predict signal classifications using statistical methods
- Match reference signals to predicted signals
- Plot signal and message data for visual decoding
- Read and write in DBC format

### TODO
**Large Enhancements**
- [ ] Expand decoding functions to more log file formats
- [ ] Decode in real-time
- [ ] CAN-FD support
    - Message sizes up to 64 bytes (512 bits)
- [ ] Improve DBC file reading/writing
- [ ] Store recorded log more efficiently
- [ ] Send CAN messages
- [ ] Tools for longer log files (trimming, improving efficiency)
- [ ] Non-time series decoding tools (single bit flips)

**Small Enhancements**
- [ ] New method: return value of `signal a` when `signal b` (a single bit) flips
- [ ] Match single bit flip to time series signals with same value (within some user-given error) at each bit flip
- [ ] New method: Plot signal alongside raw data, useful for scaling
- [ ] Account for time stamp differences when numerically comparing signals, possibly interpolate to get most accurate results
- [ ] Add option to not write raw data to `.npy` file in `PandaCANDecoder.message.Message.generate_ts_data()`
- [ ] Add option to `PandaCANDecoder.decoder.Decoder.print_msgs()` for only printing specific messages
- [ ] Send multiple PIDs on the same arbitration ID and filter responses based on the response message data

**Bug Fixes**
- [ ] `PandaCANDecoder.decoder.Decoder.plot_signal_multi()` only uses 4 colors. Should change `linestyle` for >4 signals
- [ ] `PandaCANDecoder.dbc_str_to_signal()` regex formula for signal unit value does not work if unit contains special characters
- [ ] `PandaCANDecoder.decoder.Decoder.print_msgs()` should print in ascending order

_Updated: 08/04/2022_
