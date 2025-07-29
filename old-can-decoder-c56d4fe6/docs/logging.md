# Logging

### Overview

- Log raw CAN data to `.csv` file
- Log format:

![csv format](./images/csv_format.png)

### Example
- [`log_example.py`](./examples/log_example.py)

### Example walk-through

1. Import library

    ```python
    from PandaCANDecoder.logger import Logger
    ```

2. Create logger
    ```python
    can_logger = Logger()
    ```

3. Connect pandas
    ```python
    can_logger.get_pandas()
    ```
    ```
    Plug in Diagnostic/PID Panda now... Press 'Ctl-C' when finished
    Connecting to Panda 0
    opening device 310028000551363338383037 0xddcc
    connected
    ```
    ```
    ^C
    ```
    ```
    Plug in Main HS-CAN Panda(s) now... Press 'Ctl-C' when finished
    Connecting to Panda 1
    opening device 3c0043000551363338383037 0xddcc
    connected
    ```
    ```
    ^C
    ```

4. Log
    ```python
    can_logger.log()
    ```
    ```
    Writing csv file ./logs/log.csv. Press Ctrl-C to exit...
    Message Count:
    {Panda Num: [Bus 0, Bus 1, Bus 2]}
    {0: [0, 0, 0], 1: [10527, 12810, 22114]}
    ```
    ```
    ^C
    ```
    ```
    Finished logging
    ```

5. Alternatively, log while sending PIDs
    ```python
    # pid format:
    #   {message id: message as byte array, ...}
    #   byte array is formatted as full message by pandacan
    #   byte 1: Custom Service (optional)
    #   byte 2: PID code byte 1
    #   byte 3: PID code byte 2
    pids = {0x740: b"\x22\x11\x1A"}
    can_logger.log(pids)
    ```

_Updated: 08/03/2022_
