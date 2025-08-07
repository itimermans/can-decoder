# Copilot Instructions for PandaCANDecoder

## Overview
- This project aims to create a new, extended version of an old project called `old-can-decoder`, stored in the `old-can-decoder-c56d4fe6` folder.
- The project aims to find hidden signal in CAN messages given a log file.

## Functionality in the old decoder
- A "Decoder" class is initiated with a csv containing a buffer (stream) of CAN messages
- The format of the buffer is csv with a row for each message, containing:
  - `timestamp`: Time of the message
  - `id`: CAN message ID
  - `data`: Data bytes of the message, all together, in hex (e.g. 0x12345678)
- After the decoder is instantiated, a function `generate_messages` is called, which generates a list of `Message` objects, basically filter by unique id
- After it, the generate_msg_ts_data function is called, which generates a time series for each message, containing the timestamp and data bytes
- Then, the calculate_signals function is called, which calculates the signals for each message, based on statistican decoding
- Signals found are defined by the Signal class inside each message
Additional tools provide plotting and visualization capabilities

## Things to be modified in the new version
- Old version supports byte size of up to 8 bytes
- New version should support byte size of up to 64 bytes
- The key changes will be internal format of ts_data and similar things: Instead of a np.uint64 as done now, it will be a dataframe with one column for each byte

## Workflow
- We will go very slowly, one step at a time, modifying individual parts of the code and checking that the result is similar to the old version