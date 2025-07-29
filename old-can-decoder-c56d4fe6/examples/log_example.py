from PandaCANDecoder.logger import Logger

# create logger object
can_logger = Logger()

# connect pandas according to CLI instructions
can_logger.get_pandas()

# log while sending pid
pids = {0x740: b"\x22\x11\x1A"}
can_logger.log(pids)
