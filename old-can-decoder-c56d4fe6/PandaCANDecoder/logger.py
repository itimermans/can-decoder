'''
Based on Comma ai Panda logger code: https://github.com/commaai/panda/blob/master/examples/can_logger.py
made available under MIT license: https://github.com/commaai/panda/blob/master/LICENSE
'''

import csv
from panda import Panda
from panda.isotp import isotp_send
import usb1
import os
import time


class Logger:
    def __init__(self):
        self.pid_pandas = []
        self.main_pandas = []
        self.panda_serial_numbers = []
        self.logs_dir = './logs'

    def _poll_for_pandas(self, panda_type, context):
        if panda_type == 'pid':
            panda_list = self.pid_pandas
        elif panda_type == 'main':
            panda_list = self.main_pandas

        try:
            while True:
                for device in context.getDeviceList(skip_on_error=True):
                    if device.getVendorID() == 0xbbaa and device.getProductID() in (0xddcc, 0xddee) and device.getSerialNumber() not in self.panda_serial_numbers:
                        print(f"Connecting to Panda {len(self.pid_pandas) + len(self.main_pandas)}") # TODO: num error
                        serial_num = device.getSerialNumber()
                        panda = Panda(serial_num)
                        if panda_type == 'pid':
                            panda.set_safety_mode(Panda.SAFETY_ELM327)
                        panda.can_clear(0)
                        panda_list.append(panda)
                        self.panda_serial_numbers.append(serial_num)
        except KeyboardInterrupt:
            return

    def get_pandas(self):
        '''
        Connect to Pandas
        '''
        context = usb1.USBContext()
        print("Connect Diagnostic/PID Panda now... Press 'ctl-c' when finished")
        self._poll_for_pandas('pid', context)
        if len(self.pid_pandas) == 0:
            print("\nNo Diagnostic/PID Panda device detected")

        print("Connect Main HS-CAN Panda(s) now... Press 'ctl-c' when finished")
        self._poll_for_pandas('main', context)
        if len(self.main_pandas) == 0:
            print("\nNo Main HS-CAN Panda devices detected")

    def log(self, log_file_name='log', pids=None):
        '''
        Log CAN data from Pandas to csv file

            Parameters:
                log_file_name (str)
        '''
        if len(self.pid_pandas) + len(self.main_pandas) == 0:
            print("WARNING: No Panda devices are registed. Run Logger.get_pandas() after connecting Pandas.")
            return

        if self.pid_pandas and not pids:
            print("WARNING: A Diagnostic/PID Panda was registed. PIDs must be provided to log function.")
            return

        try:
            os.makedirs(self.logs_dir, exist_ok=True)
            outputfile = open(os.path.join(self.logs_dir, log_file_name + '.csv'), 'w')
            csvwriter = csv.writer(outputfile)
            csvwriter.writerow(['TimeStampNs', 'PandaNum', 'Bus', 'MessageID', 'Message', 'MessageLength'])

            print(f"Writing csv file {self.logs_dir}/{log_file_name}.csv. Press ctl-c to exit...\n")

            bus_msg_cnt = {}
            for i in range(len(self.pid_pandas) + len(self.main_pandas)):
                bus_msg_cnt[i] = [0,0,0]
            print("Message Count:")
            print("{Panda Num: [Bus 0, Bus 1, Bus 2]}")

            start_time_stamp_ns = time.time_ns()

            while True:
                for n, p in enumerate(self.pid_pandas + self.main_pandas):
                    # if pip panda send all PIDS
                    # i have no idea how often you actually have to send the pids
                    # this might now work if you have >1 message
                    for pid_p in self.pid_pandas:
                        for m in pids.keys():
                            isotp_send(pid_p, pids[m], m)

                    can_recv = p.can_recv()
                    time_stamp_ns = time.time_ns() - start_time_stamp_ns

                    for address, _, dat, src in can_recv:
                        csvwriter.writerow([str(time_stamp_ns), str(n), str(src), str(hex(address)), f"0x{dat.hex()}", len(dat)])

                        if src in [0, 1, 2]:
                            bus_msg_cnt[n][src] += 1

                print(bus_msg_cnt, end='\r')

        except KeyboardInterrupt:
            print(bus_msg_cnt)
            print("Finished logging")
            outputfile.close()
