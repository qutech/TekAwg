#!/usr/bin/env python
"""Module for communication with and translation of data with a tektronix AWG5000 series.

06.2018 Modified by Simon Hmupohl
"""

from typing import Sequence, Union, Optional
from collections import OrderedDict
import itertools
import socket
import pyvisa
from pyvisa.resources.messagebased import MessageBasedResource
import time
import sys
import warnings
import numpy as np


WaveformId = Union[str, int]


class TekAwg:
    """Class which allows communication with a tektronix AWG5000 series (7000 series should work
     as well, but should be tested). This extends the socket class, and uses ethernet TCP/IP
     packets.

    Example:

        AWG_IP = 127.0.0.1
        AWG_PORT = 4001

        awg = tekawg5000.tekawg5000(AWG_IP,AWG_PORT)
        awg.print_waveform_list()
        awg.close()

    """

    def __init__(self, instrument: str):
        """Initialize connection and set timout to 1000ms

            Raises: socket.error"""

        if isinstance(instrument, str):
            instrument = pyvisa.ResourceManager().open_resource(instrument, read_termination='\n')

        self._inst = instrument

        self._n_channels = None

    @property
    def n_channels(self) -> int:
        if self._n_channels is None:
            self._n_channels = int(self.query('AWGControl:CONFigure:CNUMber?', expected_responses=1))
        return self._n_channels

    @property
    def instrument(self) -> MessageBasedResource:
        return self._inst

    @classmethod
    def connect_to_ip(cls, ip: str, port: int):
        return cls('TCPIP::{ip}::{port}::SOCKET'.format(ip=ip, port=port))

    def write(self, message: str, expected_responses=0) -> Optional[str]:
        """Sends text commands to the AWG5000 Series, no newline or return character required

            Args:
                message: str command to be sent to the AWG, multiple commands can be combined
                    with ";" as a separator

                expect_response: BOOL, whether a response is expected from the AWG, if true
                    then it will wait until a message is receieved from the AWG

                expected_length: INT, if a response is expected, this is the number of expected
                    responses to be recieved (usually one from each command sent)

            Returns: Str, response from AWG when expected_response=True, else it returns None

            Raises:
                IOError if a response was expected but not recieved
            """
        return self._write_helper(message, expected_responses)

    def query(self, query: str, expected_responses=1):
        result = self.write(query, expected_responses=expected_responses)
        return result[0] if expected_responses == 1 else tuple(result)

    def _write_helper(self, message, expected_responses: Union[int, bool]) -> Optional[Union[str, Sequence[str]]]:
        """This is the helper for the write command, this allows for multiple attempts to receive
        a response when a response is expected.
        """

        if expected_responses:
            result = self.instrument.query(message)

            if expected_responses is True:
                return result
            else:
                result = result.split(';')

            if len(result) != expected_responses:
                raise IOError('Got {} responses but expected  {}.'.format(len(result),
                                                                          expected_responses), result)
            return result

        else:
            self.instrument.write(message)

    def get_error_queue(self):
        err_queue = []
        err_num = int(self.write("*ESR?", True)[0])
        while err_num != 0:
            err_queue.append(self.write("SYSTEM:ERR?",True))
            err_num = int(self.write("*ESR?", True)[0])
        return err_queue


#############  GETTING SETTINGS   #########################
    def get_waveform_info(self) -> OrderedDict:
        waveform_names = self.get_waveform_names()

        result = OrderedDict()

        result['names'] = self.get_waveform_names()

        try:
            result['length'] = self.get_waveform_lengths(waveform_names)
        except IOError:
            pass

        try:
            result['type'] = self.get_waveform_types(waveform_names)
        except IOError:
            pass

        try:
            result['timestamp'] = self.get_waveform_types(waveform_names)
        except IOError:
            pass

        return result

#############  PRINTING SETTINGS   #########################

    def print_waveform_list(self):
        """Prints a formatted list of all the current waveforms in active memory of the AWG.

            Returns: 0  if printed correctly
                     -1 if there was a connection issue

        """
        con_error = False

        #get list of waveforms, and count how many we have
        try:
            waveform_list = self.get_waveform_names()
            num_saved_waveforms = len(waveform_list)
        except IOError:
            return -1

        try:
            waveform_lengths = self.get_waveform_lengths(waveform_list)
        except IOError:
            waveform_lengths = ["" for _ in range(num_saved_waveforms)]
            con_error = True

        try:
            waveform_types = self.get_waveform_types(waveform_list)
        except IOError:
            waveform_types = ["" for _ in range(num_saved_waveforms)]
            con_error = True

        try:
            waveform_date = self.get_waveform_timestamps(waveform_list)
        except IOError:
            waveform_date = ["" for _ in range(0, num_saved_waveforms)]
            con_error = True

        print("\nList of waveforms in memory:")
        print("\nIndex \t Name\t\t\t\t Data Points \tType\t\tDate")
        for i in range(num_saved_waveforms):
            print('{0:<9}{1: <32}{2: <15}{3:<16}{4:<5}'.format(i+1,
                                                                waveform_list[i],
                                                                waveform_lengths[i],
                                                                waveform_types[i],
                                                                waveform_date[i]))

        if con_error:
            print("\nConnection Error, partial list printed only")
            return -1
        else:
            return 0

    def print_config(self):
        """Print the current configuration of the AWG"""
        print("\n\nCurrent Settings\n")
        print("Hardware ID:     ", self.get_serial())
        print("Run Mode:        ", self.get_run_mode())
        print("Run State:       ", self.get_run_state())
        print("Frequency:       ", self.get_freq())

        cur_waves = self.get_cur_waveform()
        cur_amp = self.get_amplitude()
        cur_offset = self.get_offset()
        chan_state = self.get_chan_state()
        print("\nChannel Settings")
        print ('%-15s%-15s%-15s%-15s%-15s' %
               ("Setting", "Channel 1", "Channel 2", "Channel 3", "Channel 4"))
        print ('%-15s%-15s%-15s%-15s%-15s' %
               ("Waveforms:", cur_waves[0], cur_waves[1], cur_waves[2], cur_waves[3]))
        print ('%-15s%-15s%-15s%-15s%-15s' %
               ("Amplitude (V):", cur_amp[0], cur_amp[1], cur_amp[2], cur_amp[3]))
        print ('%-15s%-15s%-15s%-15s%-15s' %
               ("Offset (V):", cur_offset[0], cur_offset[1], cur_offset[2], cur_offset[3]))
        print ('%-15s%-15s%-15s%-15s%-15s' %
               ("Channel State:", chan_state[0], chan_state[1], chan_state[2], chan_state[3]))


        seq_list = self.get_seq_list()
        print("\nCurrent Sequence:")
        print ('%-15s%-15s%-15s%-15s%-15s%-15s%-15s' %
               ("Index", "Channel 1", "Channel 2", "Channel 3",
                "Channel 4", "Loop Count", "Jump Target"))
        for i in range(len(seq_list)):
            loop_count = self.get_seq_element_loop_cnt(i+1)
            jump_trg = self.get_seq_element_jmp_ind(i+1)
            print ('%-15i%-15s%-15s%-15s%-15s%-15s%-15s' %
                   (i+1, seq_list[i][0], seq_list[i][1], seq_list[i][2],
                    seq_list[i][3], loop_count, jump_trg))

        print("")


################  WAVEFORMS    #############################

    def get_waveform_names(self, waveform_indices: Union[Sequence[int], int, None]=None) -> Union[Sequence[str], str]:
        """Returns a list of all the currently saved waveforms on the AWG"""

        if waveform_indices is None:
            num_saved_waveforms = int(self.write("WLIST:SIZE?", True))
            waveform_indices = range(num_saved_waveforms)
        elif isinstance(waveform_indices, int):
            return self.get_waveform_names([waveform_indices])[0]

        waveform_indices = list(waveform_indices)

        waveform_list_cmd = 'WLIST:'
        waveform_list_cmd += ";".join(["NAME? " + str(i) for i in waveform_indices])

        waveform_names = self.write(waveform_list_cmd, len(waveform_indices))

        return waveform_names

    def get_waveform_lengths(self, waveform_ids: Union[Sequence[WaveformId], WaveformId]) -> Union[Sequence[int], int]:
        """Returns a list of lengths of all saved waveforms on the AWG"""
        if isinstance(waveform_ids, (str, int)):
            return self.get_waveform_lengths([waveform_ids])[0]

        waveform_ids = [str(name) for name in waveform_ids]

        num_requests = len(waveform_ids)

        waveform_length_cmd = 'WLIST:WAVeform:' + ";".join(["LENGTH? " + i for i in waveform_ids])
        waveform_lengths = self.write(waveform_length_cmd, num_requests)

        waveform_lengths = [int(length) for length in waveform_lengths]

        return waveform_lengths

    def get_waveform_types(self, waveform_ids: Union[Sequence[str], str]) -> Union[Sequence[str], str]:
        """returns the type of waveform which is stored on the AWG, IE: the AWG saves waveforms
        as either Integer ("INT") or Floating Point ("REAL") representations.

            Args:
                waveform_list: A single waveform name, or list of names

            Returns: list of strings containing either "INT" or "REAL" for int or float

            Raises:
                IOError if fewer types were returned then asked for"""
        if isinstance(waveform_ids, (str, int)):
            return self.get_waveform_types([waveform_ids])[0]

        elif not isinstance(waveform_ids, (list, tuple)):
            waveform_ids = list(waveform_ids)

        num_requests = len(waveform_ids)

        waveform_type_cmd = 'WLIST:WAVeform:' + ";".join(["TYPE? " + str(name) for name in waveform_ids])

        try:
            return self.write(waveform_type_cmd, num_requests)
        except IOError as err:
            raise IOError("Failed to retrieve lengths of all waveforms.") from err

    def get_waveform_timestamps(self, waveform_ids: Union[Sequence[str], str]) -> Union[Sequence[str], str]:
        """Returns the creation/edit timestamp of waveforms which are stored on the AWG,

            Args:
                waveform_list: A single waveform name, or list of names

            Returns: list of strings containing date of creation or last edit

            Raises:
                IOError if fewer types were returned then asked for"""
        if isinstance(waveform_ids, (str, int)):
            return self.get_waveform_timestamps([waveform_ids])[0]

        waveform_ids = self._parse_waveform_id(waveform_ids)

        num_requests = len(waveform_ids)

        waveform_date_cmd = 'WLIST:WAVeform:' + ";".join(["TSTAMP? " + str(name) for name in waveform_ids])

        waveform_dates = self.write(waveform_date_cmd, num_requests)

        if len(waveform_dates) == num_requests:
            return waveform_dates
        else:
            raise IOError("Failed to retrieve lengths of all waveforms.")

    def get_waveform_data(self, waveform_name, chunk_size=10*2**10):
        """Get the raw waveform data from the AWG, this will be in the packed format containing
        both the channel waveforms as well as the markers, this needs to be correctly formatted.
            Args:
                filename: name of the file to get from the AWG

            Returns: a string of binary containing the data from the AWG, header has been removed

            Raises:
                IOError if there was a timeout, most likely due to connection or incorrect name
        """
        waveform_name = self._parse_waveform_id(waveform_name)

        wf_length = self.get_waveform_lengths(waveform_name)
        data_type = self.get_waveform_types(waveform_name)
        if data_type == 'REAL':
            data_type = 'f'
        else:
            data_type = 'H'

        n_chunks = (wf_length + chunk_size - 1) // chunk_size

        waveform_data_cmd = 'WLISt:WAVeform:DATA? %s,{start}, {size}' % waveform_name

        waveform_data = []

        remaining_points = wf_length
        for chunk in range(n_chunks):
            cmd = waveform_data_cmd.format(start=chunk*chunk_size, size=min(chunk_size, remaining_points))

            received = self.instrument.query_binary_values(cmd, datatype=data_type, container=np.ndarray,
                                                           header_fmt='ieee')

            waveform_data.append(received)
            remaining_points -= chunk_size

        return np.concatenate(waveform_data)

    def new_waveform(self, filename, packed_data, packet_size=20000):
        """Creates a new waveform on the AWG and saves the data. It has error checking
            in the transmission, after every packet it asks the AWG if it had any issues
            writing the data to memory. If the AWG reports an error it resends that packet.
            This communication guarantees correct waveform on the AWG, but the requesting
            of updates from the AWG adds time to the transmission. There is a tradeoff
            between packet_size and speed of transfer, too large of packets and errors increase,
            too small and it takes longer because of waiting for the AWG to respond that it
            recieved the data correctly.

            Args:
                filename: the name of the new waveform

                packed_data: numpy ndarray or list of the already 'packed' data (both
                            the waveform and markers in an int16 format)

                packet_size: Size of the TCP/IP packet which are sent to the AWG.
                            This has a large effect on speed of transfer and stability.

            Returns:
                None

            Raises:
                IOError: if there was a connection error"""
        packed_data = ints_to_byte_str(packed_data)
        self.__new_waveform_int(filename, packed_data, packet_size)
        return None

    def _new_waveform(self, waveform_name, data: np.ndarray, chunk_size=10*2**10):
        if data.dtype == np.uint16:
            data_type = 'INT'
        elif data.dtype == np.float32:
            data_type = 'REAL'
        else:
            raise TypeError('Invalid data type', data.dtype)

        wf_length = data.size

        waveform_name = "%s" % waveform_name.strip('"')

        self.write('WLISt:WAVeform:NEW {name},{size},{data_type}'.format(name=waveform_name,
                                                                         size=wf_length,
                                                                         data_type=data_type))

        data_cmd = 'WLIST:WAVEFORM:DATA {name},{offset},{size},'

        n_chunks = (wf_length + chunk_size - 1) // chunk_size
        remaining_points = wf_length
        for chunk in range(n_chunks):
            self.instrument.write_binary_values(
                data_cmd.format(name=waveform_name,
                                offset=chunk*chunk_size,
                                size=min(chunk_size, remaining_points)),
                data[chunk*chunk_size:(chunk+1)*chunk_size],
                datatype=data.dtype.char,
                termination=self.instrument.write_termination
            )
            remaining_points -= chunk_size

    def __new_waveform_int(self, filename, packed_data, packet_size):
        """This is the helper function which actually sends the waveform to the AWG, see above."""
        errs = self.get_error_queue()
        #if errs != []:
        #    print errs,
        data_length = len(packed_data)

        if '"'+filename+'"' in self.get_waveform_names():
            self.del_waveform(filename)

        self.write('WLISt:WAVeform:NEW "'+filename+'",'+str(data_length/2)+",INT")

        if data_length >= packet_size*2:
            for i in range(0, data_length/(packet_size*2)):
                prefix = create_prefix(packed_data[i*packet_size*2:(i+1)*packet_size*2])
                packet = packed_data[i*packet_size*2:(i+1)*packet_size*2]
                success = False
                while not success:
                    success = self.write('WLIST:WAVEFORM:DATA "'+filename+'",'
                                         +str(i*packet_size)+','
                                         +str(packet_size)+','
                                         +prefix
                                         +packet
                                         +";*ESR?\r\n", True) == "0"

        remaining_data_size = data_length-data_length/(packet_size*2)*packet_size*2

        if remaining_data_size > 0:
            self.write('WLIST:WAVeform:DATA "'+filename+'",'
                       +str((data_length-remaining_data_size)/2)+','
                       +str(remaining_data_size/2)+","
                       +create_prefix(packed_data[data_length-remaining_data_size:])
                       +packed_data[data_length-remaining_data_size:]
                       +"\r\n")

        errs = self.get_error_queue()
        if errs:
            warnings.warn('ERRORS: ' + '; '.join(errs))

    def del_waveform(self, filename):
        """Delete Specified Waveform"""
        self.write('WLISt:WAVeform:DELete "'+filename+'"')



#######################   AWG SETTINGS  ############################

    def get_serial(self) -> str:
        """Returns the hardware serial number and ID as a string"""
        return self.query("*IDN?")

    def get_freq(self):
        """Returns the current sample rate of the AWG"""
        return self.query("FREQ?")

    def set_freq(self, freq):
        """Sets the current sample rate of the AWG"""
        self.write("FREQ "+str(freq))

    def get_run_mode(self):
        """Gets the current running mode of the AWG: SEQ, CONT, TRIG, GAT"""
        return self.query("AWGCONTROL:RMODE?")

    def set_run_mode(self, mode):
        """Sets the run mode of the AWG, allowed modes are:
            continuous, triggered, gated, sequence"""
        if mode.lower() in ["continuous", "cont",
                            "triggered", "trig",
                            "gated", "gat",
                            "sequence", "seq"]:
            self.write("AWGCONTROL:RMODE "+mode)
        else:
            raise RuntimeError('Invalid mode', mode)

    def get_run_state(self):
        """Gets the current state of the AWG, possible states are:
        stopped, waiting for trigger, or running"""
        state = self.query("AWGControl:RSTate?")
        if state == "0":
            return "Stopped"
        elif state == "1":
            return "Waiting for Trigger"
        elif state == "2":
            return "Running"
        raise IOError("Not valid run state")

    def run(self):
        """Start running the AWG"""
        self.write("AWGControl:RUN")

    def stop(self):
        """Stop the AWG"""
        self.write("AWGCONTROL:STOP")

    def _parse_channel(self, channel) -> Sequence[str]:
        if channel is None:
            channel = range(1, 1 + self.n_channels)
        elif isinstance(channel, (int, str)):
            channel = [channel]
        return [str(int(ch)) for ch in channel]

    def _parse_waveform_id(self, waveform_id):
        if isinstance(waveform_id, int):
            return str(waveform_id)
        elif isinstance(waveform_id, str):
            try:
                return str(int(waveform_id))
            except ValueError:
                return '"%s"' % waveform_id.strip('"')

        return [self._parse_waveform_id(wf_id)
                for wf_id in waveform_id]

    def get_amplitude(self, channel=None):
        channel = self._parse_channel(channel)
        cmd_str = ';'.join([':SOURCE' + c + ':VOLTAGE?' for c in channel])
        return [float(x) for x in self.write(cmd_str, len(channel))]

    def set_amplitude(self, amplitude, channel=None):
        channel = self._parse_channel(channel)
        if not isinstance(amplitude, (list, tuple)):
            amplitude = [amplitude]*len(channel)

        if len(amplitude) != len(channel):
            raise ValueError("Number of channels does not match number of amplitudes.")

        cmd_str = []
        for i in range(len(channel)):
            cmd_str.append(':SOURCE'+str(int(channel[i]))+':VOLTAGE '+str(amplitude[i]))
        cmd_str = ';'.join(cmd_str)
        self.write(cmd_str)

    def get_offset(self, channel=None):
        channel = self._parse_channel(channel)
        cmd_str = ';'.join([':SOURCE'+str(c)+':VOLTAGE:OFFSET?' for c in channel])
        return [float(x) for x in self.write(cmd_str, len(channel))]

    def set_offset(self, offset, channel=None):
        channel = self._parse_channel(channel)
        if not isinstance(offset, list): offset = [offset]*len(channel)

        if len(offset) != len(channel):
            raise ValueError("Number of channels does not match number of amplitudes.")

        cmd_str = []
        for i in range(len(channel)):
            cmd_str.append(':SOURCE'+str(channel[i])+':VOLTAGE:OFFSET '+str(offset[i]))
        cmd_str = ';'.join(cmd_str)
        self.write(cmd_str)

    def get_marker_high(self, marker, channel=None):
        channel = self._parse_channel(channel)
        cmd_str = ';'.join([':SOURCE'+str(c)+':MARKER'+str(int(marker))+':VOLTAGE:HIGH?' for c in channel])
        return [float(x) for x in self.write(cmd_str, len(channel))]

    def set_marker_high(self, voltage, marker, channel=None):
        """Set whether the channels are on or off, where 0 means off and 1 means on"""
        assert int(marker) in [1,2]
        channel = self._parse_channel(channel)
        if not isinstance(voltage, list): voltage = [voltage]*len(channel)

        if len(voltage) != len(channel):
            raise ValueError("Number of channels does not match number of voltages.")

        cmd_str = ''
        for i in range(len(channel)):
            cmd_str = cmd_str + ';:SOURCE{}:MARKER{}:VOLTAGE:HIGH {}'.format(int(channel[i]),int(marker),voltage[i])
        self.write(cmd_str)

    def get_marker_low(self, marker, channel=None):
        channel = self._parse_channel(channel)
        cmd_str = ';'.join([':SOURCE'+str(int(c))+':MARKER'+str(int(marker))+':VOLTAGE:LOW?' for c in channel])
        return [float(x) for x in self.write(cmd_str, len(channel))]

    def set_marker_low(self, voltage, marker, channel=None):
        """Set whether the channels are on or off, where 0 means off and 1 means on"""
        assert int(marker) in [1,2]
        channel = self._parse_channel(channel)
        if not isinstance(voltage, list): voltage = [voltage]*len(channel)

        if len(voltage) != len(channel):
            raise ValueError("Number of channels does not match number of voltages.")

        cmd_str = ''
        for i in range(len(channel)):
            cmd_str = cmd_str + ';:SOURCE{}:MARKER{}:VOLTAGE:LOW {}'.format(int(channel[i]),int(marker),voltage[i])
        self.write(cmd_str)

    def get_chan_state(self, channel=None):
        channel = self._parse_channel(channel)
        cmd_str = ';'.join([':OUTPUT'+str(c)+'?' for c in channel])
        return [int(x) for x in self.write(cmd_str, len(channel))]

    def set_chan_state(self, state, channel=None):
        """Set whether the channels are on or off, where 0 means off and 1 means on"""
        channel = self._parse_channel(channel)
        if not isinstance(state, list): state = [state]*len(channel)

        if len(state) != len(channel):
            raise ValueError("Number of channels does not match number of states.")

        cmd_str = ''
        for i in range(len(channel)):
            cmd_str = cmd_str + ';:OUTPUT'+str(channel[i])+':STATE '+str(state[i])
        self.write(cmd_str)

    def get_trig_source(self):
        return self.write("TRIG:SOUR?",True)

    def set_trig_source(self,source):
        trig_sources = ["int","internal","ext","external"]
        if source.lower() in trig_sources:
            self.write("TRIG:SOUR "+source)

    def get_trig_interval(self):
        return float(self.write("TRIG:TIM?",True))

    def set_trig_interval(self, interval):
        assert float(interval) > 0
        self.write("TRIG:TIM "+str(float(interval)))

    def trig(self):
        return self.write("*TRG")

####################  SEQUENCER ######################

    def get_cur_waveform(self, channel=None):
        channel = self._parse_channel(channel)
        cmd_str = ';'.join([':SOURCE'+str(c)+':WAV?' for c in channel])
        return self.write(cmd_str, len(channel))

    def set_cur_waveform(self, waveform_name, channel=None):
        channel = self._parse_channel(channel)
        cmd_str = ';'.join([':SOURCE'+str(c)+':WAV "'+waveform_name+'"' for c in channel])
        self.write(cmd_str)

    def set_seq_element(self, element_index, waveform_name, channel=None):
        channel = self._parse_channel(channel)
        cmd_str = ';'.join([':Sequence:ELEM'
                            +str(element_index)
                            +':WAV'+str(c)
                            +' "'
                            +waveform_name
                            +'"' for c in channel])
        self.write(cmd_str)

    def get_seq_element(self, element_index, channel=None):
        channel = self._parse_channel(channel)
        cmd_str = ';'.join([':Sequence:ELEM'+str(element_index)+':WAV'+str(c)+"?" for c in channel])
        return self.write(cmd_str, len(channel))

    def get_seq_element_loop_cnt(self, element_index) -> int:
        return int(self.query('SEQuence:ELEMent'+str(element_index)+':LOOP:COUNt?'))

    def set_seq_element_loop_cnt(self, element_index, count):
        return self.write('SEQuence:ELEMent'+str(element_index)+':LOOP:COUNt '+str(count))

    def get_seq_element_loop_inf(self, element_index) -> bool:
        return int(self.query('SEQuence:ELEMent'+str(element_index)+':LOOP:INFinite?')) == 1

    def get_seq_length(self):
        return int(self.query('SEQ:LENGTH?'))

    def set_seq_length(self, length):
        self.write('SEQ:LENGTH '+str(length))

    def get_seq_element_jmp_ind(self, element_index):
        tar_type = self.get_seq_element_jmp_type(element_index)
        if tar_type == "IND":
            return self.query('SEQuence:ELEMent'+str(element_index)+':JTARget:INDex?')
        else:
            return tar_type

    def set_seq_element_jmp_ind(self, element_index, target):
        self.set_seq_element_jmp_type(element_index, "ind")
        self.write('SEQuence:ELEMent'+str(element_index)+':JTARget:INDex '+str(target))

    def get_seq_element_jmp_type(self, element_index):
        return self.query('SEQuence:ELEMent'+str(element_index)+':JTARget:TYPE?')

    def set_seq_element_jmp_type(self, element_index, tar_type):
        if tar_type.lower() in ["index", "ind", "next", "off"]:
            return self.write('SEQuence:ELEMent'+str(element_index)+':JTARget:TYPE '+str(tar_type))

    def get_seq_element_goto_state(self, element_index) -> bool:
        return int(self.query('SEQuence:ELEMent' + str(element_index) + ':GOTO:STAT?')) == 1

    def get_seq_element_goto_ind(self, element_index) -> int:
        return int(self.query('SEQuence:ELEMent' + str(element_index) + ':GOTO:IND?'))

    def get_seq_element_wait(self, element_index) -> bool:
        return self.query('SEQuence:ELEMent' + str(element_index) + ':TWA?') == 'ON'

    def get_seq_list(self):
        """Get the current list of waveforms in the sequencer"""
        return [self.get_seq_element(i)
                for i in range(1, 1+self.get_seq_length())]

    def set_seq_list(self, seq_list, position):
        """Set the sequence list"""

        if isinstance(position, int):
            position = itertools.count(position)

        position = list(itertools.islice(position, len(seq_list)))

        seq_len = self.get_seq_length()

        cmd = []
        for pos, (entries, wait, repeat, event_jump, goto) in zip(position, seq_list):
            if pos > seq_len:
                raise ValueError('Invalid position')

            if len(entries) != self.n_channels:
                raise ValueError('Invalid channel count')

            base = ':SEQ:ELEM{pos}:'.format(pos=pos)

            for ch, entry in enumerate(entries):
                if entry is not None:
                    cmd.append(
                        base + 'WAV{ch} {entry}'.format(ch=ch+1, entry=entry)
                    )

            if wait is not None:
                cmd.append(
                    base + 'TWA {wait}'.format(wait=int(wait))
                )

            if repeat is not None:
                if repeat == float('inf'):
                    cmd.append(base + 'LOOP:INF 1')

                else:
                    cmd.append(base + 'LOOP:INF 0')

                    cmd.append(base + 'LOOP:COUNT %d' % repeat)

            if event_jump:
                raise NotImplementedError()

            if goto is not None:
                if goto is False:
                    cmd.append(base + 'GOTO:STAT 0')
                else:
                    cmd.append(base + 'GOTO:STAT 1')
                    cmd.append(base + 'GOTO:IND %d' % goto)

        chunk_size = 32
        chunk_num = (len(cmd) + chunk_size - 1) // chunk_size

        for chunk in range(chunk_num):
            cmd_chunk = ";".join(itertools.islice(cmd, chunk*chunk_size, (chunk+1)*chunk_size))
            self.write(cmd_chunk)


#These are the bit conversions needed for accurate representation on the AWG
_bit_depth_mult_offset = {8:  (127, 127),
                          12: (2047, 2047),
                          14: (8191, 8191),
                          16: (32767, 32767)}


def create_prefix(data):
    return "#"+str(len(list(str(len(data)))))+str(len(data))

def bifloat_to_uint(value, bit_depth):
    """Convert a float on the range [-1.0, 1.0] to a unsigned int.

    Not a totally straightforward conversion, this conversion will result in matching
    values seen on the AWG, however some decimals may not be represented exactly
    as certain fractions in decimal are not representable in binary.

    Args:
        value: a single float, or list of floats, or numpy array of
            floats to operate on
        bit_depth: the target AWG's bit depth, taken from the set {8, 12, 14, 16}

    Returns:
        the converted input value/list/ndarray

    Raises:
        ValueError for a bit depth outside the set of supported values.
    """
    try:
        mult, offset = _bit_depth_mult_offset[bit_depth]
    except KeyError:
        raise ValueError("No rule exists for converting a bipolar float to a bit depth of "
                         "'{}'; supported bit depths are {}."
                         .format(bit_depth, _bit_depth_mult_offset.keys()))
    # ndarray case
    if isinstance(value, np.ndarray):
        output = np.empty(value.shape, dtype=int)
        np.multiply(value, mult, output, casting='unsafe')
        output += offset
        return output

    # generic iterable case
    try:
        val_iter = iter(value)
        return [int(val*mult + offset) for val in val_iter]
    except TypeError:
        # hopefully this is a scalar
        return int(value * mult + offset)

def uint_to_bifloat(value, bit_depth):
    """Convert an unsigned int to a float on the range [-1.0, 1.0].

    This is an undo of the bifloat_to_uint function.

    Args:
        value: a single uint, or list of uints, or numpy array of
            uints to operate on
        bit_depth: the target AWG's bit depth, taken from the set {8, 12, 14, 16}

    Returns:
        the converted input value/list/ndarray

    Raises:
        ValueError for a bit depth outside the set of supported values.
    """
    try:
        mult, offset = _bit_depth_mult_offset[bit_depth]
    except KeyError:
        raise ValueError("No rule exists for converting a bipolar float to a bit depth of "
                         "'{}'; supported bit depths are {}."
                         .format(bit_depth, _bit_depth_mult_offset.keys()))
    # ndarray case
    if isinstance(value, np.ndarray):
        output = np.empty(value.shape, dtype=float)
        value = value - float(offset)
        np.divide(value, float(mult), output, casting='unsafe')
        return output

    # generic iterable case
    try:
        val_iter = iter(value)
        return [float((val- offset)/float(mult)) for val in val_iter]
    except TypeError:
        # hopefully this is a scalar
        return float((value- offset)/float(mult))



def merge_arb_and_markers(arb=None, mk1=None, mk2=None, bit_depth=14):
    """Merge arbitrary waveform and marker values into a binary array of AWG codes.

    If any of the inputs are not supplied, they will be filled with placeholder
    arrays of zeros.  This function is only set up to support 10 and 12-bit AWGs

    Args:
        arb: the arbitrary waveform data on the range [-1.0, 1.0]
        mk1, mk2: the marker data.  Can be supplied as a booleans, integers
            (0 -> off, non-zero -> on), or floats (0.0 -> off, all other values -> on)

    Returns:
        An ndarray of Tektronix-formatted AWG sample codes.

    Raises:
        ValueError if no sequences were supplied or an unsupported bit depth was
            provided.
        UnequalPatternLengths if any of the input patterns were of unequal length.
    """
    supported_bit_depths = (8, 14)
    if bit_depth not in supported_bit_depths:
        raise ValueError("Unsupported bit depth of {}; valid bit depths are {}"
                         .format(bit_depth, supported_bit_depths))
    if arb is None and mk1 is None and mk2 is None:
        raise ValueError("Must supply at least one sequence pattern to create a"
                         " merged AWG binary array.")
    if arb is not None:
        master_pat = arb
    else:
        master_pat = mk1 if mk1 is not None else mk2

    seq_len = len(master_pat)

    arb = np.zeros(seq_len, dtype=float) if arb is None else arb
    mk1 = np.zeros(seq_len, dtype=bool) if mk1 is None else mk1.astype(bool)
    mk2 = np.zeros(seq_len, dtype=bool) if mk2 is None else mk2.astype(bool)

    if len(arb) != len(mk1) or len(mk1) != len(mk2):
        raise UnequalPatternLengths("Supplied patterns of unequal length: "
                                    "len(arb) = {}, len(mk1) = {}, len(mk2) = {}"
                                    .format(len(arb), len(mk1), len(mk2)))

    # all patterns have the same length and are valid
    # convert the bipolar float to integer
    arb = bifloat_to_uint(arb, bit_depth).astype("<u2", copy=False)
    #if bit_depth == 8:
    #    np.left_shift(arb, 6, arb)

    mk1 = mk1.astype("<u2", copy=False)
    mk2 = mk2.astype("<u2", copy=False)

    # bit shift mk1 and mk2 to the correct flag bits, 15 and 16 respectively
    np.left_shift(mk1, 14, mk1)
    np.left_shift(mk2, 15, mk2)

    np.bitwise_or(arb, mk1, arb)
    np.bitwise_or(arb, mk2, arb)

    return arb

def ints_to_byte_str(codes):
    """Convert an ndarray of AWG sample codes to bytes of the proper endianness.

    Args:
        codes: ndarray of AWG sample codes

    Returns: a byte array in little-endian order.

    Raises:
        TypeError if the incoming ndarray object does not have meaningful
            endianess.
    """
    # get the endianness of the ndarray
    byte_order = codes.dtype.byteorder
    if byte_order == '=':
        # native byte order, ask the system
        byte_order = sys.byteorder
    elif byte_order == '<':
        byte_order = 'little'
    elif byte_order == '>':
        byte_order = 'big'
    else:
        raise TypeError("Got an ndarray object without meaningful endianness!")

    # if we're little-endian, return the bytes
    if byte_order == 'little':
        return codes.tobytes()
    else:
    # otherwise, byte-swap first
        return codes.byteswap().tobytes()
#.4943891
def byte_str_to_vals(codes,str_format="INT"):
    if str_format == "INT":
        vals_ints = np.fromstring(codes, dtype="<u2")
        (arb, mk1, mk2) = unmerge_arb_and_markers(vals_ints)
        return (uint_to_bifloat(arb, 14), mk1, mk2)
    elif str_format == "REAL":
        return np.fromstring(codes, dtype="<f4, <u1")

def unmerge_arb_and_markers(codes):
    seq_len = len(codes)

    arb_mask = np.zeros(seq_len, dtype="<u2")+2**14-1
    mk1_mask = np.zeros(seq_len, dtype="<u2")+2**14
    mk2_mask = np.zeros(seq_len, dtype="<u2")+2**15

    arb = np.empty(seq_len, dtype='uint16')
    mk1 = np.empty(seq_len, dtype=bool)

    mk2 = np.empty(seq_len, dtype=bool)

    np.bitwise_and(codes, arb_mask, arb)
    np.bitwise_and(codes, mk1_mask, mk1_mask)
    np.bitwise_and(codes, mk2_mask, mk2_mask)

    np.not_equal(mk1_mask, np.zeros(seq_len), mk1)
    np.not_equal(mk2_mask, np.zeros(seq_len), mk2)

    return (arb, mk1, mk2)

class UnequalPatternLengths(Exception):
    pass


