"""
AWR1642 Radar Sensor Driver for Linux
======================================
This module provides a Python interface for controlling the Texas Instruments
AWR1642 radar sensor via its serial command-line interface (CLI) and parsing
the TLV (Type-Length-Value) formatted data stream.

Key Differences from AWR2243:
- Uses TLV packet format with magic word headers
- Has 4 RX antennas, 2 TX antennas (vs 4 RX, 1 TX on AWR2243)
- Different frequency range (77 GHz vs 60 GHz)
- Processes detected objects and range profiles instead of raw ADC

Author: VitalFlow-Radar Project
Platform: Linux
"""

import serial
import time
import numpy as np
import struct


# Default configuration file path
DEFAULT_CONFIG_FILE = 'vital_signs_awr1642.cfg'

def load_config_from_file(config_file):
    """Load configuration commands from a .cfg file.
    
    Parameters
    ----------
    config_file : str
        Path to the .cfg configuration file
        
    Returns
    -------
    list of str
        List of configuration commands (non-comment, non-empty lines)
    """
    commands = []
    try:
        with open(config_file, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if line and not line.startswith('%') and not line.startswith('#'):
                    commands.append(line)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    return commands


# Reference configuration for AWR1642 - Vital Signs Monitoring
# 20 FPS, optimized for vital sign detection at 0.3-0.9m range
# NOTE: Use load_config_from_file(DEFAULT_CONFIG_FILE) for the latest config
config_commands_1642 = [
    'sensorStop',
    'flushCfg',
    'dfeDataOutputMode 1',
    'channelCfg 15 3 0',  # RX: 1111 (4 antennas), TX: 11 (2 antennas)
    'adcCfg 2 1',
    'adcbufCfg -1 0 0 1 0',
    'profileCfg 0 77 7 6 57 0 0 70 1 200 4000 0 0 48',  # 77 GHz, 200 samples
    'chirpCfg 0 0 0 0 0 0 0 1',
    'frameCfg 0 0 2 0 50 1 0',  # 3 chirps per frame (0-2), 50ms period (20 FPS)
    'lowPower 0 1',
    'guiMonitor 0 0 0 0 1',
    'calibDcRangeSig -1 0 0 0 0',
    'vitalSignsCfg 0.3 0.9 256 512 4 0.1 0.05 100000 300000',  # Vital signs: 0.3-0.9m range
    'motionDetection 1 20 2.0 0',
    'sensorStart'  # Start sensor after configuration
]


class AWR1642:
    """
    Interface class for the Texas Instruments AWR1642 radar sensor.
    
    This class handles serial communication with the radar's CLI port for
    configuration and the data port for receiving TLV-formatted packets.
    
    Key features:
    - TLV packet parsing with magic word detection
    - Detected objects extraction
    - Range profile processing
    - Frame-based data acquisition
    """
    
    # TLV Message Types
    MMWDEMO_OUTPUT_MSG_NULL = 0
    MMWDEMO_OUTPUT_MSG_DETECTED_POINTS = 1
    MMWDEMO_OUTPUT_MSG_RANGE_PROFILE = 2
    MMWDEMO_OUTPUT_MSG_NOISE_PROFILE = 3
    MMWDEMO_OUTPUT_MSG_AZIMUT_STATIC_HEAT_MAP = 4
    MMWDEMO_OUTPUT_MSG_RANGE_DOPPLER_HEAT_MAP = 5
    MMWDEMO_OUTPUT_MSG_STATS = 6
    MMWDEMO_OUTPUT_MSG_DETECTED_POINTS_SIDE_INFO = 7
    MMWDEMO_OUTPUT_MSG_VITALSIGNS = 8
    
    # Magic word for packet identification
    MAGIC_WORD = [0x02, 0x01, 0x04, 0x03, 0x06, 0x05, 0x08, 0x07]
    
    def __init__(self, cli_port='/dev/ttyACM0', data_port='/dev/ttyACM1', 
                 cli_baud=115200, data_baud=921600):
        """
        Initialize the AWR1642 radar interface.
        
        Parameters
        ----------
        cli_port : str
            Path to the CLI/User serial port (e.g., '/dev/ttyACM0')
        data_port : str
            Path to the Data serial port (e.g., '/dev/ttyACM1')
        cli_baud : int
            Baud rate for the CLI port (default: 115200)
        data_baud : int
            Baud rate for the data port (default: 921600)
        """
        self.cli_port_name = cli_port
        self.data_port_name = data_port
        self.cli_baud = cli_baud
        self.data_baud = data_baud
        
        self.cli_serial = None
        self.data_serial = None
        self.is_connected = False
        
        # Byte buffer for TLV packet assembly
        self.byte_buffer = np.zeros(2**15, dtype='uint8')
        self.byte_buffer_length = 0
        
        # Configuration parameters (populated during configure_sensor)
        self.config_params = {
            'numRxAnt': 4,
            'numTxAnt': 2,
            'numAdcSamples': 200,
            'numChirpsPerFrame': 3,
            'numDopplerBins': 0,
            'numRangeBins': 256,
            'rangeResolutionMeters': 0.0,
            'rangeIdxToMeters': 0.0,
            'dopplerResolutionMps': 0.0,
            'maxRange': 0.0,
            'maxVelocity': 0.0
        }
        
    def connect(self):
        """
        Open serial connections to both CLI and Data ports.
        
        Raises
        ------
        serial.SerialException
            If unable to open the serial ports
        """
        try:
            print(f"Connecting to CLI port: {self.cli_port_name} @ {self.cli_baud} baud")
            self.cli_serial = serial.Serial(
                port=self.cli_port_name,
                baudrate=self.cli_baud,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                bytesize=serial.EIGHTBITS,
                timeout=1
            )
            
            print(f"Connecting to Data port: {self.data_port_name} @ {self.data_baud} baud")
            self.data_serial = serial.Serial(
                port=self.data_port_name,
                baudrate=self.data_baud,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                bytesize=serial.EIGHTBITS,
                timeout=1
            )
            
            self.is_connected = True
            time.sleep(0.5)  # Allow ports to stabilize
            
            # Clear any residual data
            self.cli_serial.reset_input_buffer()
            self.data_serial.reset_input_buffer()
            
            print("✓ Serial ports connected successfully")
            
        except serial.SerialException as e:
            print(f"✗ Failed to open serial ports: {e}")
            raise
    
    def disconnect(self):
        """
        Close serial connections to both CLI and Data ports.
        """
        if self.cli_serial and self.cli_serial.is_open:
            try:
                self._send_command('sensorStop')
            except:
                pass
            self.cli_serial.close()
            print("✓ CLI port closed")
        
        if self.data_serial and self.data_serial.is_open:
            self.data_serial.close()
            print("✓ Data port closed")
        
        self.is_connected = False
    
    def _send_command(self, command, wait_for_response=True, timeout=2.0, verbose=False):
        """
        Send a command to the CLI port and optionally wait for response.
        
        Parameters
        ----------
        command : str
            Command string to send
        wait_for_response : bool
            Whether to wait for and return the response
        timeout : float
            Maximum time to wait for response (seconds)
        verbose : bool
            Print debug information
        
        Returns
        -------
        str or None
            Response from the sensor if wait_for_response is True
        """
        if not self.is_connected or not self.cli_serial:
            raise RuntimeError("Not connected to radar. Call connect() first.")
        
        # Send command with newline
        cmd_bytes = (command + '\n').encode('utf-8')
        self.cli_serial.write(cmd_bytes)
        
        if not wait_for_response:
            time.sleep(0.01)
            return None
        
        # Read response lines
        start_time = time.time()
        response_lines = []
        
        while (time.time() - start_time) < timeout:
            if self.cli_serial.in_waiting > 0:
                line = self.cli_serial.readline().decode('utf-8', errors='ignore').strip()
                if line:
                    response_lines.append(line)
                    if verbose:
                        print(f"    Response: {line}")
                    # AWR1642 may not always return "Done", just wait a bit
                    if 'Done' in line or 'Error' in line:
                        return '\n'.join(response_lines)
            time.sleep(0.001)  # Small delay to avoid busy waiting
        
        # Return whatever we got
        return '\n'.join(response_lines) if response_lines else None
    
    def parse_config_file(self, config_commands):
        """
        Parse configuration commands to extract radar parameters.
        
        Parameters
        ----------
        config_commands : list of str
            List of configuration commands
        
        Returns
        -------
        dict
            Dictionary of configuration parameters
        """
        params = self.config_params.copy()
        
        for cmd in config_commands:
            split_words = cmd.split()
            
            if not split_words:
                continue
            
            # Parse profileCfg
            if "profileCfg" in split_words[0]:
                start_freq = int(float(split_words[2]))
                idle_time = int(split_words[3])
                ramp_end_time = float(split_words[5])
                freq_slope_const = float(split_words[8])
                num_adc_samples = int(split_words[10])
                
                # Round to next power of 2
                num_adc_samples_round = 1
                while num_adc_samples > num_adc_samples_round:
                    num_adc_samples_round *= 2
                
                dig_out_sample_rate = int(split_words[11])
                
                params['numAdcSamples'] = num_adc_samples
                params['numRangeBins'] = num_adc_samples_round
                params['rangeResolutionMeters'] = (3e8 * dig_out_sample_rate * 1e3) / \
                                                   (2 * freq_slope_const * 1e12 * num_adc_samples)
                params['rangeIdxToMeters'] = (3e8 * dig_out_sample_rate * 1e3) / \
                                              (2 * freq_slope_const * 1e12 * params['numRangeBins'])
                params['maxRange'] = (300 * 0.9 * dig_out_sample_rate) / (2 * freq_slope_const * 1e3)
                
                # Store for doppler calculation
                params['_startFreq'] = start_freq
                params['_idleTime'] = idle_time
                params['_rampEndTime'] = ramp_end_time
            
            # Parse frameCfg
            elif "frameCfg" in split_words[0]:
                chirp_start_idx = int(split_words[1])
                chirp_end_idx = int(split_words[2])
                num_loops = int(split_words[3])
                
                num_chirps_per_frame = (chirp_end_idx - chirp_start_idx + 1) * num_loops
                params['numChirpsPerFrame'] = num_chirps_per_frame
                params['numDopplerBins'] = num_chirps_per_frame / params['numTxAnt']
                
                # Calculate doppler resolution if we have profile data
                if '_startFreq' in params:
                    params['dopplerResolutionMps'] = 3e8 / \
                        (2 * params['_startFreq'] * 1e9 * \
                         (params['_idleTime'] + params['_rampEndTime']) * 1e-6 * \
                         params['numDopplerBins'] * params['numTxAnt'])
                    params['maxVelocity'] = 3e8 / \
                        (4 * params['_startFreq'] * 1e9 * \
                         (params['_idleTime'] + params['_rampEndTime']) * 1e-6 * \
                         params['numTxAnt'])
        
        return params
    
    def configure_sensor(self, commands, verbose=False):
        """
        Configure the radar sensor with a list of CLI commands.
        
        Parameters
        ----------
        commands : list of str
            List of configuration commands to send
        verbose : bool
            Print detailed responses from sensor
        
        Raises
        ------
        RuntimeError
            If not connected
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to radar. Call connect() first.")
        
        print(f"Sending {len(commands)} configuration commands...")
        
        for i, cmd in enumerate(commands, 1):
            if cmd.strip():  # Skip empty lines
                print(f"  [{i}/{len(commands)}] {cmd}")
                response = self._send_command(cmd, wait_for_response=True, timeout=1.0, verbose=verbose)
                if verbose and response:
                    print(f"    Got: {response[:50]}...")
                time.sleep(0.05)  # Increased delay for stability
        
        # Parse configuration to extract parameters
        self.config_params = self.parse_config_file(commands)
        
        # Give sensor time to start if sensorStart was in commands
        if any('sensorStart' in cmd for cmd in commands):
            print("Waiting for sensor to start...")
            time.sleep(2.0)  # Give it time to initialize
        
        print(f"✓ Sensor configured successfully")
        print(f"  RX Antennas: {self.config_params['numRxAnt']}")
        print(f"  TX Antennas: {self.config_params['numTxAnt']}")
        print(f"  ADC Samples: {self.config_params['numAdcSamples']}")
        print(f"  Chirps/Frame: {self.config_params['numChirpsPerFrame']}")
        print(f"  Range Bins: {self.config_params['numRangeBins']}")
        print(f"  Range Resolution: {self.config_params['rangeResolutionMeters']:.4f} m")
    
    def read_tlv_packet(self, timeout=1.0):
        """
        Read and parse a single TLV packet from the data port.
        
        Returns
        -------
        tuple
            (data_ok, frame_number, detected_objects, tlv_data)
            - data_ok: bool indicating if valid data was received
            - frame_number: int frame sequence number
            - detected_objects: dict with object detection data
            - tlv_data: dict with all TLV messages
        """
        data_ok = False
        frame_number = 0
        detected_objects = {}
        tlv_data = {}
        
        # Read available data from port
        start_time = time.time()
        while (time.time() - start_time) < timeout:
            if self.data_serial.in_waiting > 0:
                break
            time.sleep(0.001)
        
        if self.data_serial.in_waiting == 0:
            return data_ok, frame_number, detected_objects, tlv_data
        
        read_buffer = self.data_serial.read(self.data_serial.in_waiting)
        byte_vec = np.frombuffer(read_buffer, dtype='uint8')
        byte_count = len(byte_vec)
        
        # Add to buffer
        max_buffer_size = 2**15
        if (self.byte_buffer_length + byte_count) < max_buffer_size:
            self.byte_buffer[self.byte_buffer_length:self.byte_buffer_length + byte_count] = byte_vec
            self.byte_buffer_length += byte_count
        
        # Look for magic word
        if self.byte_buffer_length > 16:
            # Find all possible magic word locations
            possible_locs = np.where(self.byte_buffer == self.MAGIC_WORD[0])[0]
            
            start_idx = []
            for loc in possible_locs:
                if loc + 8 <= len(self.byte_buffer):
                    check = self.byte_buffer[loc:loc + 8]
                    if np.all(check == self.MAGIC_WORD):
                        start_idx.append(loc)
            
            if start_idx:
                # Remove data before first magic word
                if start_idx[0] > 0:
                    self.byte_buffer[:self.byte_buffer_length - start_idx[0]] = \
                        self.byte_buffer[start_idx[0]:self.byte_buffer_length]
                    self.byte_buffer_length -= start_idx[0]
                
                if self.byte_buffer_length < 0:
                    self.byte_buffer_length = 0
                
                # Read packet length
                if self.byte_buffer_length >= 16:
                    word = np.array([1, 2**8, 2**16, 2**24], dtype='uint32')
                    total_packet_len = np.dot(self.byte_buffer[12:16].astype('uint32'), word)
                    
                    # Check if we have full packet
                    if self.byte_buffer_length >= total_packet_len:
                        data_ok, frame_number, detected_objects, tlv_data = \
                            self._parse_packet(total_packet_len)
                        
                        # Remove processed data
                        if self.byte_buffer_length > total_packet_len:
                            self.byte_buffer[:self.byte_buffer_length - total_packet_len] = \
                                self.byte_buffer[total_packet_len:self.byte_buffer_length]
                            self.byte_buffer_length -= total_packet_len
                        else:
                            self.byte_buffer_length = 0
        
        return data_ok, frame_number, detected_objects, tlv_data
    
    def _parse_packet(self, packet_len):
        """
        Parse a TLV packet from the byte buffer.
        
        Parameters
        ----------
        packet_len : int
            Length of the packet to parse
        
        Returns
        -------
        tuple
            (data_ok, frame_number, detected_objects, tlv_data)
        """
        data_ok = False
        detected_objects = {}
        tlv_data = {}
        word = np.array([1, 2**8, 2**16, 2**24], dtype='uint32')
        
        idx = 0
        
        # Parse header
        idx += 8  # Skip magic word
        version = np.dot(self.byte_buffer[idx:idx+4].astype('uint32'), word)
        idx += 4
        total_packet_len = np.dot(self.byte_buffer[idx:idx+4].astype('uint32'), word)
        idx += 4
        platform = np.dot(self.byte_buffer[idx:idx+4].astype('uint32'), word)
        idx += 4
        frame_number = np.dot(self.byte_buffer[idx:idx+4].astype('uint32'), word)
        idx += 4
        time_cpu_cycles = np.dot(self.byte_buffer[idx:idx+4].astype('uint32'), word)
        idx += 4
        num_detected_obj = np.dot(self.byte_buffer[idx:idx+4].astype('uint32'), word)
        idx += 4
        num_tlvs = np.dot(self.byte_buffer[idx:idx+4].astype('uint32'), word)
        idx += 4
        sub_frame_number = np.dot(self.byte_buffer[idx:idx+4].astype('uint32'), word)
        idx += 4
        
        # Parse TLV messages
        for tlv_idx in range(int(num_tlvs)):
            try:
                # Check if we have enough bytes for TLV header
                if idx + 8 > len(self.byte_buffer):
                    break
                    
                tlv_type = np.dot(self.byte_buffer[idx:idx+4].astype('uint32'), word)
                idx += 4
                tlv_length = np.dot(self.byte_buffer[idx:idx+4].astype('uint32'), word)
                idx += 4
                
                # Check if we have enough bytes for TLV payload
                if idx + int(tlv_length) > len(self.byte_buffer):
                    break
                
                # Parse detected points
                if tlv_type == self.MMWDEMO_OUTPUT_MSG_DETECTED_POINTS:
                    detected_objects = self._parse_detected_points(idx, int(tlv_length))
                    data_ok = True
                
                # Parse range profile
                elif tlv_type == self.MMWDEMO_OUTPUT_MSG_RANGE_PROFILE:
                    tlv_data['range_profile'] = self._parse_range_profile(idx, int(tlv_length))
                    data_ok = True  # Also mark as OK when we have range profile
                
                # Parse Range-Doppler heatmap
                elif tlv_type == self.MMWDEMO_OUTPUT_MSG_RANGE_DOPPLER_HEAT_MAP:
                    tlv_data['range_doppler_heatmap'] = self._parse_range_doppler_heatmap(idx, int(tlv_length))
                    data_ok = True
                
                # Parse stats
                elif tlv_type == self.MMWDEMO_OUTPUT_MSG_STATS:
                    tlv_data['stats'] = self._parse_stats(idx, int(tlv_length))
                
                # Store other TLV types as raw bytes
                else:
                    tlv_data[f'tlv_type_{tlv_type}'] = self.byte_buffer[idx:idx+int(tlv_length)].copy()
                
                idx += int(tlv_length)
                
            except Exception as e:
                print(f"Warning: Error parsing TLV {tlv_idx}: {e}")
                break
        
        return data_ok, int(frame_number), detected_objects, tlv_data
    
    def _parse_detected_points(self, start_idx, length):
        """Parse detected points TLV message."""
        try:
            idx = start_idx
            word16 = np.array([1, 2**8], dtype='uint16')
            
            # Check if we have enough data
            if idx + 4 > self.byte_buffer_length:
                return {}
            
            num_obj = int(np.dot(self.byte_buffer[idx:idx+2].astype('uint16'), word16))
            idx += 2
            xyz_q_format = 2**int(np.dot(self.byte_buffer[idx:idx+2].astype('uint16'), word16))
            idx += 2
            
            if num_obj == 0 or num_obj > 500:  # Sanity check
                return {'numObj': 0}
            
            # Object structure: rangeIdx(2), dopplerIdx(2), peakVal(2), x(2), y(2), z(2) = 12 bytes
            bytes_needed = idx + num_obj * 12
            if bytes_needed > self.byte_buffer_length:
                return {'numObj': 0}
            
            # Pre-allocate arrays
            range_idx = np.zeros(num_obj, dtype='int16')
            doppler_idx = np.zeros(num_obj, dtype='int16')
            peak_val = np.zeros(num_obj, dtype='int16')
            x = np.zeros(num_obj, dtype='int16')
            y = np.zeros(num_obj, dtype='int16')
            z = np.zeros(num_obj, dtype='int16')
            
            # Read object data
            for i in range(num_obj):
                # Use view to interpret bytes as int16 directly
                range_idx[i] = self.byte_buffer[idx:idx+2].view(dtype='int16')[0]
                idx += 2
                doppler_idx[i] = self.byte_buffer[idx:idx+2].view(dtype='int16')[0]
                idx += 2
                peak_val[i] = self.byte_buffer[idx:idx+2].view(dtype='int16')[0]
                idx += 2
                x[i] = self.byte_buffer[idx:idx+2].view(dtype='int16')[0]
                idx += 2
                y[i] = self.byte_buffer[idx:idx+2].view(dtype='int16')[0]
                idx += 2
                z[i] = self.byte_buffer[idx:idx+2].view(dtype='int16')[0]
                idx += 2
            
            # Convert to physical units
            range_val = range_idx * self.config_params['rangeResolutionMeters']
            
            # Handle Doppler wrapping
            doppler_idx = doppler_idx.astype('float32')
            doppler_idx[doppler_idx > (self.config_params['numDopplerBins']/2 - 1)] -= 65536
            doppler_val = doppler_idx * self.config_params.get('dopplerResolutionMps', 0.1)
            
            # Convert x, y, z to meters
            x_m = x.astype('float32') / xyz_q_format
            y_m = y.astype('float32') / xyz_q_format
            z_m = z.astype('float32') / xyz_q_format
            
            return {
                'numObj': num_obj,
                'rangeIdx': range_idx,
                'range': range_val,
                'dopplerIdx': doppler_idx.astype('int16'),
                'doppler': doppler_val,
                'peakVal': peak_val,
                'x': x_m,
                'y': y_m,
                'z': z_m
            }
            
        except Exception as e:
            print(f"Warning: Error in _parse_detected_points: {e}")
            return {'numObj': 0}
    
    def _parse_range_profile(self, start_idx, length):
        """Parse range profile TLV message.
        
        Returns range profile in dB scale (20*log10(magnitude)).
        Raw values are Q9 format (divide by 512 to get linear magnitude).
        """
        # Range profile is array of 16-bit unsigned values in Q9 format
        num_range_bins = length // 2
        
        # Direct view of buffer as uint16 for efficiency
        raw_profile = self.byte_buffer[start_idx:start_idx + num_range_bins * 2].view(dtype='<u2').copy()
        
        # Convert from Q9 to linear magnitude, then to dB
        # Q9 means divide by 2^9 = 512
        linear_magnitude = raw_profile.astype(np.float32) / 512.0
        
        # Convert to dB (add small epsilon to avoid log(0))
        range_profile_dB = 20 * np.log10(linear_magnitude + 1e-6)
        
        return range_profile_dB
    
    def _parse_range_profile_raw(self, start_idx, length):
        """Parse range profile TLV message - return raw Q9 values."""
        num_range_bins = length // 2
        return self.byte_buffer[start_idx:start_idx + num_range_bins * 2].view(dtype='<u2').copy()
    
    def _parse_range_doppler_heatmap(self, start_idx, length):
        """Parse Range-Doppler heatmap TLV message.
        
        Returns 2D array of shape (numDopplerBins, numRangeBins) in dB.
        """
        num_range_bins = self.config_params.get('numRangeBins', 256)
        num_doppler_bins = int(self.config_params.get('numDopplerBins', 16))
        
        expected_size = num_range_bins * num_doppler_bins * 2
        if length < expected_size:
            # Fallback: infer dimensions from length
            total_bins = length // 2
            num_doppler_bins = max(1, total_bins // num_range_bins)
        
        # Parse as uint16 array
        raw_data = self.byte_buffer[start_idx:start_idx + length].view(dtype='<u2').copy()
        
        # Reshape to (numDopplerBins, numRangeBins)
        try:
            heatmap = raw_data.reshape((num_doppler_bins, num_range_bins))
        except ValueError:
            # If reshape fails, return 1D
            heatmap = raw_data
        
        # Convert to dB
        heatmap_dB = 20 * np.log10(heatmap.astype(np.float32) / 512.0 + 1e-6)
        
        return heatmap_dB
    
    def _parse_stats(self, start_idx, length):
        """Parse stats TLV message."""
        word = np.array([1, 2**8, 2**16, 2**24], dtype='uint32')
        idx = start_idx
        
        stats = {}
        if length >= 24:
            stats['interFrameProcessingTime'] = np.dot(self.byte_buffer[idx:idx+4].astype('uint32'), word)
            idx += 4
            stats['transmitOutputTime'] = np.dot(self.byte_buffer[idx:idx+4].astype('uint32'), word)
            idx += 4
            stats['interFrameProcessingMargin'] = np.dot(self.byte_buffer[idx:idx+4].astype('uint32'), word)
            idx += 4
            stats['interChirpProcessingMargin'] = np.dot(self.byte_buffer[idx:idx+4].astype('uint32'), word)
            idx += 4
            stats['activeFrameCPULoad'] = np.dot(self.byte_buffer[idx:idx+4].astype('uint32'), word)
            idx += 4
            stats['interFrameCPULoad'] = np.dot(self.byte_buffer[idx:idx+4].astype('uint32'), word)
        
        return stats
    
    def capture_frames(self, num_frames, return_raw=False, verbose=False):
        """
        Capture multiple frames of data.
        
        Parameters
        ----------
        num_frames : int
            Number of frames to capture
        return_raw : bool
            If True, return raw TLV data. If False, return processed arrays.
        verbose : bool
            Print debug information about data reception
        
        Returns
        -------
        list or dict
            List of frame data dictionaries, or processed data arrays
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to radar. Call connect() first.")
        
        print(f"Capturing {num_frames} frames...")
        
        # Check if data is available on data port
        initial_waiting = self.data_serial.in_waiting
        if verbose:
            print(f"Initial data waiting: {initial_waiting} bytes")
        
        frames_data = []
        frames_captured = 0
        timeout_count = 0
        max_timeouts = 50
        bytes_received_total = 0
        
        start_time = time.time()
        
        while frames_captured < num_frames:
            data_ok, frame_number, detected_objects, tlv_data = self.read_tlv_packet(timeout=0.1)
            
            # Debug: Check data port activity
            if verbose and timeout_count % 10 == 0:
                waiting = self.data_serial.in_waiting
                if waiting > 0:
                    print(f"  Data port has {waiting} bytes waiting")
                else:
                    print(f"  No data on port (timeout count: {timeout_count})")
            
            if data_ok:
                frames_data.append({
                    'frame_number': frame_number,
                    'detected_objects': detected_objects,
                    'tlv_data': tlv_data
                })
                frames_captured += 1
                timeout_count = 0
                
                if frames_captured % 10 == 0 or verbose:
                    elapsed = time.time() - start_time
                    fps = frames_captured / elapsed if elapsed > 0 else 0
                    print(f"  Progress: {frames_captured}/{num_frames} frames ({fps:.1f} FPS)")
            else:
                timeout_count += 1
                if timeout_count > max_timeouts:
                    print(f"Warning: Timeout waiting for frames. Captured {frames_captured}/{num_frames}")
                    print(f"  Bytes in buffer: {self.byte_buffer_length}")
                    print(f"  Check: Is sensor actually started? Is data port correct?")
                    break
            
            time.sleep(0.001)  # Small delay
        
        elapsed = time.time() - start_time
        print(f"✓ Captured {frames_captured} frames in {elapsed:.2f}s ({frames_captured/elapsed if elapsed > 0 else 0:.1f} FPS)")
        
        if return_raw:
            return frames_data
        else:
            # Process into arrays compatible with existing pipeline
            return self._process_frames_to_arrays(frames_data)
    
    def _process_frames_to_arrays(self, frames_data):
        """
        Process captured frames into arrays compatible with existing pipeline.
        
        This extracts range profiles and organizes them into the expected format.
        Range profiles are already in dB scale from _parse_range_profile.
        """
        num_frames = len(frames_data)
        
        if num_frames == 0:
            print("Warning: No frames captured")
            return None
        
        # Check if we have range profiles
        if 'range_profile' in frames_data[0]['tlv_data']:
            num_range_bins = len(frames_data[0]['tlv_data']['range_profile'])
            
            # Create array: (num_range_bins, num_frames) - simple 2D for visualization
            # Range profile is already in dB from the parser
            range_data = np.zeros((num_range_bins, num_frames), dtype=np.float32)
            
            for i, frame in enumerate(frames_data):
                if 'range_profile' in frame['tlv_data']:
                    profile = frame['tlv_data']['range_profile']
                    range_data[:len(profile), i] = profile
            
            print(f"✓ Processed data shape: {range_data.shape} (range_bins x frames)")
            print(f"  Value range: {range_data.min():.1f} to {range_data.max():.1f} dB")
            return range_data
        else:
            print("Warning: No range profile data in frames")
            return None
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
        return False


# Example usage
if __name__ == "__main__":
    print("AWR1642 Driver Test")
    print("=" * 50)
    
    with AWR1642(cli_port='/dev/ttyACM0', data_port='/dev/ttyACM1') as radar:
        print("\nConfiguring sensor...")
        radar.configure_sensor(config_commands_1642)
        
        print("\nSending sensorStart...")
        radar._send_command('sensorStart')
        time.sleep(1)
        
        print("\nCapturing 20 frames...")
        frames = radar.capture_frames(num_frames=20, return_raw=True)
        
        print(f"\nCaptured {len(frames)} frames")
        if frames:
            print(f"Frame 0: {frames[0]['frame_number']}")
            if frames[0]['detected_objects']:
                print(f"  Detected objects: {frames[0]['detected_objects']['numObj']}")
