"""A simple voice activity detection tool implemented in python.

It utilizes a VAD algorithm, described in the following paper:
M. H. Moattar and M. M. Homayounpour, “A Simple But Efficient Real-Time
Voice Activity Detection Algorithm”, 17th EUSIPCO, pp. 2549-2553, 2009.

This program can be run as a standalone script or in conjunction with
a GUI interface. 8, 16 and 32-bit PCM WAV files without compression
are supported.

For more information, visit:
https://github.com/m1ev/simple-vad-python
"""
import argparse
from multiprocessing import Pool
import wave

import numpy as np


class WrongFileFormat(Exception):
    """A 24-bit PCM WAV file is being imported."""
    pass


class SimpleVAD():
    """Apply the VAD algorithm to a new WAV file."""
    FRAME_SIZE_MSEC = 10        # Frame size in milliseconds and
    ENERGY_PRIM_THRESH = 40     # threshold values.
    F_PRIM_THRESH = 185
    SF_PRIM_THRESH = 5

    def _convert_samples(self, is_int_to_float, samples, sampwidth):
        """Converts an input array of sound samples from integer to float
        format or vice versa and returns it as a new array.

        Args:
            is_int_to_float (bool):
                'True' if it is 'integer to float' conversion and
                'False' otherwise.
            samples (array of ints or floats):
                The input 1D array of sound samples.
            sampwidth (int):
                Sample width.
        """
        type_size = {
            1: 255,
            2: 32768,
            # TODO: add support for 24-bit PCM format
            4: 2147483648
        }
        file_type_size = type_size[sampwidth]
        half_fts = round(file_type_size / 2)

        if sampwidth == 1:
            if is_int_to_float:
                samples_new = samples.copy().astype(np.int32)
                samples_new = (samples_new - half_fts) / half_fts
            else:
                samples_new = (samples * half_fts) + half_fts
                samples_new = samples_new.astype(int)
        else:
            if is_int_to_float:
                samples_new = samples.copy().astype(np.int32)
                samples_new = samples_new / file_type_size
            else:
                samples_new = samples * file_type_size
                samples_new = samples_new.astype(int)

        return samples_new

    def _pad_with_zeros(self, samples, frame_size):
        """Pads input sound data with zeros to an integer number of frames.

        Args:
            samples (arr):
                The input 1D array of sound samples.
            frame_size (int):
                Number of samples in 1 frame.

        Returns:
            An array of the input sound samples padded with zeros.
        """
        num_of_samples = len(samples)
        num_of_frames = num_of_samples / frame_size
        if num_of_frames % 1 != 0:
            samples_left = int((num_of_frames % 1) * frame_size)
            zeros_to_add = frame_size - samples_left
            samples_new = np.concatenate([samples, np.zeros(zeros_to_add)])
        else:
            samples_new = samples

        return samples_new

    def _comp_vad_features(self, samples):
        """Computes Short-Time Signal Energy, Most Dominant Frequency and
        inverted Spectral Flatness Measure values for a sound frame.

        Args:
            samples (array of floats):
                1D array of sound samples.
            """
        stse = np.sum(np.square(samples))

        stft = np.fft.fft(samples)
        stft_abs = np.abs(stft)
        # Find index of the maximum amplitude
        max_amp_ind = np.argmax(stft_abs)
        # The value of the maximum amplitude
        mdf_amp = stft_abs[max_amp_ind]
        freq_step = self.frame_rate / len(stft)
        # Frequency corresponding to the maximum amplitude
        mdf_freq = max_amp_ind * freq_step

        # Handling zeros in geometric mean calculation by
        # converting all zero (0) values to one (1)
        np.place(stft_abs, stft_abs == 0, 1)
        g_mean = np.exp(np.mean(np.log(stft_abs)))
        a_mean = np.mean(stft_abs)
        sfm = -10 * np.log10(g_mean / a_mean)

        return stse, mdf_amp, mdf_freq, sfm

    def _mark_frames(self, vad_vals, nframes):
        """Marks each sound frame as speech or silence.

        Args:
            vad_vals:
                A tuple containing STSE, MDF and SFM values for each frame.
            nframes (int):
                Number of frames in speech signal.

        Returns:
            A boolean array showing if a sound frame is marked as
                speech (True) or silence (False).
        """
        min_stse = 9999999999
        min_mdf = 9999999999
        min_sfm = 9999999999
        silence_count = 0

        stse = vad_vals[:, 0]
        mdf = vad_vals[:, 1]
        sfm = vad_vals[:, 2]

        # Find the minimum values for stse, mdf and sfm
        # (ignore the last padded with zeros frame)
        for i in range(nframes - 1):
            if (min_stse > stse[i]) and (stse[i] != 0):
                min_stse = stse[i]
            if min_mdf > mdf[i]:
                min_mdf = mdf[i]
            if min_sfm > sfm[i]:
                min_sfm = sfm[i]

        thresh_stse = self.ENERGY_PRIM_THRESH * np.log10(min_stse)
        vad_frames = np.ones(nframes, dtype=bool)

        for i in range(nframes):
            cnt = 0
            if (stse[i] - min_stse) >= thresh_stse:
                cnt += 1
            if (mdf[i] - min_mdf) >= self.F_PRIM_THRESH:
                cnt += 1
            if (sfm[i] - min_sfm) >= self.SF_PRIM_THRESH:
                cnt += 1

            # True - if the current frame is marked as speech,
            # and False - otherwise
            if cnt <= 1:
                vad_frames[i] = False
                silence_count += 1
                min_stse = ((silence_count * min_stse) + stse[i])
                min_stse /= (silence_count + 1)
                thresh_stse = self.ENERGY_PRIM_THRESH * np.log10(min_stse)

        return vad_frames

    def read_new_wav(self, file_path):
        """Reads selected wav file, from which retrieves an array of sound
        samples and a number of wav parameters.

        Args:
            file_path (str):
                A full path to the wav file.

        Returns:
            A tuple (params, data_type, frame_size, samples), where
                'params' are the following parameters of the wav file:
                (nchannels, sampwidth, framerate, nframes, comptype, compname);
                'data_type' is a numpy data type used to store a sound sample;
                'frame_size' is a size of 1 frame in samples and
                'samples' is the array of samples of the input sound data
                padded with zeros to an integer number of frames.
        """
        # Numpy data type used for 8, 16 and 32-bit WAVs respectively
        types = {
            1: np.uint8,
            2: np.int16,
            # TODO: add support for 24-bit PCM format
            4: np.int32
        }
        wave_file = wave.open(file_path, 'rb')
        params = wave_file.getparams()
        sampwidth = params.sampwidth
        self.frame_rate = params.framerate
        if sampwidth == 3:
            raise WrongFileFormat("24-bit int PCM format is not supported !")
        # Read sound samples as a byte string
        buf = wave_file.readframes(wave_file.getnframes())
        wave_file.close()

        data_type = types[sampwidth]
        frame_size = int((params.framerate / 1000) * self.FRAME_SIZE_MSEC)
        # Convert samples from the byte string into an array of integers
        # and keep only 1 channel if nchannels > 1
        samples_int = np.frombuffer(buf, dtype=data_type)
        samples_int = samples_int[::params.nchannels]
        # Convert the integer values into float format of [-1.0 .. 1.0] range
        samples_float = self._convert_samples(True, samples_int, sampwidth)
        samples = self._pad_with_zeros(samples_float, frame_size)

        return params, data_type, frame_size, samples

    def save_new_wav(self, samples, params, data_type, save_file_path):
        """Saves input sound samples as a new wav file.

        Args:
            samples (array of floats):
                The input 1D array of sound samples.
            params:
                A tuple (nchannels, sampwidth, framerate, nframes, comptype,
                compname) of wav parameters.
            data_type:
                Numpy data type corresponding to sample size.
            save_file_path:
                Full path to the new wav file to be saved.
        """
        samples_int = self._convert_samples(False, samples, params.sampwidth)
        wave_file = wave.open(save_file_path, 'wb')
        wave_file.setparams(params)
        wave_file.setnchannels(1)
        wave_file.setnframes(len(samples_int))

        samples_int = np.asarray(samples_int, dtype=data_type)
        samples_byte_string = samples_int.tobytes()
        wave_file.writeframes(samples_byte_string)
        wave_file.close()

    def comp_vad_all(self, frame_size, samples):
        """Computes STSE, MDF and inverted SFM values for all frames of
        an input array of sound samples.

        Args:
            frame_size (int):
                Number of samples in 1 frame.
            samples (array of floats):
                1D array of the input sound samples.

        Returns:
            2D array of STSE, MDF frequencies and inverted SFM values for
                each sound frame.
        """
        num_of_frames = int(len(samples) / frame_size)
        samples = np.asarray(samples)
        samples_new = []
        for i in range(num_of_frames):
            start_i = i * frame_size
            end_i = (i + 1) * frame_size
            samples_new.append(samples[start_i:end_i])
        # Using multiprocessing module to speed up the computation
        with Pool(32) as pool:
            vad = pool.map(self._comp_vad_features, samples_new)
        vad = np.asarray(vad)

        return np.vstack((vad[:, 0], vad[:, 2], vad[:, 3])).T

    def find_voiced_regions(self, vad_vals, nframes, frame_size):
        """Finds regions of speech by comparing the provided STSE, MDF and SFM
        values for each frame.

        Args:
            vad_features:
                A tuple containing STSE, MDF and SFM values.
            nframes (int):
                Number of frames.
            frame_size (int):
                Number of samples in 1 frame.

        Returns:
            2D numpy array containing pairs of the 1st and last sample index
                corresponding to each detected voiced region.
        """
        speech_count = 0
        silence_count = 0
        vad_regions = self._mark_frames(vad_vals, nframes)

        # Ignore silence run less than 10 consecutive frames.
        for i in range(nframes):
            vad_frame = vad_regions[i]
            if vad_frame:
                if not vad_regions[i-1] and silence_count < 10:
                    vad_regions[i-silence_count : i] = True
                silence_count = 0
            else:
                silence_count += 1

        # Ignore speech run less than 6 consecutive frames.
        for i in range(nframes):
            vad_frame = vad_regions[i]
            if vad_frame:
                speech_count += 1
            else:
                if vad_regions[i-1] and speech_count <= 5:
                    vad_regions[i-speech_count : i] = False
                speech_count = 0

        voiced_regions = []
        start = 0
        cnt = 0
        for i in range(nframes-1):
            if not vad_regions[i] and vad_regions[i + 1]:
                start = i + 1
            elif vad_regions[i] and not vad_regions[i + 1]:
                end = i + 1
                start = start * frame_size
                end = end * frame_size
                voiced_regions.append([start, end])
            cnt += 1

        return np.asarray(voiced_regions)

    def remove_silence(self, voiced_regions, samples):
        """Removes regions of silence in an input array of sound samples.

        Args:
            voice_regions (arr):
                2D array of the 1st and last sample indices, corresponding to
                each voiced region.
            samples (arr):
                1D array of the input sound samples.

        Returns:
            An array of the input sound samples with regions of silence
                removed.
        """
        samples_new = []
        for region in voiced_regions:
            samples_new.extend(samples[region[0]:region[1]])

        return np.asarray(samples_new)

    def apply_vad(self, args_dict):
        """Applies VAD algorithm to an input speech wav file.

        Args:
            args_dict (dict):
                A dictionary containing 'input' and 'output' keys,
                values of which contain a full path to the input and
                output wav files respectively.
        """
        input_path = args_dict["input"]
        output_path = args_dict["output"]
        print('Openning WAV file...', end="", flush=True)
        params, data_type, frame_size, samples = self.read_new_wav(input_path)
        print('     DONE !!!')

        nframes = int(len(samples) / frame_size)
        print('Computing VAD features...', end="", flush=True)
        vad_vals = self.comp_vad_all(frame_size, samples)
        print('     DONE !!!')

        print('Searching for voiced regions of speech...', end="", flush=True)
        regions = self.find_voiced_regions(vad_vals, nframes, frame_size)
        print('     DONE !!!')

        print('Removing regions of silence...', end="", flush=True)
        samples_new = self.remove_silence(regions, samples)
        print('     DONE !!!')

        print('Saving as a new WAV file...', end="", flush=True)
        if not output_path:
            src_name = input_path[:input_path.rfind('.')]
            ext = input_path[input_path.rfind('.'):]
            output_path = src_name + '_vad' + ext
        self.save_new_wav(samples_new, params, data_type, output_path)
        print('     DONE !!!')

        print('File saved as: ', output_path)


def main():
    """Defines command line arguments and instanciates """
    desc = 'A simple voice activity detection tool that finds regions of' \
           ' speech in a non-compressed 8, 16 or 32-bit PCM WAV speech file.'
    parser = argparse.ArgumentParser(description=desc)
    req_name = parser.add_argument_group('required arguments')
    msg = 'Input WAV file name'
    req_name.add_argument('-i', '--input', help=msg, type=str, required=True)
    msg = "Output WAV file name"
    parser.add_argument('-o', '--output', help=msg, type=str)
    args_dict = vars(parser.parse_args())

    new_vad = SimpleVAD()
    new_vad.apply_vad(args_dict)


if __name__ == '__main__':
    main()
