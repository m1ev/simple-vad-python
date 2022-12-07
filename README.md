# simple-vad-python
<h2>Description</h2>
A simple Voice Activity Detection (VAD) tool implemented in python and based on the following paper:
<br/><br/>

>M. H. Moattar and M. M. Homayounpour, “A Simple But Efficient Real-Time
>Voice Activity Detection Algorithm”, 17th EUSIPCO, pp. 2549-2553, 2009.


<h2>The basic VAD algorithm</h2>

1. An input audio signal is framed every 10 ms with no window function applied.
2. For each audio frame the three features are computed: Short-Term Signal Energy (STSE), Spectral Flatness Measure (SFM) and Most Dominant Frequency component (MDF).
3. An audio frame is marked as speech, if more than one of the features fall over the precomputed threshold.

<h2>Usage</h2>
<h3>From the command line</h3>

`simple_vad --help`

```
usage: simple_vad.py [-h] -i INPUT [-o OUTPUT]

A simple voice activity detection tool that finds regions of speech in a non-
compressed 8, 16 or 32-bit PCM WAV speech file.

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        Output WAV file name

required arguments:
  -i INPUT, --input INPUT
                        Input WAV file name
```

<h3>From GUI application</h3>

Run `python simple_vad_gui.py` and the main GUI window opens which will allow you to open a new WAV file, plot waveform of the input audio signal and graphs of the STSE, MDF and SFM feature values, display detected regions of speech, as well as save the input speech signal with regions of silence removed as a new WAV file.

<h2>Screenshots</h2>

![Example #1](/readme_assets/example1.jpg?raw=true)

![Example #2](/readme_assets/example2.jpg?raw=true)

![Example #3](/readme_assets/example3.jpg?raw=true)

