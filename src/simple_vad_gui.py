"""A GUI for the simple voice activity detection tool.

It allows users to open a selected WAV file, to plot graphs of a waveform
of the input sound / speech signal and show detected regions of speech,
to plot the Short-Time Signal Energy, Most Dominant Frequency Component and
Spectral Flatness Measure values of the signal. The processed WAV file with
detected regions of silence removed can be saved as a new WAV file.
"""

import sys

import pyqtgraph as pg
import simple_vad as vad
from ui import Ui_MainWindow

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QLabel, QWidget, QVBoxLayout, QFileDialog

class AboutWindow(QWidget):
    """'About' window of the GUI application."""
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        self.setWindowTitle("About 'Simple Voice Activity Detection Tool'")
        self.lbl_desc = QLabel(
            'A simple voice activity detection tool that finds regions of \n'
            'speech in a non-compressed 8, 16 or 32-bit PCM WAV speech file.\n')
        self.lbl_desc.adjustSize()
        self.lbl_desc.setFont(QFont('Calibri', 12))
        layout.addWidget(self.lbl_desc)

        self.lbl_url = QLabel(
            '<a href=\"https://github.com/m1ev/simple-vad-python"> '
            '<font>https://github.com/m1ev/simple-vad-python</font></a>')
        self.lbl_url.adjustSize()
        self.lbl_url.setFont(QFont('Calibri', 11))
        self.lbl_url.setTextInteractionFlags(Qt.LinksAccessibleByMouse)
        self.lbl_url.setOpenExternalLinks(True)
        layout.addWidget(self.lbl_url)
        self.setLayout(layout)

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    """Main window of the GUI application."""
    def __init__(self):
        super().__init__()
        self.setupUi(self)  # initializing design
        self.about = None
        self.input_file = vad.SimpleVAD()
        self.menuOpenNew.triggered.connect(
            lambda: self.exec_open_file_dialog())
        self.menuExit.triggered.connect(
            lambda: close_event())
        self.menuApplyVAD.triggered.connect(
            lambda: self.apply_vad())
        self.menuVADShowInput.triggered.connect(
            lambda: self.show_input_and_vad())
        self.menuVADShowOutput.triggered.connect(
            lambda: self.show_output())
        self.menuVADSaveFile.triggered.connect(
            lambda: self.exec_save_file_dialog())
        self.menuAbout.triggered.connect(
            lambda: self.show_about_window())

    def show_about_window(self):
        """Pops up the 'about' window."""
        if not self.about:
            self.about = AboutWindow()
            self.about.show()
        else:
            self.about.close()
            self.about = None

    def exec_save_file_dialog(self):
        """Executes save file dialog."""
        dlg = QFileDialog()
        file_path = dlg.getSaveFileName(self, "Save file", "", ".wav")
        file_path = [''.join(file_path[:])][0]
        if not file_path:
            return
        samples = self.samples_new
        params = self.wave_params
        dtype = self.data_type
        self.input_file.save_new_wav(samples, params, dtype, file_path)

    def exec_open_file_dialog(self):
        """Executes open file dialog."""
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.AnyFile)
        if dlg.exec_():
            file_path = dlg.selectedFiles()[0]
            self.file_name = file_path[file_path.rfind('/')+1 :]

            self.wave_params, self.data_type, self.frame_size, \
                self.samples = self.input_file.read_new_wav(file_path)
            params = self.wave_params
            samples = self.samples

            self.setup_interface(file_path, params)
            self.plot_graph(samples)
            self.disable_controls()
            self.menuApplyVAD.setEnabled(True)
            self.clear_graph_widgets()

    def setup_interface(self, file_path, params):
        """Outputs parameters of an opened WAV file on GUI form.

        Args:
            file_path (str):
                Full path to the WAV file.
            params:
                A tuple of WAV parameters.
        """
        # File path
        self.labelFileName.setText(file_path)
        # Number of channels
        self.labelChannels.setText(str(params.nchannels))
        # Sample width
        self.labelSWidth.setText(str(params.sampwidth))
        # Frame rate
        self.labelFR.setText(str(params.framerate))
        # Number of samples
        self.labelNFrames.setText(str(params.nframes))
        # Duration (in seconds)
        duration = round(params.nframes / params.framerate, 3)
        self.labelDuration.setText(str(duration) + ' seconds')
        # Type of compression
        self.labelCompName.setText(params.compname)

    def disable_controls(self):
        """Disables the following menu controls."""
        self.menuApplyVAD.setEnabled(False)
        self.menuVADShowInput.setEnabled(False)
        self.menuVADShowOutput.setEnabled(False)
        self.menuVADSaveFile.setEnabled(False)

    def clear_graph_widgets(self):
        """Clears all graphs on the GUI form."""
        self.graphVAD_STSE.clear()
        self.graphVAD_MDF.clear()
        self.graphVAD_SFM.clear()

    def plot_graph(self, samples, mode='wf'):
        """Plots the input values using PyQtGraph library.

        Args:
            samples (arr):
                Input data to plot, 1D list or numpy numeric array.
            mode (str):
                Type of graph to plot. The default is 'wf'.
                'wf' - Waveform of an audio file.
                'stse' - Short-Time Signal Energy.
                'mdf' - Most Dominant Frequency Component.
                'sfm' - Spectral Flatness Measure.
        """
        if mode == 'wf':
            graph_widget = self.graphWaveform
            graph_title = self.file_name
            label_bottom = 'Samples'
            label_left = 'Amplitude'
        elif mode == 'stse':
            graph_widget = self.graphVAD_STSE
            graph_title = 'Short-Time Signal Energy'
            label_bottom = 'Frames'
            label_left = 'Amplitude'
        elif mode == 'mdf':
            graph_widget = self.graphVAD_MDF
            graph_title = 'Most Dominant Frequency Component'
            label_bottom = 'Frames'
            label_left = 'Frequency (Hz)'
        elif mode == 'sfm':
            graph_widget = self.graphVAD_SFM
            graph_title = 'inverted Spectral Flatness Measure'
            label_bottom = 'Frames'
            label_left = 'Magnitude (dB)'
        x_values = range(len(samples))

        graph_widget.clear()
        pg.setConfigOptions(antialias=True)
        graph_widget.setTitle(title=graph_title)
        graph_widget.setBackground('w')
        graph_widget.setLabel('left', text=label_left)
        graph_widget.setLabel('bottom', text=label_bottom)
        graph_widget.plot(x_values, samples, pen=pg.mkPen(color=(0, 0, 0),
                                                          width=1))
        graph_widget.showGrid(x=True, y=True, alpha=0.7)

    def plot_vad_features(self, vad_vals):
        """Plots STSE, MDF and SFM graphs of an opened WAV file."""
        self.plot_graph(vad_vals[:,0], 'stse')
        self.plot_graph(vad_vals[:,1], 'mdf')
        self.plot_graph(vad_vals[:,2], 'sfm')

    def show_input_and_vad(self):
        """Plots waveform of an opened WAV file with selected
        regions of speech."""
        samples = self.samples
        v_regions = self.voiced_regions
        self.plot_graph(samples)
        for frame in v_regions:
            l_region = pg.LinearRegionItem(frame, movable=False)
            self.graphWaveform.addItem(l_region)
        self.menuVADShowInput.setChecked(True)
        self.menuVADShowOutput.setChecked(False)

    def show_output(self):
        """Plots waveform of an opened WAV file with regions of silence
        removed."""
        self.plot_graph(self.samples_new)
        self.menuVADShowInput.setChecked(False)
        self.menuVADShowOutput.setChecked(True)

    def apply_vad(self):
        """Applies VAD algorithm to a WAV file."""
        samples = self.samples
        frame_size = self.frame_size
        nframes = int(len(samples) / frame_size)
        vad_vals = self.input_file.comp_vad_all(frame_size, samples)
        self.voiced_regions = self.input_file.find_voiced_regions(
            vad_vals, nframes, frame_size)
        self.samples_new = self.input_file.remove_silence(
            self.voiced_regions, samples)

        self.plot_graph(samples)
        self.plot_vad_features(vad_vals)
        self.show_input_and_vad()

        self.menuVADShowInput.setEnabled(True)
        self.menuVADShowInput.setChecked(True)
        self.menuVADShowOutput.setEnabled(True)
        self.menuVADSaveFile.setEnabled(True)
        self.menuApplyVAD.setEnabled(False)

def main():
    """Executes the application and shows the main window."""
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()

def close_event():
    sys.exit(0)

if __name__ == '__main__':
    main()
