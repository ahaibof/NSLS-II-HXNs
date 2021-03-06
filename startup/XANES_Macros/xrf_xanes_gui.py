#conda activate analysis-2019-3.0-hxn-clone2

import sys, os, time, subprocess, logging
import matplotlib.pyplot as plt
import numpy as np
import time

from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QMessageBox, QFileDialog
from pyxrf.api import *
from epics import caget

logger = logging.getLogger()



def hxn_auto_loader(wd,param_file_name,scaler_name):



    printed = False

    while True:
    
        while int(caget('XF:03IDC-ES{Sclr:2}_cts1.B')) < 5000:
    
            logger.info('beam is not available: waiting for shutter to open')
            time.sleep(60)
        
        if caget('XF:03IDC-ES{Status}ScanRunning-I')==1 and not printed:
                
            logger.info('\n**waitng for scan to complete**\n')
            printed =True
                
    
        while caget('XF:03IDC-ES{Status}ScanRunning-I')==0:
            try:         
                sid = int(caget('XF:03IDC-ES{Status}ScanID-I'))
                logger.info(f'calling scan {sid} from data broaker')
                hdr = db[(sid)]
            
            except ValueError:
                logger.error('scan maybe incomplete; skipped')
            
                pass
    
    
            if bool(hdr.stop) == True:
                try:
                    make_hdf(sid, wd = wd, file_overwrite_existing=True)
                    pyxrf_batch(sid,sid, wd = wd, param_file_name = param_file_name, scaler_name = scaler_name )
                    logger.info('\n**waitng for scan to complete**\n')
                except:
                    
                    pass
    
            
            else:
              logger.info('scan is incomplete; skipped')
              pass
                



class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi("/GPFS/XF03ID1/home/xf03id/ipython_ophyd/profile_collection/startup/XANES_Macros/xtf_xanes_gui.ui", self)
                #All the connections done here

        self.pb_wd.clicked.connect(self.get_wd)
        self.pb_param.clicked.connect(self.get_param)
        self.pb_ref.clicked.connect(self.get_ref_file)
        self.pb_start.clicked.connect(self.create_xanes_macro)
        self.pb_xrf_start.clicked.connect(self.create_pyxrf_batch_macro)
        self.pb_activate_live.clicked.connect(self.change_label)
        self.pb_live.clicked.connect(self.start_auto)
        self.pb_open_pyxrf.clicked.connect(self.open_pyxrf)
        
        self.pb_close_plots.clicked.connect(self.close_all_plots)
        #self.pb_stop_live.clicked.connect(self.stopClick)
    

        self.show()

    def get_wd(self):
        folder_path = QFileDialog().getExistingDirectory(self, "Select Folder")
        self.le_wd.setText(str(folder_path))
    
    def get_param(self):
        file_name = QFileDialog().getOpenFileName(self, "Open file")
        self.le_param.setText(str(file_name[0]))
        
    def get_ref_file(self):
        file_name = QFileDialog().getOpenFileName(self, "Open file")
        self.le_ref.setText(str(file_name[0]))
        
    
    def create_pyxrf_batch_macro(self):
    
        cwd = self.le_wd.text()
        param = self.le_param.text()
        last_sid = int(self.le_lastid_2.text())
        first_sid = int(self.le_startid_2.text())
        norm = self.le_sclr_2.text()
        all_sid = np.arange(first_sid, last_sid+1)
        logger.info(f'scans to process {all_sid}')

          
        #os.chdir(cwd)
        #return pyxrf_batch(first_sid, last_sid, wd = cwd, param_file_name =param, scaler_name=norm, save_tiff = True)
        return (make_hdf(first_sid,last_sid, wd = cwd, file_overwrite_existing=True), pyxrf_batch(first_sid, last_sid, wd = cwd, param_file_name =param, scaler_name=norm, save_tiff = True))
                                   
    def create_xanes_macro(self):
    
        cwd = self.le_wd.text()
        param = self.le_param.text()
        last_sid = int(self.le_lastid.text())
        first_sid = int(self.le_startid.text())
        ref = self.le_ref.text()
        fit_method = self.cb_fittin_method.currentText()
        elem = self.xanes_elem.text()
        align_elem = self.alignment_elem.text()
        e_shift = float(self.energy_shift.text())
        admm_lambda = int(self.nnls_lamda.text())
        work_flow = self.cb_process.currentText()
        norm = self.le_sclr.text()
        save_all = self.ch_b_save_all_tiffs.isChecked()
        pre_edge = self.ch_b_baseline.isChecked()
        align = self.cb_align.isChecked()
        

        return     build_xanes_map(first_sid, last_sid, wd = cwd, xrf_fitting_param_fln=param,
                                   scaler_name=norm,sequence=work_flow,
                                   ref_file_name=ref, fitting_method=fit_method,
                                   emission_line=elem, emission_line_alignment=align_elem,
                                   incident_energy_shift_keV=(e_shift*0.001), subtract_pre_edge_baseline = pre_edge,
                                   alignment_enable = align, output_save_all = save_all)
                                   
    
    def change_label(self):
    
        printed = False
        
        status = self.label_live_status.text()
        if status == 'Live Processing is not ready':
        
        
            choice = QMessageBox.question(self, 'Message',
                                         "Confirm you are starting a live xrf processing, Click Start Live next", QMessageBox.Yes |
                                         QMessageBox.No, QMessageBox.No)

            if choice == QMessageBox.Yes:
                self.label_live_status.setText('Live Processing is active')
                

            else:
                pass
                
                
                
    def start_auto(self):
    
        cwd = self.le_wd.text()
        param = self.le_param.text()
        norm = self.le_sclr_2.text()
        #os.chdir(cwd)
        status = self.label_live_status.text()
        
        if status == 'Live Processing is active':
                logger.info(status)
                return hxn_auto_loader(wd = cwd,param_file_name = param,scaler_name=norm )
                logger.info('waiting for next scan to complete')
              
                    
    def open_pyxrf(self):
           subprocess.Popen(['pyxrf'])

    def close_all_plots(self):
            return plt.close('all')
            
    def stopClick(self):
            stop == 1
            
           
    def write(self, record):
        msg = self.format(record)
        self.pte_status.appendPlainText(msg)

if __name__ == "__main__":

    

    formatter = logging.Formatter(fmt='%(asctime)s : %(levelname)s : %(message)s')
    
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)
    if (logger.hasHandlers()):
      logger.handlers.clear()
    logger.addHandler(stream_handler)
    
    app = QtWidgets.QApplication(sys.argv)
    window = Ui()
    window.show()
    sys.exit(app.exec_())
