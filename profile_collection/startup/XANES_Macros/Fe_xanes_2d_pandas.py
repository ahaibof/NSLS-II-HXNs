#Last Update 07/15/2021
#osaz = 5000, zpz = 10.465 mm, crl#12, -5 to +4 (increasing E)
#EXAMPLE:<zp_list_xanes2d(e_list,dets1, zpssx,-3,3,50,zpssy,-3,3,50,0.03)

import numpy as np
from datetime import datetime
import pandas as pd

user_folder = '/data/users/2021Q2/Ajith_2021Q2/' #This is to add scan details in .txt format to your folder

e_list = pd.DataFrame()

e_points_dict = {}

e_points_dict['pre'] = [7.08,7.11,4]
e_points_dict['XANES1'] = [7.111,7.131,21]
e_points_dict['XANES2'] = [7.132,7.148,33]
e_points_dict['XANES3'] = [7.15,7.18,11]
e_points_dict['post'] = [7.182,7.232,6]

energies = []
for values in e_points_dict.values():
    energies.extend(np.linspace(values[0],values[1],values[2])) 

e_list['energy'] = energies


high_e = 7.2
low_e = 7.1
high_e_ugap = 7675
low_e_ugap = 7585
ugap_slope = (high_e_ugap - low_e_ugap)/(high_e-low_e)
ugap_list = high_e_ugap + (energies - high_e)*ugap_slope
e_list['ugap'] = ugap_list

high_e_crl= 4
low_e_crl = -5
crl_slope = (high_e_crl - low_e_crl)/(high_e-low_e)
crl_list = high_e_crl + (energies - high_e)*crl_slope
e_list['crl_theta'] = crl_list

zpz1_ref = 65.858-0.37
zpz1_slope = -5.9
zpz1_list = zpz1_ref + (energies - high_e)*zpz1_slope
e_list['ZP focus'] = zpz1_list 

e_list['E Readback'] = np.nan #add real energy to the dataframe
e_list['Scan ID'] = np.nan #add scan id to the dataframe
e_list['TimeStamp'] = pd.Timestamp.now()
e_list['IC3'] = sclr2_ch4.get() #Ic values are useful for calibration
e_list['IC0'] = sclr2_ch2.get() #Ic values are useful for calibration   


#e_list = np.column_stack((energies,ugap_list,zpz1_list,crl_list))

#e_list = e_list[::-1]

def peak_the_flux():
    
    yield from bps.sleep(2)
    yield from peak_bpm_y(-2,2,4)
    yield from peak_bpm_x(-15,15,5)
    yield from peak_bpm_y(-2,2,4)
    

    
def move_energy(e_,ugap_,zpz_,crl_th_, ignoreCRL= False, ignoreZPZ = False):
    yield from bps.sleep(1)
            
    #tuning the scanning pv on to dispable c bpms
    caput('XF:03IDC-ES{Status}ScanRunning-I', 1)  

    yield from bps.mov(e,e_)
    yield from bps.sleep(1)
    yield from bps.mov(ugap, ugap_)
    yield from bps.sleep(2)
    if not ignoreZPZ: yield from mov_zpz1(zpz_)
    yield from bps.sleep(1)
    if not ignoreCRL: yield from bps.mov(crl.p,crl_th_)
    yield from bps.sleep(1)
            
    caput('XF:03IDC-ES{Status}ScanRunning-I', 0) #scan status off
    

def zp_list_xanes2d(e_list,dets,mot1,x_s,x_e,x_num,mot2,y_s,y_e,y_num,accq_t, 
                    xcen = 0, ycen = 0, doScan = True, moveOptics = True, 
                    doAlignScan = True, pdfLog = True, foilCalibScan = False, 
                    peakBeam = True):

    ic_0 = sclr2_ch2.get()
    
    #opening fast shutter for initial ic3 reading
    caput('XF:03IDC-ES{Zeb:2}:SOFT_IN:B0',1) 
    yield from bps.sleep(2)
    
    #get the initial ic3 reading for peaking the beam
    ic_3 =  sclr2_ch4.get()
     
    #close fast shutter after initial ic3 reading
    caput('XF:03IDC-ES{Zeb:2}:SOFT_IN:B0',0) 
    
    #remeber the start positions
    zpssx_i = zpssx.position
    zpssy_i = zpssy.position

    #particle location, I keep the beam away (0,0) from it when so scans done
    xcen = 2.95
    ycen = -9.23


    for i in range (len(e_list)):

        yield from check_for_beam_dump(threshold=0.1*ic_0)
        
        e_t, ugap_t, zpz_t, crl_t, *others = e_list.iloc[i]
        
        if moveOptics: 
            yield from move_energy(e_t,ugap_t,zpz_t,crl_t,
                                   ignoreCRL= foilCalibScan, 
                                   ignoreZPZ = foilCalibScan)

        else: pass
        caput('XF:03IDC-ES{Zeb:2}:SOFT_IN:B0',1) #opening fast shutter
        if sclr2_ch4.get()<ic_3*0.9:
        
            if peakBeam: yield from peak_the_flux()
        
        ic_3 = sclr2_ch4.get()
        ic_0 = sclr2_ch2.get()

        # move to particle location for alignemnt scan
        if doScan:
        
            yield from bps.mov(zpssx, xcen)
            yield from bps.mov(zpssy, ycen)
        
        #do the alignemnt scan on the xanes elem after it excited , 
        #otherwise skip or use another element

        if e_list['energy'][i]<7.120:
            
            '''
            yield from fly1d(dets,zpssx,-1,1,100,0.1)
            xcen = return_line_center(-1,'Cl',0.6)
            yield from bps.mov(zpssx, xcen)
            yield from fly1d(dets,zpssy,-1,1 ,100,0.1)
            ycen = return_line_center(-1,'Cl',0.6)
            yield from bps.mov(zpssy, ycen)
            '''
            pass

        elif doScan and if doAlign:
            yield from fly1d(dets,zpssx,-2,2,100,0.03)
            xcen = return_line_center(-1,'Fe',0.6)
            yield from bps.mov(zpssx, xcen)
            yield from fly1d(dets,zpssy,-2,2 ,100,0.03)
            ycen = return_line_center(-1,'Fe',0.6)
            yield from bps.mov(zpssy, ycen)


        print(f'Current scan: {i+1}/{len(e_list)}')

        # do the fly2d scan

        if dets == dets_fs: #for fast xanes scan, no transmission (merlin) in the list

            if doScan: yield from fly2d(dets, mot1,x_s,x_e,x_num,mot2,y_s,y_e,y_num,accq_t, dead_time=0.001) 
            #dead_time = 0.001 for 0.015 dwell

        else:

            if doScan: yield from fly2d(dets, mot1,x_s,x_e,x_num,mot2,y_s,y_e,y_num,accq_t)
        yield from bps.sleep(1)
        
        # after scan done go to 0,0 to rest
        
        if doScan: yield from bps.mov(zpssx, 0)
        if doScan: yield from bps.mov(zpssy, 0)

        #ycen, xcen = return_center_of_mass_blurr(-1,'S') 
        # some cases use 2D mass center for alignemnt
        #print(ycen,xcen)

        # get some scan details and add to the list of scan id and energy

        last_sid = int(caget('XF:03IDC-ES{Status}ScanID-I'))
        e_pos = e.position
        
        #Add more info to the dataframe
        e_list['E Readback'][i] = e_pos #add real energy to the dataframe
        e_list['Scan ID'][i] = last_sid #add scan id to the dataframe
        e_list['TimeStamp'][i] = pd.Timestamp.now()
        e_list['IC3'][i] = ic_3 #Ic values are useful for calibration
        e_list['IC0'][i] = ic_0 #Ic values are useful for calibration   

        if pdfLog: insert_xrf_map_to_pdf(-1,'Fe')# plot data and add to pdf
        if pdfLog: insert_xrf_map_to_pdf(-1,'Cl')# plot data and add to pdf

        # save the DF in the loop so quitting a scan won't affect
        filename = f"HXN_nanoXANES_{e_list['Scan ID'][0]}_to_{e_list['Scan ID'][0]}.csv"
        e_list.to_csv(os.path.join(user_folder, filename))

    #go back to max energy point if scans done reverese
    max_e_id = e_list['energy'].idxmax()
    e_max, ugap_max, zpz_max, crl_max, *others = e_list.iloc[max_e_id]
    
    if not np.isclose(e_list['energy'].max(), e.position):
    
        yield from move_energy(e_max,ugap_max,zpz_max,crl_max,
                               ignoreCRL= foilCalibScan,
                               ignoreZPZ = foilCalibScan)
        
        yield from peak_the_flux()

    
    else: pass
        
    
    if pdfLog: save_page() #save the pdf

