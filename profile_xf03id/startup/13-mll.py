from ophyd.controls import EpicsMotor, PVPositioner

ssx = EpicsMotor('XF:03IDC-ES{Ppmac:1-ssx}Mtr', name='ssx')
ssy = EpicsMotor('XF:03IDC-ES{Ppmac:1-ssy}Mtr', name='ssy')
ssz = EpicsMotor('XF:03IDC-ES{Ppmac:1-ssz}Mtr', name='ssz')

sth = EpicsMotor('XF:03IDC-ES{ANC350:1-Ax:0}Mtr', name='sth')
# anc1m1 = EpicsMotor('XF:03IDC-ES{ANC350:1-Ax:1}Mtr', name='anc1m1')
hth = EpicsMotor('XF:03IDC-ES{ANC350:1-Ax:2}Mtr', name='hth')
# anc350_ssx = EpicsMotor('XF:03IDC-ES{ANC350:1-Ax:3}Mtr', name='anc350_ssx')
# anc350_ssy = EpicsMotor('XF:03IDC-ES{ANC350:1-Ax:4}Mtr', name='anc350_ssy')
# anc350_ssz = EpicsMotor('XF:03IDC-ES{ANC350:1-Ax:5}Mtr', name='anc350_ssz')

vx = EpicsMotor('XF:03IDC-ES{ANC350:2-Ax:0}Mtr', name='vx')
vy = EpicsMotor('XF:03IDC-ES{ANC350:2-Ax:1}Mtr', name='vy')
vz = EpicsMotor('XF:03IDC-ES{ANC350:2-Ax:2}Mtr', name='vz')
vchi = EpicsMotor('XF:03IDC-ES{ANC350:2-Ax:3}Mtr', name='vchi')
vth = EpicsMotor('XF:03IDC-ES{ANC350:2-Ax:4}Mtr', name='vth')
hx = EpicsMotor('XF:03IDC-ES{ANC350:2-Ax:5}Mtr', name='hx')
sy = EpicsMotor('XF:03IDC-ES{ANC350:3-Ax:0}Mtr', name='sy')
sx1 = EpicsMotor('XF:03IDC-ES{ANC350:3-Ax:1}Mtr', name='sx1')
sz = EpicsMotor('XF:03IDC-ES{ANC350:3-Ax:2}Mtr', name='sz')
sz1 = EpicsMotor('XF:03IDC-ES{ANC350:3-Ax:3}Mtr', name='sz1')
hy = EpicsMotor('XF:03IDC-ES{ANC350:4-Ax:0}Mtr', name='hy')
hz = EpicsMotor('XF:03IDC-ES{ANC350:4-Ax:1}Mtr', name='hz')
osax = EpicsMotor('XF:03IDC-ES{ANC350:4-Ax:2}Mtr', name='osax')
osay = EpicsMotor('XF:03IDC-ES{ANC350:4-Ax:3}Mtr', name='osay')
osaz = EpicsMotor('XF:03IDC-ES{ANC350:4-Ax:4}Mtr', name='osaz')
sx = EpicsMotor('XF:03IDC-ES{ANC350:4-Ax:5}Mtr', name='sx')
bsx = EpicsMotor('XF:03IDC-ES{ANC350:5-Ax:0}Mtr', name='bsx')
bsy = EpicsMotor('XF:03IDC-ES{ANC350:5-Ax:1}Mtr', name='bsy')

# anc5m2 = EpicsMotor('XF:03IDC-ES{ANC350:5-Ax:2}Mtr', name='anc5m2')
# anc5m3 = EpicsMotor('XF:03IDC-ES{ANC350:5-Ax:3}Mtr', name='anc5m3')
# anc5m4 = EpicsMotor('XF:03IDC-ES{ANC350:5-Ax:4}Mtr', name='anc5m4')
# anc5m5 = EpicsMotor('XF:03IDC-ES{ANC350:5-Ax:5}Mtr', name='anc5m5')
# anc6m0 = EpicsMotor('XF:03IDC-ES{ANC350:6-Ax:0}Mtr', name='anc6m0')
# anc6m1 = EpicsMotor('XF:03IDC-ES{ANC350:6-Ax:1}Mtr', name='anc6m1')
# anc6m2 = EpicsMotor('XF:03IDC-ES{ANC350:6-Ax:2}Mtr', name='anc6m2')
# anc6m3 = EpicsMotor('XF:03IDC-ES{ANC350:6-Ax:3}Mtr', name='anc6m3')
# anc6m4 = EpicsMotor('XF:03IDC-ES{ANC350:6-Ax:4}Mtr', name='anc6m4')
# anc6m5 = EpicsMotor('XF:03IDC-ES{ANC350:6-Ax:5}Mtr', name='anc6m5')
# anc7m0 = EpicsMotor('XF:03IDC-ES{ANC350:7-Ax:0}Mtr', name='anc7m0')
# anc7m1 = EpicsMotor('XF:03IDC-ES{ANC350:7-Ax:1}Mtr', name='anc7m1')
# anc7m2 = EpicsMotor('XF:03IDC-ES{ANC350:7-Ax:2}Mtr', name='anc7m2')