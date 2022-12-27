'''
from epics import caput,caget

def fly1d_user(motor,start,end,num_pos,exp):
    caput('XF:03IDC-ES{Flt:2}sendCommand.VAL',"R1")
    RE(fly1d(motor,start,end,num_pos,exp))
    caput('XF:03IDC-ES{Flt:2}sendCommand.VAL',"I1")


def fly2d_user(motor1,start1,end1,num1,motor2,start2,end2,num2,exp):
    caput('XF:03IDC-ES{Flt:2}sendCommand.VAL',"R1")
    RE(fly2d(motor1,start1,end1,num1,motor2,start2,end2,num2,exp))
    caput('XF:03IDC-ES{Flt:2}sendCommand.VAL',"I1")
'''
def theta_fly2d(angle_start,angle_end,num_angle,motor1,start1,end1,num1,motor2,start2,end2,num2,exp):
    angle_current = zps.zpsth.position
    smarx_current = zps.smarx.position
    smary_current = zps.smary.position
    smarz_current = zps.smarz.position
    ssx_current = zps.zpssx.position
    ssy_current = zps.zpssy.position
    ssz_current = zps.zpssz.position

    angle_step = (angle_end - angle_start) / num_angle
    start_angle = angle_current + angle_start
    mov(zps.zpsth,start_angle)
    #RE(fly1d(zps.zpssz,-1,1,50,0.1))
    #mov_to_line_center(-1,elem='Ga',moveflag=1,threshold=0.2,movepiezoflag=1)

    for i in range(num_angle+1):
        RE(fly2d(motor1,start1,end1,num1,motor2,start2,end2,num2,exp))
        mov_to_image_cen_zpss(-1,'Ga',1)
        export(-1)
        movr(zps.zpsth,angle_step)
        #RE(fly1d(zps.zpssz,-.25,.25,20,0.1))
        #mov_to_line_center(-1,elem='Ga',moveflag=1,threshold=0.2,movepiezoflag=1)
        print('wait 2 sec ...')
        sleep(2)

    mov(zps.zpsth,angle_current)

def searchtheta(angle_start,angle_end,num_angle,motor1,start1,end1,num1):
    angle_current = zps.zpsth.position
    smarx_current = zps.smarx.position
    smary_current = zps.smary.position
    smarz_current = zps.smarz.position
    ssx_current = zps.zpssx.position
    ssy_current = zps.zpssy.position
    ssz_current = zps.zpssz.position

    angle_step = (angle_end - angle_start) / num_angle
    start_angle = angle_current + angle_start
    mov(zps.zpsth,start_angle)
    RE(fly1d(zps.zpssz,-2,2,50,0.1))
    mov_to_line_center(-1,elem='Ga',moveflag=1,threshold=0.2,movepiezoflag=1)

    for i in range(num_angle+1):
        movr(zps.zpsth,angle_step)
        RE(fly1d(zps.zpssz,-2,2,40,0.1))
        mov_to_line_center(-1,elem='Ga',moveflag=1,threshold=0.2,movepiezoflag=1)
        print('wait 2 sec ...')
        sleep(2)


def theta_fly2d_ce(angle_start,angle_end,num_angle,motor1,start1,end1,num1,motor2,start2,end2,num2,exp):
    angle_current = zps.zpsth.position
    smarx_current = zps.smarx.position
    smary_current = zps.smary.position
    smarz_current = zps.smarz.position
    ssx_current = zps.zpssx.position
    ssy_current = zps.zpssy.position
    ssz_current = zps.zpssz.position

    angle_step = (angle_end - angle_start) / num_angle
    start_angle = angle_current + angle_start
    mov(zps.zpsth,start_angle)
   # RE(fly1d(zps.zpssz,-1,1,50,0.1))
   # mov_to_line_center(-1,elem='Ga',moveflag=1,threshold=0.2,movepiezoflag=1)

    for i in range(num_angle+1):
        RE(fly2d(motor1,start1,end1,num1,motor2,start2,end2,num2,exp))
    #    mov_to_image_cen_zpss(-1,'Ga',1)
        export(-1)
        movr(zps.zpsth,angle_step)
        movr(zps.zpssz,-2*angle_step)
    #    RE(fly1d(zps.zpssz,-.25,.25,20,0.1))
    #    mov_to_line_center(-1,elem='Ga',moveflag=1,threshold=0.2,movepiezoflag=1)
        print('wait 2 sec ...')
        sleep(2)

   # mov(zps.zpsth,angle_current)

def search_smarx():
    movr(zps.smarx, -.1)
    RE(fly2d(zpssz,-15,15,60, zpssy, -15,15,60, .1))
    movr(zps.smarx, -.1)
    print('10sec pause')
    sleep(10)
    RE(fly2d(zpssz,-15,15,60, zpssy, -15,15,60, .1))
    movr(zps.smarx, -.1)
    print('10sec pause')
    sleep(10)
    RE(fly2d(zpssz,-15,15,60, zpssy, -15,15,60, .1))
    movr(zps.smarx, -.1)
    print('10sec pause')
    sleep(10)
    RE(fly2d(zpssz,-15,15,60, zpssy, -15,15,60, .1))
    movr(zps.smarx, -.1)
    print('10sec pause')
    sleep(10)
    RE(fly2d(zpssz,-15,15,60, zpssy, -15,15,60, .1))
    movr(zps.smarx, -.1)
    print('10sec pause')
    sleep(10)
    RE(fly2d(zpssz,-15,15,60, zpssy, -15,15,60, .1))
    movr(zps.smarx, -.1)
    print('10sec pause')
    sleep(10)
    RE(fly2d(zpssz,-15,15,60, zpssy, -15,15,60, .1))
    movr(zps.smarx, -.1)
    print('10sec pause')
    sleep(10)
    RE(fly2d(zpssz,-15,15,60, zpssy, -15,15,60, .1))
    movr(zps.smarx, -.1)
    print('10sec pause')
    sleep(10)
    RE(fly2d(zpssz,-15,15,60, zpssy, -15,15,60, .1))
    movr(zps.smarx, -.1)
    print('10sec pause')
    sleep(10)
    RE(fly2d(zpssz,-15,15,60, zpssy, -15,15,60, .1))
    movr(zps.smarx, -.1)
    print('10sec pause')
    sleep(10)
    RE(fly2d(zpssz,-15,15,60, zpssy, -15,15,60, .1))
    movr(zps.smarx, -.1)
    print('10sec pause')
    sleep(10)
    RE(fly2d(zpssz,-15,15,60, zpssy, -15,15,60, .1))
    movr(zps.smarx, -.1)

def vlm_tomo(angle_start,angle_end,angle_step):
    angle_0 = dsth.position
    angle_list = np.arange(angle_start,angle_end+0.1,angle_step)
    num_angle = np.size(angle_list)
    mov(dsth,angle_list[0])
    for i in range(num_angle):
        mov(dsth,angle_list[i])
        sleep(1)
        caput('XF:03IDC-ES{CAM:10}TIFF1:WriteFile',1)
    mov(dsth,angle_0)


def vlm_orth(start,end,step,angle):
    x_list = np.arange(start,end+1,step)
    y_list = np.arange(start,end+1,step)
    num = np.size(x_list)
    mov(dsth,angle)
    for i in range(num):
        mov(dssy,y_list[i])
        for j in range(num):
            if angle == 0:
                mov(dssx,x_list[j])
            else:
                mov(dssz,x_list[j])
            sleep(2)
            caput('XF:03IDC-ES{CAM:10}TIFF1:WriteFile',1)
        sleep(2)

    mov(dssy,0)
    mov(dssx,0)
    mov(dssz,0)

def vlm_orth_coarse(start,end,step,angle):
    x_list = np.arange(start,end+1,step)
    y_list = np.arange(start,end+1,step)
    num = np.size(x_list)
    dsx_0 = dsx.position
    dsy_0 = dsy.position
    dsz_0 = dsz.position
    mov(dsth,angle)
    movr(dsy,start/1000.)

    for i in range(num):

        if angle == 0:
            mov(dsx,dsx_0+start)
        else:
            mov(dsz,dsz_0+start)

        for j in range(num):
            if angle == 0:
                movr(dsx,step)
            else:
                movr(dsz,step)
            sleep(1)
            caput('XF:03IDC-ES{CAM:10}TIFF1:WriteFile',1)
        movr(dsy,step/1000.)

    mov(dsx,dsx_0)
    mov(dsy,dsy_0)
    mov(dsz,dsz_0)

def peak_up_xbpm(start_relative,end_relative,step_size,direction='y'):
    print('peaking up xbmp set point in ', direction, 'direction')
    num_points = np.int((end_relative - start_relative) // step_size + 1)
    ic = np.zeros(num_points)

    if direction == 'x':
        setpoint_pv = 'XF:03ID{XBPM:17}Fdbk:A-SP'
    elif direction == 'y':
        setpoint_pv = 'XF:03ID{XBPM:17}Fdbk:B-SP'

    init_position = caget(setpoint_pv)

    position = init_position + start_relative + np.arange(num_points) * step_size

    for i in range(num_points):
        caput(setpoint_pv,position[i])
        sleep(2)
        ic[i] = caget('XF:03IDC-ES{Sclr:2}_cts1.D')
        print(direction, 'set point: ', position[i], ', IC 3 counts: ', ic[i])

    peak_position = position[ic == np.max(ic)]
    print('peak count at ',peak_position)
    caput(setpoint_pv,peak_position)


def nightscan():
    RE(fly2d(dssx,-0.5, 0.5, 100, dssy, -0.5, 0.5, 100, 0.1))
    sleep(5)
    RE(fly2d(dssx,-0.5, 0.5, 100, dssy, -0.5, 0.5, 100, 0.05))
    sleep(5)
    movr(dssy,1)
    RE(fly2d(dssx,-0.5, 0.5, 100, dssy, -0.5, 0.5, 100, 0.1))
    sleep(5)
    RE(fly2d(dssx,-0.5, 0.5, 100, dssy, -0.5, 0.5, 100, 0.1))
    movr(dssx, 1.)
    sleep(5)
    RE(fly2d(dssx,-0.5, 0.5, 100, dssy, -0.5, 0.5, 100, 0.1))
    sleep(5)
    RE(fly2d(dssx,-0.5, 0.5, 100, dssy, -0.5, 0.5, 100, 0.05))
    RE(fly2d(dssx,-0.5, 0.5, 100, dssy, -0.5, 0.5, 100, 0.5))
    movr(dssy, 2)
    movr(dssx,-2)
    RE(fly2d(dssx,-1,1,200,dssy,-1,1,200,0.2))
    shutter('close')

