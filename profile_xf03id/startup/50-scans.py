# vim: sw=4 ts=4 sts expandtab smarttab

from hxntools.scans import (HXNDScan, HXNAScan, HXNCount)

from hxnfly import fly1d, fly2d

ct = HXNCount()

ascan = HXNAScan()
'''
ascan.default_detectors = [det_sclr1,
                           ion0, ion1, ion3, Pt_ch1, Pt_ch2, Pt_ch3,
                           ssx_rbv, ssy_rbv, ssz_rbv,
                           t_base, t_sample, t_vlens, t_hlens]
'''
ascan.default_detectors = [det_sclr1,
                           ion0, ion1, ion3, Pt_ch1, Pt_ch2, Pt_ch3,
                           Al_ch1, Al_ch2, Al_ch3,Si_ch1, Si_ch2, Si_ch3,
                           S_ch1, S_ch2, S_ch3,Ar_ch1, Ar_ch2, Ar_ch3,
                           Ca_ch1, Ca_ch2, Ca_ch3,Ti_ch1, Ti_ch2, Ti_ch3,
                           V_ch1, V_ch2, V_ch3,Cr_ch1, Cr_ch2, Cr_ch3,
                           Mn_ch1, Mn_ch2, Mn_ch3,Fe_ch1, Fe_ch2, Fe_ch3,
                           Co_ch1, Co_ch2, Co_ch3,Ni_ch1, Ni_ch2, Ni_ch3,
                           Cu_ch1, Cu_ch2, Cu_ch3,Zn_ch1, Zn_ch2, Zn_ch3,
                           Au_ch1, Au_ch2, Au_ch3,
                           ssx_rbv, ssy_rbv, ssz_rbv,
                           t_base, t_sample, t_vlens, t_hlens]


ascan.user_detectors = [xspress3.filestore, timepix1.filestore]

dscan = HXNDScan()
# Detectors are shared among the scans


def synchronize(detectors, integration_time):
    # EpicsScaler calls it preset_time;
    # AreaDetector calls it exposure_time.
    attrs = ['preset_time', 'exposure_time']
    for det in detectors:
        for attr in attrs:
            try:
                setattr(det, attr, integration_time)
            except AttributeError:
                pass
            # TODO Raise better.

    # For a step scan using the zebra or scaler 1, the scaler is the master
    # Scaler LNE -> Zebra IN1_TTL -> Pulse1 -> fanout
    if zebra in detectors or det_sclr1 in detectors:
        sclr1._preset_time.put(integration_time)
        zebra.preset_time = integration_time

    for tpx in [timepix1, timepix2]:
        if tpx.filestore in detectors:
            tpx.acquire_time.put(integration_time)
            tpx.acquire_period.put(integration_time)


def sync_dscan(positioners, start, stop, step, exposure_time):
    """Perform a normal dscan, but first sync exposure_time time.

    Parameters
    ----------
    positioners : same as dscan
    start : same as dscan
    stop : same as dscan
    step : same as dscan
    exposure_time : sensor integration time in seconds
    """
    synchronize(dscan.detectors, exposure_time)
    return dscan(positioners, start, stop, step)


def d2_scan(*args):
    """
    Parameters
    ----------
    args : (positioner, start, stop)
    steps : number of steps in the scan

    Example
    -------
    # In five steps, move sli from 0 to 5 and slo from 0 to 10.
    >>> d2_scan([sli, 0, 5], [slo, 0, 10], 5)
    """

    args = list(args)
    steps = args.pop()  # The last argument is the number of steps.
    args = [[list(lst)] for lst in zip(*args)]
    args.append(steps)
    return dscan(*args)


def mesh(*args):
    '''
    Parameters
    ----------
    args: (positioner, start, stop)
    steps: [size1, size2, ...]
    exposure_time: float

    Example
    -------
    >>> mesh([sli, 0, 5], [slo, 0, 10], [5, 5], 1.0)
    >>> mesh([sli, 0, 5], [slo, 0, 10], [slt, 0, 10], [5, 5, 5], 0.5)
    '''
    args = list(args)
    exposure_time = args.pop()
    steps = args.pop()

    args = [list(lst) for lst in zip(*args)]
    args = _elements_to_singlets(args)
    args.append(steps)
    args.append(exposure_time)
    return sync_dscan(*args)


def _elements_to_singlets(a):
    """ [[a, b], c] -> [[[a], [b]], [c]] """
    return [_elements_to_singlets(element) if isinstance(element, (list, tuple))
            else [element]
            for element in a]


def movr_hth(angle):
    radian = angle*pi/180.0
    correction = -1.*tan(radian)*34376.6
    movr(hth, angle)
    movr(hx,correction)


def movr_ssx(d):
    dx_n = d * cos(15.*pi/180.)
    dz_n = d * sin(15.*pi/180.)
    movr(ssx, dx_n)
    movr(ssz, dz_n)

def movr_sx(d):
    dx_n = d * cos(15.*pi/180.)
    dz_n = d * sin(15.*pi/180.)
    movr(sx, dx_n)
    movr(sz, dz_n)


def movr_sz(d):
    dx_n = -d * sin(15.*pi/180.)
    dz_n = d * cos(15.*pi/180.)
    movr(sx, dx_n)
    movr(sz, dz_n)


def movr_ssz(d):
    dx_n = -d * sin(15.*pi/180.)
    dz_n = d * cos(15.*pi/180.)
    movr(ssx, dx_n)
    movr(ssz, dz_n)
