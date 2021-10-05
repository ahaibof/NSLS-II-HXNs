import asyncio

from functools import partial
from bluesky.standard_config import *
from bluesky.scans import *
from bluesky.callbacks import *
from bluesky.broker_callbacks import *
from bluesky.hardware_checklist import *

from hxntools.uid_broadcast import HxnUidBroadcast
from hxntools.scan_status import HxnScanStatus

RE = gs.RE
RE.md['group'] = ''
RE.md['config'] = {}
RE.md['beamline_id'] = 'HXN'
# RE.ignore_callback_exceptions = False


class HxnScanNumberPrinter:
    def __init__(self):
        self._last_start = None

    def __call__(self, name, doc):
        if name == 'start':
            self._last_start = doc
        if self._last_start is None:
            return
        if name in ('start', 'stop'):
            print('Scan ID: {scan_id} [{uid}]'.format(**self._last_start))

uid_broadcaster = HxnUidBroadcast('XF:03IDC-ES{BS-Scan}UID-I')
hxn_scan_status = HxnScanStatus('XF:03IDC-ES{Status}ScanRunning-I')

RE.subscribe('all', HxnScanNumberPrinter())
RE.subscribe('all', uid_broadcaster)
RE.subscribe('all', hxn_scan_status)

loop = asyncio.get_event_loop()
loop.set_debug(False)
# RE.verbose = True

# sr_shutter_status = EpicsSignal('SR-EPS{PLC:1}Sts:MstrSh-Sts', rw=False,
#                                 name='sr_shutter_status')
# sr_beam_current = EpicsSignal('SR:C03-BI{DCCT:1}I:Real-I', rw=False,
#                               name='sr_beam_current')

checklist = partial(basic_checklist,
                    ca_url='http://xf03id-ca.cs.nsls2.local:4800',
                    # disk_storage=[('/', 1e9)],
                    # pv_names=['XF:23ID1-ES{Dif-Ax:SY}Pos-SP'],
                    # pv_conditions=[('XF:23ID-PPS{Sh:FE}Pos-Sts', 'front-end shutter is open', assert_pv_equal, 0),
                    # 		   ('XF:23IDA-PPS:1{PSh}Pos-Sts', 'upstream shutter is open', assert_pv_equal, 0),
                    #                ('XF:23ID1-PPS{PSh}Pos-Sts', 'downstream shutter is open', assert_pv_equal, 0)],
		    )
