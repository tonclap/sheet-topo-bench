"""Backoff patch for awc.fetch — importing this module applies it.

Why it exists: awc.fetch already retries 4 times, but back to back with no
pause. A DNS resolver outage lasting a few seconds therefore burns every
attempt at once — on 16.08 the detect_v2 dev run and the build_bothsilent
run died in the same second on the same `getaddrinfo failed`, one segment
from the end and mid-first-segment respectively. The frozen wave-2 and v1
files are not edited (the freeze covers them); new long-running scripts
import this module instead, which wraps the stock fetch in an outer loop
with growing sleeps. 404 -> None semantics pass through untouched: the
stock fetch returns None for absence without raising, so the wrapper only
ever retries genuine transport errors.

Usage, before any reader is constructed:

    import net_retry  # noqa: F401  (patches awc.fetch on import)
"""
import sys
import time

import absolute_winding_calibration as awc

# A Windows console defaults to a legacy code page, and the records below carry
# Cyrillic, Delta and the minus sign. Substitute the unrepresentable rather than
# raise: the numbers are the payload, and a UnicodeEncodeError would hide all of
# them behind the first one that does not fit.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, 'reconfigure'):
        _stream.reconfigure(errors='replace')

DELAYS_S = (5, 15, 45, 90, 180)

_stock_fetch = awc.fetch


def patient_fetch(url, byte_range=None, timeout=300, attempts=4):
    last = None
    for i, delay in enumerate((0,) + DELAYS_S):
        if delay:
            print(f'net_retry: transport error, sleeping {delay}s '
                  f'(pass {i} of {len(DELAYS_S)}): {last}',
                  file=sys.stderr, flush=True)
            time.sleep(delay)
        try:
            return _stock_fetch(url, byte_range, timeout=timeout,
                                attempts=attempts)
        except Exception as error:                                # noqa: BLE001
            last = error
    raise last


awc.fetch = patient_fetch
