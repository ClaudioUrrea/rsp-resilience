#!/usr/bin/env python3
"""
patch_fig3_thousands.py

MDPI copy-editing query on Figure 3 (empirical Lipschitz ratio):

    "Please use commas to separate thousands for numbers with five or more
     digits (not four digits) in the picture, e.g. '10000' should be '10,000'."

Figure 3 is the only figure in the paper whose axes carry numbers of five or
more digits: its y-axis (episode counts) runs to 15000.  This script edits
scripts/make_analysis.py in place so that fig_lipschitz() formats that axis
with the MDPI rule - separators from five digits upward, none at four - and
then leaves the rest of the file untouched.

It is idempotent: running it twice changes nothing the second time.  Run it
once, then regenerate the figures:

    python scripts/patch_fig3_thousands.py
    python scripts/make_analysis.py

Author: C. Urrea
License: MIT
"""

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET = os.path.join(ROOT, 'scripts', 'make_analysis.py')

ANCHOR = "    ax.set_ylabel('Episodes')\n"

INSERT = """    ax.set_ylabel('Episodes')
    # MDPI house rule: a thousands separator from five digits upward, none at
    # four.  This axis is the only one in the paper that reaches five digits.
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(
        lambda v, _: format(int(v), ',') if v >= 10000 else '%d' % v))
"""

IMPORT_LINE = 'import matplotlib.ticker as mtick\n'


def main():
    if not os.path.exists(TARGET):
        sys.exit('not found: %s\nRun this script from the repository.' % TARGET)

    src = open(TARGET, encoding='utf-8').read()

    if 'MDPI house rule' in src:
        print('already patched; nothing to do')
        return 0

    if ANCHOR not in src:
        sys.exit("could not find the anchor line %r in fig_lipschitz(); "
                 "apply the change by hand." % ANCHOR.strip())
    if src.count(ANCHOR) != 1:
        sys.exit('the anchor line is not unique; apply the change by hand.')

    # the ticker import, placed with the other matplotlib imports
    if 'matplotlib.ticker' not in src:
        m = re.search(r'^import matplotlib\.pyplot as plt\n', src, re.M)
        if not m:
            sys.exit('no "import matplotlib.pyplot as plt" line found.')
        src = src[:m.end()] + IMPORT_LINE + src[m.end():]

    src = src.replace(ANCHOR, INSERT, 1)

    open(TARGET, 'w', encoding='utf-8').write(src)
    print('patched %s' % TARGET)
    print('now run:  python scripts/make_analysis.py')
    return 0


if __name__ == '__main__':
    sys.exit(main())
