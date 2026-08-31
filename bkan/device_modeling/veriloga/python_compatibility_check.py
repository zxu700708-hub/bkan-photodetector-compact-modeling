#!/usr/bin/env python3
"""Check the remote verifier stack without using ``python -c``.

Some EDA installations expose Python through a shell wrapper that reparses
``-c`` arguments with ``eval``.  A standalone script avoids that wrapper bug
while still checking the exact interpreter and scientific packages used by
the Spectre acceptance scripts.
"""

import sys


def main():
    if sys.version_info < (3, 6):
        sys.stderr.write("Python 3.6 or newer is required.\n")
        return 1
    try:
        import numpy
        import pandas
        import scipy
    except Exception as exc:
        sys.stderr.write("Scientific Python import failed: {}\n".format(exc))
        return 1
    print(
        "Python {}.{}.{} | numpy {} | pandas {} | scipy {}".format(
            sys.version_info[0],
            sys.version_info[1],
            sys.version_info[2],
            numpy.__version__,
            pandas.__version__,
            scipy.__version__,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
