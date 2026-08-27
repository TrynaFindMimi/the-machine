import sys

from app.registry import TEST_NAMES, TESTS
from app.runner import run
from config.strings import USAGE


def main() -> None:
    start = 0
    if len(sys.argv) > 1:
        if sys.argv[1] not in TESTS:
            names = ", ".join(TEST_NAMES)
            print(USAGE.format(names))
            raise SystemExit(1)
        start = TEST_NAMES.index(sys.argv[1])
    run(start)


if __name__ == "__main__":
    main()
