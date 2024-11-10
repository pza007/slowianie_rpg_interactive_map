# Copyright (c) 2024, Przemyslaw Zawadzki
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from PyQt6.QtWidgets import QApplication
import sys
from gui_lib import Window


def main():
    app = QApplication(sys.argv)

    window = Window()
    window.display_map()
    window.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
