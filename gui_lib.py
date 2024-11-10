# Copyright (c) 2024, Przemyslaw Zawadzki
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from PyQt6.QtWidgets import QMainWindow, QWidget, QScrollArea, QVBoxLayout, QLabel, QSlider, QSpinBox, QCheckBox
from PyQt6 import QtCore, QtGui, uic
import datetime
from dateutil.relativedelta import relativedelta
import functions
from map_lib import Map
from enumerations import EnumTimeOfDay, get_time_pointer_position


class ScrollLabel(QScrollArea):
    def __init__(self, *args, **kwargs):
        QScrollArea.__init__(self, *args, **kwargs)
        self.my_map = None
        self.main_window = None

        # making widget resizable
        self.setWidgetResizable(True)
        # making qwidget object
        content = QWidget(self)
        self.setWidget(content)
        # vertical box layout
        lay = QVBoxLayout(content)
        # creating label
        self.label = QLabel(content)
        # setting alignment to the text
        self.label.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop)
        # adding label to the layout
        lay.addWidget(self.label)

        self.label.setMouseTracking(True)
        self.label.mouseMoveEvent = self.label_mouseMoveEvent

    def label_mouseMoveEvent(self, event):
        x, y = event.pos().x(), event.pos().y()  #print(f"Mouse Coordinates: ({x}, {y})")
        row, col = y, x
        f_reset_text = False

        prev_event_number = self.main_window.out_event_number.text()
        prev_event_title = self.main_window.out_event_title.text()
        new_event_number, new_event_title = self.my_map.get_event_data((row, col))

        # mouse entered event position -> change text
        if new_event_number is not None:
            self.main_window.out_event_number.setText(str(new_event_number))
        if new_event_title is not None:
            self.main_window.out_event_title.setText(new_event_title)

        # mouse left event position -> reset
        if prev_event_number != '' and new_event_number is None:
            f_reset_text = True

        # reset text?
        if f_reset_text:
            self.main_window.out_event_number.clear()
            self.main_window.out_event_title.clear()

    def mousePressEvent(self, e):
        def get_click_point():
            # get mouse positions
            mouse_pos = e.position()
            h_scrollbar_pos = self.horizontalScrollBar().value()
            v_scrollbar_pos = self.verticalScrollBar().value()
            row = mouse_pos.y() + v_scrollbar_pos - 10  # -10 because of QScrollArea padding
            col = mouse_pos.x() + h_scrollbar_pos - 10

            # adjust values to map ranges
            if row < 0:
                row = 0
            if row > self.my_map.map_height:
                row = self.my_map.map_height - 1
            if col < 0:
                col = 0
            if col > self.my_map.map_width:
                col = self.my_map.map_width - 1
            return int(row), int(col)

        def process_left_click():
            _mouse_key = "LEFT"
            _center_point = None
            # show tile(s)
            if self.main_window.chb_show_tiles.isChecked():
                if self.main_window.slider_show_tiles.value() == 0:  # single tile
                    _center_point = self.my_map.one_tile(click_point, "show")
                else:  # multiple tiles
                    _center_point = self.my_map.multiple_tiles(click_point, "show")
            # move banner
            if self.main_window.chb_move_banner.isChecked():
                _center_point = self.my_map.update_banner_position(click_point)
            # change time
            if self.main_window.chb_change_time.isChecked():
                self.main_window.sb_time_of_day.stepUp()
            return _mouse_key, _center_point

        def process_middle_click():
            _mouse_key = "MIDDLE"
            _center_point = None
            if self.main_window.chb_visit_event.isChecked():
                _center_point = self.my_map.update_events(click_point)
            return _mouse_key, _center_point

        def process_right_click():
            _mouse_key = "RIGHT"
            _center_point = None
            if self.main_window.chb_hide_tiles.isChecked():
                if self.main_window.slider_hide_tiles.value() == 0:  # one tile
                    _center_point = self.my_map.one_tile(click_point, "hide")
                else:  # neighbour tiles
                    _center_point = self.my_map.multiple_tiles(click_point, "hide")
            return _mouse_key, _center_point

        # ----------------------------------------------------------------------------------
        click_point = get_click_point()
        mouse_key, center_point = None, None

        functions.my_click_point = click_point

        if e.button() == QtCore.Qt.MouseButton.LeftButton:
            mouse_key, center_point = process_left_click()

        elif e.button() == QtCore.Qt.MouseButton.MiddleButton:
            mouse_key, center_point = process_middle_click()

        elif e.button() == QtCore.Qt.MouseButton.RightButton:
            mouse_key, center_point = process_right_click()

        print(f"click_point={click_point}, mouse_key={mouse_key}, center_point={center_point}")

        # display updated map
        self.main_window.display_map()


class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        # VARIABLES FROM *.UI -----------------------------------------
        self.out_img_winter = QLabel()
        self.out_img_spring = QLabel()
        self.out_img_summer = QLabel()
        self.out_img_autumn = QLabel()
        self.out_img_time_pointer = QLabel()
        self.out_img_time_grad_1 = QLabel()
        self.out_img_time_grad_2 = QLabel()
        self.out_img_time_grad_3 = QLabel()
        self.out_img_time_grad_4 = QLabel()
        self.out_img_time_of_day = QLabel()
        self.out_text_time_of_day = QLabel()
        self.out_img_weather = QLabel()
        self.out_text_weather = QLabel()
        #   events
        self.out_event_number = QLabel()
        self.out_event_title = QLabel()
        #   actions
        self.sb_action_exploration = QSpinBox()
        self.sb_action_shortcut = QSpinBox()
        self.sb_action_escape = QSpinBox()
        self.sb_action_vision = QSpinBox()
        self.sb_action_change = QSpinBox()
        #   chb_developer
        self.chb_developer = QCheckBox
        #   group_time
        self.sb_time_of_day = QSpinBox()
        self.sb_day = QSpinBox()
        self.sb_month = QSpinBox()
        #   group_left_click
        self.chb_show_tiles = QCheckBox()
        self.chb_move_banner = QCheckBox()
        self.chb_change_time = QCheckBox()
        self.slider_show_tiles = QSlider()
        #   group_middle_click
        self.chb_visit_event = QCheckBox()
        #   group_right_click
        self.chb_hide_tiles = QCheckBox()
        self.slider_hide_tiles = QSlider()

        # SETUP UI ----------------------------------------------
        uic.loadUi("uis\\main_window.ui", self)
        #   Window
        self.setWindowTitle("Mapa - Slowianie RPG")
        self.setGeometry(0, 0, 1900, 1000)
        #   Map()
        self.my_map = Map()
        #   seasons images
        self.init_static_images()
        #   time / weather
        self.set_time()  # functions inside: "set_time_pointer()", "set_weather()"
        #   events
        self.out_event_number.setText('')
        self.out_event_title.setText('')
        #   actions
        self.init_actions()
        self.sb_action_exploration.valueChanged.connect(lambda: self.set_action(self.sb_action_exploration))
        self.sb_action_shortcut.valueChanged.connect(lambda: self.set_action(self.sb_action_shortcut))
        self.sb_action_escape.valueChanged.connect(lambda: self.set_action(self.sb_action_escape))
        self.sb_action_vision.valueChanged.connect(lambda: self.set_action(self.sb_action_vision))
        self.sb_action_change.valueChanged.connect(lambda: self.set_action(self.sb_action_change))
        #   chb_developer
        self.chb_developer.setCheckState(QtCore.Qt.CheckState.Unchecked)
        self.chb_developer.checkStateChanged.connect(self.show_hide_developer_items)
        #   group_time
        self.group_time.setVisible(False)
        self.sb_time_of_day.valueChanged.connect(self.set_time)
        self.sb_day.valueChanged.connect(self.set_time)
        self.sb_month.valueChanged.connect(self.set_time)
        #   group_left_click
        self.group_left_click.setVisible(False)
        self.chb_show_tiles.setCheckState(QtCore.Qt.CheckState.Checked)
        self.chb_move_banner.setCheckState(QtCore.Qt.CheckState.Unchecked)
        self.chb_change_time.setCheckState(QtCore.Qt.CheckState.Unchecked)
        self.slider_show_tiles.setValue(0)  # == single
        #   group_middle_click
        self.group_middle_click.setVisible(False)
        self.chb_visit_event.setCheckState(QtCore.Qt.CheckState.Checked)
        #   group_right_click
        self.group_right_click.setVisible(False)
        self.chb_hide_tiles.setCheckState(QtCore.Qt.CheckState.Checked)
        self.slider_hide_tiles.setValue(0)  # == single

        # scroll label == moving map
        self.scroll_label = ScrollLabel(self)
        self.scroll_label.setGeometry(0, 0, 1700, 990)
        #   add manually
        self.scroll_label.my_map = self.my_map
        self.scroll_label.main_window = self

    def init_static_images(self) -> None:
        # seasons
        self.out_img_winter.setPixmap(self.my_map.map_data['pixmap_season_winter'])
        self.out_img_spring.setPixmap(self.my_map.map_data['pixmap_season_spring'])
        self.out_img_summer.setPixmap(self.my_map.map_data['pixmap_season_summer'])
        self.out_img_autumn.setPixmap(self.my_map.map_data['pixmap_season_autumn'])
        # time_indicator
        self.out_img_time_pointer.setPixmap(self.my_map.map_data['pixmap_time_indicator'])
        #   time_graduation
        self.out_img_time_grad_1.setPixmap(self.my_map.map_data['pixmap_time_graduation'])
        self.out_img_time_grad_2.setPixmap(self.my_map.map_data['pixmap_time_graduation'])
        self.out_img_time_grad_3.setPixmap(self.my_map.map_data['pixmap_time_graduation'])
        self.out_img_time_grad_4.setPixmap(self.my_map.map_data['pixmap_time_graduation'])

    def set_time(self) -> None:
        # set MAP_DATA, trigger: set_time_pointer(), set_weather()
        out_time_of_day, out_day, out_month = None, None, None

        # get values from gui
        gui_time_of_day = self.sb_time_of_day.value()
        gui_day = self.sb_day.value()
        gui_month = self.sb_month.value()

        # get values from map_data
        map_time_of_day = self.my_map.map_data['time_of_day']
        map_day = self.my_map.map_data['day']
        map_month = self.my_map.map_data['month']

        # __init__() -> values == 0 -> get values from self.my_map.map_data
        if gui_time_of_day == gui_day == gui_month == 0:
            out_time_of_day = self.my_map.map_data['time_of_day']  # 1-4
            out_day = self.my_map.map_data["day"]  # 1-31
            out_month = self.my_map.map_data["month"]  # 1-12
            # to skip calculations below
            gui_time_of_day = map_time_of_day
            gui_day = map_day
            gui_month = map_month

        # time_of_day
        delta_days = 0
        if gui_time_of_day > 4:  # day += 1
            out_time_of_day = 1
            delta_days = 1
        elif gui_time_of_day == 0:  # day -= 1
            out_time_of_day = 4
            delta_days = -1
        else:
            out_time_of_day = gui_time_of_day
        if delta_days != 0:
            date_map = datetime.date(2024, map_month, map_day)
            new_date = date_map + relativedelta(days=delta_days)
            out_day = new_date.day
            out_month = new_date.month
            #print(f'date_map={date_map}, new_date={new_date}, delta_days={delta_days}')

        # day
        delta_days = gui_day - map_day
        if delta_days != 0:
            date_map = datetime.date(2024, map_month, map_day)
            new_date = date_map + relativedelta(days=delta_days)
            out_day = new_date.day
            out_month = new_date.month
            #print(f'date_map={date_map}, new_date={new_date}, delta_days={delta_days}')

        # month
        delta_months = gui_month - map_month
        if delta_months != 0:
            date_map = datetime.date(2024, map_month, map_day)
            new_date = date_map + relativedelta(months=delta_months)
            out_day = new_date.day
            out_month = new_date.month
            #print(f'date_map={date_map}, new_date={new_date}, delta_months={delta_months}')

        # set new values to map_data, to gui elements
        for name, value in zip(["time_of_day", "day", "month"], [out_time_of_day, out_day, out_month]):
            if value is not None:  # change in values?
                # set value to map_data
                self.my_map.map_data[name] = value
                # set value to gui
                gui_obj = getattr(self, f'sb_{name}')
                gui_obj.setValue(value)

        if out_time_of_day is not None:
            # set new image for out_img_time_of_day
            time_of_day_str = EnumTimeOfDay(out_time_of_day).name
            self.out_img_time_of_day.setPixmap(self.my_map.weather_data[f'pixmap_day_{time_of_day_str}'])
            # set new text for out_text_time_of_day
            self.out_text_time_of_day.setText(time_of_day_str)

        if [out_time_of_day, out_day, out_month].count(None) < 3:
            # trigger moving time_pointer
            self.set_time_pointer()
            # trigger changing weather
            self.set_weather()

    def set_time_pointer(self) -> None:
        # self.out_img_time_pointer
        x0, y0 = 1855, 0
        day = self.my_map.map_data["day"]     # 1-31
        month = self.my_map.map_data["month"]   # 1-12

        y = get_time_pointer_position(day, month)
        self.out_img_time_pointer.move(x0, y)

    def set_weather(self) -> None:
        #   pixmap
        pixmap, text = functions.generate_combined_weather(self.my_map.map_data, self.my_map.weather_data)
        self.out_img_weather.setPixmap(pixmap)
        #   tooltip
        self.out_img_weather.setToolTip(f'<img src="images\\weather\\weather_combined_text.png">')

        # text = "temp_cold,no_clouds, ..."
        out_text = text.replace("_",' ')
        out_text = out_text.replace("temp",'')
        if len(out_text) > 10:
            out_text = out_text[:10] + '\n' + out_text[10:]
        self.out_text_weather.setText(out_text)

    def init_actions(self) -> None:
        self.sb_action_exploration.setValue(self.my_map.map_data["action_exploration"])
        self.sb_action_shortcut.setValue(self.my_map.map_data["action_shortcut"])
        self.sb_action_escape.setValue(self.my_map.map_data["action_escape"])
        self.sb_action_vision.setValue(self.my_map.map_data["action_vision"])
        self.sb_action_change.setValue(self.my_map.map_data["action_change"])

    def show_hide_developer_items(self, state: QtCore.Qt.CheckState) -> None:
        if state == QtCore.Qt.CheckState.Checked:
            self.group_time.setVisible(True)
            self.group_left_click.setVisible(True)
            self.group_middle_click.setVisible(True)
            self.group_right_click.setVisible(True)
        else:
            self.group_time.setVisible(False)
            self.group_left_click.setVisible(False)
            self.group_middle_click.setVisible(False)
            self.group_right_click.setVisible(False)

    def set_action(self, spinbox: QSpinBox) -> None:
        # set MAP_DATA
        # sb_action_exploration, sb_action_shortcut, sb_action_escape, sb_action_vision, sb_action_change
        if spinbox.objectName() in ["sb_action_exploration", "sb_action_shortcut", "sb_action_escape",
                                   "sb_action_vision", "sb_action_change"]:
            name = spinbox.objectName()[3:]  # remove prefix "sb_"
            value = int(spinbox.value())
            # save new values to map_data
            self.my_map.map_data[name] = value
        else:
            raise ValueError(f'Unknown widget name = {spinbox.objectName()}')

    def display_map(self) -> None:
        file_path = "images\\layers\\layers_combined.png"
        pixmap = QtGui.QPixmap(file_path)
        self.scroll_label.label.setPixmap(pixmap)

    def closeEvent(self, event) -> None:
        self.my_map.save_map_data()
        event.accept()  # Close the app




# OBSOLETE
"""

class StringBox(QSpinBox):
    # https://www.geeksforgeeks.org/pyqt5-creating-string-spin-box/
    def __init__(self, parent=None):
        super(StringBox, self).__init__(parent)

        # string values
        strings = ["morning", "noon", "evening", "night"]
        # calling setStrings method
        self.setStrings(strings)

    # similar to set value method
    def setStrings(self, strings):
        # making strings list
        strings = list(strings)

        # making tuple from the string list
        self._strings = tuple(strings)

        # creating a dictionary
        self._values = dict(zip(strings, range(len(strings))))

        # setting range to it the spin box
        self.setRange(0, len(strings) - 1)

        # overwriting the textFromValue method

    def textFromValue(self, value):
        # returning string from index
        # _string = tuple
        return self._strings[value]
"""
