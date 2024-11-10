# Copyright (c) 2024, Przemyslaw Zawadzki
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import cv2
import os
import json
import ast
from PyQt6.QtGui import QPixmap
from typing import Any
import functions
from enumerations import get_event_title


class Events:
    def __init__(self):
        # VARIABLES  -----------------------------------------
        self.center_points = {}  # {center_point: [event_number, visited(True or False)]}
        self.layer = None  # Image: map_height x map_width of [uint8, uint8, uint8, alpha]

        # LOAD DATA -----------------------------------------
        self.load_center_points()
        self.load_layer()

    def load_center_points(self) -> None:
        try:
            with open('json_files\\event_center_points.json') as f:
                center_points = json.load(f)
                self.center_points = {ast.literal_eval(key): value for key, value in center_points.items()}
        except FileNotFoundError:
            self.center_points = functions.generate_json_event_center_points()  # save to file in function

    def load_layer(self) -> None:
        if os.path.isfile("images\\layers\\layer_events.png"):
            self.layer = cv2.imread("images\\layers\\layer_events.png", cv2.IMREAD_UNCHANGED)
        else:
            self.layer = functions.generate_event_layer(self.center_points)  # save to file in function

    def update(self, in_center_point: tuple[int, int]) -> None:
        # center_points
        event_number, visited = self.center_points[in_center_point]
        visited = not visited
        self.center_points[in_center_point] = [event_number, visited]
        # layer
        self.layer = functions.generate_event_layer(self.center_points)  # save to file in function

        # save to file
        self.save_center_points()
        self.save_layer()

    def save_center_points(self) -> None:
        with open('json_files\\event_center_points.json', 'w') as f:
            points = {str(key): value for key, value in self.center_points.items()}
            json.dump(points, f)

    def save_layer(self) -> None:
        cv2.imwrite("images\\layers\\layer_events.png", self.layer)


class Fog:
    def __init__(self):
        # VARIABLES  -----------------------------------------
        self.center_points = {}  # {center_point: is_visible(True or False)}
        self.layer = None  # Image: map_height x map_width of [uint8, uint8, uint8, alpha]

        # LOAD DATA -----------------------------------------
        self.load_center_points()
        self.load_layer()

    def load_center_points(self) -> None:
        try:
            with open('json_files\\fog_center_points.json') as f:
                center_points = json.load(f)
                self.center_points = {ast.literal_eval(key): value for key, value in center_points.items()}
        except FileNotFoundError:
            self.center_points = functions.generate_json_fog_center_points()  # save to file in function

    def load_layer(self) -> None:
        if os.path.isfile("images\\layers\\layer_fog.png"):
            self.layer = cv2.imread("images\\layers\\layer_fog.png", cv2.IMREAD_UNCHANGED)
        else:
            self.layer = functions.generate_fog_layer(self.center_points)  # save to file in function

    def update(self, in_center_points: list[tuple[int, int]], mode: str) -> None:
        f_updated = False

        # update layer, center_points
        for (row, col) in in_center_points:
            if mode == "show":
                # no transparency -> needs to be updated
                if self.layer[row][col][3] == 255:
                    f_updated = True
                    inner_points = functions.get_inner_points((row, col))
                    for _row, _col in inner_points:
                        self.layer[_row, _col, 3] = 0  # make point transparent
                    self.center_points[(row, col)] = True   # update center_points

            else:  # mode=="hide"
                # transparent -> needs to be updated
                if self.layer[row][col][3] == 0:
                    f_updated = True
                    inner_points = functions.get_inner_points((row, col))
                    for _row, _col in inner_points:
                        self.layer[_row, _col, 3] = 255  # make point not transparent (fill with black background)
                    self.center_points[(row, col)] = False   # update center_points

        # save to files
        if f_updated:
            self.save_center_points()
            self.save_layer()

    def save_center_points(self) -> None:
        with open('json_files\\fog_center_points.json', 'w') as f:
            points = {str(key): value for key, value in self.center_points.items()}
            json.dump(points, f)

    def save_layer(self) -> None:
        cv2.imwrite("images\\layers\\layer_fog.png", self.layer)


class Items:
    def __init__(self):
        # VARIABLES  -----------------------------------------
        self.center_points = {}  # {center_point: item_name}
        self.layer = None  # Image: map_height x map_width of [uint8, uint8, uint8, alpha]

        # LOAD DATA -----------------------------------------
        self.load_center_points()
        self.load_layer()

    def load_center_points(self) -> None:
        try:
            with open('json_files\\item_center_points.json') as f:
                center_points = json.load(f)
                self.center_points = {ast.literal_eval(key): value for key, value in center_points.items()}
        except FileNotFoundError:
            self.center_points = functions.generate_json_item_center_points()  # save to file already inside function

    def load_layer(self) -> None:
        if os.path.isfile("images\\layers\\layer_items.png"):
            self.layer = cv2.imread("images\\layers\\layer_items.png", cv2.IMREAD_UNCHANGED)
        else:
            self.layer = functions.generate_item_layer(self.center_points)  # save to file already inside function

    def update(self, in_center_point: tuple[int, int]) -> None:
        # find banner
        for point, item_name in self.center_points.items():
            if item_name == "banner":
                break
        # new position?
        if point == in_center_point:
            return

        # update center_points
        self.center_points[point] = None  # reset old position
        self.center_points[in_center_point] = "banner"  # set new position

        # update layer
        self.layer = functions.generate_item_layer(self.center_points)  # save to file already inside function

        # save to file
        self.save_center_points()
        self.save_layer()

    def save_center_points(self) -> None:
        with open('json_files\\item_center_points.json', 'w') as f:
            points = {str(key): value for key, value in self.center_points.items()}
            json.dump(points, f)

    def save_layer(self) -> None:
        cv2.imwrite("images\\layers\\layer_items.png", self.layer)


class Map:
    def __init__(self):
        # VARIABLES  -----------------------------------------
        self.layer = None  # Image: map_height x map_width of [uint8, uint8, uint8, alpha]
        self.map_height, self.map_width = 0, 0
        self.map_data = {}  # {name_of_data: value} e.g.: "day": 1
        self.weather_data = {}  # dict[str:dict[str, Any]], see more in functions.generate_json_weather_data()
        self.center_points = {}  # {center_point: [inner_point, inner_point, ... ]}
        self.inner_points = {}  # {inner_point: center_point}
        self.events = None  # Events()
        self.items = None  # Items()
        self.fog = None  # Fog()
        self.layers_combined = None  # Image: map_height x map_width of [uint8, uint8, uint8, alpha]

        # LOAD DATA -----------------------------------------
        # layer, map_height, map_width
        self.load_layer()
        # map_data
        self.load_map_data()
        # weather_data
        self.load_weather_data()
        # center_points, inner_points
        self.load_points()
        # events
        self.events = Events()
        # items
        self.items = Items()
        # fog
        self.fog = Fog()
        # layers_combined
        self.load_combined_layers()

    def load_layer(self) -> None:
        if os.path.isfile("images\\layers\\layer_map.png"):
            self.layer = cv2.imread("images\\layers\\layer_map.png", cv2.IMREAD_UNCHANGED)
            self.map_height, self.map_width, _ = self.layer.shape
            # save to functions
            functions.map_height, functions.map_width = self.map_height, self.map_width
        else:
            raise FileNotFoundError

    def load_map_data(self) -> None:
        try:
            with open('json_files\\map_data.json') as f:
                self.map_data = json.load(f)
        except FileNotFoundError:
            self.map_data = functions.generate_json_map_data()  # save to file in function
        # load pixmaps
        #   seasons
        self.map_data['pixmap_season_winter'] = QPixmap("images\\static_images\\season_winter.jpg")
        self.map_data['pixmap_season_spring'] = QPixmap("images\\static_images\\season_spring.jpg")
        self.map_data['pixmap_season_summer'] = QPixmap("images\\static_images\\season_summer.jpg")
        self.map_data['pixmap_season_autumn'] = QPixmap("images\\static_images\\season_autumn.jpg")
        #   time_indicator
        self.map_data['pixmap_time_indicator'] = QPixmap("images\\static_images\\time_indicator.png")
        #   time_graduation
        self.map_data['pixmap_time_graduation'] = QPixmap("images\\static_images\\time_graduation.png")

    def load_weather_data(self) -> None:
        try:
            with open('json_files\\weather_data.json') as f:
                self.weather_data = json.load(f)
        except FileNotFoundError:
            self.weather_data = functions.generate_json_weather_data()  # save to file in function
        # load pixmaps
        #   time of day
        self.weather_data['pixmap_day_morning'] = QPixmap("images\\weather\\day_morning.png")
        self.weather_data['pixmap_day_noon'] = QPixmap("images\\weather\\day_noon.png")
        self.weather_data['pixmap_day_evening'] = QPixmap("images\\weather\\day_evening.png")
        self.weather_data['pixmap_day_night'] = QPixmap("images\\weather\\day_night.png")
        # load images
        #   weather
        #       temperature
        self.weather_data['img_weather_temp_very_cold'] = cv2.imread("images\\weather\\temp_very_cold.png", cv2.IMREAD_UNCHANGED)
        self.weather_data['img_weather_temp_cold'] = cv2.imread("images\\weather\\temp_cold.png", cv2.IMREAD_UNCHANGED)
        self.weather_data['img_weather_temp_warm'] = cv2.imread("images\\weather\\temp_warm.png", cv2.IMREAD_UNCHANGED)
        self.weather_data['img_weather_temp_hot'] = cv2.imread("images\\weather\\temp_hot.png", cv2.IMREAD_UNCHANGED)
        #       wind
        self.weather_data['img_weather_wind'] = cv2.imread("images\\weather\\wind.png", cv2.IMREAD_UNCHANGED)
        #       cloud
        self.weather_data['img_weather_no_clouds'] = cv2.imread("images\\weather\\no_clouds.png", cv2.IMREAD_UNCHANGED)
        self.weather_data['img_weather_little_clouds'] = cv2.imread("images\\weather\\little_clouds.png", cv2.IMREAD_UNCHANGED)
        self.weather_data['img_weather_cloudy'] = cv2.imread("images\\weather\\cloudy.png", cv2.IMREAD_UNCHANGED)
        self.weather_data['img_weather_overcast'] = cv2.imread("images\\weather\\overcast.png", cv2.IMREAD_UNCHANGED)
        #       snow
        self.weather_data['img_weather_snow'] = cv2.imread("images\\weather\\snow.png", cv2.IMREAD_UNCHANGED)
        self.weather_data['img_weather_snow_rain'] = cv2.imread("images\\weather\\snow_rain.png", cv2.IMREAD_UNCHANGED)
        #       fog
        self.weather_data['img_weather_fog'] = cv2.imread("images\\weather\\fog.png", cv2.IMREAD_UNCHANGED)
        #       rain
        self.weather_data['img_weather_rain'] = cv2.imread("images\\weather\\rain.png", cv2.IMREAD_UNCHANGED)
        self.weather_data['img_weather_heavy_rain'] = cv2.imread("images\\weather\\heavy_rain.png", cv2.IMREAD_UNCHANGED)
        self.weather_data['img_weather_downpour'] = cv2.imread("images\\weather\\downpour.png", cv2.IMREAD_UNCHANGED)
        #       lightning
        self.weather_data['img_weather_lightning'] = cv2.imread("images\\weather\\lightning.png", cv2.IMREAD_UNCHANGED)

    def load_points(self) -> None:
        # center_points
        try:
            with open('json_files\\center_points.json') as f:
                center_points = json.load(f)
                self.center_points = {ast.literal_eval(key): value for key, value in center_points.items()}
        except FileNotFoundError:
            self.center_points = functions.generate_json_center_points()  # (saving to file) already inside function
        # save to functions
        functions.map_center_points = self.center_points

        # inner_points
        for center_point, points in self.center_points.items():
            for point in points:
                self.inner_points[tuple(point)] = center_point
        # save to functions
        functions.map_inner_points = self.inner_points

    def load_combined_layers(self) -> None:
        if os.path.isfile("images\\layers\\layers_combined.png"):
            self.layers_combined = cv2.imread("images\\layers\\layers_combined.png", cv2.IMREAD_UNCHANGED)
        else:
            layers = [
                self.layer,  # map
                self.events.layer,
                self.fog.layer,
                self.items.layer
            ]
            self.layers_combined = functions.generate_combined_layers(layers)  # save already in function

    def update_combined_layers(self, center_point: tuple[int, int]) -> None:
        layers = [
            self.layer,  # map
            self.events.layer,
            self.fog.layer,
            self.items.layer
        ]
        self.layers_combined = functions.update_combined_layers(center_point, layers, self.layers_combined)  # save already in function

    def save_map_data(self) -> None:
        # no pixmap data!
        map_data = {key: value for key, value in self.map_data.items() if key.find('pixmap_') < 0}
        with open('json_files\\map_data.json', 'w') as f:
            json.dump(map_data, f)

    # USER TRIGGERED FUNCTIONS ---------------------------------------------------------------------
    def one_tile(self, point: tuple[int, int], mode: str) -> tuple[int, int]:
        # get center_point, that is the closest to the mouse click
        center_point = functions.get_center_point(point)
        # events layer
        pass
        # items layer
        pass
        # fog layer
        self.fog.update([center_point], mode)  # show/hide center_point tile
        # combined layers
        self.update_combined_layers(center_point)

        return center_point

    def multiple_tiles(self, point: tuple[int, int], mode: str) -> tuple[int, int]:
        # get center_point, that is the closest to the mouse click
        center_point = functions.get_center_point(point)
        # events layer
        pass
        # items layer
        pass
        # fog layer
        center_points = list(set(functions.get_neighbour_center_points(center_point) + [center_point]))
        self.fog.update(center_points, mode)  # show/hide center_point tile and its neighbour tiles
        # combined layers
        self.update_combined_layers(center_point)

        return center_point

    def update_banner_position(self, point: tuple[int, int]) -> tuple[int, int]:
        # get center_point, that is the closest to the mouse click
        center_point = functions.get_center_point(point)
        # events layer
        pass
        # items layer
        self.items.update(center_point)  # display banner at center_point
        # fog layer
        pass
        # combined layers
        self.update_combined_layers(center_point)

        return center_point

    def update_events(self, point: tuple[int, int]) -> tuple[int, int]:
        # get center_point, that is the closest to the mouse click
        center_point = functions.get_center_point(point)
        # center_point has assigned event?
        if center_point not in self.events.center_points.keys():
            print(f'ERR: center_point {center_point} has no event assigned to it.')
            return center_point
        # events layer
        self.events.update(center_point)  # mark event as visited / not visited
        # items layer
        pass
        # fog layer
        pass
        # combined layers
        self.update_combined_layers(center_point)

        return center_point

    def get_event_data(self, point: tuple[int, int]) -> tuple[int, str] | tuple[Any, Any]:
        # get center_point, that is the closest to the mouse click
        center_point = functions.get_center_point(point)
        # center_point exists in events.center_points?
        if center_point not in self.events.center_points.keys():
            return None, None
        # center_point "is_visible" in fog.center_points?
        if not self.fog.center_points[center_point]:
            return None, None

        event_number, visited = self.events.center_points[center_point]
        # event already visited?
        if visited:
            event_title = get_event_title(event_number)
        else:
            event_title = ''

        return event_number, event_title


# OBSOLETE
"""
    def user_function(func):
        @wraps(func)
        def wrapper_func(self, *args, **kwargs):
            point = kwargs.get('point')
            # get center_point, that is the closest to the mouse click
            center_point = functions.get_center_point(point)
            kwargs['center_point'] = center_point

            func(self, *args, **kwargs)

            # save combined layers
            self.save_combined_layers()
            return center_point

        return wrapper_func

    @user_function
    def one_tile(self, **kwargs) -> tuple[int, int]:
        #param kwargs: "point", "mode"
        #return: center_point -> tuple[int, int]
        
        # single point
        kwargs['center_points'] = [kwargs['center_point']]
        # update layers
        self.items.update(**kwargs)
        self.fog.update(**kwargs)
"""