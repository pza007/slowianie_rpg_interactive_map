# Copyright (c) 2024, Przemyslaw Zawadzki
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import time
from PyQt6.QtGui import QPixmap
import numpy as np
import numpy.typing as npt
from typing import Annotated, Literal, Any
import math
import cv2
import json
import random
import copy
import csv
import calendar
from enumerations import EnumTimeOfDay

# global variables
R = 10  # outer radius for hexagons
r = R * math.sqrt(3) / 2  # inner radius for hexagons
map_height, map_width = 1561, 2270
map_center_points = {}  # {center_point: [inner_point, inner_point, ... ]}
map_inner_points = {}  # {inner_point: center_point}
ArrayHxWx4 = Annotated[npt.NDArray[np.uint8], Literal[map_height, map_width, 4]]
Array500x500x4 = Annotated[npt.NDArray[np.uint8], Literal[500, 500, 4]]


def circle_intersect(x0: int, y0: int, r0: int, x1: int, y1: int, r1: int) -> list[tuple[int, int]] | Any:
    """
    Get list of points where two circles intersect
    :param x0: first circle x-position
    :param y0: first circle y-position
    :param r0: first circle radius
    :param x1: second circle x-position
    :param y1: second circle y-position
    :param r1: second circle radius
    :return: list of points
    """
    c0 = np.array([x0, y0])
    c1 = np.array([x1, y1])
    v = c1 - c0
    d = np.linalg.norm(v)

    if d > r0 + r1 or d == 0:
        return None

    u = v / np.linalg.norm(v)
    xvec = c0 + (d ** 2 - r1 ** 2 + r0 ** 2) * u / (2 * d)

    uperp = np.array([u[1], -u[0]])
    a = ((-d + r1 - r0) * (-d - r1 + r0) * (-d + r1 + r0) * (d + r1 + r0)) ** 0.5 / d

    out = (xvec + a * uperp / 2, xvec - a * uperp / 2)
    out = [(round(x), round(y)) for x, y in out]
    return out


def generate_hexagon_center_points() -> list[tuple[int, int]]:
    """
    Get center point for each hexagon that be located on map image (map_height x map_width)
    :return: list of points for hexagon centers
    """
    out_points = []

    row = R
    idx = 0
    while row + R < map_height:
        if idx % 2 == 0:
            col = r
        else:
            col = 2 * r

        while col + r < map_width:
            out_points.append((row, col))
            col += (2 * r)

        row += (R + r / 2)
        idx += 1

    # round values
    out_points = [(round(row), round(col)) for row, col in out_points]

    return out_points


def generate_hexagon_inner_points(center_point: tuple[int, int]) -> list[tuple[int, int]]:
    """
    For hexagon center point, get all points that compose hexagon, aka "inner points"
    :param center_point: center point of hexagon
    :return: list of points that compose hexagon
    """
    xc, yc = center_point[1], center_point[0]

    # (x, y-R)
    x0, y0 = [xc, yc - R]
    points = circle_intersect(xc, yc, R, x0, y0, R)
    points.sort()
    x1, y1 = points[0]  # min x
    x5, y5 = points[1]  # max x

    # (x, y+R)
    x3, y3 = [xc, yc + R]
    points = circle_intersect(xc, yc, R, x3, y3, R)
    points.sort()
    x2, y2 = points[0]  # min x
    x4, y4 = points[1]  # max x

    # make contour
    out_points = [(x0, y0), (x1, y1), (x2, y2), (x3, y3), (x4, y4), (x5, y5), (x0, y0)]
    contour = np.array(out_points).reshape((-1, 1, 2)).astype(np.int32)
    img = np.zeros((yc + R + 1, xc + R + 1, 4), dtype=np.uint8)
    cv2.drawContours(img, [contour], 0, (255, 255, 255), -1)

    # get all hexagon points
    out_points = []
    for y in range(yc - R, yc + R + 1):
        for x in range(xc - R, xc + R + 1):
            if img[y, x][0] > 0:
                out_points.append((y, x))

    return out_points


def generate_json_center_points() -> dict[tuple[int, int]: list[tuple[int, int]]]:
    """
    Save data into json file: 'json_files\\center_points.json'
    :return: {center_point: [inner_point, inner_point, ... ]}
    """
    # get hexagon centers
    points = generate_hexagon_center_points()

    # get inner points of hexagon centers
    center_points = {}
    for center_point in points:
        inner_points = generate_hexagon_inner_points(center_point)
        center_points[center_point] = inner_points

    # mark all points "white"
    image = np.zeros((map_height, map_width), dtype=np.uint8)
    for center_point, points in center_points.items():
        for (row, col) in points:
            image[row, col] = 255
        image[center_point[0], center_point[1]] = 255
    cv2.imwrite("json_files\\_center_points_1.png", image)

    # find all not assigned points ("black" ones)
    missing = list(zip(*np.where(image == 0)))
    missing = [(int(row), int(col)) for row, col in missing]
    # assign point to the closest center_point
    for (row, col) in missing:
        tmp = []
        for center_point in center_points.keys():
            dist = math.dist((row, col), center_point)
            if dist < 50:
                tmp.append([dist, center_point])
        tmp.sort()
        center_points[tmp[0][1]].append((row, col))

    # for visualisation only
    """
    image = np.zeros((map_height, map_width), dtype=np.uint8)
    val = 10
    for center_point, points in center_points.items():
        for (row, col) in points:
            image[row, col] = val
        val += 10
        val = val % 256
    for center_point, points in center_points.items():
        image[center_point[0], center_point[1]] = 0
    cv2.imwrite("json_files\\_center_points_2.png", image)
    """

    # save to file
    with open('json_files\\center_points.json', 'w') as f:
        center_points = {str(key): value for key, value in center_points.items()}
        json.dump(center_points, f)

    return center_points


def generate_json_map_data() -> dict:
    """
    Save data into json file: 'json_files\\map_data.json'
    :return: {name_of_data: value} e.g.: "day": 1
    """
    map_data = {
        "banner_position": (100, 100),
        "time_of_day": "morning",
        "day": 1,
        "month": 6}
    # save to file
    with open('json_files\\map_data.json', 'w') as f:
        json.dump(map_data, f)
    return map_data


def get_inner_points(center_point: tuple[int, int]) -> list[tuple[int, int]]:
    """
    :param center_point: hexagon center point
    :return: list of inner points
    """
    return map_center_points[center_point]  # list of inner points


def get_center_point(inner_point: tuple[int, int]) -> tuple[int, int]:
    """
    :param inner_point: one of inner point related to specific hexagon center point
    :return: hexagon center point
    """
    return map_inner_points[inner_point]    # center_point


def get_neighbour_center_points(point: tuple[int, int]) -> list[tuple[int, int]]:
    """
    Get 6 neighbour center points of given hexagon center point
    :param point: hexagon center point
    :return: list of 6 neighbour center points
    """
    neighbour_points = []   # [ center_point, ... ]
    for center_point in map_center_points.keys():
        dist = math.dist(center_point, point)
        if dist <= 2*r*1.1:
            neighbour_points.append(center_point)
    return neighbour_points


def generate_json_event_center_points() -> dict[tuple[int, int]: list[int, bool]]:
    """
    Save data into json file: 'json_files\\event_center_points.json'
    :return: {center_point: [event_number, visited(True or False)]}
    """
    center_points = {}
    # choose points from center_points, that will have events
    chosen_points = random.sample(list(map_center_points.keys()), int(0.06 * len(map_center_points.keys())))
    rand_values = [random.randint(1, 100) for point in chosen_points]
    # save points and values in dictionary
    for point, value in zip(chosen_points, rand_values):
        center_points[point] = [value, False]
    # save to file
    with open('json_files\\event_center_points.json', 'w') as f:
        points = {str(key): value for key, value in center_points.items()}
        json.dump(points, f)
    return center_points


def generate_event_layer(event_center_points: dict[tuple[int, int]: list[int, bool]]) -> ArrayHxWx4:
    """
    :param event_center_points: center points from event class
    :return: image, also stored in file: "images\\layers\\layer_events.png"
    """
    image = np.ones((map_height, map_width, 3), dtype=np.uint8)
    image[:, :, :] *= 255   # white background

    # put text for each event ()
    font = cv2.FONT_HERSHEY_TRIPLEX
    font_scale = 0.5

    for (row, col), [event_number, visited] in event_center_points.items():  # {(row, col): [event_number, visited(True or False)]}
        if not visited:
            x, y = col - 6, row + 5
            color = (0, 0, 255)
            text = '!!'
        else:
            x, y = col - 6, row + 5
            color = (0, 255, 0)
            text = 'X'
        image = cv2.putText(image, text, (x, y), font, font_scale, color, 1, cv2.LINE_AA)

    # Convert black to transparent
    # https://stackoverflow.com/questions/70223829/opencv-how-to-convert-all-black-pixels-to-transparent-and-save-it-to-png-file
    # Make a True/False mask of pixels whose BGR values sum to more than zero
    alpha = np.sum(image, axis=-1) < 3*250#255
    # Convert True/False to 0/255 and change type to "uint8" to match "na"
    alpha = np.uint8(alpha * 255)
    # Stack new alpha layer with existing image to go from BGR to BGRA, i.e. 3 channels to 4 channels
    image = np.dstack((image, alpha))

    # save to file
    cv2.imwrite("images\\layers\\layer_events.png", image)

    return image


def generate_json_item_center_points() -> dict[tuple[int, int]: list[int, bool]]:
    """
    Save data into json file: 'json_files\\item_center_points.json'
    :return: {center_point: item_name}
    """
    center_points = {center_point: None for center_point in map_center_points.keys()}

    # banner item
    #center_point = list(center_points.keys())[0]
    center_point = (855, 849)
    center_points[center_point] = "banner"

    # save to file
    with open('json_files\\item_center_points.json', 'w') as f:
        points = {str(key): value for key, value in center_points.items()}
        json.dump(points, f)
    return center_points


def generate_item_layer(item_center_points: dict[tuple[int, int]: list[int, bool]]) -> ArrayHxWx4:
    """
    :param item_center_points: center points from item class
    :return: image, also stored in file: "images\\layers\\layer_items.png"
    """
    image = np.zeros((map_height, map_width, 4), dtype=np.uint8)    # 100% transparent

    # banner item
    for center_point, item_name in item_center_points.items():
        if item_name == "banner":
            break
    banner_image = cv2.imread("images\\static_images\\banner.png", cv2.IMREAD_UNCHANGED)
    banner_height, banner_width, _ = banner_image.shape

    # put banner on image
    row_min = center_point[0] - 2  #row_min = center_point[0] - 10
    row_max = row_min + banner_height
    col_min = center_point[1] - 11  #col_min = center_point[1] - 9
    col_max = col_min + banner_width

    banner_row = 0
    for row in range(row_min, row_max):
        banner_col = 0
        for col in range(col_min, col_max):
            banner_value = banner_image[banner_row][banner_col]
            if banner_value[3] == 0:
                pass  # transparent pixel -> nothing to copy
            else:
                image[row][col] = banner_value
                image[row][col][3] = 255  # not transparent
            banner_col += 1
        banner_row += 1

    # save to file
    cv2.imwrite("images\\layers\\layer_items.png", image)

    return image


def generate_json_fog_center_points() -> dict[tuple[int, int]: bool]:
    """
    Save data into json file: 'json_files\\fog_center_points.json'
    :return: {center_point: is_visible(True or False)}
    """
    center_points = {key: False for key in map_center_points.keys()}

    # save to file
    with open('json_files\\fog_center_points.json', 'w') as f:
        points = {str(key): value for key, value in center_points.items()}
        json.dump(points, f)
    return center_points


def generate_fog_layer(fog_center_points: dict[tuple[int, int]: bool]) -> ArrayHxWx4:
    """
    :param fog_center_points: center points from fog class
    :return: image, also stored in file: "images\\layers\\layer_fog.png"
    """
    image = np.zeros((map_height, map_width, 4), dtype=np.uint8)    # all black
    image[:, :, 3] = 255  # no transparency

    # make center_points (and their inner_points) transparent if their flag: "is_visible" is set to True
    for center_point, is_visible in fog_center_points.items():
        if is_visible:
            inner_points = get_inner_points(center_point)
            for (row, col) in inner_points:
                image[row, col, 3] = 0  # make transparent

    # save to file
    cv2.imwrite("images\\layers\\layer_fog.png", image)
    return image


def merge_images(in_background: ArrayHxWx4, in_foreground: ArrayHxWx4) -> ArrayHxWx4:
    """
    Merge image of the same size == (map_height x map_width)
    :param in_background:
    :param in_foreground:
    :return: merged foreground image into background image
    """
    background = copy.copy(in_background)
    foreground = copy.copy(in_foreground)

    # normalize alpha channels from 0-255 to 0-1
    alpha_background = background[:, :, 3] / 255.0
    alpha_foreground = foreground[:, :, 3] / 255.0

    # set adjusted colors
    for color in range(0, 3):
        background[:, :, color] = alpha_foreground * foreground[:, :, color] + \
                                  alpha_background * background[:, :, color] * (1 - alpha_foreground)

    # set adjusted alpha and denormalize back to 0-255
    background[:, :, 3] = (1 - (1 - alpha_foreground) * (1 - alpha_background)) * 255

    return background


def merge_images_different_size(large_image: np.ndarray, small_image: np.ndarray, position: tuple[int, int]) -> np.ndarray:
    """
    Merges a smaller image onto a larger image at a specified position.
    :param large_image:
    :param small_image:
    :param position: position where to merge smaller image
    :return: merged smaller image into larger image
    """
    large_img = copy.copy(large_image)
    small_img = copy.copy(small_image)
    # Get the position to place the small image
    x, y = position

    # Check if the small image can fit in the specified position on the large image
    if (y + small_img.shape[0] > large_img.shape[0]) or (x + small_img.shape[1] > large_img.shape[1]):
        raise ValueError("The small image exceeds the boundaries of the large image at the given position.")

    # Split the small image into RGB and alpha channels
    small_rgb = small_img[:, :, :3]
    small_alpha = small_img[:, :, 3] / 255.0  # Normalize alpha channel to range [0, 1]

    # Prepare the region in the large image where the small image will be overlaid
    large_rgb = large_img[y:y + small_img.shape[0], x:x + small_img.shape[1], :3]
    large_alpha = large_img[y:y + small_img.shape[0], x:x + small_img.shape[1], 3] / 255.0

    # Calculate the new alpha for the output image in the region where the small image is applied
    combined_alpha = small_alpha + large_alpha * (1 - small_alpha)
    combined_alpha[combined_alpha == 0] = 1  # Avoid division by zero

    # Blend the RGB channels based on the alpha values
    merged_rgb = (small_rgb * small_alpha[:, :, None] + large_rgb * large_alpha[:, :, None] * (1 - small_alpha[:, :, None])) / combined_alpha[:, :, None]

    # Place the blended result into the large image
    large_img[y:y + small_img.shape[0], x:x + small_img.shape[1], :3] = merged_rgb
    large_img[y:y + small_img.shape[0], x:x + small_img.shape[1], 3] = (combined_alpha * 255).astype(np.uint8)

    return large_img


def generate_combined_layers(layers: list[ArrayHxWx4]) -> ArrayHxWx4:
    """
    :param layers: list of images
    :return: merged image, stored in file: "images\\layers\\layers_combined.png"
    """
    for idx in range(len(layers)-1):
        background = layers[idx]
        foreground = layers[idx + 1]
        merged_image = merge_images(background, foreground)
        layers[idx + 1] = merged_image  # merged image becomes the background for next iteration
    # save to file
    cv2.imwrite("images\\layers\\layers_combined.png", layers[-1])

    return layers[-1]


def update_combined_layers(center_point: tuple[int, int], layers: list[ArrayHxWx4], layers_combined: ArrayHxWx4) -> ArrayHxWx4:
    """
    Update the section of the layers combined image, close to user-clicked position
    :param center_point: center point where user clicked the mouse
    :param layers: list of images
    :param layers_combined: previous layers combined image
    :return: new, updated, merged image, stored in file: "images\\layers\\layers_combined.png"
    """

    t = [time.time_ns()]
    # get boundaries of snipped of image to be updated
    row_min = center_point[0]-100 if center_point[0]-100 >= 0 else 0
    row_max = center_point[0]+100 if center_point[0]+100 < map_height else map_height-1
    col_min = center_point[1]-100 if center_point[1]-100 >= 0 else 0
    col_max = center_point[1]+100 if center_point[1]+100 < map_width else map_width-1

    # crop layers to boundaries
    for idx in range(len(layers)):
        layers[idx] = layers[idx][row_min:row_max, col_min:col_max, :]

    # merge layers
    for idx in range(len(layers)-1):
        background = layers[idx]
        foreground = layers[idx + 1]
        merged_image = merge_images(background, foreground)
        layers[idx + 1] = merged_image  # merged image becomes the background for next iteration
        t.append(time.time_ns())
    snipped_img = layers[-1]

    # merge snipped image with already existing image (layers_combined)
    out_img = merge_images_different_size(layers_combined, snipped_img, (col_min, row_min))
    t.append(time.time_ns())

    # save to file
    cv2.imwrite("images\\layers\\layers_combined.png", out_img)
    t.append(time.time_ns())

    for idx in range(1, len(t)):
        print(f'{(t[idx]-t[0])/1000000:.2f}')

    return out_img


def generate_json_weather_data() -> dict[str:dict[str, Any]]:
    """
    Read data from .csv file, process data and save them into json file: 'json_files\\weather_data.json'
    # source for weather data: https://open-meteo.com/
    # source for weather icons: https://www.svgrepo.com/collection/weather-line-icons/
    :return: dictionary data for each day of year, example:
        "morning/01/01": {"avg_temperature": 7.083333333333333, "avg_rain": 0.3499999999999999, "avg_snow": 0.0, "avg_cloud": 100, "avg_wind": 38.21666666666667, "images": ["temp_cold", "overcast"]}
    """
    out_data = {}
    for month in range(1,13):
        max_day = int(calendar.monthrange(2024, month)[1])
        for day in range(1, max_day+1):
            for time_of_day in ["morning", "noon", "evening", "night"]:
                key = f"{time_of_day}/{day:02d}/{month:02d}"
                out_data[key] = {
                    "avg_temperature":0.0,
                    "avg_rain":0.0,
                    "avg_snow":0.0,
                    "avg_cloud":0,
                    "avg_wind":0.0,
                    "images":[]  #e.g.: ["cloudy", "rain", "temp_cold"]
                }

    with open("json_files\\open-meteo.csv", newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader, None)  # skip header

        # SUM values
        for row in reader:
            # header = time, temperature_2m (°C), rain (mm), snowfall (cm), cloud_cover (%), wind_speed_100m (km/h)
            # [0] time = "2022-01-01T00:00" -> f"{time_of_day}/{day}/{month}"
            month = int(row[0][5:7])
            day = int(row[0][8:10])
            hour = int(row[0][11:13])
            if 5 <= hour <= 10:
                time_of_day = "morning"
            elif 11 <= hour <= 14:
                time_of_day = "noon"
            elif 15 <= hour <= 19:
                time_of_day = "evening"
            else:  # T20,T21,T22,T23,T00,T01,T02,T03,T04
                time_of_day = "night"
            key = f"{time_of_day}/{day:02d}/{month:02d}"

            # [1] temperature = "5.6"
            out_data[key]["avg_temperature"] += float(row[1])   # SUM
            # [2] rain = "0.00"
            out_data[key]["avg_rain"] += float(row[2])   # SUM
            # [3] snow = "0.00"
            out_data[key]["avg_snow"] += float(row[3])   # SUM
            # [4] cloud = "100"
            out_data[key]["avg_cloud"] += int(row[4])   # SUM
            # [5] wind = "28.3"
            out_data[key]["avg_wind"] += float(row[5])   # SUM

    # AVG values
    for key, values in out_data.items():
        time_of_day = key.split("/")[0]  #  f"{time_of_day}/{day:02d}/{month:02d}"
        if time_of_day == "morning":
            num_data = 6  # 5 <= hour <= 10
        elif time_of_day == "noon":
            num_data = 4  # 11 <= hour <= 14
        elif time_of_day == "evening":
            num_data = 5  # 15 <= hour <= 19
        else:   # night
            num_data = 9  # T20,T21,T22,T23,T00,T01,T02,T03,T04

        values["avg_temperature"] = values["avg_temperature"]/num_data
        values["avg_rain"] = values["avg_rain"]/num_data
        values["avg_snow"] = values["avg_snow"]/num_data
        values["avg_cloud"] = int(values["avg_cloud"]/num_data)
        values["avg_wind"] = values["avg_wind"]/num_data

    # YEAR AVG MIN MAX values
    year_min = {}
    year_max = {}
    for name in ["temperature", "rain", "snow", "cloud", "wind"]:
        min_value = min([values[f'avg_{name}'] for values in out_data.values()])
        max_value = max([values[f'avg_{name}'] for values in out_data.values()])
        year_min[name] = min_value
        year_max[name] = max_value

    # images
    for key, values in out_data.items():
        # f"{time_of_day}/{day:02d}/{month:02d}"
        time_of_day = key.split("/")[0]
        day = int(key.split("/")[1])
        month = int(key.split("/")[2])

        # temperature: temp_very_cold, temp_cold, temp_warm, temp_hot
        name = "temperature"
        val = values[f"avg_{name}"]
        if val <= 0.1*year_min[name]:
            img_name = "temp_very_cold"
        elif val <= 0.4*year_max[name]:
            img_name = "temp_cold"
        elif val <= 0.8*year_max[name]:
            img_name = "temp_warm"
        else:
            img_name = "temp_hot"
        values["images"].append(img_name)

        # wind: wind
        name = "wind"
        val = values[f"avg_{name}"]
        total_range = abs(year_min[name]) + abs(year_max[name])
        if 0.70*total_range < abs(val):
            img_name = "wind"
            values["images"].append(img_name)

        # rain: rain, heavy_rain, downpour
        rain_name = None  # used for evaluation below
        name = "rain"
        val = values[f"avg_{name}"]
        # val < 0.15 * year_max[name] ->  no rain
        if val > 0.15 * year_max[name]:
            if 0.15*year_max[name] < val <= 0.45*year_max[name]:
                img_name = "rain"
            elif 0.45*year_max[name] < val <= 0.9*year_max[name]:
                img_name = "heavy_rain"
            else:
                img_name = "downpour"
            rain_name = img_name
            values["images"].append(img_name)
            #   lightning(*extra on rain)
            if random.random() > 0.8:
                values["images"].append("lightning")

        # cloud: no_clouds, little_clouds, cloudy, overcast
        name = "cloud"
        val = values[f"avg_{name}"]
        total_range = abs(year_min[name]) + abs(year_max[name])
        if abs(val) <= 0.45*total_range:
            img_name = "no_clouds"
            if rain_name is not None:
                img_name = "little_clouds"  # there must be clouds for rain
        elif 0.45*total_range < abs(val) <= 0.85*total_range:
            img_name = "little_clouds"
        elif 0.85*total_range < abs(val) <= 0.95*total_range:
            img_name = "cloudy"
        else:
            img_name = "overcast"
        values["images"].append(img_name)

        # snow: snow, snow_rain
        name = "snow"
        val = values[f"avg_{name}"]
        if val > 0:  # there is snow
            if rain_name is not None:   # there is rain and clouds
                img_name = "snow_rain"
            else:
                img_name = "snow"
            values["images"].append(img_name)

        # fog: when recently there was "rain" and now there is no "wind" -> fog instead of rain
        if time_of_day == "night":
            dates = [f"evening/{day:02d}/{month:02d}", f"noon/{day:02d}/{month:02d}", f"morning/{day:02d}/{month:02d}"]
        if time_of_day in ["evening", "noon", "morning"]:
            # night  evening  noon  morning  |  prev_night  prev_evening  prev_noon
            prev_day = day - 1
            prev_month = month
            if prev_day <= 0:
                prev_month = prev_month-1 if prev_month-1 > 0 else 12
                prev_day = int(calendar.monthrange(2024, prev_month)[1])
            if time_of_day == "evening":
                dates = [f"noon/{day:02d}/{month:02d}", f"morning/{day:02d}/{month:02d}", f"night/{prev_day:02d}/{prev_month:02d}"]
            if time_of_day == "noon":
                dates = [f"morning/{day:02d}/{month:02d}", f"night/{prev_day:02d}/{prev_month:02d}", f"evening/{prev_day:02d}/{prev_month:02d}"]
            if time_of_day == "morning":
                dates = [f"night/{prev_day:02d}/{prev_month:02d}", f"evening/{prev_day:02d}/{prev_month:02d}", f"noon/{prev_day:02d}/{prev_month:02d}"]
        f_rain = False
        for _date in dates:
            for _img in out_data[_date]["images"]:
                if _img.find('heavy_rain') >=0 or _img.find('downpour') >=0:
                    f_rain = True
                    break
            if f_rain:
                break
        if f_rain:  # recently there was "heavy_rain" or "downpour"
            if "wind" not in values["images"]:  # now there is no "wind"
                if "fog" not in out_data[dates[1]]["images"]:  # fog will not appear 3rd time in row
                    values["images"].append("fog")
                    continue

    # save to json
    with open('json_files\\weather_data.json', 'w') as f:
        json.dump(out_data, f)
    # save to csv
    """
    with open('json_files\\weather_data.csv', 'w') as f:
        f.write(f'time;temp;rain;snow;cloud;wind;images\n')
        for month in range(1,13):
            max_day = int(calendar.monthrange(2024, month)[1])
            for day in range(1, max_day+1):
                for time_of_day in ["morning", "noon", "evening", "night"]:
                    key = f"{time_of_day}/{day:02d}/{month:02d}"
                    values = out_data[key]
                    f.write(f'{key};'
                            f'{str(values["avg_temperature"]).replace(".",",")};'
                            f'{str(values["avg_rain"]).replace(".",",")};'
                            f'{str(values["avg_snow"]).replace(".",",")};'
                            f'{str(values["avg_cloud"]).replace(".",",")};'
                            f'{str(values["avg_wind"]).replace(".",",")};')
                    for img in values["images"]:
                        f.write(f'{img};')
                    f.write(f'\n')
    """

    return out_data


def generate_combined_weather(map_data, weather_data) -> tuple[QPixmap, str]:
    """
    Create image that combines multiple pictures of weather conditions and adds text description on the bottom
    :param map_data: {name_of_data: value}
    :param weather_data: dict[str:dict[str, Any]]
    :return: (pixmap, text description)
    """
    # map_data
    time_of_day = EnumTimeOfDay(map_data["time_of_day"]).name
    day = map_data["day"]
    month = map_data["month"]

    # weather_data
    key = f"{time_of_day}/{day:02d}/{month:02d}"
    img_names = weather_data[key]["images"]
    layers = [weather_data[f'img_weather_{img_name}'] for img_name in img_names]
    for idx in range(len(layers) - 1):
        background = layers[idx]
        foreground = layers[idx + 1]
        merged_image = merge_images(background, foreground)
        layers[idx + 1] = merged_image  # merged image becomes the background for next iteration

    # save to file - combined
    cv2.imwrite("images\\weather\\weather_combined.png", layers[-1])
    # save to file - combined_text
    background = np.ones((650, 500, 3), dtype=np.uint8)
    background[:, :, :] *= 255  # white background
    text = ', '.join(img_names).replace("_",' ')
    text = text.replace("temp",'')
    background = cv2.putText(background, text, (2, 640), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    #   expand with alpha channel
    alpha = np.ones((650, 500), dtype=np.uint8)
    alpha[:] *= 255
    background = np.dstack((background, alpha))
    foreground = cv2.imread("images\\weather\\weather_combined.png", cv2.IMREAD_UNCHANGED)
    merged_image = merge_images(background, foreground)
    cv2.imwrite("images\\weather\\weather_combined_text.png", merged_image)
    # make pixmap
    out_pixmap = QPixmap("images\\weather\\weather_combined.png")

    return out_pixmap, ','.join(img_names)




# OBSOLETE
"""
from functools import wraps
class ClassTmp:
    def __init__(self):
        self.tmp_variable = 1

    def my_decorator_func(func):
        @wraps(func)
        def wrapper_func(self, *args, **kwargs):
            obj = args[0]

            # before
            # ...
            obj.val = 1
            print(f"before function, val = {obj.val}")

            func(*args, **kwargs)

            # after
            # ...
            obj.val += 1
            print(f"after function, val = {obj.val}")

        return wrapper_func

    @my_decorator_func
    def fun1(self, obj, **kwargs):
        obj.val = 99
        print(f"in function, val = {obj.val}")

class ClassA:
    val = 1

a = ClassA()
tmp = ClassTmp()
tmp.fun1(a, my_var=77)
a = 1


def my_decorator_func(func):

    @wraps(func)
    def wrapper_func(*args, **kwargs):
        obj = args[0]

        # before
        # ...
        obj.val = 1
        print(f"before function, val = {obj.val}")

        func(*args, **kwargs)

        # after
        # ...
        obj.val += 1
        print(f"after function, val = {obj.val}")
    return wrapper_func

@my_decorator_func
def my_func(obj):
    obj.val = 99
    print(f"in function, val = {obj.val}")
    pass

class ClassA:
    val = 1

a = ClassA()
my_func(a)
a = 0


    def get_closest_hexagon_center_point(self, in_point: tuple[int, int]) -> tuple[int, int]:
        tmp_list = []   # [ [dist, center_point], ... ]
        for center_point in self.center_points.keys():
            dist = math.dist(center_point, in_point)
            tmp_list.append([dist, center_point])
        tmp_list.sort()
        return tmp_list[0][1]
    
    
        
    # add transparency layer (=255)
    b, g, r = cv2.split(image)
    tmp = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)
    rgba = [b, g, r, alpha]
    image = cv2.merge(rgba, 4)
"""