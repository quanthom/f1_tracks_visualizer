import fastf1 as ff
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.widgets import Button, TextBox
import matplotlib.image as mpimg
import argparse

DEBUG = False

class CoordAdjust:
    ratio = 1
    x_stretch = 1
    y_stretch = 1
    x_offset = 0
    y_offset = 0
    rotation = 0
    x_mirror = 1
    y_mirror = 1

    def __init__(self, q=1, xs=1, ys=1, xo=0, yo=0, rot=0, xmir=False, ymir=False):
        self.ratio = q
        self.x_stretch = xs
        self.y_stretch = ys
        self.x_offset = xo
        self.y_offset = yo
        self.rotation = rot
        self.x_mirror = -1 if xmir is True else 1
        self.y_mirror = -1 if ymir is True else 1

    def generate(self, axes, x, y):
        # Applying Horizontal/Vertical mirroring
        x = self.x_mirror * x
        y = self.y_mirror * y

        # Plot driver's trace
        plt_xlim = axes.get_xlim(); xq = -min(plt_xlim) + max(plt_xlim)
        plt_ylim = axes.get_ylim(); yq = -min(plt_ylim) + max(plt_ylim)
        xlen = -min(x) + max(x); ylen = -min(y) + max(y)
        x = (x / xlen) * xq; y = (y / xlen) * xq
        x = x - min(x) + min(plt_xlim)
        y = y - min(y) + min(plt_ylim)

        # Makes vectors into 2D points
        pts = np.vstack((x, y)).T

        # Rotate points around origin
        theta = np.radians(self.rotation)
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                    [np.sin(theta), np.cos(theta)]])
        pts = np.dot(pts, rotation_matrix.T)

        # Apply translation/stretch
        pts[:, 0] = pts[:, 0] * self.x_stretch + self.x_offset
        pts[:, 1] = pts[:, 1] * self.y_stretch + self.y_offset

        pts = pts.reshape(-1, 1, 2) / self.ratio
        segments = np.concatenate([pts[:-1], pts[1:]], axis=1)
        return segments


RACE_MAP = {
    "Bahrain": "maps/bahrain.png",
    "Australia": "maps/melbourne_hr.png",
    "Azerbaijan": "maps/baku.png",
    "Miami": "maps/miami.png",
    "Monaco": "maps/monaco_v2.png",
    "Barcelona": "maps/barcelona.png",
    "Montréal": "maps/canada.png",
    "Spielberg": "maps/austria.png",
    "Silverstone": "maps/silverstone.png",
    "Budapest": "maps/hungary.png",
    "Spa-Francorchamps": "maps/belgium.png",
    "Zandvoort": "maps/Netherlands.png",
    "Monza": "maps/Monza.png",
    "Yas Island": "maps/yasmarina.png", # Not very accurate
}

COORD_ADJUST_FACTORS = {
    "Bahrain": {'q': 1.0, 'xs': 0.8, 'ys': 0.81, 'xo': 87.0, 'yo': 111.0, 'rot': 0, 'xmir': False, 'ymir': True},
    "Australia": {'q': 1.0, 'xs': 0.735, 'ys': 0.745, 'xo': 137.0, 'yo': 20.0, 'rot': 0.7, 'xmir': False, 'ymir': True},
    "Azerbaijan": {'q': 1.0, 'xs': 0.794, 'ys': 0.76, 'xo': 71.0, 'yo': 185.0, 'rot': -2.0, 'xmir': False, 'ymir': True},
    "Miami": {'q': 1, 'xs': 0.914, 'ys': 0.905, 'xo': 125.0, 'yo': 128.0, 'rot': -2.65, 'xmir': False, 'ymir': True},
    "Monaco": {'q': 0.9, 'xs': 0.54, 'ys': 0.54, 'xo': 935.0, 'yo': -350.0, 'rot': 49.0, 'xmir': False, 'ymir': True},
    "Barcelona": {'q': 1.0, 'xs': 0.584, 'ys': 0.584, 'xo': 568.0, 'yo': 1098.0, 'rot': -124.7, 'xmir': True, 'ymir': False},
    "Montréal": {'q': 1.0, 'xs': 0.298, 'ys': 0.301, 'xo': 76.0, 'yo': 415.0, 'rot': -77.8, 'xmir': False, 'ymir': True},
    "Spielberg": {'q': 1.0, 'xs': 0.697, 'ys': 0.703, 'xo': 208.0, 'yo': 111.0, 'rot': -0.4, 'xmir': False, 'ymir': True},
    "Silverstone": {'q': 2.28, 'xs': 1, 'ys': 1, 'xo': 3205.0, 'yo': 250.0, 'rot': 93.0, 'xmir': True, 'ymir': False},
    "Budapest": {'q': 1.31, 'xs': 1.048, 'ys': 1.04, 'xo': -235.0, 'yo': 519.0, 'rot': -40.4, 'xmir': False, 'ymir': True},
    "Spa-Francorchamps": {'q': 1.999, 'xs': 1.0, 'ys': 0.98, 'xo': 2475.0, 'yo': 650.0, 'rot': 121.0, 'xmir': True, 'ymir': False},
    "Zandvoort": {'q': 1.26, 'xs': 1.005, 'ys': 1, 'xo': 155.0, 'yo': 105.0, 'rot': 1.1, 'xmir': False, 'ymir': True},
    "Monza": {'q': 2.038, 'xs': 1, 'ys': 1, 'xo': 3138.0, 'yo': -113.0, 'rot': 84.2, 'xmir': True, 'ymir': False},
    "Yas Island": {'q': 1.0, 'xs': 0.404, 'ys': 0.385, 'xo': 2380.0, 'yo': 290.0, 'rot': 90.0, 'xmir': False, 'ymir': True},
}


class F1Trace:
    params = {'q': 1, 'xs': 1, 'ys': 1, 'xo': 0, 'yo': 0, 'rot': 0, 'xmir': False, 'ymir': False}

    def __init__(self, year, event, session, driver):
        self.year = year
        self.driver = driver
        self.event = ff.get_event_schedule(year).get_event_by_name(event + " Grand Prix")
        if event.lower() not in [x.lower() for x in [self.event.Country, self.event.Location, self.event.OfficialEventName]]:
            raise Exception(f"No event found in {year} for {event}. Got {[self.event.Country, self.event.Location, self.event.OfficialEventName]}")
        self.session = self.event.get_session(session)

    def create_driver(self):
        self.driver_instance = self.session.laps.pick_driver(self.driver)
        self.total_laps = len(self.driver_instance)
        self.lap_number = 1

    def select_lap(self, text):
        if text.isnumeric():
            self.lap_number = int(text)
            self.clear_plot(0)
            self.update(0)

    def select_fastest_lap(self, val):
        self.lap_number = int(self.driver_instance.pick_fastest().LapNumber)
        self.textbox.set_val(self.lap_number)
        self.clear_plot(0)
        self.update(0)

    def clear_plot(self, val):
        self.ax.cla()
        self.plot_track()
        plt.show()

    def plot_track(self):
        print(f"Loading map for event: {self.event.Location}")
        mapimg = mpimg.imread(RACE_MAP[self.event.Location])
        self.ax.imshow(mapimg)
        self.ax.autoscale(enable=False)

    # Create a LineCollection object to display the telemetry data as a racing line
    def update(self, val):
        # Get the telemetry data for the lap
        lap = self.driver_instance.pick_lap(self.lap_number)
        x = lap.telemetry['X']
        y = lap.telemetry['Y']
        speed = lap.telemetry['Speed']

        # Get the event name for the session
        event_name = str(self.session)

        self.ax.set_title(f'{self.driver} Lap {self.lap_number} at {event_name}')
        self.ax.set_xlabel('X position (m)')
        self.ax.set_ylabel('Y position (m)')
        self.ax.set_aspect('equal')

        # Rotate trace to fit map
        if not DEBUG:
            adjust = CoordAdjust(**COORD_ADJUST_FACTORS[self.event.Location])
        else:
            adjust = CoordAdjust(**self.params) # Uncomment for debugging
        segments = adjust.generate(self.ax, x, y)
        lc = LineCollection(segments, cmap=mpl.cm.plasma, norm=plt.Normalize(speed.min(), speed.max()))
        lc.set_array(speed)
        lc.set_linewidth(1.0)
        self.ax.add_collection(lc)

        plt.show()

    def update_params(self, text):
        self.params = {'q': 1, 'xs': 1, 'ys': 1, 'xo': 0, 'yo': 0, 'rot': 0, 'xmir': False, 'ymir': False}
        import re
        pattern = re.compile(r"(\w+):(\-?[\w\.]+)")
        matches = pattern.findall(text)
        for key, value in matches:
            if key not in ['q', 'xs', 'ys', 'xo', 'yo', 'rot', 'xmir', 'ymir']:
                print(f"The parameter {key} is not valid")
                return
            if key in ['xmir', 'ymir']:
                self.params[key] = True if value == "True" else False
            else:
                self.params[key] = float(value)
        print(f"generated parameters: {self.params}")
        self.clear_plot(0)
        self.update(0)

    def start(self):
        # Load the session data for the given year, weekend, and session
        if self.session is None:
            print("The session was not initialised")
            return
        self.session.load()
        self.create_driver()

        # Plot the track map with the racing line and color gradient
        fig, self.ax = plt.subplots(figsize=(12, 8.75))

        button_ax = plt.axes([0.45, 0.92, 0.15, 0.05])
        text_ax = plt.axes([0.25, 0.92, 0.15, 0.05])
        clear_ax = plt.axes([0.65, 0.92, 0.15, 0.05])

        button = Button(button_ax, "Show fastest lap")
        self.textbox = TextBox(text_ax, f"Laps / {self.total_laps}", initial=str(self.lap_number))
        clear = Button(clear_ax, "Clear plot")

        if DEBUG:
            params_ax = plt.axes([0.2, 0.02, 0.65, 0.05])
            paramsbox = TextBox(params_ax, 'Enter params:', initial='q:1')
            paramsbox.on_submit(self.update_params)

        button.on_clicked(self.select_fastest_lap)
        self.textbox.on_submit(self.select_lap)
        clear.on_clicked(self.clear_plot)


        self.plot_track()
        plt.show()


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="Generate F1 driver traces on track maps")

    # Add the arguments
    parser.add_argument("driver", type=str, help="Name of the driver")
    parser.add_argument("year", type=int, help="Year of the race")
    parser.add_argument("track", type=str, help="Track of the race")
    parser.add_argument("session", type=str, help="Session of the race")
    parser.add_argument("--debug", action='store_true', help="Enable debug mode")
    args = parser.parse_args()

    year = args.year
    track = args.track
    session = args.session # Also 'R', 'SS', 'S', 'FP1', 'FP2', 'FP3'
    driver = args.driver
    DEBUG = args.debug

    generator = F1Trace(year, track, session, driver)
    generator.start()
