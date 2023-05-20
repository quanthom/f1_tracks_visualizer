import fastf1 as ff
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.widgets import Button, TextBox
import matplotlib.image as mpimg


RACE_MAP = {
    "Bahrain": "bahrain.png",
    "Australia": "melbourne.png",
    "Azerbaijan": "baku.png"
}

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

    def generatePoints(self, x, y):
        xm = x * self.x_stretch + self.x_offset
        ym = y * self.y_stretch + self.y_offset
        return np.array([xm, ym]).T.reshape(-1, 1, 2) / self.ratio

    def rotate(self, x, y):
        """Rotate a set of points by a given angle."""
        # Apply mirroring
        x = self.x_mirror * x
        y = self.y_mirror * y
        points = np.vstack((x, y)).T

        theta = np.radians(self.rotation)
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ])

        rotated_points = np.dot(points, rotation_matrix.T)
        x_rotated, y_rotated = rotated_points[:, 0], rotated_points[:, 1]
        return x_rotated, y_rotated


COORD_ADJUST_FACTORS = {
    "Bahrain": CoordAdjust(xs=0.805, ys=0.805, xo=80, yo=115, ymir=True),
    "Australia": CoordAdjust(xs=0.715, ys=0.715, xo=125, yo=30, rot=182.0, xmir=True),
    "Azerbaijan": CoordAdjust(xs=0.78, ys=0.77, xo=85, yo=160, rot=179.0, xmir=True)
}

class F1Trace:
    def __init__(self, year, event, session, driver):
        self.year = year
        self.driver = driver
        self.event = ff.get_event_schedule(year).get_event_by_name(event + " Grand Prix")
        if event not in self.event.Country:
            raise Exception(f"No event found for {event} in year {year}")
        self.session = self.event.get_session(session)

    def create_driver(self):
        self.driver_instance = self.session.laps.pick_driver(self.driver)
        self.total_laps = len(self.driver_instance)
        self.lap_number = 1

    def update_lap(self, text):
        if text.isnumeric():
            self.lap_number = int(text)

    def clear_plot(self, val):
        self.ax.cla()
        self.plot_track()
        plt.show()

    def plot_track(self):
        mapimg = mpimg.imread(RACE_MAP[self.event.Country])
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
        adjust = COORD_ADJUST_FACTORS[self.event.Country]
        x, y = adjust.rotate(x, y)

        # Plot driver's trace
        plt_xlim = self.ax.get_xlim(); xq = -min(plt_xlim) + max(plt_xlim)
        plt_ylim = self.ax.get_ylim(); yq = -min(plt_ylim) + max(plt_ylim)
        xlen = -min(x) + max(x); ylen = -min(y) + max(y)
        x = (x / xlen) * xq; y = (y / xlen) * xq
        x = x - min(x) + min(plt_xlim)
        y = y - min(y) + min(plt_ylim)

        points = adjust.generatePoints(x, y)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=mpl.cm.plasma, norm=plt.Normalize(speed.min(), speed.max()))
        lc.set_array(speed)
        lc.set_linewidth(1.0)
        self.ax.add_collection(lc)

        plt.show()


    def start(self):
        # Load the session data for the given year, weekend, and session
        if self.session is None:
            print("The session was not initialised")
            return
        
        # Load session
        self.session.load()

        # Create driver instance
        self.create_driver()

        # Plot the track map with the racing line and color gradient
        fig, self.ax = plt.subplots(figsize=(12, 8.75))

        button_ax = plt.axes([0.05, 0.7, 0.15, 0.05])
        text_ax = plt.axes([0.05, 0.5, 0.15, 0.05])
        clear_ax = plt.axes([0.05, 0.6, 0.15, 0.05])

        button = Button(button_ax, "Show next lap")
        textbox = TextBox(text_ax, f"Laps / {self.total_laps}", initial=str(self.lap_number))
        clear = Button(clear_ax, "Clear plot")

        button.on_clicked(self.update)
        textbox.on_text_change(self.update_lap)
        clear.on_clicked(self.clear_plot)

        self.plot_track()
        plt.show()


if __name__ == "__main__":
    year = 2021
    track = "Azerbaijan" # Also available: "Bahrain", "Australia"
    session = 'R' # Also 'R', 'SS', 'S', 'FP1', 'FP2', 'FP3'
    driver = 'HAM'

    generator = F1Trace(year, track, session, driver)
    generator.start()