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

        # Create an array of shape (n, 2)
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
    "Bahrain": CoordAdjust(1.12, 0.98, 1.12, 0, 1000),
    "Australia": CoordAdjust(1.12, 0.98, 1.12, 0, 1000, rot=182.0, xmir=True)
}

class F1Trace:
    # Set the year, weekend, session, and driver name
    colormap = mpl.cm.plasma
    
    session = None
    displayed = False
    clearPlot = False

    def __init__(self, year, event, session, driver):
        self.year = year
        self.driver = driver
        self.event = ff.get_event_schedule(year).get_event_by_name(event + "Grand Prix")
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

    # def rotate_points(self, x, y, angle):
    #     """Rotate a set of points by a given angle."""
    #     # Create an array of shape (n, 2)
    #     points = np.vstack((x, y)).T

    #     # Convert the angle to radians
    #     theta = np.radians(angle)

    #     # Define the rotation matrix
    #     rotation_matrix = np.array([
    #         [np.cos(theta), -np.sin(theta)],
    #         [np.sin(theta), np.cos(theta)],
    #     ])

    #     # Apply the rotation matrix to the points
    #     rotated_points = np.dot(points, rotation_matrix.T)

    #     # Split the rotated points back into separate x and y arrays
    #     x_rotated, y_rotated = rotated_points[:, 0], rotated_points[:, 1]

    #     return x_rotated, y_rotated

    # Create a LineCollection object to display the telemetry data as a racing line
    def update(self, val):
        # Get the telemetry data for the lap
        lap = self.driver_instance.pick_lap(self.lap_number)
        x = lap.telemetry['X']
        y = lap.telemetry['Y']
        speed = lap.telemetry['Speed']

        # Get the event name for the session
        event_name = self.session.event.get_race()

        # # fiddling with proportion
        # ratio = 1.12
        # xm = x * 0.98
        # ym = (y * 1.12 + 1000)
        # points = np.array([xm, ym]).T.reshape(-1, 1, 2) / ratio # Fiddled with proportion to fit the map

        # adjust = COORD_ADJUST_FACTORS[self.event.Country]
        # points = adjust.generatePoints(x, y)
        # segments = np.concatenate([points[:-1], points[1:]], axis=1)
        # lc = LineCollection(segments, cmap=mpl.cm.plasma, norm=plt.Normalize(speed.min(), speed.max()))
        # lc.set_array(speed)
        # lc.set_linewidth(1.0)

        # self.ax.add_collection(lc)
        # self.ax.plot(x, y, color='black', linestyle='--', zorder=0)
        self.ax.set_title(f'{self.driver} Qualifying Lap {self.lap_number} at {event_name} ({self.year})')
        self.ax.set_xlabel('X position (m)')
        self.ax.set_ylabel('Y position (m)')
        self.ax.set_aspect('equal')

        # dim = list(self.ax.get_xlim()) + list(self.ax.get_ylim())
        # self.ax.imshow(mapimg, extent=dim)

        # display map
        mapimg = mpimg.imread(RACE_MAP[self.event.Country])
        self.ax.imshow(mapimg)
        self.ax.autoscale(enable=False)

        # Rotate trace to fit map
        adjust = CoordAdjust(xs=0.715, ys=0.715, xo=125, yo=30, rot=182.0, xmir=True)
        x, y = adjust.rotate(x, y)
        # x, y = self.rotate_points(-x, y, 182.0)

        # Plot driver's trace
        # x = (x * self.ax.get_xlim()) / max(x)
        # y = (y * self.ax.get_ylim()) / max(y)
        plt_xlim = self.ax.get_xlim(); xq = -min(plt_xlim) + max(plt_xlim)
        plt_ylim = self.ax.get_ylim(); yq = -min(plt_ylim) + max(plt_ylim)
        xlen = -min(x) + max(x); ylen = -min(y) + max(y)
        x = (x / xlen) * xq; y = (y / xlen) * xq  # y = (y / ylen) * yq 
        x = x - min(x) + min(plt_xlim)
        y = y - min(y) + min(plt_ylim)

        # adjust = COORD_ADJUST_FACTORS[self.event.Country]

        points = adjust.generatePoints(x, y)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=mpl.cm.plasma, norm=plt.Normalize(speed.min(), speed.max()))
        lc.set_array(speed)
        lc.set_linewidth(1.0)
        self.ax.add_collection(lc)
        # self.ax.plot(x, y, color='black', linestyle='--', zorder=0)

        if not self.displayed:
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

        plt.show()


if __name__ == "__main__":
    year = 2022
    track = "Australia"
    session = 'Q'
    driver = 'HAM'

    generator = F1Trace(year, track, session, driver)
    generator.start()