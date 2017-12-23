import numpy as np
import time
import sys
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk

UNIT = 100 # scale 1 to 100 pixels
R = 0.5
B = 0.1
Epsilon = 0.1
RATE = 1.0

class Env(tk.Tk, object):
    def __init__(self):
        self.action_space = ['f', 'b', 'l', 'r', 'fl', 'fr', 'bl', 'br']
        self.n_actions = len(self.action_space)
        self.n_features = 6


    def build_map(self):
        if np.random.binomial(1, Epsilon) == 1:
            # print "MAP1"
            self.obstacle = np.random.uniform(-1.0, 1.0, size=2)
            while True:
                self.start = np.random.uniform(-1.0, 1.0, size=2)
                if np.linalg.norm(self.start - self.obstacle) > R + B:
                    break

            while True:
                self.stop = np.random.uniform(-1.0, 1.0, size=2)
                if (np.linalg.norm(self.stop - self.obstacle) > R + B) & (np.linalg.norm(self.stop - self.start) > 0.1):
                    break
        else:
            if np.random.binomial(1, 0.5) == 1:
                # print "MAP2"
                ox = np.random.uniform(-0.3, 0.3, size=1)
                oy = np.random.uniform(-1.0, 1.0, size=1)
                self.obstacle = np.array((ox, oy)).reshape(2)

                while True:
                    px1 = np.random.uniform(-1.0, -0.3, size=1)
                    py1 = np.random.uniform(-1.0, 1.0, size=1)
                    px2 = np.random.uniform(0.3, 1.0, size=1)
                    py2 = np.random.uniform(-1.0, 1.0, size=1)
                    p1 = np.array((px1, py1)).reshape(2)
                    p2 = np.array((px2, py2)).reshape(2)
                    if (np.linalg.norm(p1 - self.obstacle) > R + B) & (np.linalg.norm(p2 - self.obstacle) > R + B):
                        break

                if np.random.binomial(1, 0.5) == 1:
                    self.start = p1
                    self.stop = p2
                else:
                    self.start = p2
                    self.stop = p1

            else:
                # print "MAP3"
                oy = np.random.uniform(-0.3, 0.3, size=1)
                ox = np.random.uniform(-1.0, 1.0, size=1)
                self.obstacle = np.array((ox, oy)).reshape(2)

                while True:
                    py1 = np.random.uniform(-1.0, -0.3, size=1)
                    px1 = np.random.uniform(-1.0, 1.0, size=1)
                    py2 = np.random.uniform(0.3, 1.0, size=1)
                    px2 = np.random.uniform(-1.0, 1.0, size=1)
                    p1 = np.array((px1, py1)).reshape(2)
                    p2 = np.array((px2, py2)).reshape(2)
                    if (np.linalg.norm(p1 - self.obstacle) > R + B) & (np.linalg.norm(p2 - self.obstacle) > R + B):
                        break

                if np.random.binomial(1, 0.5) == 1:
                    self.start = p1
                    self.stop = p2
                else:
                    self.start = p2
                    self.stop = p1

        self.state = np.array((self.obstacle, self.stop, self.start)).reshape(self.n_features)
        self.current = self.start.copy()
        self.path = np.concatenate((self.current, self.current))

    def step(self, action):
        if (action == 'f') | (action == 'fl') | (action == 'fr'):  # forward
            self.current[1] -= RATE / UNIT
        if (action == 'b') | (action == 'bl') | (action == 'br'):  # backward
            self.current[1] += RATE / UNIT
        if (action == 'l') | (action == 'fl') | (action == 'bl'):  # left
            self.current[0] -= RATE / UNIT
        if (action == 'r') | (action == 'fr') | (action == 'br'):  # right
            self.current[0] += RATE / UNIT

        self.state = np.array((self.obstacle, self.stop, self.current)).reshape(self.n_features)
        self.path = np.concatenate((self.path, self.current))


if __name__ == '__main__':
    map = Env()
    map.build_map()
    for i in range(100):
        print i
        map.step('l')
        map.setup_display()
        map.display2D()
        time.sleep(1)

