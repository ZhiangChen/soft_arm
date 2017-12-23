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

class Map(tk.Tk, object):
    def __init__(self):
        super(Map, self).__init__()
        self.action_space = ['f','b','l','r']
        self.n_actions = len(self.action_space)
        self.n_features = 6
        self.title('map')
        self.geometry('{0}x{1}'.format(UNIT*2, UNIT*2))
        self._build_map()

    def _build_map(self):
        self.canvas = tk.Canvas(self, bg='white',
                                height=UNIT*2, width=UNIT*2)
        self.obstacle = np.random.uniform(-1.0, 1.0, size=2)
        while True:
            self.start = np.random.uniform(-1.0, 1.0, size=2)
            if np.linalg.norm(self.start - self.obstacle)> R+B:
                break

        while True:
            self.stop = np.random.uniform(-1.0, 1.0, size=2)
            if (np.linalg.norm(self.stop - self.obstacle)> R+B ) & (np.linalg.norm(self.stop - self.start)>0.1):
                break

        self.o = self._draw_circle(self.obstacle, R, 'red')
        self.s = self._draw_circle(self.start, 0.05, 'blue')
        self.e = self._draw_circle(self.stop, 0.05, 'green')
        self.canvas.pack()
        self.render()

        self.state = np.array((self.obstacle, self.stop, self.start)).reshape(self.n_features)
        self.current = self.start.copy()
        self.path_point = np.concatenate((self.start.copy(),self.start.copy()))
        self.path = self.canvas.create_line(*self.path_point)


    def reset_map(self):
        self.canvas.delete(self.o)
        self.canvas.delete(self.s)
        self.canvas.delete(self.e)
        self.canvas.delete(self.path)
        if np.random.binomial(1, Epsilon) == 1:
            #print "MAP1"
            self.obstacle = np.random.uniform(-1.0, 1.0, size=2)
            while True:
                self.start = np.random.uniform(-1.0, 1.0, size=2)
                if np.linalg.norm(self.start - self.obstacle) > R+B:
                    break

            while True:
                self.stop = np.random.uniform(-1.0, 1.0, size=2)
                if (np.linalg.norm(self.stop - self.obstacle) > R+B) & (np.linalg.norm(self.stop - self.start) > 0.1):
                    break
        else:
            if np.random.binomial(1, 0.5) == 1:
                #print "MAP2"
                ox = np.random.uniform(-0.3, 0.3, size=1)
                oy = np.random.uniform(-1.0, 1.0, size=1)
                self.obstacle = np.array((ox,oy)).reshape(2)

                while True:
                    px1 = np.random.uniform(-1.0,-0.3, size=1)
                    py1 = np.random.uniform(-1.0, 1.0, size=1)
                    px2 = np.random.uniform(0.3, 1.0, size=1)
                    py2 = np.random.uniform(-1.0, 1.0, size=1)
                    p1 = np.array((px1, py1)).reshape(2)
                    p2 = np.array((px2, py2)).reshape(2)
                    if (np.linalg.norm(p1 - self.obstacle) > R+B) & (np.linalg.norm(p2 - self.obstacle) > R+B):
                        break

                if np.random.binomial(1, 0.5) == 1:
                    self.start = p1
                    self.stop = p2
                else:
                    self.start = p2
                    self.stop = p1

            else:
                #print "MAP3"
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
                    if (np.linalg.norm(p1 - self.obstacle) > R+B) & (np.linalg.norm(p2 - self.obstacle) > R+B):
                        break

                if np.random.binomial(1, 0.5) == 1:
                    self.start = p1
                    self.stop = p2
                else:
                    self.start = p2
                    self.stop = p1

        self.o = self._draw_circle(self.obstacle, R, 'red')
        self.s = self._draw_circle(self.start, 0.05, 'blue')
        self.e = self._draw_circle(self.stop, 0.05, 'green')
        self.state = np.array((self.obstacle, self.stop, self.start)).reshape(self.n_features)
        self.current = self.start.copy()
        self.path_point = np.concatenate((self.start.copy(), self.start.copy()))
        self.path = self.canvas.create_line(*self.path_point)

    def step(self, action):
        self.canvas.delete(self.path)
        if action == 'f': # forward
            self.current[1] -= 1.0/UNIT
        if action == 'b': # backward
            self.current[1] += 1.0/UNIT
        if action == 'l': # left
            self.current[0] -= 1.0/UNIT
        if action == 'r': # right
            self.current[0] += 1.0/UNIT

        self.path_point = np.concatenate((self.path_point, self.current))
        self.path = self._draw_line(self.path_point)
        self.state = np.array((self.obstacle, self.stop, self.current)).reshape(self.n_features)

    def render(self):
        self.update()

    def _draw_circle(self, point, r, color):
        p = point*UNIT+UNIT
        R = int(r*UNIT)
        x, y = p
        x0 = x - R
        y0 = y - R
        x1 = x + R
        y1 = y + R
        return self.canvas.create_oval(x0, y0, x1, y1, fill = color)

    def _draw_line(self, points):
        p = (points*UNIT+UNIT).astype(int)
        return self.canvas.create_line(*p)


if __name__ == '__main__':
    map = Map()
    map.build_map()
    for i in range(100):
        print i
        map.step('l')
        map.render()
        time.sleep(1.0)