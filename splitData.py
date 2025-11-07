import numpy as np
from sklearn.model_selection import train_test_split

class DataSplitter:
    def __init__(self):
        self.x = np.load("X.npy")
        self.y = np.load("y.npy")
        self.xDur = np.load("X_dur.npy")
        self.yDur = np.load("y_dur.npy")
        self.xVel = np.load("X_vel.npy")
        self.yVel = np.load("y_vel.npy")

    def split_data(self):
        X_train, X_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.2, random_state=42)
        X_train_vel, X_test_vel, y_train_vel, y_test_vel = train_test_split(self.xVel, self.yVel, test_size=0.2, random_state=42)
        X_train_dur, X_test_dur, y_train_dur, y_test_dur = train_test_split(self.xDur, self.yDur, test_size=0.2, random_state=42)
        return (X_train, X_test, y_train, y_test), (X_train_vel, X_test_vel, y_train_vel, y_test_vel), (X_train_dur, X_test_dur, y_train_dur, y_test_dur)
