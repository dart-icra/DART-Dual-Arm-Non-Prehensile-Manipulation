import numpy as np
import matplotlib.pyplot as plt
import os
import time
import datetime

class databuffer:
    def __init__(self):
        self.values = []
        
    def as_np(self):
        return np.array(self.values)

    def add(self,value):
        self.values.append(value)

    def clear(self):
        self.values = []


class Logger:
    def __init__(self, logdir):
        self.logdir = logdir
        self.db = {}
        os.makedirs(logdir, exist_ok=True)
        self.filepath = None
    
    def set_fpath(self, fpath):
        self.filepath = os.path.join(self.logdir, fpath) if fpath else os.path.join(self.logdir, "unnamed_log.npy")

    def buffer(self, id: str, fpath: str = None):
        assert fpath is not None, "File path must be provided"
        if self.filepath is None:
            self.set_fpath(fpath)
        if self.filepath not in self.db:
            self.db[self.filepath] = {"file": self.filepath, "data": {}}
        if id not in self.db[self.filepath]["data"]:
            self.db[self.filepath]["data"][id] = databuffer()

    def log(self, id: str, value):
        assert self.filepath is not None, "File path must be set using set_fpath() or buffer() before logging."
        if self.filepath not in self.db or id not in self.db[self.filepath]["data"]:
            self.buffer(id, os.path.basename(self.filepath))
        self.db[self.filepath]["data"][id].add(value)

    def save(self):
        for fpath, entry in self.db.items():
            logts = datetime.datetime.now()
            snapshot = {k: v.as_np() for k, v in entry["data"].items()}

            if os.path.exists(fpath):
                existing = np.load(fpath, allow_pickle=True).item()
            else:
                existing = {"meta": {}}

            existing["meta"][logts] = {"data": snapshot}

            os.makedirs(os.path.dirname(fpath), exist_ok=True)
            np.save(fpath, existing)
            print(f"[Logger.save] File saved: {fpath}")

    def load(self, id: str = None, fpath: str = None) -> list:
        if fpath is None:
            fpath = os.path.join(self.logdir, "unnamed_log.npy")
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"No log file found at {fpath}")
        
        if id is None:
            return np.load(fpath, allow_pickle=True).item()

        entry = np.load(fpath, allow_pickle=True).item()
        result = []
        for ts, block in entry.get("meta", {}).items():
            data = block.get("data", {})
            if id in data:
                result.append(data[id])
        return result


if __name__ == "__main__":
    logger = Logger(logdir="./delete_me")
    logger.buffer(id="pos_error", fpath="test.npy")
    for i in range(3):
        logger.log("pos_error", np.random.rand())
    for i in range(3):
        logger.log("vel_error", np.random.rand())
    for i in range(3):
        logger.log("ts", time.time())
        
    # logger.save()
    results = logger.load(fpath="./delete_me/test.npy")
    print(results)