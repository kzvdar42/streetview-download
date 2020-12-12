from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
import time
import sys
import traceback

class PoolHelper:

    def __init__(self, max_workers=None, pool=None, write_out=True):
        self.max_workers = max_workers if pool is None else pool._max_workers
        self.pool = pool or ThreadPoolExecutor(max_workers=max_workers)
        self.write_out = write_out
        self.futures = []
        self._results = mp.Queue()

    def submit(self, func, *args, f_done=None, **kwargs):
        if len(self.futures) > self.max_workers * 2:
            self.check_status()
        future = self.pool.submit(func, *args, **kwargs)
        future.add_done_callback(lambda f: self.__done(f, f_done))
        self.futures.append(future)
        return future
    
    def wait_for_all(self):
        while self.futures:
            self.check_status()

    def shutdown(self, wait=True):
        return self.pool.shutdown(wait)
    
    def __done(self, future, f_done=None):
        if f_done is not None:
            f_done(future)
        code, res = future.result()
        # If Error, print it
        if code == False:
            traceback.print_tb(res[2])
            print(res)

    def check_status(self):
        futures, self.futures = self.futures, []
        for future in futures:
            if not future.done():
                self.futures.append(future)


class QueueIterator:

    def __init__(self, queue=None, pbar=None, batch_size=None):
        self.queue = queue
        self.pbar = pbar
        self.total_amount = 0
        self.batch_size = batch_size or -1
    
    def put(self, val, increment=True):
        if increment:
            self.total_amount += 1
        return self.queue.put(val)
    
    def put_iter(self, vals):
        for val in vals:
            self.put(val)
    
    def get(self):
        return self.queue.get()

    def empty(self):
        return self.queue.empty()

    def _update_pbar(self):
        if self.pbar is not None and self.total_amount != self.pbar.total:
            self.pbar.total = self.total_amount
            self.pbar.refresh()

    def __iter__(self):
        while True:
            res = []
            self._update_pbar()
            while not self.empty() and (len(res) < self.batch_size or self.batch_size < 1):
                r = self.get()
                # Stop iteration if read "exit"
                if isinstance(r, str) and r == "exit":
                    if res:
                        yield res
                    return
                res.append(r)
            # If something is read, yield it
            if res:
                yield res
            # If queue is empty, wait a little
            elif self.empty():
                time.sleep(2)
                continue


def return_with_code(func):

    def wrapper(*args, **kwargs):
        try:
            res = func(*args, **kwargs)
        except:
            return False, sys.exc_info()
        return True, res
    
    return wrapper