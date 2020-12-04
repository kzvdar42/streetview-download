from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

class MyQueue(mp.Queue):

    def __init__(self, *args, **kwargs):
        super(MyQueue).__init__(*args, **kwargs)
        self.is_closed


class PoolHelper:

    def __init__(self, max_workers=None, pool=None, write_out=True):
        self.max_workers = max_workers
        self.pool = pool or ThreadPoolExecutor(max_workers=max_workers)
        self.write_out = write_out
        self.futures = []
        self._results = mp.Queue()

    def submit(self, func, *args, f_done=None, **kwargs):
        while len(self.futures) > self.max_workers * 2:
            self.check_status()
        future = self.pool.submit(func, *args, **kwargs)
        future.add_done_callback(lambda f: self.__done(f, f_done))
        self.futures.append(future)
        # self.check_status()
        return future

    def shutdown(self, wait=True):
        return self.pool.shutdown(wait)
    
    def __done(self, future, f_done=None):
        if f_done is not None:
            f_done()
        code, res = future.result()
        # If Error, print it
        if code == False:
            print(res)

    def check_status(self):
        futures, self.futures = self.futures, []
        for future in futures:
            if not future.done():
                self.futures.append(future)

def return_with_code(func):

    def wrapper(*args, **kwargs):
        try:
            res = func(*args, **kwargs)
        except Exception as e:
            return False, e
        return True, res
    
    return wrapper