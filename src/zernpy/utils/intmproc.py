# -*- coding: utf-8 -*-
"""
Experiment with the multiprocessing speeding up of calculations.

@author: Sergei Klykov
@licence: MIT, @year: 2024

"""
# %% Global imports
import numpy as np
from multiprocessing import Process, Event, Queue
import os
import warnings
from typing import Callable, Any  # Callable[..., Any] - should check for any callable instance accepting / returning any types
# from collections.abc import Sequence  # for more broad typing check - Sequence[Any] | np.ndarray - accept list, set, tuple, str or np.ndarray


# %% Processes manager class
class DispenserManager():
    """Manager for the distributing calculation job on several Processes."""

    __MAX_WORKERS: int = os.cpu_count(); workers: int = __MAX_WORKERS // 2; __warn_message: str = ""
    __workers_pool: list = []; __results: list = []

    def __init__(self, compute_func: Callable[..., Any], params_list: list, workers: int = None):
        # Checking input parameters - workers
        if workers is not None and workers >= 2:
            if workers > self.__MAX_WORKERS:
                self.__warn_message = (f"Provided more workers than available detected by os.cpu_count() method: {self.__MAX_WORKERS}. "
                                       + f"By default, number of workers is set to: {self.workers}")
                warnings.warn(self.__warn_message)
            else:
                self.workers = workers
        elif workers is not None and workers < 2:
            self.__warn_message = ("Provided number of workers is meaningless for spreading jobs on workers. "
                                   + f"Default number of workers will be used: {self.workers}")
            warnings.warn(self.__warn_message)
        # Checking input parameters - compute function / method
        if not callable(compute_func):
            raise TypeError("Provided computing function isn't callable (not callable(compute_func) - True)")
        # Checking input parameters - list with iterable parameters. Restrict to list only acceptable type
        if len(params_list) == 0:
            raise TypeError("Provided parameters list has 0 length = no splitting of jobs needed")
        elif len(params_list) < self.workers:
            raise ValueError("There is no sence for trying to split number of jobs less than initialized workers")
        else:
            self.__results = [None]*len(params_list)
        # Initializing the Processes

    def get_max_workers(self) -> int:
        return self.__MAX_WORKERS


# %% Individual worker special class
class IndiWorker(Process):
    """Individual Worker based on Process() class."""

    __manager_live: bool = False; __trigger: Event; __data_queue: Queue

    def __init__(self, manager_live: bool, trigger: Event, data_queue: Queue, compute_func):
        self.__manager_live = manager_live; self.__trigger = trigger; self.__data_queue = data_queue; self.work = compute_func
        Process.__init__(self)

    def run(self):
        if self.__manager_live:
            while self.__manager_live:
                pass


# %% Testing as the main script
if __name__ == "__main__":

    cl = DispenserManager(lambda x: x+1); print(cl.get_max_workers())
