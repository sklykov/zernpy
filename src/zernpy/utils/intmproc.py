# -*- coding: utf-8 -*-
"""
Experiment with the multiprocessing speeding up of calculations.

@author: Sergei Klykov
@licence: MIT, @year: 2024

"""
# %% Global imports
from multiprocessing import Process, Event, Queue
import os
import warnings
import time
from typing import Callable, Any, Union  # Callable[..., Any] - should check for any callable instance accepting / returning any types
# from collections.abc import Sequence  # for more broad typing check - Sequence[Any] | np.ndarray - accept list, set, tuple, str or np.ndarray


# %% Processes manager class
class DispenserManager():
    """Manager for the distributing calculation job on several Processes."""

    __MAX_WORKERS: int = os.cpu_count(); workers_number: int = __MAX_WORKERS // 2; __warn_message: str = ""
    __workers_pool: list = []; __results: list = []; __triggers: list = []; __initialized: bool = False
    __global_live_trigger: Event = None; __queues: list = []; __parameters_vector: list = []

    def __init__(self, compute_func: Callable[..., Any], params_list: list, n_workers: int = None, verbose_info: bool = False):
        """
        Initialize main handling class along with provided (automatically calculated) number of workers.

        Parameters
        ----------
        compute_func : Callable[..., Any]
            Computation function that should receive each parameter stored in the list.
            Note that multiparameter function can be called with only single parameter and other parameters provided as constants with
            help of partial(compute_func, param1=value1, param2=value2), "from functools import partial".
        params_list : list
            List with parameters for which the compute_func should be calculated.
            Note that parameter should be pickleable.
        n_workers : int, optional
            Number of workers for performing computations. If None is provided, the os.cpu_count()//2 will be used. The default is None.
        verbose_info : bool, optional
            Flag for printing out some verbose information about initialization and calculation progress. The default is False.

        Raises
        ------
        TypeError
            Check the signature. Most probably there is some incosistency in n_workers, compute_func or params_list.
        ValueError
            If number of workers > number of parameters.

        Returns
        -------
        DispenserManager instance.

        """
        self.verbose_info = verbose_info
        if self.verbose_info:
            t1 = time.perf_counter(); self.done_jobs_percentage = 0
        # Checking input parameters - workers
        if n_workers is not None and n_workers >= 2:
            if n_workers > self.__MAX_WORKERS:
                self.__warn_message = (f"Provided more workers than available detected by os.cpu_count() method: {self.__MAX_WORKERS}. "
                                       + f"By default, number of workers is set to: {self.workers_number}")
                warnings.warn(self.__warn_message)
            else:
                self.workers_number = n_workers
        elif n_workers is not None and n_workers < 2:
            self.__warn_message = ("Provided number of workers is meaningless for spreading jobs on workers. "
                                   + f"Default number of workers will be used: {self.workers_number}")
            warnings.warn(self.__warn_message)
        # Checking input parameters - compute function / method
        if not callable(compute_func):
            raise TypeError("Provided computing function isn't callable (not callable(compute_func) - True)")
        # Checking input parameters - list with iterable parameters. Restrict to list only acceptable type
        if len(params_list) == 0:
            raise TypeError("Provided parameters list has 0 length = no splitting of jobs needed")
        elif len(params_list) < self.workers_number:
            raise ValueError("There is no sence for trying to split number of jobs less than initialized workers")
        else:
            self.__results = [None]*len(params_list); self.__parameters_vector = params_list
        # Initializing the pool with Processes
        self.__global_live_trigger = Event(); self.__global_live_trigger.set(); time.sleep(0.01)
        # print(f"Initializing {self.workers_number} Processes")
        for i in range(self.workers_number):
            trigger_event = Event(); queue = Queue(); self.__triggers.append(trigger_event)
            worker = IndiWorker(keep_run_trigger=self.__global_live_trigger, trigger=trigger_event, data_queue=queue, compute_func=compute_func)
            worker.start(); received_confirmation = False  # for waiting of initializaton of the Process
            while not received_confirmation:
                if not queue.empty():
                    confirmation = queue.get_nowait()
                    if confirmation == "Started":
                        received_confirmation = True; break
                    else:
                        self.__warn_message = f"Not recognized confirmation message '{confirmation}', contact the developer"
                        warnings.warn(self.__warn_message); break
                else:
                    time.sleep(0.005)
            self.__workers_pool.append(worker); self.__queues.append(queue)
        if self.verbose_info:
            elapsed_ms = int(round(1000.0*(time.perf_counter() - t1), 0))
            print(f"# of workers as Processes initialized: {self.workers_number}, took {elapsed_ms} ms")
        self.__initialized = True  # global flag that pool of workers should be initialized

    def get_max_workers(self) -> int:
        """
        Return maximum amount of workers detected by os.cpu_count() call.

        Returns
        -------
        int
            Max amount of workers detected by os.cpu_count().

        """
        return self.__MAX_WORKERS

    def compute(self):
        """
        Perform computation using provided parameters and function / method.

        Returns
        -------
        list
            List with computation results.

        """
        if self.__initialized:
            computed_tasks = 0; init_step = True; indices2process = [i for i in range(len(self.__parameters_vector))]
            processing_indices = [None]*len(self.__workers_pool); task_assigned = [False]*len(self.__workers_pool); self.done_jobs_percentage = 0
            while computed_tasks < len(self.__results):
                if init_step:
                    current_param_index = 0
                    for proc_i in range(len(self.__workers_pool)):
                        self.__queues[proc_i].put(self.__parameters_vector[current_param_index])
                        task_assigned[proc_i] = True; processing_indices[proc_i] = current_param_index
                        self.__triggers[proc_i].set(); time.sleep(0.0001)
                        # print(f"Set job for parameter {self.__parameters_vector[current_param_index]}, index {current_param_index}")
                        indices2process.remove(current_param_index); current_param_index += 1
                        # print(f"Remaining indices of parameters to compute: {indices2process}")
                    init_step = False
                # print("Checking that jobs completed...")
                for proc_i in range(len(self.__workers_pool)):
                    if task_assigned[proc_i] and not self.__triggers[proc_i].is_set():
                        current_param_index = processing_indices[proc_i]
                        self.__results[current_param_index] = self.__queues[proc_i].get()
                        computed_tasks += 1  # computation has been done
                        task_assigned[proc_i] = False  # next comptutation hasn't been yet assigned, check if still jobs are available
                        if len(indices2process) > 0:
                            current_param_index = indices2process.pop(0); task_assigned[proc_i] = True
                            self.__queues[proc_i].put(self.__parameters_vector[current_param_index])
                            self.__triggers[proc_i].set(); processing_indices[proc_i] = current_param_index
                            # print(f"Set job for parameter {self.__parameters_vector[current_param_index]}, index {current_param_index}")
                            time.sleep(0.0001)
                    else:
                        time.sleep(0.0001)
                # print(f"Remaining indices of parameters to compute: {indices2process}")
                if self.verbose_info:
                    # print(f"Remained computations: {len(self.__results) - computed_tasks}")
                    done_jobs_percent = int(round(100.0*(computed_tasks/len(self.__results)), 0))
                    if done_jobs_percent >= 100 or len(self.__results) == computed_tasks:
                        done_jobs_percent = 100; print("Computation completed")
                    if self.done_jobs_percentage == 0 and done_jobs_percent >= 10:
                        self.done_jobs_percentage = done_jobs_percent
                        print(f"Done {done_jobs_percent}% of calculations")
                    if done_jobs_percent - self.done_jobs_percentage >= 10:
                        print(f"Done {done_jobs_percent}% of calculations")
                        self.done_jobs_percentage = done_jobs_percent
            return self.__results
        else:
            self.__warn_message = ("All workers have been released, no computation has been performed")
            warnings.warn(self.__warn_message); return None

    def close(self):
        """
        Release all initialized Process()-based classes (workers).

        Returns
        -------
        None.

        """
        if self.__initialized:
            self.__global_live_trigger.clear(); time.sleep(0.005)  # set global trigger for all Processes as False
            for trigger in self.__triggers:
                trigger.set(); time.sleep(0.005)  # set all triggers for releasing wait() methods for letting all Process run() to stop
            for worker in self.__workers_pool:
                if worker.is_alive():
                    worker.join(timeout=0.1)  # wait all Processes stopped
        self.__initialized = False  # set the global flag to False for preventing calling the computation
        self.__workers_pool = []; self.__global_live_trigger = None  # replace all attributes with the default empty values
        self.__triggers = []; self.__results = []; self.__queues = []; self.__parameters_vector = []   # default empty values

    def __del__(self):
        """
        Overwrite default del method.

        Returns
        -------
        None.

        """
        self.close()

    def update_params(self, params_list: list):
        """
        Update list with parameters.

        Parameters
        ----------
        params_list : list
            List with parameters for which the compute_func should be calculated.
            Note that parameter should be pickleable.

        Raises
        ------
        TypeError
            Check the signature. Most probably there is some incosistency in params_list.
        ValueError
            If number of workers > number of parameters.

        Returns
        -------
        None.

        """
        if self.__initialized:
            # Checking input parameters - list with iterable parameters. Restrict to list only acceptable type
            if len(params_list) == 0:
                raise TypeError("Provided parameters list has 0 length = no splitting of jobs needed")
            elif len(params_list) < self.workers_number:
                raise ValueError("There is no sence for trying to split number of jobs less than initialized workers")
            else:
                self.__results = [None]*len(params_list); self.__parameters_vector = params_list


# %% Individual worker special class
class IndiWorker(Process):
    """Individual Worker based on Process() class for performing computations using provided function."""

    __keep_event: Event = None; __trigger: Event; __data_queue: Queue

    def __init__(self, keep_run_trigger: Event, trigger: Event, data_queue: Queue, compute_func):
        """
        Process from multiprocessing based class.

        Parameters
        ----------
        keep_run_trigger : Event
            Global multiprocessing Event() for checking that the main handling class keep working (initialized).
        trigger : Event
            Multiprocessing Event() for triggering computation and reporting that the calculation is done.
        data_queue : Queue
            Multiprocessing Queue() for exchanging data between this class and the main handling class.
        compute_func : callable function / method
            The function that should perform computation using the received parameter through the queue.

        Returns
        -------
        None.

        """
        self.__trigger = trigger; self.__data_queue = data_queue; self.computation = compute_func; self.__keep_event = keep_run_trigger
        Process.__init__(self)

    def run(self):
        """
        Rewrite default Process.start() behavior.

        Returns
        -------
        None.

        """
        if self.__keep_event.is_set():
            self.__data_queue.put_nowait("Started")  # send back the confirmation that the task is started
            while self.__keep_event.is_set():
                self.__trigger.wait()  # wait that the parameter is transferred
                if self.__trigger.is_set() and self.__keep_event.is_set():
                    # attempts = 0
                    # if not self.__data_queue.empty() and attempts < 11:
                    #     parameter = self.__data_queue.get_nowait()
                    # else:
                    #     time.sleep(0.001); attempts += 1
                    parameter = self.__data_queue.get()
                    # print(f"Start computing for parameter {parameter}", flush=True)
                    result = self.computation(parameter)  # computation work
                    self.__data_queue.put_nowait(result)  # returning the result
                    self.__trigger.clear()
                    # print(f"Finished computing for parameter {parameter} and report result {result}", flush=True)
                elif not self.__keep_event.is_set():
                    self.__trigger.clear(); break


# %% Pickleable functions for tests as the main script (latest section)
def test_plus(x: Union[float, int]) -> Union[float, int]:
    """
    Simulation of the time-costly computation task.

    Parameters
    ----------
    x : float | int
        Some number.

    Returns
    -------
    float | int
        x + 1.

    """
    time.sleep(0.004); return x + 1


def test_minus(x: Union[float, int]) -> Union[float, int]:
    """
    Simulation of the time-costly computation task.

    Parameters
    ----------
    x : float | int
        Some number.

    Returns
    -------
    float | int
        x - 1.

    """
    time.sleep(0.004); return x - 1


# %% Define standard exports from this module
__all__ = ['DispenserManager']


# %% Testing as the main script
if __name__ == "__main__":
    params = [10*(i+1) for i in range(177)]  # ... computation points
    # Direct computation - for loop for performance check
    t1 = time.perf_counter(); results_for = [None]*len(params)
    for i, par in enumerate(params):
        results_for[i] = test_plus(par)
    elapsed_ms = int(round(1000.0*(time.perf_counter() - t1), 0))
    print(f"Direct for loop computation took {elapsed_ms} ms")
    # Test performance of distrubuted workflow
    cl = DispenserManager(compute_func=test_plus, params_list=params, n_workers=4, verbose_info=True); time.sleep(0.25)
    t1 = time.perf_counter(); results_paral = cl.compute()
    elapsed_ms = int(round(1000.0*(time.perf_counter() - t1), 0))
    print(f"1st run Parallelized computation took {elapsed_ms} ms")
    params = [10*(i+1) for i in range(203)]  # ... computation points
    cl.update_params(params); t1 = time.perf_counter(); results_paral2 = cl.compute()
    elapsed_ms = int(round(1000.0*(time.perf_counter() - t1), 0))
    print(f"2nd run (changed parameters) Parallelized computation took {elapsed_ms} ms")
    time.sleep(0.65); cl.close()
