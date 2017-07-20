import sys
import time
import math

import denet.common.logging as logging
import denet.multi.network as network
from denet.multi.worker import WorkerProcess

#handles communication with update server
class UpdateClient():

    def __init__(self, epoch_start, subset_start, subset_num, sock=None, async=False, mpi=False, count_offset=0, count_delta=1):

        #ensure valid epoch / subset
        self.subset_num = subset_num
        if subset_start >= self.subset_num:
            self.epoch = epoch_start + 1
            self.subset = 0
        else:
            self.epoch = epoch_start
            self.subset = subset_start - 1

        #for non-async subset num must
        if async:
            self.epoch_stride = subset_num
        else:
            self.epoch_stride = count_delta*int(math.ceil(subset_num / count_delta))

        self.mpi = mpi
        self.sock = sock
        self.server_exists = not self.sock is None
        if not sock is None:
            self.server_name = "MPI" if self.mpi else sock.getsockname()

        self.async = async
        self.count_delta = count_delta
        self.count_offset = epoch_start*self.epoch_stride + subset_start
        if not async:
            self.count_offset += count_offset - count_delta - (subset_start%count_delta)

        logging.info("Start update client - epoch:", self.epoch, "epoch stride:", self.epoch_stride, "subset_num:", self.subset_num,
                     "async:", self.async, "MPI:", self.mpi, "count delta:", self.count_delta, "count offset:", self.count_offset)

    #server commands
    def cmd_count(self, peek=False):
        assert self.server_exists
        network.send_json(self.sock, {"cmd": "count", "peek": peek}, self.mpi)
        r = network.recv_json(self.sock, self.mpi);
        return r["count"]

    def cmd_update(self, model_update_delta, model_update):
        assert self.server_exists
        network.send_json(self.sock, {"cmd": "update", "data": model_update_delta.export_json()}, self.mpi)
        model_update.import_json(network.recv_json(self.sock, self.mpi))
        return model_update

    def cmd_sync(self, model_update, initial=False):
        assert self.server_exists
        cmd_json = {"cmd" : "sync", "initial":initial}
        if initial:
            cmd_json["data"] = model_update.export_json()

        network.send_json(self.sock, cmd_json, self.mpi)
        model_update.import_json(network.recv_json(self.sock, self.mpi))
        return model_update

    #returns index of next subset to process (returns -1 if epoch has finished)
    #overcomplicated to handle all three cases!
    def get_subset_next(self):
        if self.server_exists:
            if self.async:
                #peek counter to see if next subset is in new epoch
                epoch_next = int((self.cmd_count(peek = True) + self.count_offset) // self.epoch_stride)
                if (epoch_next != self.epoch):
                    self.epoch += 1
                    self.subset = -1
                else:
                    #get next update
                    self.subset = (self.cmd_count() + self.count_offset)%self.epoch_stride
            else:
                self.count_offset += self.count_delta
                epoch_next = int(self.count_offset // self.epoch_stride)
                if self.epoch != epoch_next:
                    self.count_offset -= self.count_delta
                    self.epoch += 1
                    self.subset = -1
                else:
                    self.subset = min(self.count_offset%self.epoch_stride, self.subset_num-1)
        else:
            self.subset += 1
            if self.subset >= self.subset_num:
                self.epoch += 1
                self.subset = -1

        return self.subset

    #send model to server
    def update(self, model_update_delta, model_update, workers):

       if self.server_exists:

            #send model delta and update workers
            logging.info("Sending updates to server:", self.server_name)
            ts = time.time()
            self.cmd_update(model_update_delta, model_update)
            for worker in workers:
                worker.update(model_update)
            logging.info("Update took %0.1f sec"%(time.time() - ts))

    #synchonize all nodes
    def sync(self, model_update, workers, initial=False):
        if self.async:
            logging.info("Synchronizing with update server:", self.server_name)
            ts = time.time()
            self.cmd_sync(model_update, initial)

            #update workers
            for worker in workers:
                worker.update(model_update)

            logging.info("Sync took %0.1f sec"%(time.time() - ts))

