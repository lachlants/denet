import socket
import time
import struct
import json
import argparse
import sys
import select

import denet.common.logging as logging
import denet.multi.network as network
import denet.multi.shared as shared

class UpdateServer(object):

    def __init__(self, model_dims, port=0, client_num=0, thread_num=1, momentum=0.9, use_async=False, use_mpi=False):
        self.port = port
        self.client_num = client_num
        self.thread_num = thread_num
        self.model_dims = model_dims
        self.momentum = momentum
        self.use_async = use_async
        self.use_mpi = use_mpi

    def start(self):
        if self.use_async:
            self.run_async()
        else:
            self.run_sync()

    #waits for client connects
    def connect_clients(self):
        if self.use_mpi:
            from mpi4py import MPI
            mpi_comm = MPI.COMM_WORLD
            server_socket = mpi_comm.Get_rank()
            client_sockets = list(range(mpi_comm.Get_size()))
            del client_sockets[server_socket]
            self.client_num = len(client_sockets)
            logging.info("Starting MPI update server on:", socket.gethostname(), "(%i clients)"%self.client_num)
        else:
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            logging.info("Starting update server on %s:%i (%i clients)"%(socket.gethostname(), self.port, self.client_num))

            server_socket.bind((socket.gethostname(), self.port))
            server_socket.listen(1)

            logging.info("Waiting for %i clients to connect..."%self.client_num)
            client_sockets=[]
            for _ in range(self.client_num):
                sock, addr = server_socket.accept()
                logging.info("Model Update Server - Adding new client:", addr)
                client_sockets.append(sock)

        logging.info("All clients are connected!")
        return server_socket, client_sockets

    #asynchronous updating loop
    def run_async(self):

        #connect to clients
        server_socket, client_sockets = self.connect_clients()

        #construct update object for each client / server
        client_update = shared.ModelUpdate(self.model_dims)
        server_update = shared.ModelUpdate(self.model_dims)

        #perform synchronization
        logging.info("Begin processing...")
        count = 0
        sync = []
        while True:
            try:
                #get sockets which have data
                # print("Waiting for updates...")
                if self.use_mpi:
                    from mpi4py import MPI
                    mpi_comm = MPI.COMM_WORLD
                    mpi_status = MPI.Status()
                    client_data = mpi_comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=mpi_status)
                    client_json = json.loads(client_data.decode('utf-8'))
                    read_sockets = [mpi_status.Get_source()]
                else:
                    read_sockets, _, _ = select.select(client_sockets, [], [])

                for i,sock in enumerate(client_sockets):

                    #socket is sending data
                    if sock in read_sockets:

                        #read data
                        if not self.use_mpi:
                            logging.info("client %i: recieving command"%i)
                            client_json = network.recv_json(sock)

                        #get counter
                        if client_json["cmd"] == "count":
                            logging.info("count:", count, "peek:", client_json["peek"])
                            network.send_json(sock, {"count" : count}, self.use_mpi)
                            if not client_json["peek"]:
                                count +=1

                        #apply client update to server updates
                        elif client_json["cmd"] == "update":
                            logging.info("update")
                            client_update.import_json(client_json["data"])
                            server_update.add_delta(client_update, self.momentum)
                            network.send_json(sock, server_update.export_json(), self.use_mpi)

                        #synchronize model between all clients / server
                        elif client_json["cmd"] == "sync":
                            logging.info("sync, initial:", client_json["initial"])
                            if not i in sync:
                                sync.append(i)

                            #on 'initial' sync set server update
                            if client_json["initial"]:
                                server_update.import_json(client_json["data"])

                            #perform sync after all clients have call sync
                            if len(sync) == len(client_sockets):
                                model_update = server_update.export_json()
                                for s in client_sockets:
                                    network.send_json(s, model_update, self.use_mpi)
                                    sync=[]
                        else:
                            logging.error("ERROR: Unknown client command: ", client_json["cmd"])

            except (KeyboardInterrupt, SystemExit):
                logging.info("Done")
                return 1

            except Exception as e:
                logging.error("Encounter exception: ", e)
                return 1

    #synchronous updating loop
    def run_sync(self):

        #connect to clients
        server_socket, client_sockets = self.connect_clients()

        #construct update object for each client / server
        client_updates = [shared.ModelUpdate(self.model_dims) for _ in range(self.client_num)]
        server_update = shared.ModelUpdate(self.model_dims)

        #perform synchronization
        while True:
            try:
                logging.info("Waiting for updates...")
                for i,sock in enumerate(client_sockets):
                    update_json = network.recv_json(sock)
                    client_updates[i].import_json(update_json["data"])

                logging.info("Synchronising...")
                ts = time.time()
                server_update.set_mean(client_updates, self.thread_num)
                logging.verbose("mean calc took %.2f sec"%(time.time() - ts))

                ts = time.time()
                server_json = server_update.export_json()
                logging.verbose("json export took %.2f sec"%(time.time() - ts))

                #send mean update to clients
                ts = time.time()
                for sock in client_sockets:
                    network.send_json(sock, server_json)

                logging.verbose("transferring data to clients took %.2f sec"%(time.time() - ts))

            except (KeyboardInterrupt, SystemExit):
                logging.info("Done")
                sys.exit(0)

            except Exception as e:
                logging.error("Encounter exception: ", e)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    logging.add_arguments(parser)
    parser.add_argument("--port", default=1234, type=int, help="")
    parser.add_argument("--model-dims", required=True, type=str, help="export file for shared model dimensions")
    parser.add_argument("--client-num", default=0, type=int, help="number of clients to wait for")
    parser.add_argument("--thread-num", default=1, type=int, help="number of threads for mean calculation")
    parser.add_argument("--momentum", default=0.8, type=float, help="momentum for asychronous implementation")
    parser.add_argument("--use-async", default=False, action="store_true", help="Use asychronous implementation")
    parser.add_argument("--use-mpi", default=False, action="store_true", help="use MPI for networking instead of TCP/IP")
    args = parser.parse_args()

    logging.init(args)
    server = UpdateServer(args.model_dims, args.port, args.client_num, args.thread_num, args.momentum, args.use_async, args.use_mpi)
    server.start()
    sys.exit(0)
