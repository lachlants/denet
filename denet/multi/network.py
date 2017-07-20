#Very basic networking code

import socket
import struct
import json

def send_msg(sock, msg):
    # print("socket sending %i bytes"%len(msg))
    msg = struct.pack('>I', len(msg)) + msg
    sock.sendall(msg)

def recv_msg(sock):
    n_raw = recvall(sock, 4)
    if not n_raw:
        raise Exception("Failed to read message length")

    n = struct.unpack('>I', n_raw)[0]
    # print("socket recieved %i bytes"%n)
    return recvall(sock, n)

def recvall(sock, n):
    data = bytes()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            raise Exception("Failed to read packet")
        data += packet

    return data

#send json data to socket
def send_json(sock, data, use_mpi=False):
    json_data = json.dumps(data).encode('utf-8')
    if use_mpi:
        from mpi4py import MPI
        # print("send %i:"%sock, json_data)
        MPI.COMM_WORLD.send(json_data, dest=sock)
    else:
        send_msg(sock, json_data)

#recv json data from socket
def recv_json(sock, use_mpi=False):
    if use_mpi:
        from mpi4py import MPI
        json_data = MPI.COMM_WORLD.recv(source=sock)
        # print("recv %i:"%sock, json_data)
        return json.loads(json_data.decode('utf-8'))
    else:
        return json.loads(recv_msg(sock).decode('utf-8'))
