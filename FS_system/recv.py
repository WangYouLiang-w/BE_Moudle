import socket
import time
sock_sever = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_sever.bind(('169.254.26.10', 7819))

while True:
    byte = sock_sever.recv(1024)
    print(byte)
