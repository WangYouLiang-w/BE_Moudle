import socket
import time
sock_client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
controlCenterAddr = ('169.254.26.10', 7819)
while True:
    sock_client.sendto(bytes(str(int(39)), "utf8"),controlCenterAddr)
    time.sleep(0.5)


