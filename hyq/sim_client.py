import socket

hostname, sld, tld, port = 'www', 'integralist', 'co.uk', 80
target = '{}.{}.{}'.format(hostname, sld, tld)

# create an ipv4 (AF_INET) socket object using the tcp protocol (SOCK_STREAM)
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# connect the client
# client.connect((target, port))
client.connect(('0.0.0.0', 9999))

# send some data (in this case a HTTP GET request)
client.send('GET /index.html HTTP/1.1\r\nHost: {}.{}\r\n\r\n'.format(sld, tld))

# receive the response data (4096 is recommended buffer size)
response = client.recv(4096)

print response



if __name__ == '__main__':

    # Create the folder
    utils.mkdir(FOLDER)
    ros.init_node("physics", anonymous=True)

    # Run space exploration
    n = 1
    # ros.on_shutdown(utils.cleanup)

    for kp in range(50, 1050, 50):
        for kd in range(1, 13, 4):
            for i in range(5):
                if not ros.is_shutdown():
                    print "\n ===== Simulation N=" + str(n) + \
                          " - Kp=" + str(kp) + " - Kd=" + str(kd) + " =====\n"

                    simulate(n, kp, kd)
                    n += 1
