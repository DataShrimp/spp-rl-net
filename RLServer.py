import socketserver
import binascii
import json
import numpy as np

count = 1
def convertToTensor(state):
    global count
    state = state.decode("utf-8")
    state_json = json.loads(state)
    x_dim = len(state_json['orders'])
    y_dim = len(state_json['riders'])

    if x_dim > 0 and y_dim > 0:
        z_dim = len(state_json['orders'][0]) + len(state_json['riders'][0])
        tensor = np.zeros([x_dim, y_dim, z_dim])
        for i in range(x_dim):
            for j in range(y_dim):
                tensor[i][j] = list(state_json['orders'][i].values()) + list(state_json['riders'][i].values())
        np.save("state-"+str(count)+".npy", tensor)
        count = count+1
        return tensor
    return None


class MyTCPHandler(socketserver.BaseRequestHandler):
    """
    The RequestHandler class for our server.

    It is instantiated once per connection to the server, and must
    override the handle() method to implement communication to the
    client.
    """
    def handle(self):
        # self.request is the TCP socket connected to the client
        b = self.request.recv(2)
        buffsize = int(binascii.b2a_hex(b), 16)
        #print(buffsize)
        self.state = self.request.recv(buffsize).strip()
        tensor = convertToTensor(self.state)

        # rl dispatch
        self.action = "[N]"
        if tensor is not None:
            rider_num = tensor.shape[1]
            choice = np.arange(rider_num)
            self.action = str(np.random.choice(choice, size=10, replace=True).tolist())

        #print("{} wrote:".format(self.client_address[0]))
        # just send back the action info
        #self.action = self.state.upper()
        self.action = bytes(self.action, "utf-8")
        print(self.action)
        #self.request.sendall(struct.pack('>i', len(ret)))
        self.request.sendall(self.action)

        b = self.request.recv(2)
        buffsize = int(binascii.b2a_hex(b), 16)
        self.reward = self.request.recv(buffsize).strip()
        print(self.reward)

if __name__ == "__main__":
    HOST, PORT = "localhost", 9999

    # Create the server, binding to localhost on port 9999
    server = socketserver.TCPServer((HOST, PORT), MyTCPHandler)

    # Activate the server; this will keep running until you
    # interrupt the program with Ctrl-C
    server.serve_forever()