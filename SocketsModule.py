import json
from socketIO_client_nexus import SocketIO, BaseNamespace
import time

class SocketNamespace(BaseNamespace):
    def on_connect(self):
        print('[Connected]')
   
    def on_disconnect(self):
        print('[Disconnected]')
   
    def on_event(self, event, *args):
        pass
        # print('[Event]', event, *args)
 
"""
# connect to server
socketIO = SocketIO('URL', 443, BaseNamespace)
ocrns = socketIO.define(SocketNamespace, '/ocr')

def send_data(data):
    ocrns.emit('data', data)

# send simple string
send_data('hello server - AVIHAY')                          
for i in range(10):
    send_data('hello server')  
    # send json to server: create python object
    python_object = {
        "patient_id": "AVIHAY",
        "sensors": [
            {"x": 240, "y": i, "width": 42, "height": 34, "type": None},
            {"x": 40, "y": 223, "width": 53, "height": 52, "type": None}
        ]
    }
    to_json = json.dumps(python_object)                 # convert to json    
    send_data(to_json)                                 # send json to server
    # time.sleep(2)
 
# stay connected forever
socketIO.wait()
"""