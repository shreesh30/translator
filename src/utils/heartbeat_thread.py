import threading
import time


class HeartbeatThread(threading.Thread):
    def __init__(self, connection, interval=5):
        super().__init__(daemon=True)
        self.connection = connection
        self.interval = interval
        self.running = True

    def run(self):
        while self.running:
            if self.connection and self.connection.is_open:
                # Process data events to send heartbeats
                self.connection.process_data_events()
            time.sleep(self.interval)

    def stop(self):
        self.running = False