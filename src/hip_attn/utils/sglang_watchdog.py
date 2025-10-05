import datetime
import sys
import os
import subprocess
import threading
import time
import requests

def log(*args):
    comment = " ".join([str(a) for a in args])
    timestamp = "{:%Y-%m-%d %H:%M:%S}".format(datetime.datetime.now())
    print(f"\033[91m[{timestamp} sglang_watchdog] {comment}\033[0m", flush=True)

class Watchdog:
    def __init__(
        self,
        timeout_bootup = 300,
    ):
        self.timeout_bootup = 300
        self.timeout_tick = 60
        self.sleep_step = 1
        self.proc: subprocess.Popen = None
        self.argv: list[str] = None
        self.running: bool = True

    def start_subprocess(self):
        args = [
            "python",
            "-m",
            "sglang.launch_server",
            *self.argv
        ]
        flatten_args = " ".join(args)
        log(f"Start subprocess using following command: {flatten_args}")
        self.proc = subprocess.Popen(args)
        log(f"Start subprocess communication.")
        return_code = self.proc.wait()
        log(f"Return code is {return_code}")

    def kill_subprocess(self):
        log(f"Start kill subprocess")
        self.proc.kill()
        self.proc = None
        log(f"Finish kill subprocess")

    def wait_for_health(self, timeout: int):
        t_start = time.time()
        while (time.time() - t_start) < timeout:
            try:
                response = requests.get(self.health_endpoint, timeout=timeout)
                response.raise_for_status()
                return
            except requests.ConnectionError:
                time.sleep(self.sleep_step)
        raise TimeoutError()

    def main_watchdog(self):
        while True:
            try:
                t_boot = time.time()
                booted = False
                while (
                    (time.time() - t_boot) < self.timeout_bootup
                    and self.proc.returncode is None
                    and not booted
                ):
                    try:
                        self.wait_for_health(timeout=self.timeout_bootup)
                        booted = True
                    except (TimeoutError, requests.HTTPError):
                        # NOTE: may process is not started yet
                        pass
                    time.sleep(self.sleep_step)
            
                if not booted: raise TimeoutError()
            
                while True:
                    self.wait_for_health(timeout=self.timeout_tick)
                    time.sleep(self.sleep_step)
            
            except (TimeoutError, requests.HTTPError):
                self.kill_subprocess()
            time.sleep(self.sleep_step)

    def main_starter(self):
        while True:
            self.start_subprocess()
            time.sleep(self.sleep_step)
    
    def start(self):
        if "--" in sys.argv:
            argv = sys.argv[sys.argv.index("--") + 1:]
        else:
            argv = sys.argv[1:]
        
        assert "--host" in argv
        assert "--port" in argv
        self.host = argv[argv.index("--host") + 1]
        self.port = argv[argv.index("--port") + 1]
        self.health_endpoint = f"http://{self.host}:{self.port}/health"
        log(f"Watching: {self.health_endpoint}")

        self.argv = argv

        self.thread_watchdog = threading.Thread(
            target=self.main_watchdog, 
            daemon=True
        )
        self.thread_starter = threading.Thread(
            target=self.main_starter, 
            daemon=True
        )

        self.thread_watchdog.start()
        self.thread_starter.start()

        self.thread_watchdog.join()
        self.thread_starter.join()

        self.running = False

if __name__ == '__main__':
    dog = Watchdog()
    dog.start()