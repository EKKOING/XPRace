import subprocess
from time import sleep
import argparse
from typing import List

parser = argparse.ArgumentParser()
parser.add_argument("-instances", help="num instances", required=True)
parser.add_argument("-host", help="host", required=True)

args = parser.parse_args()

workers: List[subprocess.Popen] = []
for i in range(int(args.instances)):
    workers.append(subprocess.Popen(["python3", "workernode.py", "-instance", str(i), "-host", args.host]))
    sleep(1)

while True:
    for i, worker in enumerate(workers):
        if worker.poll() is not None:
            print(f"Worker {i} died!")
            worker.kill()
            worker.terminate()
            workers[i] = subprocess.Popen(["python3", "workernode.py", "-instance", str(i), "-host", args.host])
    sleep(1)
