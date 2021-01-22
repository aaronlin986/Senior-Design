import GUI
import yoloObject
import cheatModel
import multiprocessing
import time

import VisualDetection

start = time.perf_counter()

p1 = multiprocessing.Process(target=yoloObject.run)
p2 = multiprocessing.Process(target=cheatModel.run)
p1.start()
p2.start()
p1.join()
p2.join()

#GUI.run()
# yoloObject.run()
# cheatModel.run()

finish = time.perf_counter()
print(f'Finished in {finish-start} seconds')