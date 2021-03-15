import GUI
import yoloObject
import cheatModel
import multiprocessing
import time

import VisualDetection
import plotly.graph_objects as go
high = []
high_t = []
low = []
low_t = []

# start = time.perf_counter()
#
# p1 = multiprocessing.Process(target=yoloObject.run)
# p2 = multiprocessing.Process(target=cheatModel.run)
# p1.start()
# p2.start()
# p1.join()
# p2.join()
#
# #GUI.run()
# # yoloObject.run()
# # cheatModel.run()
#
# finish = time.perf_counter()
# print(f'Finished in {finish-start} seconds')

high, high_t, low, low_t = VisualDetection.run()
fig = go.Figure()
fig.add_trace(go.Scatter(x=low_t, y=low, mode='markers', name='low risk'))
fig.add_trace(go.Scatter(x=high_t, y=high, mode='markers', name='high risk', marker_size=10, marker_line_width=2))
fig.show()