import matplotlib.pyplot as plt
import numpy as np
import math
import random


class traffic_distribution():

    def __init__(self):
        # PRV2 Data: Flow inter arrival rate
        self.data_x = [10, 70, 115, 1050, 9000, 12000]
        self.data_y = [0.1, 0.3, 0.32, 0.8, 0.95, 1]
        self.interval_num = len(self.data_x)-1
        self.a = []
        self.b = []
        self.avg_arr_rate = 0
        self.calculate_distribution()
        self.calculate_integral()

    def calculate_distribution(self):
        for i in range(self.interval_num):
            temp_a = (self.data_y[i+1]-self.data_y[i])/(math.log(self.data_x[i+1], 10)-math.log(self.data_x[i], 10))
            temp_b = self.data_y[i]-temp_a*math.log(self.data_x[i], 10)
            self.a.append(temp_a)
            self.b.append(temp_b)

    def draw_distribution(self):
        plt.figure()
        for i in range(self.interval_num):
            x = np.linspace(self.data_x[i], self.data_x[i+1])
            y = self.a[i]*np.log10(x)+self.b[i]
            plt.semilogx(x, y, basex=10)
            # plt.plot(x, y)
        plt.savefig('real_traffic_distribution.pdf')
        plt.show()

    def calculate_integral(self):
        # we cannot calculate the average arrival rate in this way,
        # it should be the probability density function not the accumulated probability
        area = 0
        for i in range(self.interval_num):
            length = math.log(self.data_x[i+1], 10) - math.log(self.data_x[i], 10)
            height = self.data_y[i+1]
            temp_area = length*height - 0.5*length*(self.data_y[i+1]-self.data_y[i])
            area += temp_area
        avg_inter_arr_time = math.pow(10, area)   # Transform x from log to linear
        # print("average inter arrival time: %s us" % avg_inter_arr_time)
        self.avg_arr_rate = math.pow(10, 6) / avg_inter_arr_time  # Transformation from arrival time to arrival rate
        # print("average pkt arrival rate: %s pkt/s" % self.avg_arr_rate)

    def sample(self, t, sw_num):
        pos_array = [[] for _ in range(sw_num)]

        # random.seed(1)
        for sw in range(sw_num):
            current_time = 0.
            current_pkt_num = 0
            while current_time < t:
                y = random.random()
                # obtain x according to the sampled y
                if y < self.data_y[0]:
                    y = self.data_y[0]
                    ind = 0
                else:
                    for i in range(self.interval_num):
                        if y < self.data_y[i + 1]:
                            ind = i
                            break
                x = math.pow(10, ((y - self.b[ind]) / self.a[ind]))
                # print("sample inter arrival time: %s us" % x)
                current_pkt_num += 1
                current_time += x*math.pow(10, -6)*0.375
                pos_array[sw].append(current_time)

        return pos_array






# d = traffic_distribution()
# num_list = []
# avg = 0.0
# arr_time = d.sample(1, 10)
# for i in range(len(arr_time)):
#     print(len(arr_time[i]))
#     avg += len(arr_time[i])
#     # print(arr_time[i])
#
# print(avg/10.0)