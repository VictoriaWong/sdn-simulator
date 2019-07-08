import numpy as np
import random
from simsetting import Setting
from packet import Packet
from simqueue import SimQueue
import logging
from collections import deque


def sample_possion(rate, time):  # Sample from a possion process
    pos_array = []
    current = 0
    while True:
        pos = -(np.log(1 - random.random())) / rate
        current += pos
        if current < time:
            pos_array.append(current)
        else:
            return pos_array



class sdn_simulator(object):

    def __init__(self, ts_per_actorbatch, num):
        self.set = Setting(num)
        self.ts_per_actorbatch = ts_per_actorbatch
        self._init()

    def close(self):
        print("End of the episode")

    def _init(self):
        self.tot_pktNum = self.ts_per_actorbatch * (sum(self.set.pktRate) * self.set.maxSimTime // self.ts_per_actorbatch)   # used to indicate the end of an episode
        # self.stat = Stats(self.set)
        self.currentTime = 0.
        self.sch_queues = []
        self.ctl_queues = []
        self.ctl_pktLeaveTime = []  # nested list, [[list]*ctlNum]
        self.schNum = self.set.schNum
        self.ctlNum = self.set.ctlNum
        for i in range(self.schNum):
            self.sch_queues.append(SimQueue())
        for i in range(self.ctlNum):
            self.ctl_queues.append(SimQueue())
            self.ctl_pktLeaveTime.append(deque([]))
        self.remainPktNum = 0
        # self.tot_ctlRate = sum(self.set.ctlRate)  # used in vf_ob
        # self.tot_arrRate = sum(self.set.pktRate)  # used in vf_ob
        # self.tot_latency = 0.  # used in vf_ob
        # for i in range(self.schNum):
        #     for j in range(self.ctlNum):
        #         self.tot_latency += self.set.pktRate[i] * self.set.sch2ctlLink[j][i] / self.tot_arrRate
        self.pkt_departure_generator()
        self.pkt_generator()
        self.sch = self.mapping_firstPkt2scheduler()  # the scheduler that has the earliest packet

    # def _init_ob(self):
    #     ac_ob = []
    #     for ctl in range(self.ctlNum):
    #         ac_ob.append([])
    #         ac_ob[ctl] = [self.set.ctlRate[ctl], self.set.sch2ctlLink[ctl][self.sch], self.set.pktRate[self.sch]]
    #         ac_ob[ctl] += [0.] * self.set.history_len * 2
    #     vf_ob = [self.tot_ctlRate, self.tot_latency, self.tot_arrRate]
    #     vf_ob += [0.] * self.set.history_len * 2
    #     return vf_ob, ac_ob

    def pkt_generator(self):
        for i in range(self.schNum):
            nextArrivalTime = []
            while len(nextArrivalTime) == 0:
                logging.info("Packets are generated from sch")
                nextArrivalTime = sample_possion(self.set.pktRate[i], self.set.timeStep)
                nextArrivalTime = [x + self.currentTime for x in nextArrivalTime]
            logging.info("%s packets are generated" % (len(nextArrivalTime)))
            self.remainPktNum += len(nextArrivalTime)
            for x in nextArrivalTime:
                pkt = Packet(x, i)
                self.sch_queues[i].enqueue(pkt, x)
                logging.info("Put the packet %s into the scheduler queue %s" % (pkt.enqueueTime, i))
        self.currentTime += self.set.timeStep

    def pkt_departure_generator(self):
        for i in range(0, self.ctlNum):
            pktLeaveTime = []
            while len(pktLeaveTime) == 0:
                logging.info("Packets' leaving time is generated from ctl")
                pktLeaveTime = sample_possion(self.set.ctlRate[i], self.set.timeStep)
                pktLeaveTime = [x + self.currentTime for x in pktLeaveTime]
            self.ctl_pktLeaveTime[i].extend(pktLeaveTime)

    def mapping_firstPkt2scheduler(self):
        firstPktTime = []
        for i in range(self.schNum):
            temp = self.sch_queues[i].getFirstPktTime()
            if temp is None:
                logging.info("No packet in the scheduler")
                temp = 100000
            firstPktTime.append(temp)
        return firstPktTime.index(min(firstPktTime))

    def reset(self):
        self._init()
        # vf_ob, ac_ob = self._init_ob()
        # return vf_ob, ac_ob

    # Function prototype is vf_ob, ac_ob, rew, new, _ = env.step(ac)
    def step(self, action):
        if type(action).__module__ == np.__name__:
            schDecision = action[0]
        else:
            schDecision = action
        done = False
        # Take Action: distribute the first packet from scheduler to controller
        pkt = self.sch_queues[self.sch].dequeue()
        self.remainPktNum -= 1
        self.tot_pktNum -= 1
        # todo: communication delay between switches and schedulers can be added later
        enqueueTime = pkt.generateTime + self.set.sch2ctlLink[schDecision][self.sch]  # Communication latency between schedulers and controllers
        self.ctl_queues[schDecision].enqueue(pkt, enqueueTime)
        logging.info("Put the packets into the controller queue %s" % (schDecision))


        # generate new packets for schedulers and generate packet departure time for controllers
        if self.remainPktNum == 0:  # time - self.currentTime > self.set.timeStep:
            self.pkt_departure_generator()
            self.pkt_generator()

        # update self.sch for next state and next action
        self.sch = self.mapping_firstPkt2scheduler()  # the scheduler that has the earliest packet

        time = self.sch_queues[self.sch].getFirstPktTime()  # time for next state


        resp_time = 0.
        pkt_num = 0
        noPktbyCtl = [0]*self.ctlNum
        avgCtlRespTime = [0.]*self.ctlNum

        # remove packets from controllers from current state to next state
        for i in range(0, self.ctlNum):
            while len(self.ctl_pktLeaveTime[i]) != 0:
                x = self.ctl_pktLeaveTime[i][0]
                firstPktTime = self.ctl_queues[i].getFirstPktTime()
                if firstPktTime is None:  # no packets in ctl_queues[i]
                    break
                elif firstPktTime > time:  # finish processing packets for this time period
                    break
                elif x < firstPktTime:
                    self.ctl_pktLeaveTime[i].popleft()
                    continue
                else:
                    self.ctl_pktLeaveTime[i].popleft()
                    pkt = self.ctl_queues[i].dequeue()
                    t = pkt.generateTime
                    resp_time += x - t + self.set.sch2ctlLink[i][pkt.scheduler]
                    pkt_num += 1
                    noPktbyCtl[i] += 1
                    avgCtlRespTime[i] += x - t + self.set.sch2ctlLink[i][pkt.scheduler]

                    # self.stat.add_response_time(x, pkt.scheduler, i, x - t + self.set.sch2ctlLink[i][pkt.scheduler])  # for state update
                    logging.info("Remove packets from controller queue %s" % (i))
        self.set.update_noPktbyCtl(noPktbyCtl)
        self.set.update_avgCtlRespTime(avgCtlRespTime)

        # todo: whether it is the end of the episode
        if self.tot_pktNum == 0:
            done = True
            logging.info("A trajectory is sampled")
            self._init()
            # vf_ob, ac_ob = self._init_ob()
        # else:
            # update response time history for both observation space, both vf_resp_his and ac_resp_his are lists
            # vf_resp_his, ac_resp_his = self.stat.update_response_time_history(self.sch)

            # update utilization info
            # vf_util_his, ac_util_his = self.stat.update_utilization_history()

            # vf_ob = [self.tot_ctlRate, self.tot_latency, self.tot_arrRate] + vf_resp_his + vf_util_his
            # ac_ob = []
            # for ctl_id in range(self.ctlNum):
            #     ac_ob.append([])
            #     ac_ob[ctl_id] = [self.set.ctlRate[ctl_id], self.set.sch2ctlLink[ctl_id][self.sch], self.set.pktRate[self.sch]]
            #     ac_ob[ctl_id] += ac_resp_his[ctl_id] + ac_util_his[ctl_id]

        # return vf_ob, ac_ob, resp_time, pkt_num, done, {}
        return resp_time, pkt_num, done, {}
















