import numpy as np
import random
import logging
from collections import deque
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from sdn_env_used_in_CNPA.simsetting import Setting
from sdn_env_used_in_CNPA.packet import Packet
from sdn_env_used_in_CNPA.simqueue import SimQueue



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

    def __init__(self, ts_per_actorbatch, num, arrRate, sch_loc, file):
        self.set = Setting(num, arrRate, sch_loc, file)
        self.ts_per_actorbatch = ts_per_actorbatch
        self._init()

    def close(self):
        print("End of the episode")

    def _init(self):
        self.tot_pktNum = self.ts_per_actorbatch * (sum(self.set.pktRate) * self.set.maxSimTime // self.ts_per_actorbatch)   # used to indicate the end of an episode
        # self.stat = Stats(self.set)
        self.currentTime = 0.
        self.swNum = self.set.swNum
        self.schNum = self.set.schNum
        self.ctlNum = self.set.ctlNum
        self.sch_queues = []
        self.ctl_queues = []
        self.ctl_pktLeaveTime = []  # nested list, [[list]*ctlNum]
        for i in range(self.schNum):
            self.sch_queues.append(SimQueue())
        for i in range(self.ctlNum):
            self.ctl_queues.append(SimQueue())
            self.ctl_pktLeaveTime.append(deque([]))
        self.ctl_avail = [1] * self.ctlNum  # indicate whether a controller is available, 1 means available
        self.remainPktNum = 0
        self.sch = 0  # self.mapping_firstPkt2scheduler()  # the scheduler that has the earliest packet
        self.pkt_departure_generator()
        self.pkt_generator()
        self.ctl, self.nextTime = self.mapping_firstPkt2controller()  # the controller that finishes processing the pkt at the earliest time


    def pkt_generator(self):
        for i in range(self.swNum):
            nextArrivalTime = []
            while len(nextArrivalTime) == 0:
                logging.info("Packets are generated from sch")
                nextArrivalTime = sample_possion(self.set.pktRate[i], self.set.timeStep)
                nextArrivalTime = [x + self.currentTime for x in nextArrivalTime]
            logging.info("%s packets are generated" % (len(nextArrivalTime)))
            self.remainPktNum += len(nextArrivalTime)
            for x in nextArrivalTime:
                pkt = Packet(x, i)
                self.sch_queues[self.sch].enqueue(pkt, x + self.set.sw2schLink[i])
                logging.info("Put the packet %s from switch %s into the scheduler queue" % (pkt.enqueueTime, i))
        self.currentTime += self.set.timeStep
        # send each controller a packet if they are available
        for ctl in range(self.ctlNum):
            if self.ctl_avail[ctl]:
               self.fetch_pkt_from_sch(ctl)

    def fetch_pkt_from_sch(self, ctl):
        self.ctl_avail[ctl] = 0
        pkt = self.sch_queues[self.sch].dequeue()
        pkt.controller = ctl
        self.remainPktNum -= 1
        self.tot_pktNum -= 1
        # todo: communication delay between switches and schedulers can be added later
        enqueueTime = pkt.enqueueTime
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!enqueueTime = pkt.enqueueTime + self.set.sch2ctlLink[ctl]  # Communication latency between schedulers and controllers
        self.ctl_queues[ctl].enqueue(pkt, enqueueTime)
        logging.info("Put the packets into the controller queue %s" % (ctl))


    def pkt_departure_generator(self):
        for i in range(0, self.ctlNum):
            pktLeaveTime = []
            while len(pktLeaveTime) == 0:
                logging.info("Packets' leaving time is generated from ctl")
                pktLeaveTime = sample_possion(self.set.ctlRate[i], self.set.timeStep)
                pktLeaveTime = [x + self.currentTime for x in pktLeaveTime]
            self.ctl_pktLeaveTime[i].extend(pktLeaveTime)

    def mapping_firstPkt2controller(self):
        ctl_avil_time = [0] * self.ctlNum

        # for i in range(0, self.ctlNum):
        #     if len(self.ctl_pktLeaveTime[i]) == 0:
        #         self.pkt_departure_generator()
        #         self.pkt_generator()

        # remove packets from controllers from current state to next state
        for i in range(0, self.ctlNum):
            while True:
                if len(self.ctl_pktLeaveTime[i]) != 0:
                        x = self.ctl_pktLeaveTime[i][0]
                        firstPktTime = self.ctl_queues[i].getFirstPktTime()
                        if firstPktTime is None:  # no packets in ctl_queues[i]
                            print("impossible!!!!!!! no pkt in controller queue??")
                            break
                        elif x < firstPktTime:
                            self.ctl_pktLeaveTime[i].popleft()
                            continue
                        else:
                            ctl_avil_time[i] = x
                            break
                else:
                    self.pkt_departure_generator()
                    self.pkt_generator()

        # # remove packets from controllers from current state to next state
        # for i in range(0, self.ctlNum):
        #     while len(self.ctl_pktLeaveTime[i]) != 0:
        #         x = self.ctl_pktLeaveTime[i][0]
        #         firstPktTime = self.ctl_queues[i].getFirstPktTime()
        #         if firstPktTime is None:  # no packets in ctl_queues[i]
        #             print("impossible!!!!!!! no pkt in controller queue??")
        #             break
        #         elif x < firstPktTime:
        #             self.ctl_pktLeaveTime[i].popleft()
        #             continue
        #         else:
        #             ctl_avil_time[i] = x
        #             break
        return ctl_avil_time.index(min(ctl_avil_time)), min(ctl_avil_time)


    def reset(self):
        self._init()


    # Function prototype is vf_ob, ac_ob, rew, new, _ = env.step(ac)
    def step(self, action):
        # temp = self.sch_queues[0].getFirstPktTime()
        # if temp > self.currentTime:
        #     self.pkt_departure_generator()
        #     self.pkt_generator()

        # remove a packet from self.ctl
        done = False
        resp_time = 0.
        pkt_num = 0
        noPktbyCtl = [0] * self.ctlNum
        avgCtlRespTime = [0.] * self.ctlNum

        x = self.ctl_pktLeaveTime[self.ctl].popleft()
        pkt = self.ctl_queues[self.ctl].dequeue()
        self.ctl_avail[self.ctl] = 1
        t = pkt.generateTime
        resp_time += x - t + self.set.sch2ctlLink[self.ctl]*2 + self.set.sw2schLink[pkt.scheduler]
        pkt_num += 1
        noPktbyCtl[self.ctl] += 1
        avgCtlRespTime[self.ctl] += x - t + self.set.sch2ctlLink[self.ctl]*2 + self.set.sw2schLink[pkt.scheduler]

        # self.stat.add_response_time(x, pkt.scheduler, i, x - t + self.set.sch2ctlLink[i][pkt.scheduler])  # for state update
        logging.info("Remove packets from controller queue %s" % self.ctl)
        self.set.update_noPktbyCtl(noPktbyCtl)
        self.set.update_avgCtlRespTime(avgCtlRespTime)

        # fetch a packet from a switch to self.ctl
        self.fetch_pkt_from_sch(self.ctl)


        # generate new packets for schedulers and generate packet departure time for controllers
        if self.remainPktNum == 0:  # time - self.currentTime > self.set.timeStep:
            self.pkt_departure_generator()
            self.pkt_generator()

        self.ctl, self.nextTime = self.mapping_firstPkt2controller()  # time for next state

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
















