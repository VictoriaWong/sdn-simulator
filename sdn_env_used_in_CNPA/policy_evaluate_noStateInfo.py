from baselines import logger
import baselines.common.tf_util as U
import numpy as np
import time
import struct
import csv
import pandas as pd
# from sdn_env_real_traffic_distribution import sdn_simulator
from baselines.common import set_global_seeds
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from sdn_env_used_in_CNPA.sdn_env_noStateInfo import sdn_simulator

envNodeNum = {0: 49, 1: 33, 2: 16, 3: 14, 4: 3}

def traj_segment_generator(env, horizon):
    t = 0
    ac = 0  # not used, just so we have the datatype
    new = True  # marks if we're on first timestep of an episode
    # vf_ob, ac_ob = env.reset()

    cur_ep_ret = 0  # return in current episode
    cur_ep_len = 0  # len of current episode
    cur_ep_pkt = 0
    ep_rets = []  # returns of completed episodes in this segment
    ep_lens = []  # lengths of ...
    ep_pkts = []

    # Initialize history arrays
    # vf_obs = np.array([vf_ob for _ in range(horizon)])
    # ac_obs = np.array([ac_ob for _ in range(horizon)])
    rews = np.zeros(horizon, 'float32')
    pkt_nums = np.zeros(horizon, 'int32')
    news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])

    while True:
        # ==================Define your policy=======================================
        # ac = pi.act(stochastic, ac_ob)
        if env.set.algo == 'RANDOM':
            ac = np.array([np.random.randint(env.ctlNum)])
        elif env.set.algo == 'WEIGHTED_RANDOM':
            pr = np.array(env.set.ctlRate)  # capacity-based probability
            pr = pr.astype(float)
            pr /= pr.sum()
            ac = np.array([np.random.choice(range(env.ctlNum), replace=True, p=pr)])
        elif env.set.algo == 'DELAY_RANDOM':
            sch = np.array(env.sch)
            laten = env.set.laten[:,sch]
            pr = 1./laten
            pr /= pr.sum()
            ac = np.array([np.random.choice(range(env.ctlNum), replace=True, p=pr)])
        elif env.set.algo == 'CAPA_DELAY_RANDOM':
            sch = np.array(env.sch)
            laten = env.set.laten[:,sch]
            pr = np.array([x/y for x,y in zip(env.set.ctlRate, laten)])
            pr /= pr.sum()
            ac = np.array([np.random.choice(range(env.ctlNum), replace=True, p=pr)])
        elif env.set.algo == 'GD':
            allpr = [[0.2182068040551804, 0.33318083341580557, 0.44861236252901404], [0.2180336699174153, 0.33300769927804025, 0.44895863080454446], [0.21821684081678996, 0.3332812010319011, 0.4485019581513089], [0.2182093132455828, 0.33355972116656674, 0.44823096558785047], [0.2182093132455828, 0.3334166973136304, 0.44837398944078677], [0.21821182243598516, 0.3331833426062078, 0.448604834957807], [0.21767485568987313, 0.3326463758600957, 0.4496787684500312], [0.21786555416045497, 0.33283456514027515, 0.44929988069926985], [0.21777773249637128, 0.33274172509538674, 0.449480542408242], [0.21786304497005263, 0.33283958352108, 0.4492973715088674], [0.2180311607270129, 0.333005190087638, 0.44896364918534903], [0.2178680633508574, 0.33283958352108, 0.4492923531280626], [0.21820429486477796, 0.3334116789328256, 0.4483840262023965], [0.2180487250598297, 0.33302275442045465, 0.44892852051971566]]
            pr = np.array(allpr[env.sch])  # capacity-based probability
            pr /= pr.sum()
            ac = np.array([np.random.choice(range(env.ctlNum), replace=True, p = pr)])
        elif env.set.algo == 'Deterministic':
            ac = np.array([0])
        elif env.set.algo == 'CNPA':
            ac = 0
        else:
            print("wrong scheduling algorithms !!!!!!!!!!!")
            break
        if t > 0 and t % horizon == 0:
            yield {"rew": rews, "pkt_num": pkt_nums, "new": news,
                   "ac": acs, "ep_rets": ep_rets, "ep_lens": ep_lens, "ep_pkts": ep_pkts}
            # yield {"vf_ob": vf_obs, "ac_ob": ac_obs, "rew": rews, "pkt_num": pkt_nums, "new": news,
            #        "ac": acs, "ep_rets": ep_rets, "ep_lens": ep_lens, "ep_pkts": ep_pkts}
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []
            ep_pkts = []
        i = t % horizon
        # vf_obs[i] = vf_ob
        # ac_obs[i] = ac_ob
        news[i] = new
        acs[i] = ac

        rew, pkt_num, new, _ = env.step(ac)  # vf_ob, ac_ob, rew, pkt_num, new, _ = env.step(ac)
        rews[i] = rew
        pkt_nums[i] = pkt_num

        cur_ep_ret += rew
        cur_ep_len += 1
        cur_ep_pkt += pkt_num
        if new:
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            ep_pkts.append(cur_ep_pkt)
            cur_ep_ret = 0
            cur_ep_len = 0
            cur_ep_pkt = 0
            env.reset()  # vf_ob, ac_ob = env.reset()
        t += 1


def learn(env,
          timesteps_per_actorbatch,  # timesteps per actor per update
          max_timesteps=0, max_seconds=0,  # time constraint
          callback=None,  # you can do anything in the callback, since it takes locals(), globals()
          schedule='constant'  # annealing for stepsize parameters (epsilon and adam)
          ):
    # Open a file to record the accumulated rewards
    # file = open("response_time/sDCsameCtl/linear_beta%d.txt" % (0.5), "ab")

    U.initialize()

    # Prepare for rollouts
    # ----------------------------------------
    seg_gen = traj_segment_generator(env, timesteps_per_actorbatch)

    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()

    assert sum([max_timesteps > 0,
                max_seconds > 0]) == 1, "Only one time constraint permitted"

    while True:
        if callback: callback(locals(), globals())
        if max_timesteps and timesteps_so_far >= max_timesteps:
            # file.close()
            print("============================================")
            print("arrival rate: %s" % sum(env.set.pktRate))
            print("total number of packets: %s" % seg["ep_pkts"][0])
            print("average response time: %s" % (seg["ep_rets"][0]/seg["ep_pkts"][0]))
            return seg["ep_pkts"][0], seg["ep_rets"][0]/seg["ep_pkts"][0]
            # break
        elif max_seconds and time.time() - tstart >= max_seconds:
            break

        logger.log("********** Iteration %i ************" % iters_so_far)
        logger.log("Processing %s" % (timesteps_so_far/max_timesteps))

        seg = seg_gen.__next__()

        print("avg resp: %s" % (sum(seg["rew"])/sum(seg["pkt_num"])))

        # record_reward(file, sum(seg["rew"]))
        # print("total rewards for Iteration %s: %s" % (iters_so_far, sum(seg["rew"])))
        # prob = collections.Counter(seg["ac"])  # a dict where elements are stored as dictionary keys and their counts are stored as dictionary values.
        # for key in prob:
        #     prob[key] = prob[key]/len(seg["ac"])
        # print("percentage of choosing each controller: %s" % (prob))


        iters_so_far += 1
        timesteps_so_far += len(seg["ac"])


def record_reward(file, num):
    num = struct.pack("d", num)
    file.write(num)
    file.flush()


def make_sdn_env(ts_per_actorbatch, args):
    seed = args["seed"]
    num = args["envid"]
    sch_loc = args["scheduler location"]
    set_global_seeds(seed)
    arrRate = args["arrival Rate"]
    file = args["file"]
    return sdn_simulator(ts_per_actorbatch, num, arrRate, sch_loc, file)


def train(num_timesteps, ts_per_actorbatch, env):
    U.make_session(num_cpu=1).__enter__()
    throughput, avgResp = learn(env, max_timesteps=num_timesteps,
        timesteps_per_actorbatch=ts_per_actorbatch, schedule='linear',)
    env.close()
    return throughput, avgResp


def main():
    args = {"seed": int(sys.argv[1]), "envid": int(sys.argv[2]), "arrival Rate": int(sys.argv[3])}
    # args = {"seed": 0, "envid": 4, "arrival Rate": 5}
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    result_file = os.path.join(__location__, r'pkt_trace_seed%s_env%s_arr%s.csv' % (args["seed"], args["envid"], args["arrival Rate"]))
    args["file"] = result_file
    for sch in range(envNodeNum[args["envid"]]):
        args["scheduler location"] = sch
        starttime = time.time()
        timesteps_per_actorbatch = 100 * 610*args["arrival Rate"]
        logger.configure()
        env = make_sdn_env(timesteps_per_actorbatch, args)

        if sch == 0:
            with open(result_file, 'w', newline='') as outcsv:
                writer = csv.writer(outcsv)
                writer.writerow(["controller Number", env.set.ctlNum, "control plane capacity", sum(env.set.ctlRate), "switch Number", env.set.swNum, "arrival rate", "610*%s"% args["arrival Rate"]])
            df = pd.DataFrame([["scheduler location", "each controller Throughput", "each controller RespTime", "total throughput", "overall average RespTime"]])
            df.to_csv(result_file, index=False, header=False, mode='a')


        print("total arrival rate %s" % sum(env.set.pktRate))
        traj_len = sum(env.set.pktRate) * (env.set.maxSimTime)
        num_timestep = traj_len * 1 // timesteps_per_actorbatch * timesteps_per_actorbatch
        print("Num of actorbatch: %s"% (num_timestep/timesteps_per_actorbatch))
        throughput, avgResp = train(num_timesteps=num_timestep, ts_per_actorbatch=timesteps_per_actorbatch, env=env)
        print("Num of packets processed by each controller: %s"%env.set.noPktbyCtl)
        # print("RespTime of each controller: %s"%env.set.avgCtlRespTime)
        tempVar = [x/y for x,y in zip(env.set.avgCtlRespTime, env.set.noPktbyCtl)]
        print("Average RespTime of each controller: %s" % tempVar)
        endtime2 = time.time()
        print("running time: %s" % (endtime2 - starttime))

        df = pd.DataFrame([[sch, env.set.noPktbyCtl, tempVar, throughput, avgResp]],columns=["scheduler location", "each controller Throughput", "each controller RespTime", "total throughput", "overall average RespTime"])
        df.to_csv(result_file, index=False, header=False, mode='a')


if __name__ == '__main__':
    main()