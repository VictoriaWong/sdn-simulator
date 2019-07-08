from baselines import logger
import baselines.common.tf_util as U
import numpy as np
import time
import struct
# from sdn_env_noStateInfo import sdn_simulator
from sdn_env_real_traffic_distribution import sdn_simulator
from baselines.common import set_global_seeds


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
        # =======random scheduling===========
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
            allpr = [[0.3435640490095485, 0.3959110846067223, 0.2605248663837292], [0.22921876954874854, 0.28156580514592233, 0.48921542530532913], [0.3501927608623485, 0.4621982031347223, 0.18760903600292922], [0.3452212269727485, 0.6461449570499224, 0.008633815977329151], [0.3452212269727485, 0.5516858131475223, 0.10309295987972922], [0.34687840493594857, 0.39756826256992234, 0.25555333249412904], [0.0, 0.04293217844512236, 0.9570678215548777], [0.11818784601434855, 0.16722052568512238, 0.7145916283005291], [0.060186617302348526, 0.10590494104672232, 0.8339084416509291], [0.11653066805114856, 0.17053488161152236, 0.712934450337329], [0.2275615915855485, 0.2799086271827224, 0.4925297812317291], [0.11984502397754851, 0.17053488161152236, 0.7096200944109291], [0.3419068710463485, 0.5483714572211223, 0.10972167173252911], [0.23912389866029848, 0.2916025447289481, 0.46927355661075343]]
            pr = np.array(allpr[env.sch])  # capacity-based probability
            pr /= pr.sum()
            ac = np.array([np.random.choice(range(env.ctlNum), replace=True, p = pr)])
        elif env.set.algo == 'Deterministic':
            ac = np.array([0])
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
            break
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


def make_sdn_env(ts_per_actorbatch, seed, num):
    set_global_seeds(seed)
    return sdn_simulator(ts_per_actorbatch, num)


def train(num_timesteps, ts_per_actorbatch, env):
    U.make_session(num_cpu=1).__enter__()
    learn(env, max_timesteps=num_timesteps,
        timesteps_per_actorbatch=ts_per_actorbatch, schedule='linear',)
    env.close()


def main():
    seed = 0
    j = 3
    timesteps_per_actorbatch = 29000
    logger.configure()
    env = make_sdn_env(timesteps_per_actorbatch, seed, j)
    print("total arrival rate %s" % sum(env.set.pktRate))
    traj_len = sum(env.set.pktRate) * (env.set.maxSimTime)
    num_timestep = traj_len * 1 // timesteps_per_actorbatch * timesteps_per_actorbatch
    print(num_timestep/timesteps_per_actorbatch)
    train(num_timesteps=num_timestep, ts_per_actorbatch=timesteps_per_actorbatch, env=env)
    print("Num of packets processed by each controller: %s"%env.set.noPktbyCtl)
    print("RespTime of each controller: %s"%env.set.avgCtlRespTime)
    tempVar = [x/y for x,y in zip(env.set.avgCtlRespTime, env.set.noPktbyCtl)]
    print("Average RespTime of each controller: %s" % tempVar)


if __name__ == '__main__':
    main()
