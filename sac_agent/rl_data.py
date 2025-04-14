import os
import sys
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

# from stable_baselines3 import DQN
from pathlib import Path
sys.path.append(str(Path.cwd())+'/envs/') 
from envs.trafficenv import SumoEnv

import pickle
from sac_agent.sac_online import SAC
import random


if __name__ == '__main__':

    # ts_ids_index = ['1159176756', '1388442962', '1492574988', '1492574990', '1492645720', '1916386518', '1916386562', '276556799', '2852225599', '314622964', '314635491', '314655170', '3523298662', '3562004404', '432429372', '432452908', '432452909', '432452987', '432452989', '4550018600', '4701464324', '4779165760', '4794975945', '5488650286', '671587313', '8013476764', 'J29', 'J52', 'J54', 'cluster_1159517082_4794944089', 'cluster_1159517085_5436607864', 'cluster_1916386555_432429395', 'cluster_3028924422_314614336', 'cluster_432429373_5213238455', 'cluster_4548141054_4548141062_4548142597_4548142605', 'cluster_4548211590_8013476765', 'cluster_4548211609_5789411073_cluster_4548211261_5789411072', 'cluster_4550018629_4550018932', 'cluster_4550018780_4550018980_4550019087_4550019099', 'cluster_4580811894_J22_J30_J5', 'cluster_5213238458_J22', 'cluster_672362123_8640094718', 'cluster_J100_J94_J95_J99', 'cluster_J18_J5', 'cluster_J18_cluster_1159519336_J5']

    # nobs = []
    nobs2 = []
    # nobs_next = []
    nobs2_next = []
    nact = []
    nrew = []
    ndone = []

    train_id = ['1159176756', '1159176757', '1916386495', '1916386553', '2712421554', 
                '2852225599', '3523298662', '3562004404', '3780408667', 
                '432429372', '432429376', '4779165760', '5514165402', '671587213',
                'cluster_1916386555_432429395', 'cluster_2712421553_5213237283', 
                'cluster_3028924422_314614336', 'cluster_432429373_5213238455', 'cluster_5213238458_J22', 
                'cluster_5213238478_5628750476_5628750477_J18', 'cluster_5436607838_J47_J69_J76_J78']
    test_id = ['314655170', '314635491', '8013476764', '432452987', '1159176227', 'J54']
    ts_ids_index = train_id+test_id
    env = SumoEnv(
        netfile = 'nets/',
        sumocfg='cd_region/online.sumocfg',
        out_csv_name=None,
        single_agent=False,
        use_gui=False,
        begin_time=0,
        num_seconds=3899,
        delta_time=10,
        min_green=10,
        yellow_time=3,
        reward_fn='queue',
        sumo_seed=42,
        # ts_ids_index=ts_ids_index
        )
    
    SAC_agent = {}
    for agent_id in ts_ids_index:
        SAC_agent[agent_id] = SAC(env, agent_id)

    for _ in range(27):
        print("prepare for collect {}.rou.xml".format(_))
        cfgfile = 'nets/cd_region/Chengdu.sumocfg'
        with open("nets/cd_region/Chengdu.sumocfg",'w',encoding='utf-8')as f:
            f.write("<configuration>\n<input>\n<net-file value=\"ChengduCity.net.xml\"/>\n")
            f.write("<route-files value=\"rou/{}.rou.xml\"/>\n".format(_))
            f.write("</input>\n<time>\n<begin value=\"0\"/>\n<end value=\"3900\"/>\n</time>\n")
            f.write("<output>\n<tripinfo-output value=\"rl_final/{}_tripinfo.xml\" />\n</output>\n</configuration>".format(_))

        env = SumoEnv(
            netfile = 'nets/',
            sumocfg='cd_region/Chengdu.sumocfg',
            out_csv_name=None,
            single_agent=False,
            use_gui=False,
            begin_time=0,
            num_seconds=3899,
            delta_time=10,
            min_green=10,
            yellow_time=3,
            reward_fn='queue',
            sumo_seed=42,
            # ts_ids_index=ts_ids_index
        )

        ts_ids_index = env.ts_ids

        obs = env.reset()
        
        warm_up=int(300/10)
        for i in range(0,warm_up):
            actionk = {}
            for id in env.ts_ids:
                actionk[id] = random.choice([0,1]) # env.traffic_signals[id].action_space.sample()
            next_obs, rew, done, info = env.step(actionk)
            obs = next_obs        

        for i in range(0,360):
            actionk = {}
            for id in env.ts_ids:
                actionk[id] = SAC_agent[id].take_action(obs[id])

            next_obs, rew, done, info = env.step(actionk)

            for id in env.ts_ids:
                SAC_agent[id].replay_buffer.add(obs[id], actionk[id], rew[id], next_obs[id], done[id])
            obs = next_obs
            for id in env.ts_ids:
                SAC_agent[id].train_agent()

        # nobs.append(env.last_obs_nd)
        # nobs_next.append(env.next_obs_nd)
        nobs2.append(env.last_obs_nd2)
        nobs2_next.append(env.next_obs_nd2)
        nact.append(env.actions_nd)
        nrew.append(env.rewards_nd)
        ndone.append(env.dones_nd)

    with open("sac_data.pkl", "wb") as file:
        pickle.dump([nobs2, nobs2_next, nact, nrew, ndone], file)
