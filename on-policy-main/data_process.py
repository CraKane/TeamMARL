from tensorboard.backend.event_processing import event_accumulator
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--sample_num",
                    default=5, type=int,
                    help="Batch size for training")
parser.add_argument("--env_type", default='sc2')
parser.add_argument("--model_name", default='easy_v2_c0')
parser.add_argument("--env_id", default='3m')
parser.add_argument("--frame", default='QMIX')
parser.add_argument("--game_mode", default='easy_test2')
parser.add_argument("--para_change", type=int, default=5)
parser.add_argument("--check_end", type=bool, default=False)
parser.add_argument("--run_num", default=None, type=str)
parser.add_argument("--single_run", default=False, type=bool)
parser.add_argument("--width", type=float, default=18.0)
parser.add_argument("--height", type=float, default=2.5)
parser.add_argument("--wspace", type=float, default=None)
parser.add_argument("--hspace", type=float, default=0.4)
parser.add_argument("--draw_mode", type=str, default='single_alg')
parser.add_argument("--clear_plot", type=bool,default=False)
parser.add_argument("--ex_name_list", type=str)
config = parser.parse_args()
run_num_list = None
if config.run_num is not None:
    config.run_num = eval(config.run_num)
    run_num_list = config.run_num
frame = config.frame
check_end = config.check_end
game_mode = config.game_mode
sample_num = config.sample_num
model_name = config.model_name
single_run = config.single_run
ex_name_list = config.ex_name_list.split(',')
print(config)
draw_name_map = {}
save_name_map = {}
# alg_name_list = ['COMA','COMA-DIV']

if frame == 'MAAC':
    alg_name_list = ['MAAC', 'MAAC-DIV']
elif frame == 'MAAC-share':
    alg_name_list = ['MAAC-share', 'MAAC-share-DIV']
elif frame == 'COMA':
    alg_name_list = ['COMA', 'COMA-DIV', 'COMA-SAC']
elif frame == 'QMIX':
    alg_name_list = ['QMIX', 'QMIX-DIV', 'QMIX-SAC']
elif frame == 'MAPPO':
    alg_name_list = ['mappo']
frame_alg_dict = {}
# frame_alg_dict['MAAC-share'] = ['MAAC-share', 'MAAC-share-DIV']
# frame_alg_dict['COMA'] = ['COMA', 'COMA-DIV', 'COMA-SAC']
# frame_alg_dict['QMIX'] = ['QMIX', 'QMIX-DIV', 'QMIX-SAC']
frame_alg_dict['DOP'] = ['DOP', 'DOP+DMAC']
frame_list = ['DOP']
prefix_list = ['origin_1M_1_episode_1_update_offpg_smac','mode1_mix_sample_1M_1_episode_4_update_offpg_dmac_smac']

alg_to_color = {}
alg_to_color['DOP+DMAC'] = 'r'
alg_to_color['DOP'] = 'b'
alg_to_color['MAAC-share-DIV'] = 'r'
alg_to_color['MAAC-share'] = 'b'
alg_to_color['COMA-DIV'] = 'r'
alg_to_color['COMA'] = 'b'
alg_to_color['COMA-SAC'] = 'g'
alg_to_color['QMIX-DIV'] = 'r'
alg_to_color['QMIX'] = 'b'
alg_to_color['QMIX-SAC'] = 'g'

alg_draw_name = {}
alg_from_list = ['DOP','DOP+DMAC','MAAC', 'MAAC-DIV', 'MAAC-share', 'MAAC-share-DIV', 'COMA', 'COMA-DIV', 'COMA-SAC', 'QMIX', 'QMIX-DIV',
                 'QMIX-SAC']
alg_to_list = ['DOP','DOP+DMAC','MAAC', 'MAAC+DMAC', 'MAAC', 'MAAC+DMAC', 'COMA', 'COMA+DMAC', 'COMA+ENT', 'QMIX', 'QMIX+DMAC',
               'QMIX+ENT']
for a, b in zip(alg_from_list, alg_to_list):
    alg_draw_name[a] = b
if frame == 'MAPPO':
    env_id_list = [config.env_id]
else:
    env_id_list = ['3m','2s3z','8m','1c3s5z']
# fig=plt.figure(figsize=(6,4))

critical_value = 1.96


def get_data(model_dir, sample_num,prefix = None):
    if prefix is not None:
        exist_run_nums = [str(folder.name) for folder in
                          model_dir.iterdir() if
                          str(folder.name).startswith(prefix)]
    else:

        check = [str(folder.name).split('run') for folder in
                 model_dir.iterdir() if
                 str(folder.name).startswith('run')]
        print('check = {}'.format(check))
        exist_run_nums = sorted([int(str(folder.name).split('run')[1]) for folder in
                          model_dir.iterdir() if
                          str(folder.name).startswith('run')])


    ret_steps = {}
    ret_performance = {}
    # key = 'test_battle_won_mean'
    min_length = 1000000000
    total_run_num = len(exist_run_nums)

    eg_run_name = 'run{}'.format(exist_run_nums[total_run_num - 1])
    eg_log_dir = model_dir / eg_run_name / 'logs'
    print('exist_run_nums = {}'.format(exist_run_nums))
    print('eg_log_dir = {}'.format(eg_log_dir))
    # final_key = [str(folder.name) for folder in eg_log_dir.iterdir() if not
    #                       os.path.isdir(folder.name)]
    final_key = []
    name_list = os.listdir(eg_log_dir)
    for f in name_list:
        print('f = {}, is file = {}, is dir = {}'.format(f,os.path.isfile(eg_log_dir/f),os.path.isdir(eg_log_dir/f) ))
        if os.path.isdir(eg_log_dir/f) and not str(f).startswith('agent'):
            final_key.append(f)

    print('final_key = {}'.format(final_key))
    for key in final_key:
        for i in range(sample_num):
            run_name = 'run{}'.format(exist_run_nums[total_run_num - 1 - i])
            print('get_data run_name = {}'.format(run_name))
            log_dir = model_dir / run_name / 'logs/{}/{}'.format(key,key)
            print('log_dir = {}'.format(log_dir))
            for name in os.listdir(log_dir):
                if name.startswith('events'):
                    event_name = name
                    break

            ea = event_accumulator.EventAccumulator(str(log_dir / event_name))
            ea.Reload()


            print('load data for ::\nmodel_dir = {}\nsample_idx = {}\nkey = {}'.format(model_dir, i, key))
            performance = ea.scalars.Items(key)
            if key == 'test_battle_won_mean':
                print('in model_dir: {}\nin run_name {} eval_reward_length = {}'.format(model_dir, run_name,len(performance)))
            run_steps = [i.step for i in performance]
            run_val = [i.value for i in performance]
            min_length = min(min_length, len(run_val))
            new_key = key.replace('/', '_')
            # final_key.append(new_key)
            if new_key not in ret_steps:
                ret_steps[new_key] = run_steps
            if new_key not in ret_performance:
                ret_performance[new_key] = []
            # print('ret_performance = {}'.format(ret_performance))
            ret_performance[new_key].append(run_val)

        for i in range(len(ret_performance[key])):
            ret_performance[key][i] = ret_performance[key][i][0:min_length]
        ret_steps[key] = ret_steps[key][0:min_length]
        for key in ret_performance:
            ret_performance[key] = np.array(ret_performance[key])
    return ret_performance, ret_steps, final_key


def draw():
    dir_name = frame + '_' + model_name
    if not Path(dir_name).exists():
        os.mkdir(dir_name)
    for count, env_id in enumerate(env_id_list):

        steps = {}
        performance_val = {}
        all_key = {}
        sub_plot = plt
        for a_id, alg_name in enumerate(alg_name_list):

            tmp_model_name = model_name
            model_dir = Path('./' + alg_name + '/models') / env_id / tmp_model_name
            print('model_dir = {}'.format(model_dir))
            try:
                if run_num_list is not None:
                    performance_val[alg_name], steps[alg_name], all_key[alg_name] = get_data(model_dir, sample_num,
                                                                                             run_num_list[a_id])
                else:
                    performance_val[alg_name], steps[alg_name], all_key[alg_name] = get_data(model_dir, sample_num)
            #
            except:
                continue

        all_key_eg = all_key[alg_name_list[1]]

        if single_run:
            for sample_idx in range(sample_num):
                single_dir_name = dir_name + '/run{}'.format(sample_idx + 1)
                if not Path(single_dir_name).exists():
                    os.mkdir(single_dir_name)
                for key in all_key_eg:
                    # plt.figure(figsize=(6.0, 4.0))
                    print('draw_plot for {} in run {}'.format(key, sample_idx + 1))
                    if key.startswith('agent0'):
                        name_key = 'rewards'
                    else:
                        name_key = key
                    for alg_name in performance_val:
                        if key not in performance_val[alg_name].keys() or key != 'eval_reward':
                            continue
                        performance_val[alg_name][key] = np.array(performance_val[alg_name][key])
                        mean_val = performance_val[alg_name][key][sample_idx]
                        std_val = np.zeros_like(performance_val[alg_name][key][sample_idx])
                        bound_val = critical_value * std_val / sample_num
                        # print('bound_val = {}'.format(bound_val))
                        run_step = steps[alg_name][key]

                        # length = len(mean_val)
                        # idx = range(0, length, 10)
                        # mean_val, std_val, bound_val, run_step = np.array(mean_val), np.array(std_val), np.array(
                        #     bound_val), np.array(run_step)
                        # if key != "eval_reward":
                        #     mean_val, std_val, bound_val, run_step = mean_val[idx], std_val[idx], bound_val[idx], \
                        #                                              run_step[
                        #                                                  idx]

                        sub_plot.title(env_id)
                        sub_plot.xlabel('episode')
                        sub_plot.ylabel(name_key)
                        sub_plot.plot(run_step, mean_val, label=alg_draw_name[alg_name], alpha=0.5)
                        sub_plot.fill_between(run_step, mean_val - bound_val, mean_val + bound_val, alpha=0.3)
                        plt.legend(loc='lower right')

                    print('name_key = {}'.format(name_key))
                    # print('run_step = {}'.format(run_step))

                    plt.savefig(
                        single_dir_name + '/' + frame + '_' + env_id + '_' + name_key + '_result.png',
                        dpi=300)
                    plt.clf()
        else:
            for key in all_key_eg:
                # plt.figure(figsize=(6.0, 4.0))
                print('draw_plot for {}'.format(key))
                if key.startswith('agent0'):
                    name_key = 'rewards'
                else:
                    name_key = key
                for alg_name in performance_val:
                    if key not in performance_val[alg_name].keys():
                        continue
                    performance_val[alg_name][key] = np.array(performance_val[alg_name][key])
                    mean_val = performance_val[alg_name][key].mean(axis=0)
                    std_val = performance_val[alg_name][key].std(axis=0)
                    bound_val = critical_value * std_val / sample_num
                    # print('bound_val = {}'.format(bound_val))
                    run_step = steps[alg_name][key]

                    # length = len(mean_val)
                    # idx = range(0, length, 10)
                    # mean_val, std_val, bound_val, run_step = np.array(mean_val), np.array(std_val), np.array(
                    #     bound_val), np.array(run_step)
                    # if key != "eval_reward":
                    #     mean_val, std_val, bound_val, run_step = mean_val[idx], std_val[idx], bound_val[idx], run_step[
                    #         idx]

                    sub_plot.title(env_id)
                    sub_plot.xlabel('episode')
                    sub_plot.ylabel(name_key)
                    sub_plot.plot(run_step, mean_val, label=alg_draw_name[alg_name], alpha=0.5)
                    sub_plot.fill_between(run_step, mean_val - bound_val, mean_val + bound_val, alpha=0.3)
                    plt.legend(loc='lower right')

                print('name_key = {}'.format(name_key))
                # print('run_step = {}'.format(run_step))

                plt.savefig(dir_name + '/' + frame + '_' + env_id + '_' + name_key + '_result.png',
                            dpi=300)
                plt.clf()


def single_alg_draw():
    dir_name = frame + '_' + model_name
    if not Path(dir_name).exists():
        os.mkdir(dir_name)
    for count, env_id in enumerate(env_id_list):

        steps = {}
        performance_val = {}
        all_key = {}
        sub_plot = plt
        for a_id, alg_name in enumerate(alg_name_list):
            performance_val[alg_name], steps[alg_name], all_key[alg_name] = {}, {} , {}
            for ex_id,ex_name in enumerate(ex_name_list):
                if config.env_type == 'sc2':
                    model_dir = Path('./StarCraft2/') / env_id / alg_name / ex_name
                elif config.env_type == 'MPE':
                    model_dir = Path('./MPE/') / env_id / alg_name / ex_name
                elif config.env_type == 'matrix_game':
                    model_dir = Path('./matrix_game/') / env_id / alg_name / ex_name
                else:
                    model_dir = Path('./{}/'.format(config.env_type)) / env_id / alg_name / ex_name
                print('model_dir = {}'.format(model_dir))
                performance_val[alg_name][ex_name], steps[alg_name][ex_name], all_key[alg_name][ex_name] = get_data(model_dir, sample_num)

        print('all_key.key = {}'.format(all_key.keys()))
        print('all_key[{}].key = {}'.format(alg_name_list[0],all_key[alg_name_list[0]].keys()))
        all_key_eg = all_key[alg_name_list[0]][ex_name_list[0]]
        print('all_key_eg = {}'.format(all_key_eg))
        if single_run:
            print('fuck')
            for sample_idx in range(sample_num):
                single_dir_name = dir_name + '/run{}'.format(sample_idx + 1)
                if not Path(single_dir_name).exists():
                    os.mkdir(single_dir_name)
                for key in all_key_eg:
                    # plt.figure(figsize=(6.0, 4.0))
                    print('draw_plot for {} in run {}'.format(key, sample_idx + 1))
                    if key.startswith('agent0'):
                        name_key = 'rewards'
                    else:
                        name_key = key
                    for alg_name in performance_val:
                        for ex_name in performance_val[alg_name]:
                            if key not in performance_val[alg_name][ex_name].keys() or key != 'eval_reward':
                                continue
                            performance_val[alg_name][ex_name][key] = np.array(performance_val[alg_name][ex_name][key])
                            mean_val = performance_val[alg_name][ex_name][key][sample_idx]
                            std_val = np.zeros_like(performance_val[alg_name][ex_name][key][sample_idx])
                            bound_val = critical_value * std_val / sample_num
                            # print('bound_val = {}'.format(bound_val))
                            run_step = steps[alg_name][ex_name][key]

                            length = len(mean_val)
                            idx = range(0, length, 10)
                            mean_val, std_val, bound_val, run_step = np.array(mean_val), np.array(std_val), np.array(
                                bound_val), np.array(run_step)
                            if config.clear_plot:
                                mean_val, std_val, bound_val, run_step = mean_val[idx], std_val[idx], bound_val[idx], \
                                                                         run_step[
                                                                             idx]

                            sub_plot.title(env_id)
                            sub_plot.xlabel('episode')
                            sub_plot.ylabel(name_key)
                            sub_plot.plot(run_step, mean_val, label=ex_name, alpha=0.5)
                            sub_plot.fill_between(run_step, mean_val - bound_val, mean_val + bound_val, alpha=0.3)
                            plt.legend(loc='lower right')

                        print('name_key = {}'.format(name_key))
                        # print('run_step = {}'.format(run_step))

                        plt.savefig(
                            single_dir_name + '/' + frame +'_{}_{}'.format(model_name,alg_name) + '_' + env_id + '_' + name_key + '_result.png',
                            dpi=300)
                        plt.clf()
        else:
            for key in all_key_eg:
                # plt.figure(figsize=(6.0, 4.0))
                print('draw_plot for {}'.format(key))
                if key.startswith('agent0'):
                    name_key = 'rewards'
                else:
                    name_key = key
                for alg_name in performance_val:
                    for ex_name in performance_val[alg_name]:
                        print('key = {}, performance_val[{}][{}].keys = {}'.format(key,alg_name,ex_name,performance_val[alg_name][ex_name].keys()))
                        if key not in performance_val[alg_name][ex_name].keys():
                            continue
                        performance_val[alg_name][ex_name][key] = np.array(performance_val[alg_name][ex_name][key])
                        mean_val = performance_val[alg_name][ex_name][key].mean(axis=0)
                        std_val = performance_val[alg_name][ex_name][key].std(axis=0)
                        bound_val = critical_value * std_val / sample_num
                        # print('bound_val = {}'.format(bound_val))
                        run_step = steps[alg_name][ex_name][key]

                        length = len(mean_val)
                        idx = range(0, length, 10)
                        mean_val, std_val, bound_val, run_step = np.array(mean_val), np.array(std_val), np.array(
                            bound_val), np.array(run_step)
                        if config.clear_plot:
                            mean_val, std_val, bound_val, run_step = mean_val[idx], std_val[idx], bound_val[idx], run_step[
                                idx]

                        sub_plot.title(env_id)
                        sub_plot.xlabel('episode')
                        sub_plot.ylabel(name_key)
                        sub_plot.plot(run_step, mean_val, label=ex_name, alpha=0.5)
                        sub_plot.fill_between(run_step, mean_val - bound_val, mean_val + bound_val, alpha=0.3)
                        plt.legend(loc='lower right')

                    print('name_key = {}'.format(name_key))
                    # print('run_step = {}'.format(run_step))

                    plt.savefig(dir_name + '/' + frame +'_{}_{}'.format(model_name,alg_name) + env_id + '_' + name_key + '_result.png',
                                dpi=300)
                    plt.clf()


def draw_multi_plot():
    dir_name = 'All_plot' + '_' + model_name
    if not Path(dir_name).exists():
        os.mkdir(dir_name)
    myfontsize = 8
    plt.rcParams.update({"font.size": myfontsize})
    # plt.rcParams['figure.figsize'] = (4.0, 8.0)
    # plt.rcParams['savefig.dpi'] = 300  # 图片像素
    # plt.rcParams['figure.dpi'] = 300  # 分辨率
    plt.figure(figsize=(config.width, config.height), dpi=300)

    fig = plt.figure(1)
    fig.subplots_adjust(wspace=config.wspace, hspace=config.hspace)
    steps = {}
    performance_val = {}
    all_key = {}
    for count, env_id in enumerate(env_id_list):
        steps[env_id] = {}
        performance_val[env_id] = {}
        all_key[env_id] = {}
        for f_id, frame in enumerate(frame_list):
            alg_name_list = frame_alg_dict[frame]
            steps[env_id][frame] = {}
            performance_val[env_id][frame] = {}
            all_key[env_id][frame] = {}
            # frame_run_num_list = run_num_list[f_id]
            for a_id, alg_name in enumerate(alg_name_list):
                # tmp_model_name = model_name
                model_dir = Path('./' + 'results' + '/final_logs')
                prefix = prefix_list[a_id]
                prefix = '{}_{}'.format(prefix,env_id)
                print('model_dir = {}, prefix = {}'.format(model_dir,prefix))
                performance_val[env_id][frame][alg_name], steps[env_id][frame][alg_name], \
                all_key[env_id][frame][alg_name] = get_data(model_dir, sample_num, prefix)

    key = 'test_battle_won_mean'


    for f_id, frame in enumerate(frame_list):
        for count, env_id in enumerate(env_id_list):
            # plot_num = len(frame_list) * count + f_id + 1
            plot_num = len(env_id_list) * f_id + count + 1
            ax = plt.subplot(len(frame_list),len(env_id_list),  plot_num)
            for alg_name in performance_val[env_id][frame]:
                if key not in performance_val[env_id][frame][alg_name].keys():
                    continue
                performance_val[env_id][frame][alg_name][key] = np.array(performance_val[env_id][frame][alg_name][key])
                mean_val = performance_val[env_id][frame][alg_name][key].mean(axis=0)
                std_val = performance_val[env_id][frame][alg_name][key].std(axis=0)
                bound_val = critical_value * std_val / sample_num
                # print('bound_val = {}'.format(bound_val))
                run_step = np.arange(len(mean_val)) * 20000

                length = len(mean_val)
                idx = range(0, length, 10)
                mean_val, std_val, bound_val, run_step = np.array(mean_val), np.array(std_val), np.array(
                    bound_val), np.array(run_step)

                # print('draw_name_map = {}'.format(draw_name_map))
                print('env_id = {}, frame = {}, alg_name = {}, plot_num = {}'.format(env_id, frame, alg_name, plot_num))

                # 设置坐标刻度字体大小
                # plt.ylim((-1, 0))
                plt.xticks(fontsize=myfontsize)
                plt.yticks(fontsize=myfontsize)
                plt.title(env_id, fontsize=myfontsize)


                plt.xlabel('step', fontsize=myfontsize)
                if plot_num % len(env_id_list) == 1:
                    plt.ylabel(key, fontsize=myfontsize)

                plt.plot(run_step, mean_val, label=alg_draw_name[alg_name], alpha=0.5, color=alg_to_color[alg_name])
                plt.fill_between(run_step, mean_val - bound_val, mean_val + bound_val, alpha=0.3,
                                 color=alg_to_color[alg_name])
                # plt.legend(loc='lower right')
                if plot_num <= len(env_id_list):
                    plt.legend(loc='lower right', ncol=3, frameon=False, borderpad=0, framealpha=0.2)

            print('name_key = {}'.format(key))
            # print('run_step = {}'.format(run_step))

    plt.savefig(dir_name + '/' + 'DOP' + '_' + 'eval_reward' + '_result.pdf', dpi=300, bbox_inches='tight',
                pad_inches=0)
    plt.clf()



if config.draw_mode == 'multi_plot':
    draw_multi_plot()
elif config.draw_mode == 'single_alg':
    single_alg_draw()
