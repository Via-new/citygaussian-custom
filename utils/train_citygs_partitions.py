import os
import argparse
import yaml
import concurrent
import add_pypath
import subprocess
import traceback
import time
import selectors
import py3nvml
import numpy as np
import functools
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from auto_hyper_parameter import auto_hyper_parameter, to_command_args
from argparser_utils import split_stoppable_args, parser_stoppable_args
from internal.utils.general_utils import parse

def get_project_output_dir_by_name(project_name: str) -> str:
    return os.path.join(os.path.join(os.path.dirname(os.path.dirname(__file__))), "outputs", project_name)

def srun_output_dir(project_name: str) -> str:
    return os.path.join(get_project_output_dir_by_name(project_name), "srun-outputs")

def run_subprocess(args, output_redirect) -> int:
    sel = selectors.DefaultSelector()

    with subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as p:
        sel.register(p.stdout, selectors.EVENT_READ)
        sel.register(p.stderr, selectors.EVENT_READ)

        while True:
            if len(sel.get_map()) == 0:
                break

            events = sel.select()
            for key, mask in events:
                line = key.fileobj.readline()
                if len(line) == 0:
                    sel.unregister(key.fileobj)
                    continue
                output_redirect(line.decode("utf-8").rstrip("\n"))
        p.wait()
        return p.returncode

def train_a_partition(
        config_args,
        extra_training_args,
        srun_args,
        partition_idx,
        gpu_id,
    ):
    config_file = os.path.join(config_args.config_dir, f"{config_args.config_name}.yaml")
    project_name = config_args.project_name
    dry_run = config_args.dry_run

    # build args
    # basic
    args = [
        "python",
        "main.py", "fit",
        "--config", config_file,
        "--data.parser.block_id", str(partition_idx),
    ]

    # extra
    args += extra_training_args

    experiment_name = config_args.config_name
    args += [
        "-n={}".format(experiment_name),
        "--project", project_name,
        "--logger", "wandb",
    ]

    print_func = print
    run_func = functools.partial(subprocess.run, env=dict(**os.environ, CUDA_VISIBLE_DEVICES=str(gpu_id)))
    if len(srun_args) > 0:
        def tqdm_write(i):
            tqdm.write("[{}] #{}: {}".format(
                time.strftime('%Y-%m-%d %H:%M:%S'),
                partition_idx,
                i,
            ))

        def run_with_tqdm_write(args):
            return run_subprocess(args, tqdm_write)

        run_func = run_with_tqdm_write
        print_func = tqdm_write

        output_filename = os.path.join(srun_output_dir(config_args.config_name), "block_{}.txt".format(partition_idx))
        args = [
            "srun",
            "--output={}".format(output_filename),
            "--job-name={}-{}".format(config_args.project_name, experiment_name),
        ] + srun_args + args

    ret_code = -1
    if dry_run:
        print(" \\\n  ".join(args))
    else:
        try:
            print_func(str(args))
            ret_code = run_func(args)
        except KeyboardInterrupt as e:
            raise e
        except:
            traceback.print_exc()

    return partition_idx, ret_code


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", "-n", type=str, required=True)
    parser.add_argument("--config_dir", "-c", type=str, default="./configs")
    parser.add_argument("--project_name", "-p", type=str, required=True)
    parser.add_argument("--dry-run", action="store_true", default=False)

    args, training_and_srun_args = parser_stoppable_args(parser)
    training_args, srun_args = split_stoppable_args(training_and_srun_args)

    config_path = os.path.join(args.config_dir, f"{args.config_name}.yaml")
    with open(config_path, 'r') as f:
        config = parse(yaml.load(f, Loader=yaml.FullLoader))
    num_blocks = config.data.parser.init_args.block_dim[0] * config.data.parser.init_args.block_dim[1]

    # ########## 添加：定义需要训练的块 ##########
    trainable_blocks = [1, 4, 9, 10, 12]  # 仅训练这些块
    print(f"本次训练的目标块：{trainable_blocks}，共{len(trainable_blocks)}个块")  # 新增打印：显示训练目标

    if len(srun_args) == 0:
        with ProcessPoolExecutor(max_workers=len(trainable_blocks)) as executor:
            for block_id in trainable_blocks:
                print(f"\n开始处理块 {block_id} 的GPU分配...")  # 新增打印：块处理开始
                gpu_available = False
                fail_cnt = 0
                while not gpu_available:
                    # 获取当前空闲GPU列表（py3nvml返回1表示可用，0表示占用）
                    free_gpus = py3nvml.get_free_gpus()
                    # 新增打印：显示检测到的GPU状态
                    print(f"当前检测到的GPU空闲状态（1=可用, 0=占用）：{free_gpus}，总可用数：{sum(free_gpus)}")

                    if sum(free_gpus) > 0:
                        gpu_available = True
                        # 新增打印：找到可用GPU，准备分配
                        selected_gpu = np.argmax(free_gpus)
                        print(f"找到可用GPU！选中GPU {selected_gpu} 用于块 {block_id} 的训练")
                    elif fail_cnt > 90:
                        # 超过90次等待（3小时）则退出
                        print("No free GPUs available in 3 hour, exiting.")
                        exit()
                    else:
                        # 未找到可用GPU，继续等待
                        fail_cnt += 1
                        waited_time = fail_cnt * 2  # 已等待时间（分钟）
                        remaining_time = (90 - fail_cnt) * 2  # 剩余等待时间（分钟）
                        # 新增打印：详细等待信息
                        print(f"未检测到可用GPU！当前等待次数：{fail_cnt}/90，已等待{waited_time}分钟，剩余{remaining_time}分钟")
                        print("等待2分钟后重新检查...")
                        subprocess.run(["sleep", "120"])

                # 提交训练任务
                executor.submit(train_a_partition, args, training_args, srun_args, block_id, np.argmax(free_gpus))
                print(f"块 {block_id} 已提交训练任务，等待60秒后处理下一个块...")  # 新增打印：任务提交提示
                subprocess.run(["sleep", "60"])
    else:
        print("SLURM mode enabled")
        trainable_partition_idx_list = trainable_blocks
        total_trainable_partitions = len(trainable_partition_idx_list)
        print(f"SLURM模式：训练块列表 {trainable_partition_idx_list}，共{total_trainable_partitions}个块")  # 新增打印

        with ThreadPoolExecutor(max_workers=total_trainable_partitions) as tpe:
            futures = [tpe.submit(
                train_a_partition,
                args,
                training_args,
                srun_args,
                i,
            ) for i in trainable_partition_idx_list]
            finished_count = 0
            with tqdm(
                    concurrent.futures.as_completed(futures),
                    total=total_trainable_partitions,
                    miniters=1,
                    mininterval=0,
                    maxinterval=0,
            ) as t:
                for future in t:
                    finished_count += 1
                    try:
                        finished_idx, ret_code = future.result()
                    except KeyboardInterrupt as e:
                        raise e
                    except:
                        traceback.print_exc()
                        continue
                    tqdm.write("[{}] #{} exited with code {} | {}/{}".format(
                        time.strftime('%Y-%m-%d %H:%M:%S'),
                        finished_idx,
                        ret_code,
                        finished_count,
                        total_trainable_partitions,
                    ))