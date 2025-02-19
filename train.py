import argparse
import functools
import os

from masr.trainer import MASRTrainer
from masr.utils.utils import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('configs',          str,    'configs/conformer.yml',       '配置文件')
add_arg("local_rank",       int,    0,                             '多卡训练的本地GPU')
add_arg("use_gpu",          bool,   True,                          '是否使用GPU训练')
add_arg('augment_conf_path',str,    'configs/augmentation.json',   '数据增强的配置文件，为json格式')
add_arg('save_model_path',  str,    'models/',                  '模型保存的路径')
add_arg('resume_model',     str,    None,                       '恢复训练，当为None则不使用预训练模型')
add_arg('pretrained_model', str,    None,                       '预训练模型的路径，当为None则不使用预训练模型')
add_arg('verbose',          bool,   True,                       '是否打印详细信息')
add_arg('save_interval',    int,    100,                          '保存模型的间隔（单位：epoch）')
args = parser.parse_args()

if int(os.environ.get('LOCAL_RANK', 0)) == 0 and args.verbose:
    print_arguments(args=args)

# 获取训练器
trainer = MASRTrainer(configs=args.configs, use_gpu=args.use_gpu)

# 假设训练总轮数已知
num_epochs = 101
for epoch in range(num_epochs):
    if epoch % args.save_interval == 0:
        # 每到保存间隔时，调用保存模型的方法
        trainer.save_model(args.save_model_path, epoch)
    # 调用 train 方法进行单轮训练
    # 这里需要根据 MASRTrainer 类的实际情况修改 train 方法的调用逻辑
    trainer.train(save_model_path=args.save_model_path,
                  resume_model=args.resume_model,
                  pretrained_model=args.pretrained_model,
                  augment_conf_path=args.augment_conf_path,
                  num_epochs=100)  # 假设 train 方法支持 num_epochs 参数，这里设置为 1 进行单轮训练
