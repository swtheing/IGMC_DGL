#!/usr/bin/env bash

# python3 train_sampling.py --data_name=ml-100k \
#                           --use_one_hot_fea \
#                           --gcn_agg_accum=stack \
#                           --gpu -1

# python3 train_flixster.py --data_name=flixster \
#                           --use_one_hot_fea \
#                           --gcn_agg_accum=stack \
#                           --train_lr=0.02 \
#                           --train_min_lr=0.000001 \
#                           --minibatch_size=128 \
#                           --train_max_epoch=1000000 \
#                           --gpu -1 

# nohup python3 -u train_flixster.py --data_name=flixster \
#                           --use_one_hot_fea \
#                           --gcn_agg_accum=stack \
#                           --train_lr=0.02 \
#                           --train_min_lr=0.000001 \
#                           --minibatch_size=128 \
#                           --train_max_epoch=1000000 \
#                           --gpu -1 \
#  >1  2>&1 &

# nohup python3 -u train_flixster.py --data_name=flixster \
#                       --use_one_hot_fea \
#                       --gcn_agg_accum=stack \
#                       --gpu -1 \
# >log03  2>&1 &



# nohup python3 -u train_custom.py --data_name=douban \
#                       --use_one_hot_fea \
#                       --gcn_agg_accum=stack \
#                       --gpu -1 \
# >douban.log01  2>&1 &


# nohup python3 -u train_custom.py --data_name=yahoo_music \
#                       --use_one_hot_fea \
#                       --gcn_agg_accum=stack \
#                       --gpu -1 \
# >yahoo.log01  2>&1 &


nohup python3 -u train_all_custom.py --data_name=flixster \
                      --use_one_hot_fea \
                      --gcn_agg_accum=sum \
                      --device -1 \
>flixster.log.2.sum  2>&1 &

nohup python3 -u train_all_custom.py --data_name=douban \
                      --use_one_hot_fea \
                      --gcn_agg_accum=sum \
                      --device -1 \
>douban.log.2.sum  2>&1 &

# nohup python3 -u train_all_custom.py --data_name=yahoo_music \
#                       --use_one_hot_fea \
#                       --gcn_agg_accum=sum \
#                       --gcn_agg_units=1700 \
#                       --data_valid_ratio=0.1 \
#                       --device -1 \
# >yahoo.log.1.sum  2>&1 &

# nohup python3 -u train_all_custom.py --data_name=yahoo_music \
#                       --use_one_hot_fea \
#                       --gcn_agg_accum=stack \
#                       --gcn_agg_units=1700 \
#                       --data_valid_ratio=0.1 \
#                       --device -1 \
# >yahoo.log.2.stack  2>&1 &
