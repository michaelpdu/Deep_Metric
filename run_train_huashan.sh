#!/usr/bin/env bash

echo $data_dir
echo $bn_inception
echo $model_save_dir
echo $log_dir

DATA=cub
Gallery_eq_Query=True
LOSS=LiftedStructure
CHECKPOINTS=ckps
R=.pth.tar
NET=BN-Inception
DIM=512
ALPHA=40
LR=1e-5
BatchSize=80
RATIO=0.16

SAVE_DIR=${CHECKPOINTS}/${LOSS}/${DATA}/${NET}-DIM-${DIM}-lr${LR}-ratio-${RATIO}-BatchSize-${BatchSize}
if_exist_mkdir ${SAVE_DIR}


# if [ ! -n "$1" ] ;then
echo "Begin Training!"
# CUDA_VISIBLE_DEVICES=0
python train.py --net ${NET} \
--data $DATA \
--data_dir $data_dir \
--pretrained_model_dir $bn_inception \
--lr $LR \
--dim $DIM \
--alpha $ALPHA \
--num_instances 5 \
--batch_size ${BatchSize} \
--epoch 600 \
--loss $LOSS \
--width 227 \
--save_dir ${SAVE_DIR} \
--save_step 50 \
--ratio ${RATIO} 

echo "Begin Testing!"

Model_LIST=`seq  50 50 600`
for i in $Model_LIST; do
    # CUDA_VISIBLE_DEVICES=0
    python test.py --net ${NET} \
    --data $DATA \
    --batch_size 100 \
    -g_eq_q ${Gallery_eq_Query} \
    --width 227 \
    -r ${SAVE_DIR}/ckp_ep$i$R \
    --pool_feature ${POOL_FEATURE:-'False'} \
    | tee -a result/$LOSS/$DATA/${NET}-DIM-$DIM-Batchsize-${BatchSize}-ratio-${RATIO}-lr-$LR${POOL_FEATURE:+'-pool_feature'}.txt

done


