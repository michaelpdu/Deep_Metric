#!/usr/bin/env bash
DATA=cub
DATA_ROOT=data
Gallery_eq_Query=True
LOSS=LiftedStructure
CHECKPOINTS=ckps
R=.pth.tar

if_exist_mkdir ()
{
    dirname=$1
    if [ ! -d "$dirname" ]; then
    mkdir $dirname
    fi
}

if_exist_mkdir ${CHECKPOINTS}
if_exist_mkdir ${CHECKPOINTS}/${LOSS}
if_exist_mkdir ${CHECKPOINTS}/${LOSS}/${DATA}

if_exist_mkdir result
if_exist_mkdir result/${LOSS}
if_exist_mkdir result/${LOSS}/${DATA}

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
--data_root ${DATA_ROOT} \
--init random \
--lr $LR \
--dim $DIM \
--alpha $ALPHA \
--num_instances   5 \
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
    CUDA_VISIBLE_DEVICES=0 python test.py --net ${NET} \
    --data $DATA \
    --data_root ${DATA_ROOT} \
    --batch_size 100 \
    -g_eq_q ${Gallery_eq_Query} \
    --width 227 \
    -r ${SAVE_DIR}/ckp_ep$i$R \
    --pool_feature ${POOL_FEATURE:-'False'} \
    | tee -a result/$LOSS/$DATA/${NET}-DIM-$DIM-Batchsize-${BatchSize}-ratio-${RATIO}-lr-$LR${POOL_FEATURE:+'-pool_feature'}.txt

done


