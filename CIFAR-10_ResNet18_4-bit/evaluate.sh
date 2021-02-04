PYTHON="/home/jmeng15/anaconda3/bin/python3"

############ directory to save result #############

if [ ! -d "$DIRECTORY" ]; then
    mkdir ./save
    mkdir ./dataset
fi

model=resnet18_quant_tanh
dataset=cifar10
epochs=200
batch_size=128
optimizer=SGD
group_ch=16
wbit=4
abit=4
mode=sawb
k=2
ratio=0.7
wd=0.0005
lr=0.1
WhichBit=clamp
q_file=$4

save_path="./save/${model}/${model}_w${wbit}_a${abit}_mode_${mode}_k${k}_lambda${ub}_wd${wd}_swpFalse_${WhichBit}/"
log_file="${model}_w${wbit}_a${abit}_mode${mode}_k${k}_lambda${ub}_wd${wd}_swpFalse_${WhichBit}.log"
pretrained_model="./save/resnet18_quant_tanh/resnet18_quant_tanh_w${wbit}_a${abit}_mode_mean_k2_lambda_wd0.0005_swpFalse_${WhichBit}/model_best.pth.tar"
quant_file="../mat/${q_file}.mat"

$PYTHON -W ignore train.py --dataset ${dataset} \
    --data_path ./dataset/ \
    --model ${model} \
    --save_path ${save_path} \
    --epochs ${epochs} \
    --log_file ${log_file} \
    --lr  ${lr} \
    --schedule 60 120 \
    --gammas 0.1 0.1 \
    --batch_size ${batch_size} \
    --ngpu 1 \
    --wd ${wd} \
    --k ${k} \
    --group_ch ${group_ch} \
    --wbit ${wbit} \
    --abit ${abit} \
    --fine_tune \
    --eval \
    --resume ${pretrained_model} \
    --offset_noise $2 \
    --bitline_noise $3 \
    --quant_file ${quant_file} \
    --sram_depth $1 \

    
