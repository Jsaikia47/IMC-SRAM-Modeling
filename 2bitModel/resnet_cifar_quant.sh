############ directory to save result #############

if [ ! -d "$DIRECTORY" ]; then
    mkdir ./save
    mkdir ./dataset
fi

model=resnet18_quant_tanh
dataset=cifar10
epochs=10
batch_size=20
optimizer=SGD
group_ch=16
wbit=2
abit=2
mode=mean
k=2
ratio=0.7
wd=0.0005
lr=0.1
#q_file=$4

save_path="./save/${model}/${model}_w${wbit}_a${abit}_mode_${mode}_k${k}_lambda${ub}_wd${wd}_swpFalse_g02_Eval/"
log_file="${model}_w${wbit}_a${abit}_mode${mode}_k${k}_lambda${ub}_wd${wd}_swpFalse_Eval.log"
pretrained_model="./save/resnet18_quant_tanh/resnet18_quant_tanh_w2_a2_mode_mean_k2_lambda_wd0.0005_swpFalse_g02/model_best.pth.tar"

CUDA_LAUNCH_BLOCKING=1 python -W ignore train.py --dataset ${dataset} \
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
    --resume ${pretrained_model} \
    --fine_tune \
    --eval \
    --offset_noise $2 \
    --bitline_noise $3 \
    --sram_depth $1 \
    --quant_bound $4 \
    --number_levels $5
    
