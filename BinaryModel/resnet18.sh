save_path="resnet18_binary_inflation1_1bit_act"
model=resnet_binary

python main_binary.py --model ${model} \
    --save ${save_path} \
    --dataset cifar10 \
    --depth 18
