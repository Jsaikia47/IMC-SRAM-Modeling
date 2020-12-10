export CUDA_VISIBLE_DEVICES=0; python -W ignore imagenet.py --arch ResNet_multibit4bit --data /opt/imagenet/imagenet_compressed/ --batch_size 128 --evaluate ./checkpoint/resnet18_w4_a4_4bit_eval/model_best.pth.tar 
#--sd 256 --qb 60 --pt probs/xnor_sram/prob_0p6_0p3_linear_chip4.mat --n9std 2.8 --n9mean -0.8;
# python main.py --arch ResNet18_a2_w2 --evaluate checkpoint/resnet18_a2_w2_prob.pth --sd 256 --qb 60
