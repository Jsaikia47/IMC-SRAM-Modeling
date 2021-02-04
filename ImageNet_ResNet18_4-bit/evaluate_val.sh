export CUDA_VISIBLE_DEVICES=4; python -W ignore imagenet_val.py --arch ResNet_multibit --data ~/imagenet/ --batch_size 128 --evaluate ./checkpoint/resnet18_imagenet_quant_w4_a4_mode_mean_k4_lambda_wd0.0001_swpFalse/model_best.pth.tar 
#--sd 256 --qb 60 --pt probs/xnor_sram/prob_0p6_0p3_linear_chip4.mat --n9std 2.8 --n9mean -0.8;
# python main.py --arch ResNet18_a2_w2 --evaluate checkpoint/resnet18_a2_w2_prob.pth --sd 256 --qb 60
