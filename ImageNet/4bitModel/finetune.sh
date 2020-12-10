export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5; python -W ignore imagenet.py --arch ResNet_multibit --data ~/imagenet/ --batch_size 128 --resume ./checkpoint/resnet18_w2_a2_sawb_2bit_eval/model_best.pth.tar --sd 256 --qb 60 --lr 1e-3 --epochs 10 
# --n9std 2.8 --n9mean -0.8;
# python main.py --arch ResNet18_a2_w2 --evaluate checkpoint/resnet18_a2_w2_prob.pth --sd 256 --qb 60
