python train_GSD-Net.py --exp GSD_net  --gpu 0 --datasets kvasir --noise_type 02
python train_GSD-Net.py --exp GSD_net  --gpu 0 --datasets kvasir --noise_type 08
python train_GSD-Net.py --exp GSD_net  --gpu 0 --datasets kvasir --noise_type DE

python train_GSD-Net.py --exp GSD_net  --gpu 0 --datasets Shenzhen --noise_type 02
python train_GSD-Net.py --exp GSD_net  --gpu 0 --datasets Shenzhen --noise_type 08 --ratecp 0.1
python train_GSD-Net.py --exp GSD_net  --gpu 0 --datasets Shenzhen --noise_type DE

python train_GSD-Net.py --exp GSD_net  --gpu 0 --datasets BUSUC --noise_type 02
python train_GSD-Net.py --exp GSD_net  --gpu 0 --datasets BUSUC --noise_type 08 --ratecp 0.1
python train_GSD-Net.py --exp GSD_net  --gpu 0 --datasets BUSUC --noise_type DE