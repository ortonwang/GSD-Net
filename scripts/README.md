# For generete simulated noise
## use generate_simulated_noise.py scripts
### Require: make sure channels for clean ground truth is 1 channel
### Usage:

Our noise simulated code is based on the [SpatialCorrection](https://github.com/michaelofsbu/SpatialCorrection)
```
# For SR simulated noise:
python generate_simulated_noise.py --gts_root "Directory root of groundtruth masks" --save_root "Directory root for saving generated noisy masks" 

# For SE simulated noise:
python generate_simulated_noise.py --gts_root "Directory root of groundtruth masks" --save_root "Directory root for saving generated noisy masks" --theta1 0.2

# For SDE simulated noise:
python generate_simulated_noise.py --gts_root "Directory root of groundtruth masks" --save_root "Directory root for saving generated noisy masks" --noisetype DE
```

# For generate superpixel Structural prior via SLIC 
```
python generate_SLIC.py 
```