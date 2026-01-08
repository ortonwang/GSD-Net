# For generete simulated noise
## use generate_simulated_noise.py scripts
### Attention:
### Require: make sure channels for clean ground truth is 1 channel
### Usage:

Our noise simulated code is based on the [SpatialCorrection](https://github.com/michaelofsbu/SpatialCorrection)
```
# For SR simulated noise:
python generate_simulated_noise.py --gts_root "Directory root of groundtruth masks" --save_root "Directory root for saving generated noisy masks" --T 200 --theta1 0.8  --theta2 0.05

# For SE simulated noise:
python generate_simulated_noise.py --gts_root "Directory root of groundtruth masks" --save_root "Directory root for saving generated noisy masks" --T 200 --theta1 0.2 --theta2 0.05

# For SDE simulated noise:
python generate_simulated_noise.py --gts_root "Directory root of groundtruth masks" --save_root "Directory root for saving generated noisy masks" --noisetype DE --range [9,11]
```
Here, --T represent the Markov process step number, 
      --theta1 represent the Bernoulli parameter controlling preference,
      --theta2 represent the Bernoulli parameter controlling preference.
For our study, the hyperparameters used for simulated noisy labels are shown in Table X. 
| Noise Setting | Kvasir             | Shenzhen           | BU_SUC             | BraTS2019        |
|---------------|------------------|------------------|------------------|----------------|
| S_R           | M(200,0.8,0.05)  | M(200,0.8,0.05)  | M(200,0.8,0.05)  | M(200,0.8,0.05) |
| S_E           | M(200,0.2,0.05)  | M(200,0.2,0.05)  | M(200,0.2,0.05)  | M(50,0.4,0.05)  |
| S_DE          | K(9-11)          | K(9-11)          | K(9-11)          | K(2-4)          |

**Notes:**  
- `M(T,P,V)`: Markov-based boundary perturbation strategy, with parameters T = step number, P = Bernoulli preference, V = Bernoulli variance.  
- `K(k1 - k2)`: Morphology-based erosion/dilation kernel, randomly varying between k1 and k2 for each sample.




# For generate superpixel Structural prior via SLIC 
```
python generate_SLIC.py 
```
