import numpy as np
import torch
import random
# from lib.utils.torch_deform import deform_grid
import cv2 as cv
import numpy
import torch
import elasticdeform
def gasuss_noise(img,mean=0, std=0.001):
    '''
        添加高斯噪声
        mean : 均值
        std : 标准差
    '''

    # image = np.array(image / 255, dtype=float)
    #noise = np.random.normal(mean, var ** 0.5, image1.shape)
    #noise = torch.Tensor(noise).cuda()
    noise = torch.normal(mean,std,size=img.size()).cuda()
    return torch.clamp(img + noise,0,1.0)


def flip(img,flipCode=None):
    '''
        img: input ,tensor
        flipCode: flip or not
    '''
    if flipCode is None:
        flipCode = random.choice([ 0, 2, 3, 4])
    if flipCode > 0:
        if flipCode==4:
            img = torch.flip(img, [2,3])
        else:
            img = torch.flip(img,[flipCode])
    return img, flipCode

def deform(img, displacements=None, rotates=None, zooms=None):
    n,ch,h,w=img.shape
    if displacements is None:
        displacements = []
    if rotates is None:
        rotates = []
    if zooms is None:
        zooms = []
    imgnew = []
    for i in range(n):
        imgtmp = img[i]
        if  len(displacements) < n :
            num = random.random() * 25 + 1
            displacement = np.random.randn(2,3,3) * num
            displacements.append(displacement)
        else:
            displacement = displacements[i]
        if len(rotates) < n :
            rotate = np.random.uniform(0, 60)
            rotates.append(rotate)
        else:
            rotate = rotates[i]
        if len(zooms) < n :
            zoom = np.random.uniform(1, 2)
            zooms.append(zoom)
        else:
            zoom = zooms[i]
        imgnewtmp = deform_grid(imgtmp, torch.Tensor(displacement), order=3, mode='nearest', rotate=rotate, zoom=zoom, axis=(1,2))
        imgnew.append(imgnewtmp)
    return torch.stack(imgnew,0), displacements, rotates, zooms



class ElasticDeform(torch.autograd.Function):
    @staticmethod
    def forward(ctx, displacement, deform_args, deform_kwargs, *xs):
        ctx.save_for_backward(displacement)
        ctx.deform_args = deform_args
        ctx.deform_kwargs = deform_kwargs
        ctx.x_shapes = [x.shape for x in xs]

        xs_numpy = [x.detach().cpu().numpy() for x in xs]
        displacement = displacement.detach().cpu().numpy()
        ys = elasticdeform.deform_grid(xs_numpy, displacement, *deform_args, **deform_kwargs)
        return tuple(torch.tensor(y, device=x.device) for x, y in zip(xs, ys))

    @staticmethod
    def backward(ctx, *dys):
        displacement, = ctx.saved_tensors
        deform_args = ctx.deform_args
        deform_kwargs = ctx.deform_kwargs
        x_shapes = ctx.x_shapes

        dys_numpy = [dy.detach().cpu().numpy() for dy in dys]
        displacement = displacement.detach().cpu().numpy()
        dxs = elasticdeform.deform_grid_gradient(dys_numpy, displacement,
                                                 *deform_args, X_shape=x_shapes, **deform_kwargs)
        return (None, None, None) + tuple(torch.tensor(dx, device=dy.device) for dx, dy in zip(dxs, dys))



def deform_grid(X, displacement, *args, **kwargs):
    """
    Elastic deformation with a deformation grid, wrapped for PyTorch.

    This function wraps the ``elasticdeform.deform_grid`` function in a PyTorch function
    with a custom gradient.

    Parameters
    ----------
    X : torch.Tensor or list of torch.Tensors
        input image or list of input images
    displacement : torch.Tensor
        displacement vectors for each control point

    Returns
    -------
    torch.Tensor
       the deformed image, or a list of deformed images

    See Also
    --------
    elasticdeform.deform_grid : for the other parameters
    """
    if not isinstance(X, (list, tuple)):
        X_list = [X]
    else:
        X_list = X
    displacement = torch.as_tensor(displacement)
    y = ElasticDeform.apply(displacement, args, kwargs, *X_list)

    if isinstance(X, (list, tuple)):
        return y
    else:
        return y[0]


