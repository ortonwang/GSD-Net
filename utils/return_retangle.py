import torch


def return_retangle(image):
    # output_tensor = torch.zeros_like(input_tensor)

    # 遍历每张图像
    # for b in range(args.batch_size):
    #     image = input_tensor[b]  # 取出第 b 张图像

        # 找到目标区域的非零位置
    non_zero_indices = torch.nonzero(image)
    if non_zero_indices.shape[0] > 0:  # 如果有非零元素
        min_row, min_col = torch.min(non_zero_indices, dim=0).values
        max_row, max_col = torch.max(non_zero_indices, dim=0).values
    else:
        min_row,  max_row , min_col,max_col=0,0,255,255
        # 在对应的矩形区域内设置为 1
        # output_tensor[b, min_row:max_row + 1, min_col:max_col + 1] = 1

    return  min_row,  max_row , min_col,max_col