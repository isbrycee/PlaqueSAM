import numpy as np

def encode_mask_rle(mask):
    """
    高效实现 Run-Length Encoding (RLE) 对 2D numpy 数组的编码。
    
    Parameters:
        mask (numpy.ndarray): 2D numpy array 表示语义分割的 mask。
    
    Returns:
        dict: RLE 编码结果，包含:
              - 'shape': 原始数组形状
              - 'rle': 压缩后的 [值, 长度] 序列
    """
    # 将二维数组展平为一维
    flat_mask = mask.flatten()

    # 找到值变化的位置 (即不同值的分界点)
    diff_indices = np.where(flat_mask[1:] != flat_mask[:-1])[0] + 1

    # 计算每段的开始和结束位置
    starts = np.concatenate([[0], diff_indices])
    ends = np.concatenate([diff_indices, [len(flat_mask)]])

    # 计算每段的长度
    run_lengths = ends - starts
    values = flat_mask[starts]

    # 将值和长度交替存储到 RLE 编码
    encoded_rle = np.empty(2 * len(values), dtype=int)
    encoded_rle[0::2] = values
    encoded_rle[1::2] = run_lengths

    return {
        'shape': mask.shape,
        'rle': encoded_rle.tolist()  # 转为 Python 列表
    }

def decode_mask_rle(encoded):
    """
    高效解码 Run-Length Encoded (RLE) 数据，恢复为 2D numpy 数组。
    
    Parameters:
        encoded (dict): RLE 编码结果，包含:
                        - 'shape': 原始数组形状
                        - 'rle': [值, 长度] 序列
    
    Returns:
        numpy.ndarray: 解码后的 2D numpy 数组。
    """
    shape = encoded['shape']
    rle = np.array(encoded['rle'], dtype=int)

    # 提取值与运行长度
    values = rle[0::2]
    run_lengths = rle[1::2]

    # 重建一维数组
    flat_mask = np.repeat(values, run_lengths)

    # 恢复为原始形状
    mask = flat_mask.reshape(shape)
    return mask
