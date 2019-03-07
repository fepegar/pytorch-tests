"""
Adapted from
https://github.com/pytorch/pytorch/issues/1249#issuecomment-305088398
"""

def dice_loss(output, target, pseudocount=1):
    iflat = output.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()
    num = 2 * intersection + pseudocount
    denom = (iflat**2).sum() + (tflat**2).sum() + pseudocount
    return 1 - num / denom
