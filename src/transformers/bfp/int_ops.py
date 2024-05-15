import torch

def _quantize(x, scale, zero, maxq):
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return scale * (q - zero)

def int_quantize(x, bits, device, identifier):
    maxq = torch.tensor(2**bits-1).to(device=device)
    perchannel=True
    sym=False
    mse=False
    weight=True if identifier == 'w' else False
    norm=2.4
    grid=100
    maxshrink=0.8
    grouprows=1
    shape = x.shape
    scale=torch.zeros(1)
    zero=torch.zeros(1)

    if perchannel:
        if weight:
            x = x.flatten(1)
            if self.grouprows > 1: 
                x = x.reshape((x.shape[0] // self.grouprows, -1))
        else:
            if len(shape) == 4:
                x = x.permute([1, 0, 2, 3])
                x = x.flatten(1)
            if len(shape) == 3:
                x = x.reshape((-1, shape[-1])).t()
            if len(shape) == 2:
                x = x.t()
    else:
        x = x.flatten().unsqueeze(0)

    tmp = torch.zeros(x.shape[0], device=device)
    xmin = torch.minimum(x.min(1)[0], tmp)
    xmax = torch.maximum(x.max(1)[0], tmp)

    if sym:
        xmax = torch.maximum(torch.abs(xmin), xmax)
        tmp = xmin < 0
        if torch.any(tmp):
            xmin[tmp] = -xmax[tmp]
    tmp = (xmin == 0) & (xmax == 0)
    xmin[tmp] = -1
    xmax[tmp] = +1

    scale = (xmax - xmin) / maxq
    if sym:
        zero = torch.full_like(scale, (maxq + 1) / 2)
    else:
        zero = torch.round(-xmin / scale)

    if mse:
        best = torch.full([x.shape[0]], float('inf'), device=device)
        for i in range(int(self.maxshrink * self.grid)):
            p = 1 - i / grid 
            xmin1 = p * xmin
            xmax1 = p * xmax
            scale1 = (xmax1 - xmin1) / maxq
            zero1 = torch.round(-xmin1 / scale1) if not sym else zero
            q = _quantize(x, scale1.unsqueeze(1), zero1.unsqueeze(1), maxq)
            q -= x
            q.abs_()
            q.pow_(self.norm)
            err = torch.sum(q, 1)
            tmp = err < best
            if torch.any(tmp):
                best[tmp] = err[tmp]
                scale[tmp] = scale1[tmp]
                zero[tmp] = zero1[tmp]
    
    if not perchannel:
        if weight:
            tmp = shape[0]
        else:
            tmp = shape[1] if len(shape) != 3 else shape[2]
        scale = scale.repeat(tmp)
        zero = zero.repeat(tmp)

    if weight:
        if grouprows > 1:
            scale = scale.unsqueeze(1).repeat(1, grouprows)
            zero = zero.unsqueeze(1).repeat(1, grouprows)
        shape = [-1] + [1] * (len(shape) - 1)
        scale = scale.reshape(shape)
        zero = zero.reshape(shape)
    if len(shape) == 4:
        scale = scale.reshape((1, -1, 1, 1))
        zero = zero.reshape((1, -1, 1, 1))
    if len(shape) == 3:
        scale = scale.reshape((1, 1, -1))
        zero = zero.reshape((1, 1, -1)) 
    if len(shape) == 2:
        scale = scale.unsqueeze(0)
        zero = zero.unsqueeze(0)

    return _quantize(x, scale, zero, maxq)

