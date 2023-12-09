import time

import numpy as np
import torch
from generalized_lstm import Conv2dLSTM, LinearLSTM

if __name__ == "__main__":
    """
    I recommend you use torch.jit.trace to speed-up the transformations if not all the for-loops are not computed efficiently.
    An example in the following code:
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Conv
    # B x seq.length x C x H x W
    m = Conv2dLSTM(3, 32, 3, num_layers=2).eval().to(device)
    # out = m(torch.rand(4, 15, 3, 92, 92))
    inputx = torch.rand(4, 15, 3, 92, 92).to(device)
    h0x = m.init_h_or_c(inputx)
    m_fix = torch.jit.trace(
        m,
        example_kwarg_inputs={"input": inputx, "hx": (h0x, h0x)},
        check_trace=False,  # For one unkown reason the line 177 is generating different symbols (h_0, c_0 = hx).
    )
    print("ConvLSTM:")
    times = []
    for i in range(1000):
        a0 = time.time()
        out_fix = m_fix(torch.rand(4, 15, 3, 92, 92).to(device), (h0x, h0x))
        a1 = time.time()
        times.append(1000 * (a1 - a0))
    times = times[5:]  # remove first 5
    print(f"Output Shape: {out_fix[0].shape}")
    print(
        f"Elapsed time: {np.mean(times)} (+-{np.std(times)}, last-5: {times[-5:]}, max: {np.max(times)}), min: {np.min(times)})"
    )

    # Linear
    # B x seq.length x C
    m = LinearLSTM(3, 32, num_layers=2).to(device)
    inputx = torch.rand(4, 15, 3).to(device)
    h0x = m.init_h_or_c(inputx)
    m_fix = torch.jit.trace(
        m,
        example_kwarg_inputs={"input": inputx, "hx": (h0x, h0x)},
        check_trace=False,  # For one unkown reason the line 177 is generating different symbols (h_0, c_0 = hx).
    )
    # out = m(torch.rand(4, 15, 3))
    print("LinearLSTM:")

    times = []
    for i in range(1000):
        a0 = time.time()
        out_fix = m_fix(torch.rand(4, 15, 3).to(device), (h0x, h0x))
        a1 = time.time()
        times.append(1000 * (a1 - a0))

    times = times[5:]  # remove first 5
    print(f"Output Shape: {out_fix[0].shape}")
    print(
        f"Elapsed time: {np.mean(times)} (+-{np.std(times)}, last-5: {times[-5:]}, max: {np.max(times)}), min: {np.min(times)})"
    )
