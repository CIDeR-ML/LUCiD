"""Identity numerical response for the minimal JUNO closure.

The continuous PPP uses a Gaussian interpolation kernel to evaluate discrete
time-basis masses at exact PE timestamps.  That kernel is numerical and does
not model PMT TTS, electronics, or a measured impulse response.
"""


def make_identity_response_forward(ensemble_forward):
    """Return the optical ensemble without a physical response convolution.

    输入 ~~~ ensemble_forward: PMT-by-time mass forward function.
    输出 ~~~ The same callable, unchanged.
    """
    if not callable(ensemble_forward):
        raise TypeError("ensemble forward must be callable")
    return ensemble_forward
