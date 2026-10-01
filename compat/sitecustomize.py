"""Opt-in Python startup hooks for the pinned competition environment."""

import functools
import os


if os.environ.get("ROBOTWIN_DISABLE_CUROBO_FUSED_LBFGS") == "1":
    from curobo.opt.newton.lbfgs import LBFGSOpt

    _original_init = LBFGSOpt.__init__

    if not getattr(_original_init, "_vla_fused_lbfgs_fallback", False):

        @functools.wraps(_original_init)
        def _fallback_init(self, config=None):
            if config is not None and hasattr(config, "use_cuda_kernel"):
                config.use_cuda_kernel = False
            _original_init(self, config)
            self.use_cuda_kernel = False

        _fallback_init._vla_fused_lbfgs_fallback = True
        LBFGSOpt.__init__ = _fallback_init
        print("VLA_COMPAT_CUROBO_FUSED_LBFGS_DISABLED")
