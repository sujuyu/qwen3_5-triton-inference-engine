"""开启 Triton 的 autotune 磁盘缓存。

这个包唯一的作用是在任何 kernel 模块被导入**之前**把 `TRITON_CACHE_AUTOTUNING`
打开。Python 在 `import triton_kernels.xxx` 时会先执行本文件，而
`@triton.autotune` 装饰器是在 kernel 模块导入时求值的
（`Autotuner.__init__` 里 `self.cache_results = cache_results or knobs.autotuning.cache`），
所以这里设置正好赶得上。

为什么需要它
------------
Triton 的 **JIT 编译产物会自动缓存到磁盘，但 autotune 的结果默认不缓存**——
每个进程启动都要把所有 config 重新 benchmark 一遍。本项目 15 个 kernel 合计 71 个
config，而实际 benchmark 次数远不止：`gemm_2d` 的 autotune key 是 `(N, K, IS_DECODE)`，
一次 forward 里有 7~9 个不同的 `(N,K)` 组合，每个都要跑 12 个 config，
prefill 和 decode 还各来一遍。

实测（demo，生成 8 个 token）：

    关闭          第 1 次 18.6s   第 2 次 18.2s   第 3 次 18.2s
    开启          第 1 次 18.6s   第 2 次  0.4s   第 3 次  0.4s     ← 45x

不开的话，重复运行时 JIT 缓存只能省下约 3.4s（21.3 -> 17.9s），剩下的 ~18s
全是 autotune 在重跑。开启后启动时间的主体变回加载 1.4 GiB 权重本身。

局限
----
直接运行单个 kernel 文件（`python triton_kernels/gemm_2d.py`）时，该模块以
`__main__` 身份执行，本文件不会被执行，所以那条路径拿不到缓存。这些都是一次性的
自测脚本，影响不大；真要用的话在命令前加 `TRITON_CACHE_AUTOTUNING=1` 即可。

缓存落在 `TRITON_CACHE_DIR`（默认 `~/.triton/cache`），键里含 Triton 版本、
backend hash、kernel 源码 hash 和 config 列表，所以改了 kernel 或换了卡会自动失效，
不会用到过期的调优结果。
"""

import os

os.environ.setdefault("TRITON_CACHE_AUTOTUNING", "1")
