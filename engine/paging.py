"""物理页的分配与页表管理.

调度层在 paged 之后变了什么
===========================
整块方案里"调度"几乎不存在: 一条请求 = 一整块 `[H, T_max, D]`,
位置就是 `pos` 一个显存标量, `reset()` 清零就完事. paged 之后多了三件事.

**一, 多了一层间接, 而且这一层在 host 上. **
物理页从哪来, 还回哪去, 是 host 的决定; kernel 只负责读 `block_table` 这张表.
所以状态被劈成两半:

```text
host 侧   PagePool._free        哪些物理页是空的
显存      block_table[max_pages] 逻辑页 -> 物理页
```

**二, "填充既往 KV"从连续拷贝变成 scatter. **
整块方案的 prefill 是一句 `k_cache[:, :T, :].copy_(k4[0])` -- 逻辑位置和物理位置
一一对应, 连续的一段. paged 下逻辑上相邻的 token 可能落在物理上不相邻的页,
所以必须按 token 算出物理行号再散布(`paged_kv_fill`):

```text
row(t, h) = (block_table[t // PAGE] * H + h) * PAGE + (t % PAGE)
```

decode 的追加同理, 只是 T=1(`paged_kv_append`).

**三, 扩页的时机. **
序列每长 PAGE 个 token 就要一页. 有两种做法:

- **预分配**: prefill 时按 `prompt_len + max_new_tokens` 一次性把页要够,
  decode 全程页表不变. 实现最简单, 也是本文件默认的做法
  (`SequencePages.reserve`). 代价是又回到了"按上限预留",
  paged 的省显存优势没了 -- 但**页表的机制本身是对的**, 凑批时改成按需扩即可.
- **按需扩**: 每写满一页就要一页新的, host 在两次 replay 之间往 `block_table`
  里写新页号.

**按需扩不会破坏 CUDA Graph. ** 图里烧进去的只是 `block_table` 的**地址**,
表的内容是 kernel 执行时才读的. 这和 `pos` 进显存是同一个道理(见
`triton_kernels/gqa_attention_decode.py` 顶部那段长注释): 只要变化的东西在显存里,
形状不变, 图就不用重新捕获. 所以 host 每步只需要判断"是否跨过了页边界",
跨过了就写一个 int32 进去.

**什么没变: ** `pos` 仍然是唯一的位置来源, split/combine 仍然从它算 seq_len,
GDN 的两个 cache 完全不受影响(它们是定长的, 与上下文长度无关).
"""

from __future__ import annotations

import torch

from triton_kernels.gqa_attention_decode_paged import PAGE_SIZE


def allocate_block_table(
    max_pages: int, *, batch: int = 1, device: torch.device | str = "cuda"
) -> torch.Tensor:
    """`[B, max_pages]` INT32, 逻辑页 -> 物理页, 一条序列一行.

    **必须在显存里, 不能是 host 侧的 list. ** kernel 执行时才读它,
    所以 CUDA Graph 捕获之后 host 仍然可以往里写新分配的页号, replay 会看到 --
    图里烧的只是它的地址. 这和 `pos` 进显存是同一个模式.

    凑批时这里会变成 `[max_batch, max_pages]`, 一条 session 一行. **形状固定**
    很重要: kernel 里 `block_table_ptr + pid_b * stride_bt_b` 的 stride 才能是
    编译期常量; 若每条 session 一张独立长度的表, 就得传指针数组, 多一层解引用.
    """
    return torch.zeros((batch, max_pages), dtype=torch.int32, device=device)


class PagePool:
    """物理页的空闲列表, 纯 host 侧.

    用 list 而不是显存上的 bitmap: 分配决策本来就在 host 做,
    而且 batch=1 时页数只有几十到几百, list 的开销可以忽略.
    真做起 continuous batching 再换数据结构不迟.
    """

    def __init__(self, num_pages: int):
        assert num_pages > 0
        self.num_pages = num_pages
        self._free: list[int] = list(range(num_pages))

    @property
    def num_free(self) -> int:
        return len(self._free)

    def alloc(self, count: int) -> list[int]:
        assert count <= len(self._free), (
            f"页不够了: 要 {count} 页, 只剩 {len(self._free)} 页(共 {self.num_pages} 页,"
            f"每页 {PAGE_SIZE} 个 token). 加大 num_pages 或减小 max_tokens."
        )
        # 从尾部取, free 时再 append 回去, 这样重复 alloc/free 不会让页号漂移,
        # 便于复现问题.
        taken = self._free[-count:]
        del self._free[-count:]
        return taken

    def free(self, pages: list[int]) -> None:
        self._free.extend(pages)

    def reset(self) -> None:
        self._free = list(range(self.num_pages))


class SequencePages:
    """一批序列的页表.`block_table` 是 `[B, max_pages]` 的显存张量, 分配记录在 host.

    **一块显存, B 行, 不是 B 个独立张量. ** 原因是 CUDA Graph: 图里烧的是地址,
    必须固定一块; kernel 靠 grid 的 batch 维偏到自己那一行
    (`block_table_ptr + pid_b * stride_bt_b`), stride 才能是编译期常量.
    若每条序列一张独立长度的表, 就得传指针数组, 多一层解引用.

    **页池只有一份, 所有序列共用**(`k_cache`/`v_cache`). 这正是分页的意义:
    B 条请求不再各自按 max_len 预留, 而是共同从一个池子里按实际用量取.
    """

    def __init__(
        self,
        max_pages: int,
        *,
        batch: int = 1,
        device: torch.device | str = "cuda",
    ):
        self.max_pages = max_pages
        self.batch = batch
        self.block_table = allocate_block_table(max_pages, batch=batch, device=device)
        # 每条序列各自持有的物理页号, host 侧
        self.pages: list[list[int]] = [[] for _ in range(batch)]

    def reserve(self, token_num: int, pool: PagePool, *, seq: int = 0) -> None:
        """确保第 `seq` 条序列能放下 `token_num` 个 token, 不够就从池里要页.

        可以反复调用, 只补差额 -- 所以既能在 prefill 时一次要够(预分配),
        也能在 decode 里每步调一次(按需扩), 两种策略共用这一个入口.

        **必须在会写到那个位置的 forward 之前调用. ** 写完再发现越界就晚了:
        越界写会在 device 上 assert, 报错点离原因很远.
        """
        need = (token_num + PAGE_SIZE - 1) // PAGE_SIZE
        assert need <= self.max_pages, (
            f"{token_num} 个 token 需要 {need} 页, 超出页表容量 {self.max_pages}"
        )
        held = self.pages[seq]
        if need <= len(held):
            return
        new_pages = pool.alloc(need - len(held))
        # 只写新增的那一段. block_table 在显存里, kernel 执行时才读,
        # 所以 CUDA Graph 捕获之后写它也是安全的 -- 图里烧的只是地址.
        self.block_table[seq, len(held) : need] = torch.tensor(
            new_pages, dtype=torch.int32, device=self.block_table.device
        )
        held.extend(new_pages)

    def reserve_all(self, token_num: int, pool: PagePool) -> None:
        """给所有序列都留够 `token_num`. B=1 或等长场景的便捷入口."""
        for seq in range(self.batch):
            self.reserve(token_num, pool, seq=seq)

    def release(self, pool: PagePool, *, seq: int | None = None) -> None:
        """归还页.`seq=None` 表示全部归还.

        不需要清空页表 -- kernel 只遍历 `lp < cdiv(seq_len, PAGE)`, 而 seq_len 来自
        pos, pos 归零后残留的旧页号永远读不到. 但页必须还, 否则池子会漏.
        """
        targets = range(self.batch) if seq is None else [seq]
        for b in targets:
            pool.free(self.pages[b])
            self.pages[b] = []
        if seq is None:
            self.block_table.zero_()
        else:
            self.block_table[seq].zero_()

    def capacity(self, seq: int = 0) -> int:
        """第 `seq` 条序列当前已分配的页能放下多少 token."""
        return len(self.pages[seq]) * PAGE_SIZE
