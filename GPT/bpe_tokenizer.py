# Byte Pair Encoding (BPE) Tokenizer — 字符级 BPE 实现
#
# 算法思路：
#   1. 初始词表 = 所有不重复字符（字符级）
#   2. 统计语料中所有相邻 token pair 的出现频次
#   3. 合并频次最高的 pair，生成新 token，加入词表
#   4. 重复步骤 2~3，直到词表达到目标大小
#
# 与 CharTokenizer 接口完全一致，可直接替换进 train_gpt.py：
#   - encode(text, add_bos, add_eos) -> list[int]
#   - decode(ids, skip_special_tokens) -> str
#   - save(path) / load(path)
#   - vocab_size 属性
#
# 性能优化：增量更新 pair_counts（而非每步重新统计全语料）
#   - 初始化时统计一次全语料 pair_counts
#   - 每步只更新含 best_pair 的 word 的 pair_counts
#   - 复杂度从 O(corpus × merges) 降为 O(affected_words × merges)

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from tqdm import tqdm


SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>", "<unk>")


class BPETokenizer:
    def __init__(
        self,
        stoi: dict[str, int],
        itos: dict[int, str],
        merges: list[tuple[str, str]],
        special_tokens: tuple[str, ...] = SPECIAL_TOKENS,
    ) -> None:
        self.stoi = stoi
        self.itos = itos
        # merges 是有序的合并规则列表，顺序就是训练时合并的先后顺序
        # 编码时按同样的顺序应用，保证结果一致
        self.merges = merges
        self.special_tokens = special_tokens

        self.pad_id = stoi["<pad>"]
        self.bos_id = stoi["<bos>"]
        self.eos_id = stoi["<eos>"]
        self.unk_id = stoi["<unk>"]

        # 编码时用来快速查找某个 pair 的优先级（rank 越小 = 越先学到 = 优先合并）
        self._merge_rank: dict[tuple[str, str], int] = {
            pair: rank for rank, pair in enumerate(merges)
        }

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    # 训练
    @classmethod
    def train(
        cls,
        text: str,
        vocab_size: int,
        min_freq: int = 2,
        special_tokens: tuple[str, ...] = SPECIAL_TOKENS,
        verbose: bool = True,
    ) -> "BPETokenizer":
        """在给定文本上训练 BPE。

        Args:
            text:       训练语料（纯文本字符串）
            vocab_size: 目标词表大小（必须 > 初始字符数 + 特殊 token 数）
            min_freq:   pair 至少出现多少次才合并（过滤噪声）
            verbose:    是否显示 tqdm 进度条
        """
        # Step 0: 构建初始字符级词表
        # 按行分割语料：行是天然的处理单元（诗词每行独立，合并不跨行）
        lines = [line for line in text.split("\n") if line.strip()]

        # 统计每行出现的频次（诗词中大量重复行，可大幅减少运算）
        line_freq: Counter[str] = Counter(lines)

        # 初始词表 = 特殊 token + 所有不重复字符（按字符排序保证确定性）
        all_chars = sorted(set(text))
        init_tokens = list(special_tokens) + [
            c for c in all_chars if c not in special_tokens
        ]
        stoi: dict[str, int] = {tok: idx for idx, tok in enumerate(init_tokens)}
        itos: dict[int, str] = {idx: tok for tok, idx in stoi.items()}

        num_merges = vocab_size - len(stoi)
        if num_merges <= 0:
            raise ValueError(
                f"vocab_size={vocab_size} 必须大于初始词表大小 {len(stoi)}"
            )

        if verbose:
            print(f"初始词表大小 : {len(stoi)}")
            print(f"目标词表大小 : {vocab_size}")
            print(f"计划合并次数 : {num_merges}")

        # Step 1: 把语料表示为 word_freqs
        # word: tuple of tokens（训练开始时每个 token 就是一个字符）
        # freq: 这个 word 在语料中出现的次数
        word_freqs: dict[tuple[str, ...], int] = {}
        for line, freq in line_freq.items():
            if line:
                word = tuple(line)
                word_freqs[word] = word_freqs.get(word, 0) + freq

        # Step 2: 初始化一次全语料 pair_counts（增量更新的基础）
        pair_counts: dict[tuple[str, str], int] = {}
        for word, freq in word_freqs.items():
            for i in range(len(word) - 1):
                k = (word[i], word[i + 1])
                pair_counts[k] = pair_counts.get(k, 0) + freq

        # Step 3: BPE 合并循环（增量更新 pair_counts，避免全量重扫）
        merges: list[tuple[str, str]] = []

        pbar = tqdm(
            range(num_merges),
            desc="BPE训练",
            unit="merge",
            disable=not verbose,
            dynamic_ncols=True,
        )

        for step in pbar:
            if not pair_counts:
                if verbose:
                    tqdm.write(f"  第 {step} 步：语料中无更多 pair，提前终止")
                break

            # 找频次最高的 pair
            best_pair = max(pair_counts, key=lambda p: pair_counts[p])
            best_count = pair_counts[best_pair]

            if best_count < min_freq:
                if verbose:
                    tqdm.write(f"  第 {step} 步：最高频次 {best_count} < min_freq={min_freq}，停止")
                break

            a, b = best_pair
            new_token = a + b
            new_id = len(stoi)
            stoi[new_token] = new_id
            itos[new_id] = new_token
            merges.append(best_pair)

            pbar.set_postfix(merge=f"'{a}'+'{b}'→'{new_token}'", freq=best_count)

            # 增量更新：只处理含有 best_pair 的 word
            # 找出所有含 (a, b) 的 word（遍历一次过滤，远少于全语料）
            affected: list[tuple[tuple[str, ...], int]] = [
                (w, f) for w, f in list(word_freqs.items())
                if any(w[i] == a and w[i + 1] == b for i in range(len(w) - 1))
            ]

            for word, freq in affected:
                new_word = cls._merge_word(word, a, b)

                # 从 pair_counts 中减去 old word 的 pair 贡献
                for i in range(len(word) - 1):
                    k = (word[i], word[i + 1])
                    pair_counts[k] = pair_counts.get(k, 0) - freq
                    if pair_counts.get(k, 0) <= 0:
                        pair_counts.pop(k, None)

                # 加上 new_word 的 pair 贡献
                for i in range(len(new_word) - 1):
                    k = (new_word[i], new_word[i + 1])
                    pair_counts[k] = pair_counts.get(k, 0) + freq

                # 更新 word_freqs
                old_total = word_freqs.pop(word)
                word_freqs[new_word] = word_freqs.get(new_word, 0) + old_total

            # best_pair 已被合并，从 pair_counts 中移除
            pair_counts.pop(best_pair, None)

        pbar.close()

        if verbose:
            print(f"训练完成，最终词表大小 : {len(stoi)}")

        return cls(stoi=stoi, itos=itos, merges=merges, special_tokens=special_tokens)

    @staticmethod
    def _merge_word(word: tuple[str, ...], a: str, b: str) -> tuple[str, ...]:
        """把 word 中所有相邻的 (a, b) 替换为 a+b。

        例：_merge_word(("孟","子","見","孟","子"), "孟","子")
              → ("孟子","見","孟子")
        """
        result: list[str] = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                result.append(a + b)
                i += 2
            else:
                result.append(word[i])
                i += 1
        return tuple(result)

    # 编码 / 解码
    def _tokenize(self, text: str) -> list[str]:
        """把文本转为 BPE token 字符串列表。

        编码策略：贪心优先合并（Greedy Merge by Priority）
          - 从字符级 token 开始
          - 每次找所有相邻 pair 中 rank 最低（最早学到的）那个合并
          - 重复直到没有可合并的 pair
          - 这与训练时的合并顺序一致，保证编码结果最优
        """
        if not text:
            return []

        # 按行处理，与训练时边界对齐
        all_tokens: list[str] = []
        for line in text.split("\n"):
            if not line:
                all_tokens.append("\n")
                continue
            tokens = list(line)
            tokens = self._apply_merges(tokens)
            all_tokens.extend(tokens)
            all_tokens.append("\n")

        # 去掉末尾多余的 \n
        if all_tokens and all_tokens[-1] == "\n":
            all_tokens.pop()

        return all_tokens

    def _apply_merges(self, tokens: list[str]) -> list[str]:
        """对 token 列表反复应用 merge，直到没有可合并的 pair。

        每次只合并优先级最高（rank 最小）的那个 pair，
        然后重新扫描，直到没有任何 merge 可用。
        """
        while len(tokens) >= 2:
            # 在所有相邻 pair 中找 rank 最小的
            best_rank = len(self.merges)   # 哨兵值：比所有 rank 都大
            best_i = -1

            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                rank = self._merge_rank.get(pair, len(self.merges))
                if rank < best_rank:
                    best_rank = rank
                    best_i = i

            if best_i == -1:
                break  # 没有任何可合并的 pair，结束

            a, b = tokens[best_i], tokens[best_i + 1]
            # 合并 best_i 和 best_i+1 处的 pair
            tokens = tokens[:best_i] + [a + b] + tokens[best_i + 2:]

        return tokens

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        ids: list[int] = []
        if add_bos:
            ids.append(self.bos_id)
        for token in self._tokenize(text):
            ids.append(self.stoi.get(token, self.unk_id))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(
        self,
        ids: list[int],
        skip_special_tokens: bool = True,
    ) -> str:
        chars: list[str] = []
        for idx in ids:
            token = self.itos.get(idx, "<unk>")
            if skip_special_tokens and token in self.special_tokens:
                continue
            chars.append(token)
        return "".join(chars)

    # 持久化
    def save(self, path: str | Path) -> None:
        path = Path(path)
        payload = {
            "stoi": self.stoi,
            "merges": self.merges,
            "special_tokens": list(self.special_tokens),
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        stoi = {str(k): int(v) for k, v in payload["stoi"].items()}
        itos = {v: k for k, v in stoi.items()}
        merges = [tuple(m) for m in payload["merges"]]
        special_tokens = tuple(payload.get("special_tokens", list(SPECIAL_TOKENS)))
        return cls(stoi=stoi, itos=itos, merges=merges, special_tokens=special_tokens)


# Demo
def demo() -> None:
    import time
    from pathlib import Path

    corpus_path = Path(__file__).parent / "_train_text_large.txt"
    if not corpus_path.exists():
        print("找不到训练文本，使用内置小样本演示")
        text = (
            "孟子見梁惠王。王曰叟不遠千里而來亦將有以利吾國乎。\n"
            "孟子對曰王何必曰利亦有仁義而已矣。\n"
            "王曰何以利吾國大夫曰何以利吾家士庶人曰何以利吾身。\n"
            "上下交征利而國危矣萬乘之國弒其君者必千乘之家。\n" * 20
        )
    else:
        text = corpus_path.read_text(encoding="utf-8")
        # 截取前 3 万字加速 demo
        text = text[:30000]

    print(f"语料长度   : {len(text):,} 字符")

    # ── 训练 ──
    # 先扫一遍拿到初始词表大小，再决定目标（初始 + 500 次合并）
    init_chars = len(set(text)) + len(SPECIAL_TOKENS)
    target_vocab = init_chars + 500
    t0 = time.time()
    tok = BPETokenizer.train(text, vocab_size=target_vocab, min_freq=3, verbose=True)
    print(f"训练耗时   : {time.time() - t0:.2f}s")
    print(f"合并规则数 : {len(tok.merges)}")

    # ── 编码 / 解码一致性验证 ──
    sample = "孟子見梁惠王。王曰叟不遠千里而來。"
    ids = tok.encode(sample)
    restored = tok.decode(ids)

    print(f"\n原文  : {sample}")
    print(f"token : {tok._tokenize(sample)}")
    print(f"ids   : {ids}")
    print(f"还原  : {restored}")
    print(f"一致  : {'✓' if restored == sample else '✗'}")

    # ── 压缩率统计 ──
    char_count = len(sample)
    bpe_count = len(ids)
    print(f"\n字符数 / BPE token 数 = {char_count} / {bpe_count}  "
          f"(压缩率 {bpe_count/char_count:.2%})")

    # ── 最高频 merge 前 10 ──
    print("\n前 10 条 merge 规则（训练最早学到的）：")
    for i, (a, b) in enumerate(tok.merges[:10]):
        print(f"  {i+1:2d}. '{a}' + '{b}' → '{a+b}'")

    # ── 保存 / 加载 ──
    save_path = Path(__file__).parent / "checkpoints" / "bpe_demo.tokenizer.json"
    tok.save(save_path)
    tok2 = BPETokenizer.load(save_path)
    assert tok2.encode(sample) == ids, "save/load 后编码结果不一致！"
    print(f"\n✓ save/load 验证通过，保存在 {save_path}")


if __name__ == "__main__":
    demo()
