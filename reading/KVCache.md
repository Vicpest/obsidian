https://huggingface.co/docs/transformers/cache_explanation


缓存机制（Caching）

Transformer 原生自回归生成规则：

1. 生成第t个token，依赖1~t-1全部历史token；

2. 模型计算注意力时，要对所有历史token重新计算Q/K/V，做完整矩阵乘法； 

3. 每多生成一个token，历史序列只会变长，重复计算量指数级上涨。

键值缓存（KV Cache）：

生成阶段，历史token的K,V一旦算完就永远不会改变；因为历史token embedding和位置不会改变；只有新生成token需要计算新的Q、K、V 

1.预处理输入提示词（prompt）时，一次性算出所有prompt token的K,V，存入缓存； 

2.后续每生成1个新词元，只计算这个新词元的K,V,追加到缓存尾部； 

3.计算注意力时，直接读取缓存里全部历史KV，只拿当前token的Q做计算，历史KV不再计算

KV Cache 只能推理用，训练禁止开启：

1.训练是并行计算整段序列；训练时一次性输入完整句子，所有token同时并行算损失，不存在“逐一生成、复用历史KV”的场景；

2.缓存的KV是固定只读的，无法参与反向传播求梯度，开启缓存会导致梯度出错、模型无法收敛 

3.训练数据序列是动态变化的，每一轮训练样本长度不同，缓存无法复用，还会占用大量显存 而推理只有前向传播，不需要反向求梯度；序列是逐token自回归生成，历史固定不变，缓存完全安全，提速省缓存

常见问题
KV Cache 为什么能够加速推理？

LLM采用自回归生成方式，每生成一个token，都需要关注之前所有历史token。如果没有KV Cache，每一步都会重新计算历史token的Key和Value，导致大量重复计算。KV Cache会保存历史token在每层Attention中的Key和Value。生成新token时，只需要计算当前token的新Q、K、V，然后利用缓存中的历史K、V完成Attention计算。因此避免了重复计算，提高了Decoder生成速度。
为什么只缓存K,V？不缓存Q？

因为在Decoder自回归生成过程中，每一步Attention计算只需要当前token的Query去查询历史token的信息。历史token的Key和Value在生成之后不会发生变化，因此可以缓存。而Query只对应当前正在生成的token，每生成一个新token都会产生新的Query，并且之前token的Query不会再参与后续计算，因此没有缓存价值。
为什么速度提升？

大模型逐 token 自回归生成文本时，如果不使用 KV Cache，每生成第 t 个新 token，都要把前面所有历史 token 对应的 Key、Value 全部重新计算一遍，反复做冗余运算；而开启 KV Cache 之后，过往已经算好的历史 Key 和 Value 会被提前存下来，生成新 token 时只需要计算当前这个 token 的 Key、Value，直接从缓存取出全部历史的 KV 数据，和当前 token 的 Query 做注意力计算就行，把大量重复的算力计算改成了低成本的缓存读取，从而大幅削减了整体计算量。
为什么显存越来越大？

每生成一个token，都需要保存所有Transformer层中该token对应的Key和Value。KV Cache大小约为：2 x L x n x h其中：

    L：Transformer层数
    n：序列长度
    h：hidden size
    随着上下文长度增加：n↑ KV Cache线性增长。例如：一个70B模型：长上下文推理时，KV Cache可能占用几十GB显存。因此KV Cache虽然减少计算，但是成为大模型推理显存瓶颈。
