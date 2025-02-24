

国内第一本RAG书籍对应github，比较系统  https://github.com/Nipi64310/RAG-Book

- 第一章：介绍了RAG的发展，和LLM微调的对比，优缺点、以及RAG解决了LLM直接落地时候的哪些难点

- 第二章:介绍了Transformer相关的基本原理，包括embedding、编码器、解码器等。

- 第三章：介绍了RAG环节里比较核心的文本向量化模型。首先介绍了和文本向量化模型比较相关的基础概念，比如对齐性、均匀性、句向量表示方法、对称检索非对称检索等。然后介绍了一些稠密向量检索模型，包括SimCSE、SBERT、CoSENT、WhiteBERT、SGPT等。注意，本章并没有特别介绍排行榜上的各种模型（如bge等），榜上的大部分模型（在写作本书的时间点上）基本都是按照SimCSE的方式训练的，**本章更侧重介绍不同的向量化模型训练范式**，让读者了解更本质的文本向量表示方法。最后，也介绍了稀疏向量检索模型和重排序模型。

- 第四章：这一张比较琐碎，内容也比较多：

  （1）LLM基础的提示词工程以及在RAG场景下的提示词技巧。

  （2）文本切块方法，包含基于规则的以及基于模型的。

  （3）向量数据库的基本原理以及一些开源向量数据库

  （4）召回环节优化策略：短文本全局信息增强、上下文扩充、文本多向量表示、元数据召回、重排序等。

  （5）召回环节的评估以及模型回答评估

  （6）RAG场景下的LLM优化，包括微调FLARE、Self-RAG等。

- 第五章：介绍了RAG的范式演变，从基础的RAG系统开始到agent再到多模态RAG
- 第六章：介绍了RAG系统相关的训练内容。除了文本向量化模型和LLM可以独立训练外，还可以将二者联合起来，进行序贯训练以及联合训练。
- 第七章：介绍了如何基于langchain构建一个简单的RAG系统。包括langchain基础模块介绍，以及构建一个ChatPDF可视化应用。
- 第八章：本章从实战角度出发，讲解了向量化模型和LLM的选型、训练数据构造、训练方法等。







