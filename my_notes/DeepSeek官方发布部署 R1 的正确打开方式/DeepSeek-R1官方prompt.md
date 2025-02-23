

```
https://github.com/deepseek-ai/DeepSeek-R1/pull/399/files
```

[TOC]

# 1. 模型 “小毛病” 及解决方法

研究人员发现，这个系列的模型在回答某些问题时，会偷偷绕过一种关键的思考模式，专业表述就是 “输出‘<think>\n\n</think>’” 。

为了让模型 “重回正轨”，好好思考再作答，官方给出了一个实用建议：每次让模型输出答案时，强制它以 “<think>\n” 开头。这样一来，模型就能更认真地 “思考” 问题，给出更靠谱的回答啦。

# 2. 官方prompt模板

在 DeepSeek 官方的网页和应用程序里，不再使用传统的系统提示，而是针对文件上传和网络搜索，专门打造了两款定制化提示模板，而且应用中的温度参数设置为 0.6，这个数值可以让模型生成的答案既不会太机械，也不会过于随意。

## 2.1 文件上传prompt模板

先来看文件上传提示模板，以后大家用 DeepSeek-R1 处理文件相关问题时，就得按这个 “套路” 来。模板长这样：

```plaintext
[file name]: {file_name}
[file content begin]
{file_content}
[file content end]
{question}
```

这里面的`{file_name}`是文件的名字，`{file_content}`是文件里的具体内容，`{question}`就是你针对这个文件提出的问题。举个例子，假如你上传了一份 “年度财务报表.xlsx” 文件，里面记录了公司一年的各项财务数据，你想知道公司去年的净利润是多少，那给模型的提示就可以写成：

```plaintext
[file name]: 年度财务报表.xlsx
[file content begin]
[粘贴年度财务报表中的详细数据内容]
[file content end]
公司去年的净利润是多少？
```

## 2.2 网络搜索prompt模板

中文和英文查询的模板虽然有些区别，但核心规则是相通的。

### 2.2.1 中文查询prompt

中文查询模板的规则非常详细：

```plaintext
search_answer_zh_template = \
'''# 以下内容是基于用户发送的消息的搜索结果:
{search_results}
在我给你的搜索结果中，每个结果都是[webpage X begin]...[webpage X end]格式的，X代表每篇文章的数字索引。请在适当的情况下在句子末尾引用上下文。请按照引用编号[citation:X]的格式在答案中对应部分引用上下文。如果一句话源自多个上下文，请列出所有相关的引用编号，例如[citation:3][citation:5]，切记不要将引用集中在最后返回引用编号，而是在答案对应部分列出。
在回答时，请注意以下几点：
- 今天是{cur_date}。
- 并非搜索结果的所有内容都与用户的问题密切相关，你需要结合问题，对搜索结果进行甄别、筛选。
- 对于列举类的问题（如列举所有航班信息），尽量将答案控制在10个要点以内，并告诉用户可以查看搜索来源、获得完整信息。优先提供信息完整、最相关的列举项；如非必要，不要主动告诉用户搜索结果未提供的内容。
- 对于创作类的问题（如写论文），请务必在正文的段落中引用对应的参考编号，例如[citation:3][citation:5]，不能只在文章末尾引用。你需要解读并概括用户的题目要求，选择合适的格式，充分利用搜索结果并抽取重要信息，生成符合用户要求、极具思想深度、富有创造力与专业性的答案。你的创作篇幅需要尽可能延长，对于每一个要点的论述要推测用户的意图，给出尽可能多角度的回答要点，且务必信息量大、论述详尽。
- 如果回答很长，请尽量结构化、分段落总结。如果需要分点作答，尽量控制在5个点以内，并合并相关的内容。
- 对于客观类的问答，如果问题的答案非常简短，可以适当补充一到两句相关信息，以丰富内容。
- 你需要根据用户要求和回答内容选择合适、美观的回答格式，确保可读性强。
- 你的回答应该综合多个相关网页来回答，不能重复引用一个网页。
- 除非用户要求，否则你回答的语言需要和用户提问的语言保持一致。

# 用户消息为：
{question}'''
```

简单概括一下，就是模型要依据搜索结果回答问题，引用网页内容时要遵循规范格式，不同类型的问题回答方式各有讲究。列举类问题，答案控制在 10 个要点，还要告诉用户获取完整信息的渠道；创作类问题，要在文章中间合理引用参考编号，把答案写得又长又有深度；回答较长时，要分好段落、条理清晰。

### 2.2.2 英文查询prompt

英文查询模板和中文的类似，同样强调规范引用搜索结果，以及回答各类问题的要点：

```plaintext
search_answer_en_template = \
'''# The following contents are the search results related to the user's message:
{search_results}
In the search results I provide to you, each result is formatted as [webpage X begin]...[webpage X end], where X represents the numerical index of each article. Please cite the context at the end of the relevant sentence when appropriate. Use the citation format [citation:X] in the corresponding part of your answer. If a sentence is derived from multiple contexts, list all relevant citation numbers, such as [citation:3][citation:5]. Be sure not to cluster all citations at the end; instead, include them in the corresponding parts of the answer.
When responding, please keep the following points in mind:
- Today is {cur_date}.
- Not all content in the search results is closely related to the user's question. You need to evaluate and filter the search results based on the question.
- For listing-type questions (e.g., listing all flight information), try to limit the answer to 10 key points and inform the user that they can refer to the search sources for complete information. Prioritize providing the most complete and relevant items in the list. Avoid mentioning content not provided in the search results unless necessary.
- For creative tasks (e.g., writing an essay), ensure that references are cited within the body of the text, such as [citation:3][citation:5], rather than only at the end of the text. You need to interpret and summarize the user's requirements, choose an appropriate format, fully utilize the search results, extract key information, and generate an answer that is insightful, creative, and professional. Extend the length of your response as much as possible, addressing each point in detail and from multiple perspectives, ensuring the content is rich and thorough.
- If the response is lengthy, structure it well and summarize it in paragraphs. If a point-by-point format is needed, try to limit it to 5 points and merge related content.
- For objective Q&A, if the answer is very brief, you may add one or two related sentences to enrich the content.
- Choose an appropriate and visually appealing format for your response based on the user's requirements and the content of the answer, ensuring strong readability.
- Your answer should synthesize information from multiple relevant webpages and avoid repeatedly citing the same webpage.
- Unless the user requests otherwise, your response should be in the same language as the user's question.

# The user's message is:
{question}'''
```

```
笔者能力有限，欢迎批评指正或者在留言区讨论
```