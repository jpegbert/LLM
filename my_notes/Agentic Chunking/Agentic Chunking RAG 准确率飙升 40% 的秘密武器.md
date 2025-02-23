[TOC]

最近，AI 圈又有大动作！一种名为 Agentic Chunking 的分块技术横空出世，号称能让 RAG 准确率飙升 40%，这可把 LLM 开发者们的好奇心拉满了。今天，咱们就来深度扒一扒，它到底有啥神奇之处！

# 1. 传统分块 “翻车”，问题出在哪？

在 RAG 模型的构建里，文本分块是打头阵且超关键的环节。就拿常见的递归字符分割来说，它操作简单，按照固定的 token 长度一刀切。但这也带来了大麻烦，一个完整的主题常常被拆得七零八落，分到不同文本块中，上下文连贯不起来，就像拼图被打乱了顺序，根本没法看。

还有语义分割法，听起来好像聪明点，它根据句子间语义变化来分割。但遇到文档话题频繁切换时，还是会 “翻车”，把相关内容分到不同块，信息又断了。

给大伙举个例子：“小明介绍了 Transformer 架构... （中间插入 5 段其他内容）... 最后他强调，Transformer 的核心是自注意力机制。” 用传统方法处理，要么把这两句话拆到不同区块，要么被中间内容干扰，导致语义断裂。但要是人工分块，我们肯定会把它们归到 “模型原理” 这一组。这种跨越文本距离的关联性问题，正是 Agentic Chunking 要解决的。

# 2. Agentic Chunking，凭啥能逆袭？

Agentic Chunking 的核心思路超有创意，让大语言模型（LLM）主动 “出击”，评估每一句话，把它们分到最合适的文本块里。和传统分块方法不同，它不依赖固定 token 长度，也不单纯看语义变化，全靠 LLM 的智能判断，把相隔较远但主题相关的句子归到一组。

假设我们有这样一段文本：

```
On July 20, 1969, astronaut Neil Armstrong walked on the moon. He was leading the NASA’s Apollo 11 mission. Armstrong famously said, “That’s one small step for man, one giant leap for mankind” as he stepped onto the lunar surface.
```

在 Agentic Chunking 中，LLM 会先进行 propositioning 处理，把每个句子独立化，让每个句子都有自己的主语，处理后就变成 ：

```
On July 20, 1969, astronaut Neil Armstrong walked on the moon.
Neil Armstrong was leading the NASA’s Apollo 11 mission.
Neil Armstrong famously said, “That’s one small step for man, one giant leap for mankind” as he stepped onto the lunar surface.
```

 这样，LLM 就能单独检查每个句子，再把它们分到合适的文本块。propositioning 就像是给文档里的句子做了次 “整容”，让它们都能独立完整地 “展示自己”。

# 3. Agentic Chunking 怎么实现？

实现 Agentic Chunking 主要靠 propositioning 和文本块的动态创建与更新，借助 Langchain 和 Pydantic 等工具就能搞定，下面一步步来拆解。

## 3.1 Propositioning 文本

可以用 Langchain 提供的提示词模板，让 LLM 自动完成这步。代码示例安排上：

```python
from langchain.chains import create_extraction_chain_pydantic
from langchain_core.pydantic_v1 import BaseModel
from typing import Optional
from langchain.chat_models import ChatOpenAI
import uuid
import os
from typing import List

from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from pydantic import BaseModel

obj = hub.pull("wfh/proposal-indexing")
llm = ChatOpenAI(model="gpt-4o")

class Sentences(BaseModel):
    sentences: List[str]

extraction_llm = llm.with_structured_output(Sentences)
extraction_chain = obj | extraction_llm

sentences = extraction_chain.invoke(
    """
    On July 20, 1969, astronaut Neil Armstrong walked on the moon.
    He was leading the NASA's Apollo 11 mission.
    Armstrong famously said, "That's one small step for man, one giant leap for mankind" as he stepped onto the lunar surface.
    """
)
```

## 3.2 创建和更新文本块

写个函数来动态生成和更新文本块。每个文本块都包含主题相似的 propositions，随着新 propositions 加入，文本块的标题和摘要也会不断更新。

```python
def create_new_chunk(chunk_id, proposition):
    summary_llm = llm.with_structured_output(ChunkMeta)
    summary_prompt_template = ChatPromptTemplate.from_messages([
        ("system", "Generate a new summary and a title based on the propositions."),
        ("user", "propositions:{propositions}"),
    ])
    summary_chain = summary_prompt_template | summary_llm
    chunk_meta = summary_chain.invoke({"propositions": [proposition]})
    chunks[chunk_id] = {
        "summary": chunk_meta.summary,
        "title": chunk_meta.title,
        "propositions": [proposition],
    }
```

## 3.3 将 proposition 推送到合适的文本块

还得有个 AI Agent 来判断新的 proposition 该加到哪个文本块。要是没合适的，就创建一个新的。

```python
def find_chunk_and_push_proposition(proposition):
    class ChunkID(BaseModel):
        chunk_id: int = Field(description="The chunk id.")

    allocation_llm = llm.with_structured_output(ChunkID)
    allocation_prompt = ChatPromptTemplate.from_messages([
        ("system", "Find the chunk that best matches the proposition. If no chunk matches, re"),
        ("user", "proposition:{proposition} chunks_summaries:{chunks_summaries}"),
    ])
    allocation_chain = allocation_prompt | allocation_llm
    chunks_summaries = {chunk_id: chunk["summary"] for chunk_id, chunk in chunks.items()}
    best_chunk_id = allocation_chain.invoke({"proposition": proposition, "chunks_summaries": 
    if best_chunk_id not in chunks:
        create_new_chunk(best_chunk_id, proposition)
    else:
        add_proposition(best_chunk_id, proposition)
```

# 4. Agentic Chunking 效果如何？

采用新加坡圣淘沙著名景点 Wings of Time 的介绍文本作为测试用例，用 GPT-4 模型处理。这段文本涵盖景点介绍、票务信息、开放时间等好多方面，很适合测试。

```
Product Name: Wings of Time

Product Description: Wings of Time is one of Sentosa's most breathtaking attractions, combining water, laser, fire, and music to create a mesmerizing night show about friendship and courage. Situated on the scenic  (https://www.sentosa.com.sg/en/things-to-do/attractions/siloso-beach/) Siloso Beach , this award-winning spectacle is staged nightly, promising an unforgettable experience for visitors of all ages. Be wowed by spellbinding laser, fire, and water effects set to a majestic soundtrack, complete with a jaw-dropping fireworks display. A fitting end to your day out at Sentosa, it’s possibly the only place in Singapore where you can witness such an awe-inspiring performance.  Get ready for an even better experience starting 1 February 2025 ! Wings of Time Fireworks Symphony, Singapore’s only daily fireworks show, now features a fireworks display that is four times longer!   Important Note: Please visit  (https://www.sentosa.com.sg/sentosa-reservation) here if you need to change your visit date. All changes must be made at least 1 day prior to the visit date.

Product Category: Shows

Product Type: Attraction

Keywords: Wings of Time, Sentosa night show, Sentosa attractions, laser show Sentosa, water show Singapore, Sentosa events, family activities Sentosa, Singapore night shows, outdoor night show Sentosa, book Wings of Time tickets

Meta Description: Experience Wings of Time at Sentosa! A breathtaking night show featuring water, laser, and fire effects. Perfect for a memorable evening.


Product Tags: Family Fun,Popular experiences,Frequently Bought

Locations: Beach Station

[Tickets]

Name: Wings of Time (Std)
Terms: • All Wings of Time (WOT) Open-Dated tickets require prior redemption at Singapore Cable Car Ticketing counters and are subjected to seats availability on a first come first serve basis. • This is a rain or shine event. Tickets are non-exchangeable or nonrefundable under any circumstances. • Once timeslot is confirmed, no further amendments are allowed. Please proceed to WOT admission gates to scan your issued QR code via mobile or physical printout for admission. • Gates will open 15 minutes prior to the start of the show. • Show Duration: 20 minutes per show. • Please be punctual for your booked time slot. • Admission will be on a first come first serve basis within the allocated timeslot or at the discretion of the attraction host. • Standard seats are applicable to guest aged 4 years and above. • No outside Food & Drinks are allowed. • Refer to  (https://www.mountfaberleisure.com/attraction/wings-of-time/) https://www.mountfaberleisure.com/attraction/wings-of-time/ for more information on Wings of Time.
Pax Type: Standard
Promotion A: Enjoy $1.90 off when you purchase online! Discount will automatically be applied upon checkout.
Price: 19



Opening Hours: Daily  Show 1: 7.40pm  Show 2: 8.40pm


Accessibilities: Wheelchair


[Information]

Title: Terms & Conditions
Description: For more information, click  (https://www.sentosa.com.sg/en/promotional-general-store-terms-and-conditions) here for Terms & Conditions


Title: Getting Here
Description: By Sentosa Express: Alight at Beach Station  By Public Bus: Board Bus 123 and alight at Beach Station  By Intra-Island Bus: Board Sentosa Bus A or B and alight at Beach Station     Nearest Car Park   Beach Station Car Park


Title: Contact Us
Description: Beach Station  +65 6361 0088   (mailto:guestrelations@mflg.com.sg) guestrelations@mflg.com.sg
```

系统先把原文转化成 50 多个独立陈述句（propositions），神奇的是，系统自动把每句话主语统一成了 “Wings of Time”，这 AI 对主题把握太准了！像 “Wings of Time is one of Sentosa's most breathtaking attractions.”“Wings of Time combines water, laser, fire, and music to create a mesmerizing night show.” 这些句子，都被整理得明明白白。

```
[
    "Wings of Time is one of Sentosa's most breathtaking attractions.",
    'Wings of Time combines water, laser, fire, and music to create a mesmerizing night show.',
    'The night show of Wings of Time is about friendship and courage.',
    'Wings of Time is situated on the scenic Siloso Beach.',
    'Wings of Time is an award-winning spectacle staged nightly.',
    'Wings of Time promises an unforgettable experience for visitors of all ages.',
    'Wings of Time features spellbinding laser, fire, and water effects set to a majestic soundtrack.',
    'Wings of Time includes a jaw-dropping fireworks display.',
    'Wings of Time is a fitting end to a day out at Sentosa.',
    'Wings of Time is possibly the only place in Singapore where such an awe-inspiring performance can be witnessed.',
    'Wings of Time will offer an even better experience starting 1 February 2025.',
    'Wings of Time Fireworks Symphony is Singapore’s only daily fireworks show.',
    'Wings of Time Fireworks Symphony now features a fireworks display that is four times longer.',
    'Visitors should visit the provided link if they need to change their visit date to Wings of Time.',
    'All changes to the visit date must be made at least 1 day prior to the visit date.',
    'Wings of Time is categorized as a show.',
    'Wings of Time is a type of attraction.',
    'Keywords for Wings of Time include: Wings of Time, Sentosa night show, Sentosa attractions, laser show Sentosa, water show Singapore, Sentosa events, family activities Sentosa, Singapore night shows, outdoor night show Sentosa, book Wings of Time tickets.',
    'The meta description for Wings of Time is: Experience Wings of Time at Sentosa! A breathtaking night show featuring water, laser, and fire effects. Perfect for a memorable evening.',
    'Product tags for Wings of Time include: Family Fun, Popular experiences, Frequently Bought.',
    'Wings of Time is located at Beach Station.',
    'Wings of Time (Std) tickets require prior redemption at Singapore Cable Car Ticketing counters.',
    'Wings of Time (Std) tickets are subjected to seats availability on a first come first serve basis.',
    'Wings of Time is a rain or shine event.',
    'Tickets for Wings of Time are non-exchangeable or nonrefundable under any circumstances.',
    'Once the timeslot for Wings of Time is confirmed, no further amendments are allowed.',
    'Visitors should proceed to Wings of Time admission gates to scan their issued QR code via mobile or physical printout for admission.',
    'Gates for Wings of Time will open 15 minutes prior to the start of the show.',
    'The show duration for Wings of Time is 20 minutes per show.',
    'Visitors should be punctual for their booked time slot for Wings of Time.',
    'Admission to Wings of Time will be on a first come first serve basis within the allocated timeslot or at the discretion of the attraction host.',
    'Standard seats for Wings of Time are applicable to guests aged 4 years and above.',
    'No outside food and drinks are allowed at Wings of Time.',
    'More information on Wings of Time can be found at the provided link.',
    'The pax type for Wings of Time is Standard.',
    'Promotion A for Wings of Time offers $1.90 off when purchased online.',
    'The discount for Promotion A will automatically be applied upon checkout.',
    'The price for Wings of Time is 19.',
    'Wings of Time has opening hours daily with Show 1 at 7.40pm and Show 2 at 8.40pm.',
    'Wings of Time is accessible by wheelchair.',
    "The title for terms and conditions is 'Terms & Conditions'.",
    'More information on terms and conditions can be found at the provided link.',
    "The title for getting to Wings of Time is 'Getting Here'.",
    'Visitors can get to Wings of Time by Sentosa Express by alighting at Beach Station.',
    'Visitors can get to Wings of Time by Public Bus by boarding Bus 123 and alighting at Beach Station.',
    'Visitors can get to Wings of Time by Intra-Island Bus by boarding Sentosa Bus A or B and alighting at Beach Station.',
    'The nearest car park to Wings of Time is Beach Station Car Park.',
    "The title for contacting Wings of Time is 'Contact Us'.",
    'The contact location for Wings of Time is Beach Station.',
    'The contact phone number for Wings of Time is +65 6361 0088.',
    'The contact email for Wings of Time is guestrelations@mflg.com.sg.']
```

经过 AI 智能分块（agentic chunking），整个文本自然地分成了四个主要部分：

- 主体信息块，装着景点核心介绍、特色、位置等综合信息；
- 日程政策块，专门处理预约变更相关内容；
- 价格优惠块，聚焦折扣和支付信息；
- 法律条款块，归纳各项条款规定。

这么一分块，每个块主题明确，不重叠，重要信息在前，辅助信息分类存放。放进向量库后，召回率提高了，RAG 准确率也跟着飙升。

# 4. 啥时候用 Agentic Chunking 才划算？

Agentic Chunking 确实厉害，能把主题相关的句子归到一组，提升 RAG 模型效果。但它也有缺点，成本和延迟相对较高。虽然准确率提升了 40%，可成本也增加了 3 倍。

那啥时候适合用它呢？根据项目经验，下面这些场景用 Agentic Chunking 就很合适

- 主题反复切换的内容；
- 非结构化文本，比如客服对话记录；
- 需要跨段落关联的系统。

对于结构清晰的论文、说明书，传统分块和语义分块性价比更高。

















