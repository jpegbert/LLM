[TOC]

# 1. 传统短期记忆管理的困境

现在大多数聊天机器人系统在处理短期记忆时，采用的方法都很 “实在”—— 把对话历史中的每一句话都存储下来。大模型通常借助 ChatML 的 prompt 模板，自动将这些对话拼接进去。就像下面这样（这里的示例代码有部分乱码，但不影响理解大致结构）：

```plaintext
<lin startl>system
You are ChatGPT,a large language model trained by OpenAI, Answer as concisely as possible.
Knowledge cutoff:2021-09-01
Current date:2023-03-01<|im_endl>
<im_start>user
How are you|im_end> 
<im_start>assistant
I am doing well|im_end> 
<im_start>user
How are you now<|in_end|>
```

但这种方式就像我们聊天时，非要记住对方说的每一个字、每一个标点符号一样，并不合理。实际应用中，它带来了不少麻烦：

- **Token 用量暴增**：每次对话都要消耗大量 token。token 就像是大模型处理文本的 “燃料”，消耗太快成本可吃不消。
- **上下文窗口溢出**：模型处理上下文的能力是有限的，存的对话太多，很容易超出它的处理上限，导致信息丢失或出错。
- **响应延迟**：要处理的历史信息太多，模型思考的时间就变长了，我们等待回复的时间也跟着变长，体验感直线下降。

# 2. 用 langgraph 模拟ChatBot

```python
from langgraph.graph import MessagesState, START, END, StateGraph


def chat_model_node(state: MessagesState):
    response = llm.invoke(state["messages"])
    return {"messages": response}


builder = StateGraph(MessagesState)

builder.add_node("chat_model_node", chat_model_node)

builder.add_edge(START, "chat_model_node")
builder.add_edge("chat_model_node", END)

graph = builder.compile()


graph_response = graph.invoke({"messages": messages})

for m in graph_response["messages"]:
    m.pretty_print()
```

![image-20250217221500550](让 AI 智能体秒变 "记忆大师"：3 招搞定短期记忆管理.assets/image-20250217221500550.png)



# 3. 三种短期记忆管理策略

## 3.1 RemoveMessage：简单粗暴的 “记忆瘦身法”

RemoveMessage 是 langgraph 里一个简单有效的短期记忆管理方法，原理就是只保留最近的 N 条对话记录。比如说，我们想让聊天机器人只 “记住” 最后 3 条消息，可以这样写代码：

```python
from langchain_core.messages import RemoveMessage


def filter_messages(state):
    messages = state["messages"]
    # 只保留最后3条消息
    messages = [RemoveMessage(m.id) for m in messages[:-3]]
    return {"messages": messages}


builder = StateGraph(MessagesState)

builder.add_node("filter_messages", filter_messages)
builder.add_node("chat_model_node", chat_model_node)

builder.add_edge(START, "filter_messages")
builder.add_edge("filter_messages", "chat_model_node")
builder.add_edge("chat_model_node", END)

graph = builder.compile()
```

这个方法虽然简单，但也有缺点。重要的历史信息可能会被误删，而且每次输入 prompt 的轮次固定，可能过长或过短，影响效果。

## 3.2 trim_messages：精准控制 “记忆量”

trim_messages 方法同样是保留一定量的记忆历史，但它更智能，会计算保留的历史信息符合我们设定的特定 tokens 大小。代码如下：

```python
from langchain_core.messages import trim_messages


def chat_model_node(state: MessagesState):
    messages = trim_messages(
        allow_partial=True,   # 允许消息在中间部分部分拆分；这种方法可能会丢失上下文。
        strategy="last",   # 从哪里开始算，last最后开始
        max_tokens=100,  # 最大tokens数量
        token_counter=ChatOpenAI(model="gpt-3.5-turbo"),
        state=state["messages"]
    )
    response = llm.invoke(messages)
    return {"messages": response}
    

builder = StateGraph(MessagesState)

builder.add_node("chat_model_node", chat_model_node)

builder.add_edge(START, "chat_model_node")
builder.add_edge("chat_model_node", END)

graph = builder.compile()
```

这里设置`allow_partial=True`，表示允许消息部分拆分，但可能会丢失上下文；`strategy="last"`是从最后开始计算；`max_tokens=100`指定了最大 tokens 数量；`token_counter`则用于计算 token 数量。这样就能更精准地控制记忆量，避免出现 token 用量过多和上下文窗口溢出的问题。

## 3.3 动态摘要：抓住对话 “重点”

动态摘要方法是定期对对话历史进行总结，只保留关键信息。实现这个方法有个小技巧，当积累到 K 轮对话才开始总结，这样能避免频繁总结带来的过多消耗。来看代码：

```python
def summarize_conversation(state: State):
    summary = state.get("summary", "")
    if summary:
        summary_message = (
            "This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
    else:
        summary_message = "Create a summary of the conversation above:"
    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = model.invoke(messages)
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"summary": response.content, "messages": delete_messages}


def should_continue(state: State) -> Literal["summarize_conversation", END]:
    messages = state["messages"]
    if len(messages) > 6:
        return "summarize_conversation"
    return END


workflow = StateGraph(State)
workflow.add_node("conversation", call_model)
workflow.add_node(summarize_conversation)
workflow.add_edge(START, "conversation")
workflow.add_conditional_edges(
    "conversation",
    should_continue,
)
workflow.add_edge("summarize_conversation", END)
```

在`should_continue`函数里，设置当对话消息数量超过 6 条时，就调用`summarize_conversation`函数进行总结。总结时，会根据之前有没有总结过生成不同的提示消息，让模型生成新的摘要，然后删除旧消息，只保留关键信息。

