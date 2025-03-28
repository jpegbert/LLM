

```
github：https://github.com/yuruotong1/autoMate/
```

[TOC]

你是否有过这样的经历：深夜还在办公室对着电脑，一遍又一遍处理那些重复又繁琐的工作，累得头晕眼花，却又没办法摆脱？这些琐碎任务不仅消耗了我们大量的时间和精力，还让我们根本没时间去做那些真正能发挥创造力、实现自我价值的事。别担心，今天就给大家介绍一款神器 ——autoMate，它就像你的专属 “数字同事”，能帮你轻松搞定这些难题，找回工作与生活的平衡！

# 1. autoMate 是什么？

autoMate 可不是一般的工具，它是处于 AGI 第三阶段的智能助手，更是一款融合了 AI 和 RPA（机器人流程自动化）技术的革命性产品。它基于 OmniParser 构建，简单来说，就是能让 AI 化身不知疲倦的 “数字员工”，帮你处理各种电脑上的工作。

它有多厉害呢？看看这些超能力你就知道了：

- **自动操作电脑界面**：不管多复杂的工作流程，它都能自动完成。比如日常办公中涉及多个软件的协同操作，它能行云流水般依次执行，完全不用你手动干预。
- **智能理解屏幕内容**：就像有了人类的视觉一样，它能看懂屏幕上显示的各种信息，然后根据内容模拟人类的操作，精准又高效。
- **自主决策**：遇到任务时，它能根据具体需求进行判断，然后选择最合适的行动方案，就像一个经验丰富的员工，总能做出正确选择。
- **支持本地化部署**：现在大家都很重视数据安全和隐私，autoMate 支持本地化部署，把数据稳稳地保护在本地，再也不用担心隐私泄露的问题啦。

而且，和传统 RPA 工具相比，autoMate 简直是 “降维打击”。传统工具设置规则特别繁琐，而 autoMate 借助大模型的强大能力，你只要用自然语言把任务描述清楚，它就能自动完成复杂的自动化流程，哪怕你完全不懂编程也没关系！

# 2. 为什么它能改变你的工作方式？

咱们来看看一位虚构的财务经理的真实反馈：“在我使用 autoMate 之前，我每天花费 3 小时处理报表；现在，我只需 10 分钟设置任务，然后去做真正重要的事。” 是不是很惊人？以前要花几个小时的工作，现在短短 10 分钟就能搞定，节省下来的时间可以用来提升自己、陪伴家人，或者做更有价值的工作。

想象一下，每天早上醒来，打开电脑，发现昨晚安排的数据整理、报表生成、邮件回复这些繁琐工作都已经完成了，摆在你面前的只有那些需要智慧和创造力的任务，这种感觉简直不要太爽！autoMate 带给我们的，不仅仅是效率的大幅提升，更是对创造力的解放，让我们有机会在工作中展现更多的价值。

# 3. autoMate 功能特点大揭秘

autoMate 之所以这么受欢迎，还得靠它那些实用又贴心的功能：

- **无代码自动化**：不会编程？完全不用担心！用日常说话的方式把任务告诉它就行，它听得懂，也能做得好。
- **全界面操控**：不管是办公软件、专业设计软件，还是其他各种可视化界面，它都能轻松操作，不受特定软件的限制。
- **简化安装**：安装过程超简单，比官方版本还要简洁，而且支持中文环境，一键就能部署成功，对新手超级友好。
- **本地运行**：前面提到过，数据安全有保障，在本地运行，不用把数据上传到云端，隐私安全稳稳拿捏。
- **多模型支持**：主流的大型语言模型它都能兼容，大家可以根据自己的需求和喜好选择，灵活性拉满。
- **持续成长**：它就像你的专属小助手，会随着你的使用，越来越了解你的工作习惯和需求，越来越懂你，提供更个性化的服务。

# 4. 快速上手教程

这么厉害的工具，要怎么安装和使用呢？别着急，教程来啦：

## 4.1 安装

- 首先，在命令行中输入`git clone https://github.com/yuruotong1/autoMate.git`，克隆项目。
- 接着，进入项目文件夹：`cd autoMate`。
- 然后创建一个新的 conda 环境：`conda create -n "automate" python==3.12`。
- 激活这个环境：`conda activate automate`。
- 最后安装所需依赖：`pip install -r requirements.txt`。

## 4.2 启动应用

安装完成后，在命令行输入`python main.py`，然后在浏览器中打开`http://localhost:7888/` ，在这里配置你的 API 密钥和基本设置，就可以开始使用啦！

# 5. 常见问题及解决办法

有些小伙伴在启动的时候可能会遇到 “显卡驱动不适配，请根据 readme 安装合适版本的 torch” 这个问题。遇到这种情况，有两种解决办法：

- 要是不着急，对速度要求不高，你可以不用管这条信息，只用 CPU 运行，不过运行速度会比较慢。
- 要是想让它跑得更快，那就按照下面的步骤来：
  - 运行`pip list`查看 torch 版本；
  - 从官网查看支持的 cuda 版本；
  - 重新安装 Nvidia 驱动。











