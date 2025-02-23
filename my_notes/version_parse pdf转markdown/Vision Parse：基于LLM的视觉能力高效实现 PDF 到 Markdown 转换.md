```
github链接：https://github.com/iamarunbrahma/vision-parse
```

[TOC]

# 1. Vision Parse是什么

Vision Parse 是一款功能强大的文档处理工具，利用视觉语言模型的力量改变文档处理方式，旨在将 PDF 文档转换为格式精美的 Markdown 内容，为用户提供高效便捷的文档转换体验。

# 2. 功能特点

- **精准识别与提取**：在扫描文档处理方面表现出色，能够智能识别扫描文档中的文本、表格和 LaTeX 方程，并高精度地提取为 Markdown 格式的内容。例如，对于包含复杂数学公式的学术论文 PDF，它可以准确地将 LaTeX 方程转换为 Markdown 中可编辑的格式，同时保留文本的原始排版和逻辑结构。
- **内容格式保留**：高度重视内容格式的完整性，能够保留 Markdown 格式内容的 LaTeX 方程、超链接、图像和文档层次结构。这意味着转换后的 Markdown 文档在视觉和逻辑上与原始 PDF 文档高度相似，方便用户直接进行后续的编辑和使用。比如，文档中的图片会以合适的方式在 Markdown 中呈现，超链接也依然有效。
- **多模型集成优势**：支持与 OpenAI、Gemini 和 Llama 等多个 Vision LLM 提供商无缝集成。这种多 LLM 支持的特性使得用户可以根据不同的需求和场景选择最合适的模型，从而实现最佳的准确性和速度。例如，在处理一些需要高度语言理解能力的文档时，可以选择 OpenAI 的模型；而对于某些对特定领域知识有要求的文档，可能 Llama 模型会更合适。
- **本地模型托管功能**：支持 Ollama 本地模型托管，为用户提供了安全、免费、私密和离线的文档处理环境。这对于那些对数据安全有严格要求，或者在网络环境不稳定的情况下需要处理文档的用户来说非常重要。用户可以在本地部署模型，无需担心数据传输和隐私问题。

# 3. 如何使用

## 3.1 环境配置

满足先决条件

- 确保系统安装了 Python 3.9 及以上版本。
- 若要使用本地模型，需安装 Ollama。
- 若计划使用 OpenAI 或 Google Gemini 模型，要获取相应的 API 密钥。

安装 vision-parse 

```
pip install vision-parse
```

安装 OpenAI 或 Gemini 的附加依赖项：

```
# For OpenAI support
pip install 'vision-parse[openai]'
```

```
# For Gemini support
pip install 'vision-parse[gemini]'
```

```
# To install all the additional dependencies
pip install 'vision-parse[all]'
```

或者从源代码安装

```
pip install 'git+https://github.com/iamarunbrahma/vision-parse.git#egg=vision-parse[all]'
```

设置 Ollama（可选）

查阅Examples/ollama_setup.md 。

## 3.2 用法

### 3.2.1 基本用法

```python
from vision_parse import VisionParser

# Initialize parser
parser = VisionParser(
    model_name="llama3.2-vision:11b", # For local models, you don't need to provide the api key
    temperature=0.4,
    top_p=0.5,
    image_mode="url", # Image mode can be "url", "base64" or None
    detailed_extraction=False, # Set to True for more detailed extraction
    enable_concurrency=False, # Set to True for parallel processing
)

# Convert PDF to markdown
pdf_path = "path/to/your/document.pdf" # local path to your pdf file
markdown_pages = parser.convert_pdf(pdf_path)

# Process results
for i, page_content in enumerate(markdown_pages):
    print(f"\n--- Page {i+1} ---\n{page_content}")
```

### 3.2.2 自定义 Ollama 配置以获得更好的性能

```python
from vision_parse import VisionParser

custom_prompt = """
Strictly preserve markdown formatting during text extraction from scanned document.
"""

# Initialize parser with Ollama configuration
parser = VisionParser(
    model_name="llama3.2-vision:11b",
    temperature=0.7,
    top_p=0.6,
    num_ctx=4096,
    image_mode="base64",
    custom_prompt=custom_prompt,
    detailed_extraction=True,
    ollama_config={
        "OLLAMA_NUM_PARALLEL": 8,
        "OLLAMA_REQUEST_TIMEOUT": 240,
    },
    enable_concurrency=True,
)

# Convert PDF to markdown
pdf_path = "path/to/your/document.pdf"
markdown_pages = parser.convert_pdf(pdf_path)
```

### 3.2.3 OpenAI 或 Gemini 模型使用示例

```python
from vision_parse import VisionParser

# Initialize parser with OpenAI model
parser = VisionParser(
    model_name="gpt-4o",
    api_key="your-openai-api-key", # Get the OpenAI API key from https://platform.openai.com/api-keys
    temperature=0.7,
    top_p=0.4,
    image_mode="url",
    detailed_extraction=True, # Set to True for more detailed extraction
    enable_concurrency=True,
)

# Initialize parser with Google Gemini model
parser = VisionParser(
    model_name="gemini-1.5-flash",
    api_key="your-gemini-api-key", # Get the Gemini API key from https://aistudio.google.com/app/apikey
    temperature=0.7,
    top_p=0.4,
    image_mode="url",
    detailed_extraction=True, # Set to True for more detailed extraction
    enable_concurrency=True,
)
```

# 4. 支持的LLM模型

Vision Parse 支持多种 Vision LLM 模型，具体如下：

- **OpenAI**：包括 gpt-4o、gpt-4o-mini。这些模型在语言处理能力上表现出色，能够对 PDF 中的文本进行准确的理解和转换，适用于多种类型文档的处理，尤其是对语言逻辑和语义理解要求较高的场景。
- **Google Gemini**：有 gemini-1.5-flash、gemini-2.0-flash-exp、gemini-1.5-pro 等。Google Gemini 系列模型以其强大的性能和创新的架构为 Vision Parse 提供了可靠的支持，在处理复杂文档结构和内容提取方面具有一定优势。
- **Meta Llama 和 LLava（来自 Ollama）**：涵盖 llava:13b、llava:34b、llama3.2-vision:11b、llama3.2-vision:70b。这些模型在与 Vision Parse 的集成中，能够利用自身特点，为用户提供不同性能和功能侧重的选择，例如在某些特定领域知识的处理或对本地模型托管需求的场景下发挥重要作用。