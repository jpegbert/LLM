[TOC]

![image-20250111220802662](魔镜(MagicMirror) 轻量级 AI换脸 换装 换发型 CPU可运行.assets/image-20250111220802662.png)

## 1. MagicMirror 诞生的背景

在时尚探索过程中，人们常常好奇不同发型和服装上身效果，却因现有 AI 人脸应用的复杂设置、高端 GPU 硬件依赖及隐私风险而受阻。传统应用要么需繁琐操作和专业知识，要么借助损害隐私的云处理，难以满足大众便捷尝试新造型的需求。在此背景下，MagicMirror 应运而生，旨在提供如自拍般简单、无需昂贵硬件且保障隐私的解决方案，让用户轻松打造全新形象。

![image-20250111220838469](魔镜(MagicMirror) 轻量级 AI换脸 换装 换发型 CPU可运行.assets/image-20250111220838469.png)

## 2. MagicMirror 简介

MagicMirror 是一款即时人工智能换脸，项目基于 Tauri、FaceFusion、InsightFace、Nuitka 等优秀开源项目构建，用于个人娱乐和创意目的，禁止商业使用，并对用户行为有明确伦理和法律约束。目前版本为 v1.0.0，支持 macOS 13（Ventura）和 Windows 10 及以上系统。

![image-20250111221324271](魔镜(MagicMirror) 轻量级 AI换脸 换装 换发型 CPU可运行.assets/image-20250111221324271.png)

## 3. MagicMirror 的功能亮点

- **操作极简**：采用拖放照片方式即可瞬间换脸、变换发型和服装，无需复杂设置，用户无需技术专业知识就能轻松上手。
- **硬件适配广**：能在标准计算机上流畅运行，摆脱对专用 GPU 硬件的依赖，降低使用门槛，使更多普通设备用户受益。
- **隐私保障强**：完全离线处理，确保用户图像数据始终在本地设备，有效避免隐私泄露风险，让用户安心使用。
- **轻巧便携**：安装程序小于 10MB，模型文件小于 1GB，占用空间小，便于下载、存储和安装，对设备存储要求低。

![image-20250111220546426](魔镜(MagicMirror) 轻量级 AI换脸 换装 换发型 CPU可运行.assets/image-20250111220546426.png)

## 4. MagicMirror 使用方法

## 4.1 安装

安装指南：https://github.com/idootop/MagicMirror/blob/main/docs/en/install.md

下载并安装适用于您的操作系统的 MagicMirror 安装程序：

1. Windows: [MagicMirror_1.0.0_windows_x86_64.exe](https://github.com/idootop/MagicMirror/releases/download/app-v1.0.0/MagicMirror_1.0.0_windows_x86_64.exe)
2. macOS: [MagicMirror_1.0.0_macos_universal.dmg](https://github.com/idootop/MagicMirror/releases/download/app-v1.0.0/MagicMirror_1.0.0_macos_universal.dmg)
3. Other: [Go to Release](https://github.com/idootop/MagicMirror/releases/app-v1.0.0) 

## 4.2 下载模型

首次启动该应用程序时，它将自动下载所需的模型文件。

如果下载进度卡在 0% 或中途停止，请按照以下步骤进行手动设置：

选择与操作系统匹配的模型文件：

- [server_windows_x86_64.zip](https://github.com/idootop/MagicMirror/releases/download/server-v1.0.0/server_windows_x86_64.zip)（Windows）
- [server_windows_aarch64.zip](https://github.com/idootop/MagicMirror/releases/download/server-v1.0.0/server_windows_aarch64.zip)（Windows，适用于 ARM64 设备）
- [server_macos_aarch64.zip](https://github.com/idootop/MagicMirror/releases/download/server-v1.0.0/server_macos_aarch64.zip)（macOS、Apple Silicon，如 M1、M4 芯片）
- [server_macos_x86_64.zip](https://github.com/idootop/MagicMirror/releases/download/server-v1.0.0/server_macos_x86_64.zip)（macOS、Intel 芯片）

解压缩下载的文件。将获得一个文件夹 - 将其重命名为 `MagicMirror`。将此文件夹移动到计算机的 `HOME` 目录。

重新启动 MagicMirror，它现在应该可以正常工作了

注意：下载模型文件后，首次启动可能需要一些时间。

## 5. 与其他 AI 换脸应用的区别

当前市面上流行的AI换脸工具是Facefusion、DeepLiveCamde。

MagicMirror 与 Facefusion、DeepLiveCamde 在多方面存在区别。

- 硬件方面，MagicMirror 可在标准计算机流畅运行无需专用 GPU，Facefusion 和 DeepLiveCamde 对硬件配置要求高。
- 隐私方面，MagicMirror 完全离线处理保障隐私，后两者部分功能涉及云端处理有隐私风险。
- 操作方面，MagicMirror 拖放照片即可操作极为便捷，Facefusion 和 DeepLiveCamde 相对复杂。
- 功能侧重也不同，MagicMirror 主打换脸、发型和服装变换，Facefusion 侧重人脸融合，DeepLiveCamde 倾向实时视频流图像处理与特效添加。

```
github：https://github.com/idootop/MagicMirror
演示视频：https://www.bilibili.com/video/BV1TTzfYDEUe
安装指南：https://github.com/idootop/MagicMirror/blob/main/docs/en/install.md
release版本：https://github.com/idootop/MagicMirror/releases/tag/app-v1.0.0
```









