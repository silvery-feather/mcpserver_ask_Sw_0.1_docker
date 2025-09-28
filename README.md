# macOS / Linux（终端 / VSCode 终端）

##  选择一个提前准备好的空文件夹
cd ---

## 0) 安装 uv（包管理 + Python 管理）
curl -LsSf https://astral.sh/uv/install.sh | sh

## 1) 安装一个可用的 Python 版本（与你的依赖最稳的版本）
uv python install 3.12

## 2) 克隆仓库 // 直接下载可以忽略
git clone https://github.com/silvery-feather/mcpserver_ask_Sw_0.1
cd mcpserver_ask_Sw_0.1

## 3) 设置 DashScope Key（仅本次终端会话）
export DASHSCOPE_API_KEY=替换为你的Key

## 4) 启动 MCP（stdio）
##    --python 3.12：强制用上面装好的 3.12 运行时
##   首次运行会按 pyproject.toml 自动安装依赖（faiss/numpy 等）
uv run --python 3.12 python main.py




# Windows（PowerShell）

## 0) 安装 uv
iwr https://astral.sh/uv/install.ps1 -UseBasicParsing | iex

## 1) 安装 Python 3.12（由 uv 管理，无需单独装 Python）
uv python install 3.12

## 2) 克隆仓库
git clone https://github.com/silvery-feather/mcpserver_ask_Sw_0.1
cd mcpserver_ask_Sw_0.1

## 3) 设置 DashScope Key（仅当前窗口）
$env:DASHSCOPE_API_KEY="替换为你的Key"

## 4) 启动 MCP（stdio），依赖会自动按 pyproject 安装
uv run --python 3.12 python main.py

