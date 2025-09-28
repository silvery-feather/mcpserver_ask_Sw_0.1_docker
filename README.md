##如果你想要在cursor与trae中配置这个mcpserver

## macOS / Linux

```bash
git clone https://github.com/silvery-feather/mcpserver_ask_Sw_0.1_docker  
cd mcpserver_ask_Sw_0.1_docker  

export DASHSCOPE_API_KEY="你的Key"  
docker compose up -d --build  
```

## Windows PowerShell

```bash
git clone https://github.com/silvery-feather/mcpserver_ask_Sw_0.1_docker  
cd mcpserver_ask_Sw_0.1_docker  

$env:DASHSCOPE_API_KEY="替换为你的Key"  
docker compose up -d --build  

```

## cursor的进一步配置（必须进行）
1.打开cursor的 cursor setting  
2.在左侧列表找到mcp  
3.点击add new custom mcp server  
4.此时你会发现打开了一个叫做mcp.json的文件，直接复制粘贴即可  
###  MCP 配置示例（JSON）

```json
{
  "mcpServers": {
    "swanlab-docs": {
      "transport": "sse",
      "url": "http://localhost:8765/sse"
    }
  }
}
```
5.如果你想要配置到trae中，逻辑同上，json示例为通用格式，直接复制粘贴即可。
