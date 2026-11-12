# SIDRA LangChain + LangGraph 综合集成说明

## 状态

本分支基于现有 LangGraph 集成 PR #15，加入 **真实 LangChain 组件**：
`langchain` + `langchain-openai` + `langchain-core` + `langchain-text-splitters`。
保留原有 LLM 适配器、BM25 排序、证据门禁、AgentTrace、TRAIN-only
SFT/GRPO；新增功能均为 opt-in，并记录在独立实验 manifest 中。

**“已接入并测试”不等于模型正确率提升。**

## 架构分工

~~~text
CLI / Evaluation
    ├── runtime=legacy: SIDRA SearchAgent
    └── runtime=langgraph: StateGraph
             ├── Planner / Researcher / Solver (可选独立模型)
             ├── Search（受控检索）
             ├── Read（只读已检索 doc_id 的全文）
             └── Verifier（已有 source-only / read-before-cite）
                 ↓
        LangChain 组件层
             ├── ChatOpenAI (明确传入模型与 API)
             ├── RecursiveCharacterTextSplitter
             ├── Embeddings (接口与 OpenAIEmbeddings 适配)
             ├── InMemoryVectorStore
             ├── BM25 旧基线
             └── Hybrid：BM25 + Dense RRF(排名融合)
                 ↓
        SIDRA Research
           Evidence State / AgentTrace
           TRAIN-only rewards / On-policy GRPO
~~~

重点不是再创建一个与 LangGraph 冲突的高层 LangChain Agent Loop。
LangGraph 负责图节点的生命周期，LangChain 负责模型与 RAG 组件；
独立 Verifier 不能由 LLM 自己给分。

## 安装与依赖

~~~bash
python -m pip install -e '.[test,graph,langchain]'
python -m pytest -q tests/test_langchain_integration.py
~~~

为了保留仓库原有 **Python 3.10 / 3.12 CI**，
使用支持 Python 3.10 的 LangChain 1.4.x 版本，而不是直接升级到
最近要求 Python >=3.11 的 LangChain 1.5.x。
版本在 pyproject.toml 锁定，修改须重新运行全部 offline CI。

## CLI：LangChain ChatModel，保留 BM25

~~~bash
sidra eval --dataset /path/to/licensed/dev_data --split dev \
  --policy multi-openai \
  --runtime langgraph --graph-role-mode multi \
  --model-backend langchain \
  --endpoint http://127.0.0.1:8000/v1/chat/completions \
  --model your-local-openai-compatible-model \
  --retriever bm25 --limit 32 \
  --out /outside-repo/graph-langchain-chat
~~~

ChatModel 不会产生隐藏的另一个 Agent 循环：每轮仍只产生一个 SIDRA Action。
API Key 从 --key-env 指定环境变量读取；远程 HTTP 被拒绝，
只有显式配置本机 localhost/127.0.0.1 时允许无密钥。

## CLI：LangChain Dense / Hybrid Retrieval

~~~bash
sidra eval --dataset /path/to/licensed/dev_data --split dev \
  --policy multi-openai \
  --runtime langgraph --graph-role-mode multi \
  --model-backend langchain \
  --endpoint http://127.0.0.1:8000/v1/chat/completions \
  --model your-local-model \
  --retriever hybrid \
  --embedding-endpoint http://127.0.0.1:8001/v1 \
  --embedding-model your-local-embedding-model \
  --retrieval-chunk-size 1200 --retrieval-chunk-overlap 120 \
  --retrieval-dense-weight 0.5 \
  --out /outside-repo/graph-langchain-hybrid
~~~

**必须由用户自行提供可使用的模型和 Embeddings 服务。**
没有 embedding endpoint / model 时只允许 bm25；
不会自动上传文档到默认云服务。
**索引时外部 Embedding Provider 会看到被选取的来源文档文本**，
必须确认文档许可与隐私后才能使用远程地址。

召回接口保持 `search(query, k, scope) -> Hit[doc_id, title, snippet, score]`，
因此 `read(doc_id)`、引用校验与训练奖励均使用原始文档 ID，
而不是 LangChain 内部 chunk ID。跨问题 scope 搜索结果必须严格过滤。

- Dense：LangChain 文档切分 → Embeddings → InMemoryVectorStore → 余弦检索；
  多个命中的 chunk 归并到同一个原始 doc_id。
- Hybrid：分别排序 BM25 和 Dense；按排名做 RRF，不能直接相加
  BM25 分数和余弦分数。
- 离线测试用合成 Embeddings 和假的 ChatModel，不产生 API 费用。
- 原始 BM25 未改变，所有新选项写入 manifest；没有新模型参数更新。

## 公平对照与边界

建议预声明三个控制组：BM25、Dense、Hybrid，分别在 Legacy 和
LangGraph 下运行，固定同一数据哈希、embedding 模型及版本、LLM、
工具预算和种子。报告答案 EM、严格证据正确、Support Recall、
调用成本/Token/延迟、未完成和拒答比例。

目前 LangGraph + LangChain 不等于已经改善 M8 的失败问题：
正确候选答案可达性、语义支持判定仍然是核心研究难题。
TRL/GRPO 的 raw policy 概率采样机制不因为添加 LangChain 而自动变更；
Graph RL 单策略路径仍走原 OnPolicyHFAction，防止改变采样分布后
错误宣称 on-policy 更新。

## 文件与验收入口

- src/sidra/langchain_policy.py：ChatOpenAI、OpenAIEmbeddings、显式 endpoint。
- src/sidra/langchain_retrieval.py：Splitter/VectorStore/Dense/Hybrid。
- src/sidra/agent.py / langgraph_agent.py：检索器注入。
- src/sidra/experiment.py：manifest/指标复用。
- src/sidra/cli.py：实际 CLI 开关。
- tests/test_langchain_integration.py：作用域、文档 ID、模型接口、奖励边界等。
