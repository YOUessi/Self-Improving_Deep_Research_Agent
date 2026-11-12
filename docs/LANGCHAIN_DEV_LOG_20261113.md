# 2026-11-13 · LangChain + LangGraph + Multi-Agent + RAG + Agentic RL 集成验收日志

## 背景和版本

用户要求将 LangChain 真正接入此前已经实现 LangGraph 的 Self-Improving Deep Research Agent，
而不是仅添加依赖或写一个独立 Demo。原项目中 multi-hop SearchAgent、BM25、Evidence State、
source-only Verifier 与 TRAIN-only on-policy GRPO 脚本必须保留。

- GitHub: YOUessi/Self-Improving_Deep_Research_Agent
- PR: #15（Draft），base=feat/m8-chain-state-v3
- 开发分支：feat/langgraph-multiagent-rag-rl-integration
- 这轮最初的 HEAD：ee46b83ad696b85d5f2b79937ef7ffcc4ea3ae51
- 实现与全部新增测试的 HEAD：a0fd6340ecb9c259bfab310dd61ef6a648be4f3c
- 旧 M7/M8 的真实 Qwen/GRPO 负结果不覆盖、不回填、不当成这次成绩。

## 1. 实际代码改动

| 提交 | 作用 |
|---|---|
| 4330d90dc7 | 新增 langchain_policy.py：ChatOpenAI、OpenAIEmbeddings 适配器；新增 langchain_retrieval.py：Splitter + VectorStore + Dense + Hybrid RRF |
| 76651c7d5c | 注入 LangChain RAG 到原 SearchAgent、LangGraph GraphResearchAgent、统一 evaluate 和 CLI；pyproject 添加 optional langchain extra；CI 同时安装依赖 |
| ee493dea2e | 新增 tests/test_langchain_integration.py、中文集成说明、README 运行示例 |
| cc19fd9eb4 | 修复测试对固定工具次数的错误假设，保留严格文档与引用断言 |
| a0fd6340ec | 增加实际本地 HTTP OpenAI-compatible ChatCompletions / Embeddings 的 CLI + LangGraph 端到端集成测试 |

新增运行参数：
- `--model-backend native|langchain`（默认 native）
- `--retriever bm25|dense|hybrid`（默认 bm25）
- `--embedding-endpoint`、`--embedding-model`、`--embedding-key-env`
- `--retrieval-chunk-size`、`--retrieval-chunk-overlap`、`--retrieval-dense-weight`

选择 Dense/Hybrid 时必须显式配置 Embedding Provider。默认路径不会进行外部 embedding API 调用。
模型/向量检索改动不改变原 GRPO 采样分布与 loss 实现。

## 2. 真实问题、原因与修复

**首次 CI 失败**：新测试中 `test_langchain_retrieval_executes_in_real_graph_and_train_scorer`
的 Dense / Hybrid 两种参数均断言 `trace.tool_calls == 4`；
实际输出为 3，其他答案和引用断言已经通过。

**确认的原因**：语义检索改变候选文档的暴露顺序；第一次 Search 就能返回两篇相关文档，
Agent 可以一次 Search + 两次 Read + 一次 Finish，不再必然需要第二次 Search。
这是检索结果变化，而不是工具预算绕过或引用泄漏。

**实际修复**：不再硬编码 Search 必须发生两次；要求两个独立 Read、
1–2 次 Search、工具数始终等于 Search + Read，仍要求答案 Quartz9、
两份已读引用、源限定检索和 TRAIN-only strict reward 都通过。

没有改变 Verifier、准许跨 scope 读取、关闭测试或更改参考答案。

## 3. 已完成验证

### GitHub Actions（精确提交 a0fd6340ecb9c259bfab310dd61ef6a648be4f3c）

- Python 3.10：workflow run 43295251081，**300 passed / 27 skipped**，compileall 成功
- Python 3.12：workflow run 43295251081，**300 passed / 27 skipped**，compileall 成功
- 同一提交另一条 PR workflow run 43295239520 也成功
- 27 项 skipped 均为未安装 torch 的 GPU/训练相关可选测试，
  不能表示 TRAIN-only 实际 GPU 实验已重新跑过。

新测试已覆盖：

1. LangChain 实际 RecursiveCharacterTextSplitter / InMemoryVectorStore，
   Chunk ID 映射回原 source doc_id，并强制 per-question scope。
2. Dense/Hybrid RRF、与旧 BM25 baseline 同输出合同、确定性排序。
3. LangGraph 和旧 Agent Loop 使用同一检索器的端到端 Search/Read/Finish。
4. Verifier / TRAIN-only reward 与 traces 维持原接口，DEV/TEST 仍禁止产生训练奖励。
5. ChatOpenAI adapter 使用真实 LangChain message 对象、结构化动作校验、用量记录、
   显式 provider endpoint 和缺少 key 时 fail closed。
6. 本机 OpenAI-compatible HTTP server（确定性响应，不是真实模型）下，
   真正通过 CLI 经过 ChatOpenAI + OpenAIEmbeddings + LangGraph + Hybrid
   完成一次标准问答，并确保不泄漏 evaluator gold 字段及跨问题 doc_id。

## 4. 明确没有完成的事情

- 还没有真实 Embedding 模型 + Qwen3 在未见 HotpotQA 题目上做 BM25/Dense/Hybrid
  三臂消融；HTTP mock 只验证协议/工程连接，不证明检索性能。
- 没有启动新的 GRPO 梯度更新；旧的 TRAIN-only GRPO 与运行轨迹不变。
- 没有证明 LangChain 提高 Agent 的答案 EM、证据成功率或降低成本。
- 没有实现多代 RSI；LangChain 和 LangGraph 是架构组件，不是自动训练算法。
- 本 PR 仍为 Draft，旧 M7/M8 分支未合并到仓库空白的 main。

## 5. 下一轮研究门禁

- 在明确许可的模型服务、冻结文档集及独立数据上进行公平的 BM25/Dense/Hybrid
  真实检索覆盖比较，再执行 Graph single/multi 的真实 LLM A/B。
- 控制 source SHA、embedding/model/provider、采样和 token/tool budget，
  记录检索 Hit@k、严格证据成功、EM、模型调用、token、API 费用。
- 在发现证据质量真正改善之前，避免无目的地扩展 Agentic GRPO 训练。
- RL 只能从 TRAIN-only 原始 action-token + 工具轨迹计算奖励和梯度；
  不得将 Multi role Prompt 与 single-policy logits 混算。
