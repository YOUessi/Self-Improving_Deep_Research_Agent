# SIDRA LangGraph + Multi-Agent + RAG + Agentic RL 集成合同

本分支以 feat/m8-chain-state-v3 为冻结开发基线。当前实现是**可选的研究型编排架构**，
不宣称改进既有模型能力，也不将 M7/M8 的负结果抹掉。

## 架构及模块

~~~text
START
  → observe (可选择 evidence_state v1/v2/v3；默认关闭)
  → supervisor role routing
       ├─ Planner（首次规划）
       ├─ Researcher（还缺实际 READ 来源时）
       ├─ Solver（证据较完整时）
       └─ Single Policy（与旧 RL 行为分布对齐）
  → role LLM/Python Policy
  → search_or_read(BM25 / scoped READ) → observe
  → verify_finish(source-only) → observe 或 END
~~~

- LangGraph: StateGraph、条件边、真实节点执行、可选 SQLite checkpoint。
- Multi-Agent：显式角色化节点，复用同一真实模型；不是多个独立模型进程。
- RAG：原 BM25Index，不偷偷替换数据集；先 search 才能 read，严格限制 scope。
- Evidence State：复用原 derive_evidence_state；只有实际得到的 Search/Read 结果可进入，
  从不读取 gold answer / supporting_doc_ids。M8 V3 的负结果要求默认 off。
- Verifier：复用 EvidenceAudit，验证来源数、read-before-cite 和字面答案包含。
  **不能把字面包含等同于语义蕴含**；不把 evaluator 标签送给模型。
- AgentTrace：沿用原 trace schema，使 score_trace、TRAIN-only reward 可读取同一轨迹。

## 无 GPU 的实际图运行

~~~bash
python -m pip install -e '.[test,graph]'
python -m pytest -q tests/test_langgraph_agent.py
python scripts/generate_fixture.py --out examples/fixture --n 40 --seed 13
sidra eval --dataset examples/fixture --split dev \
  --policy fixture-multihop --runtime langgraph \
  --graph-role-mode multi --graph-evidence-state off \
  --out results/graph_smoke
~~~

fixture policy 是固定规则解析器：测试通过只证明 LangGraph、RAG 工具和状态路径成立。

使用真实 LLM + 授权数据时：

~~~bash
sidra eval --dataset /path/to/hotpot_dev --split dev \
  --policy multi-openai --model YOUR_MODEL \
  --endpoint https://YOUR_ENDPOINT/v1/chat/completions \
  --constrain-actions --constrain-document-ids \
  --runtime langgraph --graph-role-mode multi --limit 32 \
  --out /outside_repo/graph_dev32
~~~

仍是 HotpotQA per-question candidate documents，不是全网 Internet Search。

## 状态持久化与安全限制

~~~bash
sidra eval --dataset examples/fixture --split dev --policy fixture-multihop \
  --runtime langgraph --out /tmp/sidra-graph-run \
  --graph-checkpoint-db /tmp/sidra-graph-run.private.sqlite
~~~

- 需要全新 SQLite 文件，不允许覆盖历史 Run；
  graph_threads.jsonl 记录 thread_id；manifest.json 绑定 SQLite SHA256。
- checkpoint 内可能保存问题文本、READ 原文和 Action，必须置于受控目录，不提交仓库。
- 当前没有“自动恢复未结算 LLM 调用”的安全策略，也没有持久化 provider receipt；
  SQLite 的 node checkpoint **不等于** exactly-once token 计费和安全恢复。
- 旧实验仍可用默认 runtime=legacy 重复，不能拼接不同源码/模型/模式的指标。

## Agentic RL：保持采样与梯度的一致性

现有脚本现在有可选参数：

~~~bash
python scripts/train_onpolicy_tool_grpo.py \
  --train-data /path/to/TRAIN --dev-manifest /path/to/FROZEN_DEV.json \
  --model /path/to/Qwen3-0.6B-Base --adapter /path/to/STOP_LORA \
  --out /outside_repo/new_onpolicy_run \
  --agent-runtime langgraph --updates 1 --groups-per-update 6 --generations 4
~~~

该训练运行使用 GraphResearchAgent(role_mode=single, evidence_state_mode=off)。
它调用原有 OnPolicyHFAction.decide，仍然由同一个原始 SYSTEM prompt 对模型采样，
而不是将 Multi 模式的角色指令强加于原 GRPO 的旧 logprob 计算。

原有训练奖励通过 score_train_rollout 在**完整轨迹之后**读取 TRAIN-only gold；
训练脚本依旧使用原有 group-normalized reward 与 clipped LoRA 优化器。
这次只是新增一个图环境选项，**没有**进行 GPU 训练，也**不代表**改进了 RL 效果。

训练 Multi 角色模型需要先将完整 role-specific prompt 同步进入采样/重算概率合同；
否则 old/new logprob 不一致，不可宣称标准 on-policy RL。

## 验收门禁

1. 真 LangGraph 运行、Planner/Researcher/Solver/Verifier、BM25 搜索及源限定 READ。
2. 未读引用拒绝、来源不足反馈、最大 steps/tool 调用约束、Gold-blind。
3. SQLite 检查点存在且和 manifest 对齐，历史输出不可覆盖。
4. TRAIN-only reward 可评价 Graph 单策略轨迹，DEV/TEST 禁止加入 reward。
5. GitHub CI Python 3.10/3.12 安装 graph extra，运行真实框架测试。
6. 未来必须用新冻结问题对照 legacy/graph/multi/evidence-state；
   并且控制模型、提示词、预算、seed，结果不提升时保留负结果。

## 仍未完成的研究目标

- 还没有真实 LLM + HotpotQA 新批次的 LangGraph A/B 提升证据。
- 还没有 checkpoint 后的精确安全恢复 / provider 幂等。
- 还没有训练出经过独立评测改善的 Graph Multi-Agent policy。
- 没有证明有效的多代 RSI；不能从创建新状态图推断自我改进。


## 7. 独立模型的 Multi-Agent 角色分配（可选）

默认的三个角色可以共享一个模型；若希望测试独立 Planner/Researcher/Solver
的协作，可通过下列命令将其分别绑定到不同模型：

~~~bash
sidra eval --dataset /path/to/hotpot_dev --split dev \
  --policy multi-openai --endpoint https://YOUR_ENDPOINT/v1/chat/completions \
  --model DEFAULT_MODEL --runtime langgraph --graph-role-mode multi \
  --graph-planner-model PLAN_MODEL \
  --graph-researcher-model RESEARCH_MODEL \
  --graph-solver-model SOLVER_MODEL \
  --out /outside_repo/role_split_eval
~~~

三个模型使用同一受控工具执行环境、GraphState 和权限检查，但有独立 API
请求/使用量计数；在 manifest 的 graph_role_models 中记录模型归属。
不改变底层 Dataset 与 TRAIN-only 权限。不同角色模型的能力不同，
因此与单模型基线比较时**不能把混合模型提升归因于多 Agent 编排**，
应在相同模型设置下另外完成角色分工消融。


## 8. 真实 Qwen3 负结果与引用恢复门禁

Tang 独立目录在本分支截止提交 483238fe 上运行过已经使用过的 DEV 前四题。
模型采用缓存的 Qwen3-0.6B-Base 与既有 TRAIN-only Stop-SFT LoRA，参数一致：

- 原 Agent Loop：EM 0/4，严格验证成功 0/4；
- LangGraph 单策略：EM 0/4，严格验证成功 0/4；
- LangGraph 多角色：EM 0/4，严格验证成功 0/4；
- 每题平均模型 3 次、工具 2 次；都在一次 Search + READ 后引用尚未 READ 的文档。

本次新增可选开关 --recover-unread-citations：它对 source-only 引用门禁失败
提供简明非 Oracle 反馈，消耗当前 Agent Step 并要求模型自行继续 Search/Read。
开关默认关闭，不改变原评测合同或早前冻结实验证据。
修复后的新运行必须另行记录，不能拿同一批已见 DEV 反复调参宣称泛化。
