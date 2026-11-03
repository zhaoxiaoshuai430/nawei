# 总结 Agent v2

按《总结agent整体概括(5).txt》实现的后台阶段理解模块。独立调用文本模型，不进入主对话工具循环，不负责 TTS。

## 内容与在线使用

| 文档内容 | 保存与使用方式 |
| --- | --- |
| 明确个人事实 | personal_facts，保存用户明确提供的姓名、宠物名称等，校验用户证据后可投影为 profile_fact；相关回忆问题按预算选择 |
| 最近主题 | topics，涵盖学习、生活、兴趣、计划；证据轮次数辅助判断频率 |
| 问题进展 | open_task + progress：open、partial、resolved、cancelled；续讲只选择未完成项 |
| 完成与取消记录 | task_outcomes，最多24条独立保存 resolved/cancelled 事项，保留证据供回忆，与当前 claims 一起按最新状态归并 |
| 目标计划 | goal，记录安排及 open、partial、resolved、cancelled、changed；只使用明确用户目标 |
| 学习变化 | learning_observations + trend：进步、停滞、可能遗忘、反复错误、独立完成；趋势至少有两个不同时刻的用户证据 |
| 沟通偏好 | 明确表达直接使用；observed 偏好至少三个用户轮次、两个会话支持后使用 |
| 教学方式 | teaching_strategy_observations，相关条目进入主 Agent（含 Tutor）的提示，指导讲解与练习 |
| 临时状态 | temporary_states，自最近用户证据起两小时过期，单独指导回答节奏 |
| 稳定画像 | profile_change_proposals 经服务器核验至少三个用户轮次、两个会话、supported、非 inferred、无反证后形成 stable_profile；最长参考一年 |

稳定画像是可撤销、可追溯的行为总结，不写入正式人格字段，不修改知识点数值。关闭 SUMMARY_PROFILE_PROMOTION_ENABLED 可禁止生成和使用这层画像。消息 ID 合法不等于语义已被证明，真实效果仍取决于原始对话和模型判断。

## 生成与证据

默认每累计 50 个完整轮次触发，读取最近 100 轮、30 天内的完整对话。按 UTF-8 字节上界控制输入，系统指令和 JSON schema 计入预算；不截断半个轮次，遗漏记录在 coverage。

生成时比较上一版总结。最多约三分之一输入预算用于旧结论和原始证据，允许引用近期窗口外的证据。重新读取同一用户、同一记忆 epoch 的原文，并与上一版快照 hash 匹配；缺失或修改过的证据使对应旧条目不再进入输入。旧总结自身不是证据。coverage 分别记录近期输入、reference_turn_ids、previous_summary_revision，不复制旧摘要全文。

密码、验证码、银行卡号、身份证号所在轮次在保存任务快照前排除；输出再次过滤敏感字段、心理诊断、智商和人格标签，并保留本地及外部安全审核。这仅限制总结模块，不改变原有聊天记录保存策略，也不会自动扫描改写旧版本已经保存的快照。

严格校验消息归属、用户/助手角色、反证、趋势和字段所属。用户事实的时间不因助手回复而刷新。发布和重试重新核验所有近期及引用原文；删除引用原文后读路径不再提供对应总结。

完整轮次和触发需求原子提交。PostgreSQL 持久任务、单并发槽位、可续租 lease、epoch/revision fencing 和固定重试快照保证多 worker 一致性。重复 request_id 返回已提交答案；不同输入复用 ID 被拒绝。预览、失败或未验证轮次不进入账本。默认重试三次，无空闲触发；相同失败窗口不无限重试，新轮次可产生新目标。

停机状态区分 `summary_stop_requested`（已请求停止）和 `summary_scheduler_stopped`（调度任务已退出且无处理中任务）。初始化和发现任务返回后会再次检查停止请求，避免继续领取任务；已经在领取中的任务完成处理及租约清理后才允许报告 drained。`summary_inflight` 包含租约清理阶段。外部取消仍保留租约到期恢复机制；已退出的调度任务可由 `start()` 重新启动。

## 在线选择

全局启用后，新用户默认自动参考总结。既有用户设置保持原值；历史 false 无法区分默认值与主动关闭，因此不自动批量开启，可通过 settings API 修改。memory_mode=off 不读取总结；删除与关闭操作保留 epoch 失效机制。

只选择相关、supported、非 inferred、无反证、未过期内容。general 的偏好、状态、行为习惯可跨主题使用，具体学科信息要求文本相关。已完成/取消任务不用于无主题续讲；相关提问时以“已完成/已取消”提供背景。同一任务以最新条目为准。

任务清单查询（例如“列出已完成和未完成的事项”）不作为纠正拦截；明确陈述任务完成、修改偏好等仍优先于历史。清单查询优先分配任务/目标的提示预算。重复任务、个人事实、偏好和临时状态按原始证据时间与轮次选择最新值，不能由模型数组顺序决定。实体关联仍依赖稳定的 kind、subject、scope，不做无证据的模糊合并。

发布总结时，摘要派生的计划记忆仅保留本版重新确认的条目；被遗漏的旧计划软删除为 summary_plan_not_current，不能据此断言任务已完成。明确完成/取消仍使用 summary_claim_retracted。此规则只影响 conversation_summary 来源的 plan，不修改其他来源或其他用户的记录；个人事实和偏好不会因本版遗漏而自动清除。历史版本及证据保留，重新确认的计划可再次生成。持久教学偏好从仓库返回的 teaching_preferences 读取。

当前纠正、最近会话历史和直接工具请求优先。总结作为不可信历史背景注入，主 Agent 自然回答，不展示内部字段。选择受独立超时和提示预算约束；故障不阻塞正常回复。内部 metadata.runtime.summary 记录选中条数、revision、claim IDs、耗时与预算；health 策略版本为 summary-online-v3。

## 升级与配置

1. 通过现有 drain 流程停止旧 LLM workers，保留数据库及原始对话。
2. 在包含 DATABASE_URL 的环境运行 `python -m deploy.scripts.migrate_conversation_summary`，依次执行 001、002、003。第二次迁移保留用户设置、更新新行默认值，并废弃旧版本未完成任务；第三次迁移建立教学偏好表。应用镜像包含迁移脚本及 SQL 文件，迁移仍需显式运行，不随服务启动自动执行。
3. 启动全部新 workers，不长期混用 v1/v2。启动时检查 v2 迁移。
4. SUMMARY_ENABLED=true 启用后台，SUMMARY_APPLY_TO_PROMPT=true 启用在线参考。部署模板保持这两个开关关闭，当前项目本地配置已开启。
5. 回退在线参考可关闭 SUMMARY_APPLY_TO_PROMPT 后重启；停止后台前正常 drain。

2026-11-02 增加的 personal_facts、task_outcomes 使用现有 JSON 文档和证据表，无新增 SQL 迁移。部署新代码需重启 LLM workers 并重新生成总结，旧文档不会自动补出遗漏的事实或完成状态。生成模型仍可能遗漏或误判，必须结合实际回放结果验收。

默认：trigger_turns=50、window_turns=100、window_max_days=30、input_max_tokens=12000、output_max_tokens=2000、prompt_max_tokens=600、read_timeout_seconds=0.1。预算名称沿用 token，但采用保守 UTF-8 字节上界。当前本地长对话配置为输入 110000、输出 8000、在线 2000、生成超时 300 秒。小预算可能无法覆盖所有类别或全部轮次。

## 内部 API

现有内部认证保护 `/api/v1/llm/memory/{user_id}/summary`：

| 方法与后缀 | 功能 |
| --- | --- |
| GET | 状态、总结、覆盖范围、开关 |
| POST /refresh | 异步生成，返回 202/job_id；无可用轮次返回 409 |
| GET /jobs/{job_id} | 同一用户的任务状态 |
| DELETE | 清除总结并推进排除水位，旧轮次不再重建 |
| GET /settings | 用户和全局有效开关，不创建状态 |
| PATCH /settings | enabled、apply_to_prompt；关闭时使旧结果失效 |

原有 DELETE /memory/{user_id}?session_id=... 仍会清除全局画像、长期记忆、知识点，并删除指定会话消息，不是仅删除该会话的接口。

## 验证与边界

- `python -m pytest tests/test_conversation_summary.py tests/test_summary_understanding.py tests/test_summary_online.py tests/test_summary_model_adapter.py`
- `python -m pytest tests/test_summary_quality_regressions.py` 覆盖任务状态查询、真实纠正、完成状态乱序、临时状态恢复、个人事实证据/快照/在线读取及持久教学偏好。
- `python -m pytest tests/test_summary_worker_lifecycle.py tests/test_knowledge_cancellation.py` 验证停止与领取竞争、租约清理和检索失败后的子任务回收。
- 设置 SUMMARY_TEST_DATABASE_URL 后运行 `python -m pytest tests/test_summary_postgres.py`，每个用例仅创建和清理自己的随机 schema。
- `python scripts/check_summary_live.py --verify-online` 使用真实服务、真实模型和一次性测试用户，完成后清理该用户。
- `python -m scripts.run_summary_acceptance` 可在临时 8013 实例做同样验证，不修改常规服务配置。

短期历史、明确长期记忆、知识点状态继续由原模块维护。总结只提供有证据的阶段理解；词语相关度和纠正规则不是完整语义推理，不能保证每次回答采用所有建议。旧历史不会在启动时自动回填；已有显式核验报告回填脚本仍可使用。正常答复完成也不代表语音已被用户听到。
