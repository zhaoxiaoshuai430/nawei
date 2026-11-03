# 云端部署与机器人接入

目标链路：机器人通过 HTTPS 获取配置、通过 WSS 发送语音，Caddy 转发到 Gateway，Gateway 调用 ASR、LLM、TTS；LLM 调用知识库与持久化存储。数据库和内部服务不开放公网端口。

当前方案为 Linux x86_64 单机、单 NVIDIA GPU。ASR 在 CPU 上使用 Paraformer 流式识别，同时保留 SenseVoice 完整音频识别；TTS 使用 GPU；LLM 调用 DeepSeek API；中文向量模型使用本地 BGE small，维度 512。

默认边听边识别，收到最终识别结果后生成并播报。`GATEWAY_VOICE_PREVIEW_ENABLED=false` 关闭提前生成；此前长句测试没有命中可复用答案。开启提前生成时必须保持 `LLM_WORKERS=1`，因为目前缓存只在单个进程内共享。

## 1. 服务器与网络

- 候选配置：8 vCPU、至少 30 GiB 内存、A10 24 GB 或兼容 GPU。该配置还没有通过云端并发测试，不能据此承诺 15 台设备同时对话。
- 建议至少 150 GB SSD，并根据知识库、模型镜像和备份增长留出空间。
- 安装 NVIDIA 驱动、NVIDIA Container Toolkit、Docker Engine 和支持 `up --wait` 的 Compose v2。执行 `nvidia-smi -L`，并确认 Docker 可以使用 GPU。
- 将真实域名 A 记录指向服务器公网 IP。有 AAAA 记录时也必须可达。
- 安全组开放 TCP 80、443，UDP 443 可选；SSH 仅对管理来源开放。不要开放 8000、8001、8002、8003、19000 或数据库端口。
- 服务器需要能访问镜像源、Python 包源、DeepSeek 和已启用的工具 API。大陆公网域名按云厂商要求完成备案等接入手续。

## 2. 本地准备模型与代码

在 `nawei` 根目录执行，Windows 使用现有 `.venv\Scripts\python.exe`：

```powershell
.\.venv\Scripts\python.exe -m deploy.scripts.cloud prepare-models
```

此命令将 `.runtime/embedding-cache` 中的 BGE 缓存复制到 `models/embeddings`，解引用 Windows 链接，并执行一次完全离线的 512 维推理。已有目标缓存不会被覆盖。

上传 `nawei` 项目源码、`deploy`、`scripts` 和 `models` 到服务器，例如 `/srv/nawei`。不要上传本机虚拟环境、日志、私有 `.env` 或 `.runtime`。`models/embeddings` 是宿主机挂载目录，虽然被 `.dockerignore` 排除，上传时仍必须保留。

TTS 的三个 `model.safetensors.part000/001/002` 必须保留；镜像构建会校验 SHA256 并合并。仅上传合并后的大文件不能构建当前镜像。ASR 三个模型目录和 TTS 的 tokenizer、配置文件也必须完整上传。

## 3. 初始化生产配置

下面是服务器 Linux Bash 命令。将域名和邮箱换成实际值。初始化不会读取本地开发密钥，也不会覆盖已有配置。

```bash
cd /srv/nawei
python3 -m venv .deploy-venv
.deploy-venv/bin/pip install 'python-dotenv>=1,<2'
.deploy-venv/bin/python -m deploy.scripts.cloud init \
  --domain voice.your-domain.cn --tls-email ops@your-domain.cn
```

编辑 `deploy/.env.production`，填写真实 `LLM_API_KEY`。内部服务密钥、JWT 密钥、数据库密码已独立随机生成。文件在 Linux 上仅供所有者读写；不要提交或公开它。

模板的阿里云内容审核默认关闭，与当前本地启用状态不同。需要保持本地审核策略时，将 `ALIYUN_CONTENT_SAFETY_ENABLED=true` 并配置对应访问凭证。搜索等可选工具也需单独配置。

```bash
.deploy-venv/bin/python -m deploy.scripts.cloud check
.deploy-venv/bin/python -m deploy.scripts.cloud up
.deploy-venv/bin/python -m deploy.scripts.cloud status
```

`check` 验证密钥、HTTPS/WSS 域名一致性、模型文件存在性及 Compose 配置，不代表模型已加载或云端链路已通过。`up` 在 Linux 上构建、启动，并最多等待 1200 秒就绪。首次下载镜像和安装依赖另需时间。知识库会离线加载向量模型；TTS 必须加载成功；LLM 必须连通记忆和知识库，Gateway 才会被判定就绪。

首次启动是空数据环境。容器健康不代表本机用户、设备绑定和知识库已经迁移。

## 4. 验证公网并连接机器人

证书签发需要公网 DNS 和 80/443 可达。容器就绪后再验证公网，禁止用跳过证书校验的方式验收：

```bash
.deploy-venv/bin/python -m deploy.scripts.smoke_test \
  --base-url https://voice.your-domain.cn
```

在管理员机器上生成一个短期用户 JWT，`robot-owner` 是将要拥有设备的用户 ID。以下命令不会将 JWT 放入命令行参数：

```bash
export NAWEI_SMOKE_TOKEN="$(docker compose --env-file deploy/.env.production \
  -f deploy/docker-compose.production.yml run --rm --no-deps config-check \
  python /app/deploy/scripts/create_jwt.py --user-id robot-owner)"
.deploy-venv/bin/python -m deploy.scripts.smoke_test \
  --base-url https://voice.your-domain.cn --user-id robot-owner
```

当前固件源码在 `firmware/xiaozhi-esp32`，没有 `ROBOT/` 目录。先通过其板型配置或 menuconfig 设置：

```ini
CONFIG_NAWEI_SERVER_MODE=y
CONFIG_OTA_URL="https://voice.your-domain.cn/xiaozhi/ota/"
```

固件要求 ESP-IDF >= 6.0.1，推荐 6.1；在固件目录使用 `python scripts/build.py --list-boards` 确认真实板型，再按固件 README 的板型/variant 流程构建和烧录。不要将用户 JWT 或内部服务密钥写入固件。

**当前实体接入缺口：** 这份固件的 `main/ota.cc` 尚未发送后端默认要求的 `X-Device-Key`，仅修改 OTA 地址仍会返回 401。必须先实现设备独立初始化密钥的持久化和 OTA 请求头，再进行下面的绑定流程。协议模拟器已实现该请求头，不能用模拟器通过代替这份固件的激活验收。详见 [机器人协议说明](../services/gateway/xiaozhi/README.md)。

设备联网后取得激活码。使用上面的用户 JWT 调用 `POST /api/v1/devices/bind`，JSON 为 `{"code":"设备给出的激活码","name":"我的机器人"}`。例如在受控终端中执行：

```bash
curl --fail-with-body https://voice.your-domain.cn/api/v1/devices/bind \
  -H "Authorization: Bearer $NAWEI_SMOKE_TOKEN" \
  -H 'Content-Type: application/json' \
  --data '{"code":"设备给出的激活码","name":"我的机器人"}'
unset NAWEI_SMOKE_TOKEN
```

绑定成功后，设备再次请求 OTA 接口取得专用令牌和 `wss://域名/xiaozhi/v1/` 地址。验证至少一次长句提问、停顿后播报、打断、断网重连和解绑后拒绝连接。

现有 `scripts/simulate_robot.py` 支持 `--url https://域名 --user-id robot-owner --mode auto --text "测试长句"`；通过环境变量 `NAWEI_SIMULATOR_USER_TOKEN` 传入同一用户的 JWT。请在安装了主项目依赖的环境执行，部署管理虚拟环境不包含模拟器依赖。模拟器会建立和解绑自己的测试设备，并产生真实 API 请求。

## 5. 数据迁移、备份与回滚

需迁移的状态包括 PostgreSQL（记忆与知识库元数据）、Qdrant（向量）、Neo4j（关系）、Redis，以及 Gateway 的 `devices.sqlite3`（设备绑定）。还要保留 TTS 自定义数据和生产配置。仅复制源码不会带走这些状态。

第一次上云时，在明确是否保留旧用户/设备之前，不要覆盖任何目标数据。可以先用新用户和空库验收，再安排迁移。迁移知识库应同时迁移三种数据库，保持 `QDRANT_COLLECTION`、BGE 模型及 512 维配置一致；若重建索引，从原始知识资料重新导入。

单机部署可使用停机磁盘快照做一致性备份，步骤如下。此流程会中断对话，安排维护窗口执行；本次准备没有停止本机服务。

1. 记录当前镜像版本、配置和数据库版本。给 `NAWEI_IMAGE_TAG` 使用每次发布唯一的版本号，保留上一个版本的镜像。
2. 在服务器根目录执行 `docker compose --env-file deploy/.env.production -f deploy/docker-compose.production.yml stop`。
3. 在云控制台对承载 Docker 数据目录和 `/srv/nawei` 的磁盘创建同一停机点的快照。确认所有数据卷和生产配置都包含在内，等待快照完成。
4. 执行 `.deploy-venv/bin/python -m deploy.scripts.cloud up`，再运行公网 smoke test 和机器人语音验收。
5. 定期在独立测试服务器恢复快照，验证用户、知识检索、设备绑定与语音。快照恢复演练前不能声称备份可用。

数据库密码修改还需在数据库中执行对应改密操作，不能只改环境文件。配置初始化脚本拒绝覆盖已有文件，避免意外轮换这些凭证。

回滚无数据变更的应用版本：恢复上一个版本的源码及配置，将 `NAWEI_IMAGE_TAG` 指回已保留镜像，再执行 Compose `up -d --no-build --wait`。若发布涉及数据格式或数据库版本变更，应在维护窗口恢复一致快照后验收。不要执行 `docker compose down -v`，它会删除持久化数据卷。

## 6. 上云后的验收指标

先测一台真机，再测 15 个在线连接、4/5/6 路同时提问。记录识别正确率、说完到首段播报的 P50/P95、失败率、排队、CPU/RAM/显存峰值和重连成功率。当前配置限制每用户一路进行中的请求，整机最多六路；这只是限流设置，实际吞吐必须实测。

健康检查不包含真实 DeepSeek 推理、TLS 可达性或真实机器人麦克风测试，这些都需要上述公网和真机验收补齐。
