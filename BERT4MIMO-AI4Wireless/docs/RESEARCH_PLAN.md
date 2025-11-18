# BERT4MIMO 研究计划（学生项目）

> 目标：在学期内将项目“开放/复现到当前公开仓库的最新稳定状态”，并完成可复现的训练、验证与 WebUI 演示，形成报告与可复用的工程基线。
> 时间跨度：12–16 周（可按学期实际周数调整）
> 适用对象：本科/硕士学生（1–2 人小组）

---

## 一、项目背景（Background）

- 无线通信系统（尤其是 5G/6G Massive MIMO）对高精度、低延迟的信道状态信息（CSI）获取与压缩反馈提出更高要求。传统方法在大规模天线、宽带多载波场景下难以兼顾精度与效率。
- 近年，Transformer/BERT 等自注意力模型在序列建模与特征表达方面表现优异，已被尝试用于 CSI 压缩、重构与鲁棒性建模。
- 本项目基于“CSIBERT”思路，提供了从数据生成（MATLAB）、模型训练（PyTorch）、验证评估（多项实验）、到 WebUI 交互演示（Gradio）的端到端工程框架，且已在仓库中实现可运行的训练、验证与五个“高级实验”入口。
- 学生研究目标：在可控的资源与时间内，完整复现当前开源仓库的功能集（训练/验证/WebUI），补充复现实验记录与文档，使其成为后续研究与工程化的“可复用基线”。

---

## 二、研究问题与目标（Problem & Objectives）

- 核心研究问题：
  - 如何以标准化流程复现 CSIBERT 在 Massive MIMO 场景中的训练与验证结果？
  - 在统一数据与超参数设置下，得到稳定的压缩-重构、鲁棒性与推理性能基线？
  - 如何以工程化方式封装训练/验证流程并提供交互式演示（WebUI），便于教学与对比实验？

- 明确目标：
  1. 搭建与验证可复现的环境（Python/依赖/CUDA/MATLAB）与数据管线。
  2. 复现训练与验证：得到 `best_model.pt` 与 `validation_results/*` 全套产物。
  3. 打通 WebUI 全流程：加载模型、运行 5 个高级实验并生成图表/报告。
  4. 形成“可复现报告+脚本+文档”，包括参数、日志、种子、版本信息。
  5. 工程开放性：按规范组织目录与 .gitignore，保证仓库可直接拉起与演示。

---

## 三、预期方法（Methodology）

- 方法聚焦于“用什么模型实现什么目标”。以 CSIBERT 为主模型，配合若干对照基线，围绕压缩-重构、预测、鲁棒性与高效推理四类目标构建评测闭环。

1) 主模型（Primary Model）：CSIBERT（Transformer 编码器-瓶颈-解码器）
- 架构要点：`model.py` 中的 Transformer 堆叠（默认 Hidden=256, Layers=4, Heads=4，dropout=0.1），加入位置编码与可调“压缩瓶颈”（token 维度/长度约束）。
- 表示方式：将 CSI 的实部/虚部映射为两个通道并序列化为 tokens（或采用幅度/相位表示，二者择一并保持一致）。
- 量化/比特率（可选）：在瓶颈处接入 k-bit 量化（2/4/8-bit）或熵模型，以支持压缩比实验与率失真的权衡。

2) 目标→模型映射（Tasks → Models）
- 目标A：CSI 压缩与重构（High-fidelity Reconstruction）
  - 模型：CSIBERT 编码器→瓶颈→解码器。
  - 损失：NMSE 主损失 + 可选 MAE；率失真可采用 L = NMSE + λ·Bitrate。
  - 产出：`reconstruction_error.png`，`validation_report.json` 中 NMSE 指标。
- 目标B：时序预测/外推（Sequence Prediction/Extrapolation）
  - 模型：CSIBERT + 因果/Seq2Seq 预测头（在编码器顶端接前馈预测头）。
  - 损失：MSE/NMSE；可加入时间平滑正则。
  - 产出：`prediction_accuracy.png`（准确率或误差曲线）。
- 目标C：SNR 鲁棒性（Robustness across SNR）
  - 模型：CSIBERT + 训练时噪声注入（SNR∈[0,30]dB 的课程式采样），可选 SNR 条件编码（concat 条件向量）。
  - 协议：同一模型在不同 SNR 测试集上评估 NMSE/Accuracy 曲线。
  - 产出：`snr_robustness.png`。
- 目标D：压缩比-质量权衡（Rate–Distortion Trade-off）
  - 模型：CSIBERT 的瓶颈宽度/序列长度可调 + 量化位宽（2/4/8-bit）。
  - 协议：遍历瓶颈与位宽，统计压缩比与 NMSE 的帕累托曲线。
  - 产出：`compression_analysis.png`。
- 目标E：高效推理（Efficient Inference）
  - 模型：轻量化 CSIBERT-S（Hidden=128, Layers=2, Heads=2）与基线 CSIBERT-B（默认配置）对比；可选 TorchScript/ONNX 导出。
  - 指标：ms/样本（GPU/CPU 分别统计）。
  - 产出：`inference_speed.json`。

3) 对照与基线（Baselines）
- CNN 自编码器（CsiNet 类）：卷积编码-解码结构，对比非注意力模型的压缩-重构性能。
- RNN 序列自编码（LSTM/GRU）：评估序列模型在预测与重构上的表现。
- 传统降维（PCA）：提供简单可解释的压缩基线。
以上基线采用相同训练/验证划分、相同损失（NMSE 为主），用于横向对比。

4) 训练与优化（Training Protocol）
- 优化器与调度：AdamW（lr=1e-4, weight_decay=1e-2），cosine/plateau 调度可选；早停 patience≈15。
- 数据增强：随机 SNR 噪声注入、子载波/天线子采样或遮挡（与验证协议对齐）。
- 预训练（可选）：Masked Token Modeling（MTM）自监督预训练 → 重构微调。
- 复现性：固定随机种子，记录软件/驱动版本；所有输出归档到 `validation_results/`。

5) 模型配置矩阵与消融（Ablations）
- CSIBERT-S（128/2/2）、CSIBERT-B（256/4/4，默认）、CSIBERT-L（384/6/6）。
- 消融维度：是否量化、瓶颈宽度、注意力头数、层数、表示方式（RI vs Mag/Phase）。
- 目标：在等算力预算下寻找最优“精度-速度-压缩比”折中点。

---

## 四、里程碑与时间表（Milestones & Timeline）

- 第 1–2 周：
  - 完成环境搭建与依赖安装；确认 GPU 可用与 MATLAB 可运行。
  - 生成 `.mat` 数据，检查数据格式；完成项目结构熟悉与代码走读。
- 第 3–4 周：
  - 启动训练，得到第一版 `best_model.pt`；整理训练日志与参数表。
  - 记录关键版本与种子，确保同配置可重复得到相近指标。
- 第 5–6 周：
  - 运行验证与 5 项高级实验；生成 `validation_results/*` 全套产物。
  - 撰写阶段中期报告（方法、设置、初步指标与问题）。
- 第 7–8 周：
  - 打通 WebUI 全流程；补全交互说明与常见问题（FAQ）。
  - 优化图表与报告呈现，形成标准化导出模板。
- 第 9–12 周：
  - 稳定复现基线；开展小幅度超参/数据扰动实验，验证鲁棒性。
  - 整理最终文档、录制演示视频（可选）、准备答辩/展示材料。

---

## 五、评估指标（Evaluation Metrics）

- 复现性：同一配置下多次训练/验证指标方差小；结果产物一致性高。
- 性能指标：
  - 重构误差（MSE/NMSE）：达到或优于仓库基线（例如 MSE < 0.01）。
  - 预测准确率：≥ 85%（基于项目默认设定）。
  - 压缩比-质量：4:1–8:1 范围保持可用重构质量。
  - 推理延迟（GPU）：< 10ms / 样本；CPU 指标单独记录。
- 工程质量：
  - 文档完整度（Setup/Quick Start/WebUI/复现实验记录）。
  - 可运行性（新机器“按文档即可启动”通过率）。
  - 产物完备性（模型、结果、报告、脚本均可用）。

---

## 六、风险与应对（Risks & Mitigations）

- 环境不一致 / 依赖冲突：固定 Python 次版本；使用 `requirements.txt`；在两台不同主机交叉验证。
- GPU 资源不足：提供 CPU 运行路径与较小批次/短训练回合设置；必要时使用云 GPU。
- 数据规模/格式变化：在 `model_validation.py` 保持 `.npy/.mat` 双支持并做显式断言。
- 结果波动：固定随机种子；记录所有关键超参数与版本；进行三次以上独立复现实验。
- 时间管理：按里程碑提交阶段性产物（日志、图表、文档），教师/导师滚动检查。

---

## 七、伦理与合规（Ethics & Compliance）

- 数据来源合法与匿名化处理，避免包含可识别的个人信息。
- 遵循开源许可与第三方依赖协议，不上传大体量或受限数据。
- 复现实验报告公开透明，避免选择性报告与过拟合结论。

---

## 八、预期成果（Expected Outcomes）

- 代码与模型：
  - `checkpoints/best_model.pt`（可复现实验得到的最佳权重）。
  - 完整可运行的训练/验证脚本与 WebUI。
- 实验与报告：
  - `validation_results/` 下图表、JSON/MD 报告与可复现实验记录。
  - 最终技术报告（方法、设置、指标、误差分析、局限与展望）。
- 文档与演示：
  - `docs/` 下 Setup/Quick Start/WebUI/Research Plan/FAQ 全集。
  - 可选 5–10 分钟演示视频（含训练启动、WebUI 演示、结果解读）。
- 工程基线：
  - 新机器按文档“零障碍”拉起与演示；一键脚本与启动入口（`START.bat`）。

---

## 九、资源需求（Resources）

- 计算：1 块中端 GPU（如 RTX 3060）或云端等效；存储≥10GB；内存≥16GB。
- 软件：Python 3.11+/PyTorch（CUDA 对齐版本）、MATLAB（可选）。
- 时间：12–16 周（每周 6–8 小时投入）。

---

## 十、工作分工建议（可选）

- 学生 A：环境与数据、训练与日志、复现实验记录。
- 学生 B：验证与高级实验、WebUI 演示、文档与展示材料。

---

## 十一、验收标准与提交物（Deliverables）

- 可运行仓库（含 `.gitignore`、不含大文件）；
- 训练与验证可复现脚本与日志；
- `validation_results/*` 全套指标与图表；
- 最终技术报告（PDF）与演示幻灯片（PPT/Keynote）；
- 简短演示视频（可选）。

---

> 注：本研究计划面向“将项目开放/复现到当前稳定状态”的基线目标，后续可在此基础上继续进行模型结构改进、数据域迁移（不同信道模型/频段/天线规模）、蒸馏与量化、在线适配等研究方向。