# BondCliQ 数据质量研究计划

更新：2026-10-01（ET）

目标：先理解重复、多价、更新、crossing 和连续偏离，判断报价有哪些可用信息，再决定如何清洗和构造 feature。不能把“离同行远”“曲线不光滑”直接等同于 corruption。

## 当前进度与推进规则

| 阶段 | 状态 | 本阶段要得到的结论 |
| --- | --- | --- |
| 1. Quote quantity 总览 | 已完成本轮；已审阅九个 issuer 截图 | 零值覆盖很广，正值的经济含义尚未确认 |
| 2. 同 dealer、同时间、同边多价 | 下一步，待实现 | 重复与多价有多少，quantity 能解释多少 |
| 3. 更新行为与连续偏离 | 待执行 | 反复刷新、稳定偏移和异常 run 如何区分 |
| 4. Crossing 分解 | 待执行 | 同 dealer / 跨 dealer、同步 / 异步、多候选分别怎样 cross |
| 5. 报价信息验证 | 待执行 | 报价水平或变化是否被独立后续观测支持 |
| 6. 清洗建议及 feature 设计 | 等待前述证据 | 哪些情况能合并、需保留候选或降低使用权重 |

**第一步不再追加一轮独立的 quantity 总览。直接推进第二步。**Quantity 的剩余问题作为解释多价、更新和 crossing 的条件，嵌入后续研究。单位或零值含义无法确认不阻止推进；也不要求解释完所有案例才进入下一阶段。每一步记录“已回答、未回答、对下一步的影响”。

## 固定前提

- Universe：给定三个月 `data_ig` 中至少有一次 trade 的 bonds；保留其中短 quote 窗口没有 trade 的 bonds。
- 直接沿用第一步的加载 cell：TRACE cache / `data.load_merged_prints`、BondCliQ parquet、CUSIP 筛选、UTC 转 ET 和 issuer 映射。不重复要求手动添加加载代码。
- Spread 是 bond yield 减对应 UST yield；multiplier 已正确，TRACE 已 clean。均不重新验证。
- Event time 就是 known time。没有 message / quote ID、stream / desk、sequence、撤回、修订或有效期字段，不再查找这些字段。
- Trade quantity 是面值；quote quantity 保留原值，单位和经济含义未知。0、正值、真缺失、其他值分开。
- Dealer bid 对应 client sell，dealer ask 对应 client buy；沿用已确认的实际 side 映射。
- 零或负 spread 保留；quantity=0、多价、crossing、更新少或不可直接成交，都不单独证明 corruption。
- 所有阶段先用原始记录。旧 EDA 的 median 合并、30min 过期、zero / crossing / peer 排除规则只是待评估的处理假设。

## 第一步结果：够用，进入下一步

来源：用户提供的 IMG_3175–IMG_3183。百分比按各 issuer 全部原始行计算，包含重复播报、合并两侧；只描述这些截图样本。

| Issuer | 原始行数 | Quantity=0 |
| --- | ---: | ---: |
| 180 Medical | 4,971 | 60.5% |
| Adobe | 24,510 | 43.3% |
| Airbnb | 11,355 | 18.8% |
| Alphabet | 222,478 | 17.6% |
| Bank of America | 295,007 | 45.0% |
| Coterra Energy | 9,359 | 37.0% |
| Ford Holdings | 74 | 100.0% |
| Parker-Hannifin | 35,824 | 86.8% |
| Yale-New Haven | 152 | 100.0% |

九张图的 Missing、Other 均为 0。Dealer 537332… 在 Alphabet 图中约全为正值，在 Ford 的 66 行中全为 0；dealer b3f94… 在 180 Medical 的 2,184 行中全为 0，在 Parker 的 744 行中全为正值。不能给 dealer 贴固定的“报量 / 不报量”标签。

Alphabet、Coterra 的正值中 quantity=5 分别占 80.0%、85.1%。尚不能区分常用报价量、固定填报值或重复播报权重；不能据此确认 MM 或可成交量。0.097、0.098 等原值也不凭非整数判为异常。

**后续约束：不筛除 quantity=0，不把相同正值当作已确认相同经济条件，不把两个 0 当作 size 匹配。**Ford / Yale 保留作低覆盖案例，因样本少优先个案描述。

## 第二步：同 dealer、同时间、同边多价

### 要回答什么

1. 多条记录主要是相同内容重复，还是不同 spread 的真实观测候选？
2. 不同 spread 是否与不同正值 quantity 稳定对应？相同正值或全零时，是否仍有多价？
3. 多价集中于少数 dealer / bond / 日期，还是覆盖广？它是否形成反复出现的双轨或 A/B 交替？

### 只做这些计算

- 以 `firm × cusip × side × quote_timestamp_ET` 为组，使用原始 timestamp，不先按秒取整，不先取 latest 或 median。
- 每组保留原始行数、现有字段完全相同的重复数、不同 spread 数、spread 范围及 quantity 候选。相同内容重复不证明是同一消息重放。
- 有至少两个不同有限 spread 的组称为“同 timestamp 多价组”。非有限 spread 的覆盖单独计数，不因删除它们把组标为正常。
- Quantity 类别互斥，依次判定：包含真缺失 / 其他值；否则包含 0；否则全部正值且数值不同；否则全部正值且数值相同。包含 0 的案例再区分全零与零 / 正混合。
- 每个 timestamp 组只投一次票，不将组内候选组合展开成独立样本。分别报告总体多价率、各 quantity 类别内多价率、发生过多价的 dealer-bond-side-day 比例，写清分母和样本量。
- 总体按组计数；另按 dealer-bond-side-day 等权汇总对照，避免高频播报支配结论。仅用于统计，不删除原始重复记录。

### 交付两张图

**总览图，最多三个 panel：**按 dealer 的多价发生率及组数；按 quantity 类别的多价率及组数；多价组 spread 范围的分布。Issuer 下拉选择，dealer 长列表分页。

**案例图：**选择 bond、dealer、side、日期；上层为局部原始 spread 散点，下层为 quantity 散点，直接标注同 timestamp 的候选值。0 使用独立标识，不通过点大小画成“零报价量”，也不把无法识别身份的候选连成唯一曲线。

先检查 3–5 个有代表性的多价案例及少量非多价对照，覆盖相同正值、不同正值、全零 / 混合；没有观测到某类就记为未观察。每例只写“观察、支持的解释、尚不能判断什么”。不 print 大表。

### 完成后怎么推进

能区分重复与多价，给出分母明确的发生比例，并展示 quantity 能解释 / 不能解释的典型案例，就进入第三步。若同 timestamp 多价很少，报告这个结果，并把短时间 A/B 交替交给第三步；不为凑案例扩张“同时”的定义。

这一步不做唯一报价重建、去除 outlier、强制合并、crossing 排除或曲线拟合。相同 quantity 下多价仍可能对应未记录条件；不同 quantity 对应不同价格只是条件差异证据，尚不是已验证的 size 曲线。

## 第三步：更新行为与连续偏离

先从第二步的案例看完整时间过程，再扩大统计。分别看相同 spread / quantity 的刷新、spread 改变、quantity 改变、A/B 交替、持续双轨、突然跳变及稳定偏移。

对同 timestamp 多候选保持候选集合，不任意排序制造“上一笔”和改价次数。候选 batch 仅表示同 dealer 同时跨 bond 发布，不认定为已确认的算法 run。

输出两类图：消息间隔与实际改价间隔的分布；局部时间线对照该 dealer、其他 dealer 及同 issuer bonds。记录偏移幅度、持续时间、覆盖 bond 数和恢复方式。固定 quantity 与零值切换在这些图里一起看，不另开新的 quantity 总览阶段。

完成标准：有重复刷新、持续偏移、跳变 / 恢复或交替的案例及反例；明确哪些变化有同行支持，哪些参考不足。更新规律只作行为描述，不能凭频率认定算法 / 人工或错误。

## 第四步：Crossing 分解

先研究同 dealer、同 timestamp 的双侧候选，再扩展至不同 timestamp 和跨 dealer。Benchmark spread 坐标下，signed width = bid spread − ask spread；小于 0 为 crossing。零 / 负 spread 本身与 crossing 无关。

每类按 quantity 条件、两侧时间差与较旧一侧 age 分层。没有 ID，不声称同 timestamp 是同一双边消息；两个零不证明 size 相同。

多候选时分别记录全部组合都 cross、部分组合 cross、没有组合 cross；每个 pair-bond-time 只贡献一次分类。异步状态明确使用的 age 假设，不把假设下持续时间当作可执行持续时间。

输出 crossing 类型 / 幅度总览与局部双侧案例。重点看由哪一侧触发、如何结束，以及逐个 dealer 排除后的变化。先完成同步层，再做少量 freshness 敏感性；不展开巨大参数网格。

完成标准：能分开异步拼接、条件不明、单来源极端值与持续分歧；没有可执行性证据不称套利，也不因 crossing 自动删两侧。

## 第五步：验证报价是否有信息

分别评价水平与变化。用排除被研究 dealer 的同行参考，以及预先指定时间段内的后续独立 dealer 报价和已 clean 的 TRACE；不事后挑最接近的一笔成交。

Quote quantity 未确认前不与 TRACE 面值精确匹配。Client buy / sell、odd lot 和大额成交分层，允许几十 bps 的差异作为待解释结果。没有可比后续观测记为无法验证，不记为错误。

先发现案例机制，再用后段日期复核。输出水平 / 变化信息图、前后段对照及验证覆盖数；按日或 episode 汇总，重复消息不作独立证据。

完成标准：能说明哪些报价含有可重复的信息、哪些只增加消息数、哪些还无法判断。与同行不同或缺少预测增量都不单独证明 corruption。

## 第六步：根据证据提出处理建议

对每条建议写明证据、适用条件、受影响记录 / bond / dealer 比例及可能误伤。候选结论包括：内容去重但保留刷新行为；按有证据的条件保留不同报价；保留多候选；降低特定 episode 的使用权重；确定记录问题后修复或排除。

先比较原始数据与拟议规则，再实现 cleaner。Bayesian 平滑 bid / ask 与 uncertainty band 放在这里讨论：先确定它观测的报价条件和候选结构，再评估模型。平滑程度不作为清洗正确的证据。

事后使用未来数据得到的判读与当时可得标记分开。进入 point-in-time feature 阶段前明确哪些条件仍未知，不强制制造唯一“正确价格”。

## 所有阶段的展示要求

- 简短 Jupyter notebook，沿用加载代码和 issuer dropdown；少量 cell，不搭建研究用不到的 production 框架。
- 不 print / display 大 DataFrame；明细保留后台。每次一张图，最多三个 panel，直接标注日期、ET、bps、关键数值、样本量和分母，支持保存 PNG。
- 每步结束就更新本文件状态和结论，未解决问题带入下一步；不反复增加前置检查。
