# exp(u) → exp(u,v) 细粒度化可行性分析

## ⚠️ 重要说明

**本文档分析的是操作粒度细化（exp(u) → exp(u,v)），这并不等同于实现自适应动态搜索计划 Φ。**

- **exp(u) → exp(u,v)**：将粗粒度操作细化为更细粒度的操作（操作粒度改进）
- **动态计划 Φ**：根据搜索过程中的实时状态动态调整计划（决策时机改进）

两者是**独立的改进维度**，可以分别实现，也可以组合实现。详见 `dynamic_plan_analysis.md`。

## 一、当前实现分析

### 1.1 当前架构

**Python 端（RL 决策层）**：
- `Actor`: 负责选择 `gen(u)` 操作（选择哪个查询节点 u 进行 build_candidate_pool）
- `BinaryClassifier`: 负责决定是否对节点 u 执行 `exp(u)`（二分类：0=跳过，1=展开）
- 当前 `exp(u)` 是粗粒度操作：一旦决定 `exp(u)`，就会在 C++ 端遍历 LC(u) 中的所有候选节点 v

**C++ 端（执行层）**：
- `Execute_expansion()`: 执行 `exp(u)` 操作
- 在 `EvaluateQuery.cpp:1001-1058` 中，有一个 `while` 循环遍历 `valid_candidate[candidate_depth]` 中的所有 v
- 对每个 v，尝试将 u 匹配到 v，然后递归进入下一层

### 1.2 关键代码位置

**Python 端决策**：
```python
# train.py:132-152
for node_id in Bc_ppo.old_actor.candidate_expansion_pool:
    action, dist = Bc_ppo.old_actor.act_actor(gnn_feature[node_id].detach())
    if action == 1:  # 决定 exp(u)
        expand_search_tree[node_id] = True
        subg_actions.append(node_id)  # 将 u 加入匹配计划
```

**C++ 端执行**：
```cpp
// EvaluateQuery.cpp:1001-1058
while(idx[candidate_depth] < idx_count[candidate_depth]) {
    VertexID v = valid_candidate[candidate_depth][idx[candidate_depth]];
    // 自动遍历所有 v，无法跳过
    embedding[current_vertex] = v;
    // ... 递归匹配 ...
}
```

## 二、改进方案设计

### 2.1 核心改变

**原流程**：
```
gen(u1) → exp(u1) [遍历所有 v in LC(u1)] → gen(u2) → exp(u2) [遍历所有 v in LC(u2)]
```

**改进后流程**：
```
gen(u1) → exp(u1,v1) → exp(u1,v2) → exp(u1,v3) → gen(u2) → exp(u2,v1) → ...
```

### 2.2 需要修改的部分

#### 2.2.1 Python 端修改

**1. 状态表示增强**
- 当前状态：只包含查询图节点 u 的特征
- 需要增加：对每个 (u, v) 对的状态表示
- 方案：使用双图交互 GNN 编码 (u, v) 对的特征

**2. 动作空间重构**
- 当前：BinaryClassifier 输出 2 维（exp(u) 或跳过）
- 改进后：需要输出 |LC(u)| 维，每个维度对应一个候选 v
- 或者：使用注意力机制，动态选择 v

**3. 匹配计划格式**
- 当前：`actions2str()` 格式为 `start-start-u1-u2-...`
- 改进后：需要包含 (u,v) 对，如 `start-start-u1:v1-u1:v2-u2:v1-...`

#### 2.2.2 C++ 端修改

**1. 匹配计划解析**
- 当前：解析操作序列（gen/exp）
- 改进后：解析 (u,v) 对序列

**2. Execute_expansion 重构**
- 当前：自动遍历所有候选 v
- 改进后：只尝试计划中指定的 v

**3. 候选池管理**
- 需要：在 Python 端获取 LC(u) 列表
- 需要：将 LC(u) 传递给 RL 模型进行决策

## 三、可行性评估

### 3.1 ✅ 高度可行的部分

1. **Python 端 RL 模型修改**
   - BinaryClassifier 可以改为输出 |LC(u)| 维分布
   - 可以使用注意力机制处理变长的 LC(u)
   - 状态编码可以使用现有的 GNN 架构扩展

2. **匹配计划格式扩展**
   - `actions2str()` 函数易于修改
   - C++ 端解析逻辑可以扩展支持 (u,v) 格式

3. **C++ 端执行逻辑**
   - `Execute_expansion` 可以改为只处理指定的 v
   - 需要添加从匹配计划中读取 v 的逻辑

### 3.2 ⚠️ 需要仔细设计的部分

1. **候选池获取时机**
   - **问题**：LC(u) 在 C++ 端计算，Python 端需要提前知道
   - **方案 A**：在 gen(u) 时，C++ 返回 LC(u) 列表给 Python
   - **方案 B**：Python 端实现候选池计算（可能重复代码）
   - **推荐**：方案 A，修改 C++ 接口返回候选信息

2. **动作空间大小问题**
   - **问题**：如果 |LC(u)| 很大（如 1000+），动作空间过大
   - **方案 A**：使用注意力机制，动态选择 top-k 候选
   - **方案 B**：分层决策（先选 u，再选 v）
   - **推荐**：方案 A，使用 Transformer 或 Graph Attention

3. **状态表示复杂度**
   - **问题**：需要编码 (u, v) 对的特征
   - **方案**：使用双图交互 GNN
     - 输入：查询图节点 u 的特征 + 数据图节点 v 的特征
     - 输出：(u, v) 对的匹配概率

4. **训练效率**
   - **问题**：决策步数可能大幅增加（从 O(n) 到 O(n×|LC|)）
   - **缓解**：
     - 使用课程学习，从简单查询开始
     - 使用重要性采样，跳过明显无价值的 (u,v)
     - 使用 early stopping，提前终止无望的分支

### 3.3 ❌ 潜在挑战

1. **C++/Python 交互开销**
   - 当前：每个 gen/exp 操作调用一次 C++
   - 改进后：可能需要更频繁的交互（获取候选池）
   - **缓解**：批量处理，或缓存候选池信息

2. **搜索状态爆炸**
   - 状态空间从 O(n) 增加到 O(n×|LC|)
   - **缓解**：使用更强大的状态编码（如 Transformer）

3. **奖励信号设计**
   - 需要为每个 exp(u,v) 设计合适的奖励
   - 可能需要延迟奖励（只有完整匹配成功才有奖励）

## 四、实现建议

### 4.1 分阶段实现

**阶段 1：基础框架**
1. 修改匹配计划格式，支持 (u,v) 对
2. 修改 C++ 端，支持按计划执行单个 (u,v)
3. 保持 Python 端决策逻辑不变（仍使用 BinaryClassifier）

**阶段 2：RL 模型增强**
1. 实现双图交互 GNN 编码 (u,v) 对
2. 修改 BinaryClassifier 为 CandidateSelector（输出 v 的选择）
3. 使用注意力机制处理变长候选列表

**阶段 3：优化与加速**
1. 实现候选池预过滤（跳过明显无价值的 v）
2. 实现批量决策（一次选择多个 v）
3. 优化 C++/Python 交互

### 4.2 关键技术点

1. **状态编码**：
```python
# 伪代码
def encode_candidate_pair(u_feat, v_feat, query_graph, data_graph):
    # u_feat: 查询节点 u 的 GNN 特征
    # v_feat: 数据节点 v 的 GNN 特征
    # 使用交互注意力机制
    pair_feat = interaction_attention(u_feat, v_feat, query_graph, data_graph)
    return pair_feat
```

2. **动作选择**：
```python
# 伪代码
def select_candidate(u, LC_u, state):
    # LC_u: u 的候选列表 [v1, v2, ..., vk]
    # 使用注意力机制选择
    candidate_scores = candidate_selector(state, u, LC_u)
    v = sample_from_distribution(candidate_scores)
    return v
```

3. **C++ 接口修改**：
```cpp
// 伪代码
// 新增：获取候选池接口
std::vector<VertexID> get_candidates(VertexID u);

// 修改：执行单个 (u,v) 匹配
void execute_single_expansion(VertexID u, VertexID v);
```

## 五、预期收益

1. **搜索树缩小**：可以跳过无价值的 v，避免深入无效分支
2. **决策更精细**：RL 可以在更细粒度上学习
3. **性能提升**：预期 1-2 个数量级的速度提升（取决于查询图复杂度）

## 六、风险评估

1. **实现复杂度**：中等偏高，需要修改多个模块
2. **训练稳定性**：需要仔细设计奖励函数和课程学习
3. **兼容性**：需要保持与现有代码的兼容性（至少支持回退）

## 七、结论

**总体可行性：✅ 高度可行**

这个改进在技术上完全可行，主要挑战在于：
1. 设计合适的 (u,v) 对状态编码
2. 处理变长候选列表的动作空间
3. 优化 C++/Python 交互效率

建议采用分阶段实现，先验证基础框架，再逐步优化。

## 八、与动态计划的关系

### 8.1 当前实现的局限性

**重要**：exp(u) → exp(u,v) 的改进**并不会自动实现动态计划**。

- ✅ **可以实现**：更细粒度的操作（exp(u,v)）
- ❌ **不会实现**：动态调整计划（仍然是在搜索前生成完整计划）

### 8.2 两种改进的关系

| 改进类型 | exp(u) → exp(u,v) | 固定计划 → 动态计划 |
|---------|------------------|-------------------|
| **改进内容** | 操作粒度细化 | 决策时机改变 |
| **当前状态** | ❌ 未实现 | ❌ 未实现 |
| **可以独立实现** | ✅ 是 | ✅ 是 |
| **组合效果** | 更细粒度的动态计划 | 最佳性能 |

### 8.3 实现建议

1. **先实现动态计划框架**（使用 exp(u) 粒度）
   - 验证动态计划的有效性
   - 建立 Python-C++ 实时交互机制

2. **再细化操作粒度**（exp(u) → exp(u,v)）
   - 在动态计划基础上实现
   - 实现候选池获取和状态编码

3. **组合实现**：细粒度的动态计划 = 最佳性能

详见 `dynamic_plan_analysis.md` 了解如何实现真正的自适应动态搜索计划 Φ。

