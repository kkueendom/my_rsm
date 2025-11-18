# 自适应动态搜索计划 Φ 实现分析

## 一、当前实现问题分析

### 1.1 当前架构：固定计划生成

**Python 端（计划生成阶段）**：
```python
# train.py:108-154
while len(subg_actions) < 2 * (query_graph.vertices_count - 1):
    # Actor 选择 gen(u)
    action = Ac_ppo.old_actor.choose_action(...)
    subg_actions.append(action)
    
    # BinaryClassifier 决定 exp(u)
    for node_id in Bc_ppo.old_actor.candidate_expansion_pool:
        action = Bc_ppo.old_actor.act_actor(...)
        if action == 1:
            subg_actions.append(node_id)

# 生成完整计划（一次性）
matching_plan = actions2str(start_node, subg_actions)  # "start-start-u1-u2-..."
```

**C++ 端（计划执行阶段）**：
```cpp
// StudyPerformance.cpp:362-368
// 一次性解析完整计划
std::stringstream ss(input_order_type);
while (std::getline(ss, item, '-')) {
    tokens.push_back(std::stoul(item));
}
// 按照固定计划执行，无实时交互
EvaluateQuery::QuickSIStyleForRL(..., RL_matching_order, ...);
```

### 1.2 核心问题

❌ **计划是固定的**：
- 在搜索开始前，Python 端就生成了完整的操作序列
- C++ 端一次性接收完整计划，按顺序执行
- **无法根据搜索过程中的实际状态动态调整**

❌ **缺乏实时反馈**：
- Python 和 C++ 之间只有一次交互（传递完整计划）
- C++ 执行完成后才返回结果
- 无法在搜索过程中获取中间状态

❌ **exp(u) → exp(u,v) 改进的局限性**：
- 虽然可以让计划更细粒度（从 exp(u) 到 exp(u,v)）
- 但计划仍然是在搜索前一次性生成的
- **仍然是固定计划，不是动态的**

## 二、自适应动态 Φ 的定义

### 2.1 什么是自适应动态计划？

**固定计划（当前）**：
```
搜索前：生成完整计划 Φ = [gen(u1), exp(u1), gen(u2), exp(u2), ...]
搜索中：严格按照 Φ 执行，无法调整
```

**动态计划（目标）**：
```
搜索中：每执行一步操作后
  → 获取当前搜索状态（已匹配节点、候选池大小、搜索树深度等）
  → RL 模型根据状态决定下一步操作
  → 执行该操作
  → 循环，直到搜索完成
```

### 2.2 动态计划是否需要预先生成计划？

**答案：❌ 不需要生成完整计划，但需要初始状态**

#### 2.2.1 不需要的部分

❌ **不需要生成完整操作序列**：
- 不需要预先决定所有 `gen(u)` 和 `exp(u)` 的顺序
- 不需要预先决定每个 `exp(u)` 要尝试哪些 `v`
- **完全可以在搜索过程中动态决策**

#### 2.2.2 需要初始化的部分

✅ **需要初始状态**（在搜索开始前确定）：

1. **起始节点 `start_node`**：
   ```python
   # 当前：train.py:94
   start_node = query_graph.Get_max_degree()  # 选择度最大的节点
   ```
   - 可以选择度最大、候选数最少等启发式方法
   - 或者用 RL 模型选择（但这是单次决策，不是完整计划）

2. **初始候选池**：
   ```python
   # 当前：train.py:95-96
   build_candidate_pool[start_node] = True
   expand_search_tree[start_node] = True
   ```
   - 为起始节点构建候选池 `LC(start_node)`
   - 这是搜索的起点，必须预先确定

3. **查询图结构**：
   - 查询图的节点、边、标签等信息
   - 数据图的完整信息

#### 2.2.3 动态计划的初始化流程

**方案 A：最小初始化（推荐）**
```python
# 1. 初始化：只确定起始节点和初始候选池
start_node = select_start_node(query_graph, data_graph)  # 启发式或 RL 单次决策
init_candidate_pool = build_candidate_pool(start_node)

# 2. 开始动态搜索
state = initialize_search_state(start_node, init_candidate_pool)

# 3. 动态决策循环（无需预先生成计划）
while not is_search_complete(state):
    action = rl_model.decide_next_action(state)  # 动态决策
    state = execute_action(action, state)         # 执行并更新状态
```

**方案 B：粗略框架（可选）**
```python
# 1. 初始化：确定要匹配的节点集合（但不决定顺序）
nodes_to_match = set(query_graph.nodes) - {start_node}

# 2. 动态决策：决定匹配顺序和展开时机
while nodes_to_match:
    # 动态选择下一个要处理的节点
    next_node = rl_model.select_next_node(state, nodes_to_match)
    # 动态决定是否展开
    if rl_model.should_expand(state, next_node):
        # 动态选择要尝试的候选
        candidates = rl_model.select_candidates(state, next_node)
        # ...
```

#### 2.2.4 对比总结

| 阶段 | 固定计划（当前） | 动态计划（目标） |
|------|----------------|----------------|
| **搜索前** | 生成完整计划 Φ | 只初始化起始状态 |
| **搜索中** | 按计划执行 | 动态决策每一步 |
| **计划长度** | 2×(n-1) 个操作 | 不确定（可能更短或更长） |
| **调整能力** | ❌ 无法调整 | ✅ 可随时调整 |

**关键区别**：
- **固定计划**：`搜索前生成完整计划 → 搜索中按计划执行`
- **动态计划**：`搜索前只初始化 → 搜索中边决策边执行`

### 2.3 动态计划的优势

1. **自适应调整**：根据搜索过程中的实际反馈调整策略
2. **早期剪枝**：发现无效分支时，可以提前终止或改变方向
3. **状态感知**：利用搜索过程中的实时信息（如候选池缩小、匹配失败等）

## 三、实现自适应动态 Φ 的方案

### 3.1 架构设计

**当前架构**：
```
Python (计划生成) → 完整计划字符串 → C++ (执行) → 结果
```

**动态架构**：
```
Python (RL决策) ←→ C++ (执行引擎)
     ↓                ↓
  决策下一步     执行操作 + 返回状态
     ↓                ↓
  循环交互直到搜索完成
```

### 3.2 关键技术点

#### 3.2.1 Python-C++ 实时交互

**方案 A：进程间通信（IPC）**
- 使用管道（pipe）或共享内存
- Python 和 C++ 通过消息传递
- 优点：高效，适合频繁交互
- 缺点：实现复杂

**方案 B：Python C++ 扩展**
- 将 C++ 编译为 Python 扩展模块
- 直接调用 C++ 函数
- 优点：性能好，调用简单
- 缺点：需要重新编译

**方案 C：RPC/网络通信**
- 使用 gRPC 或 socket
- Python 和 C++ 作为独立服务
- 优点：解耦，易于调试
- 缺点：有网络开销

**推荐**：方案 B（Python C++ 扩展），性能最好

#### 3.2.2 状态表示与传递

**需要传递的状态信息**：
```python
class SearchState:
    current_depth: int              # 当前搜索深度
    matched_nodes: List[Tuple]      # 已匹配的 (u, v) 对
    candidate_pools: Dict           # 每个 u 的候选池 LC(u)
    search_tree_size: int          # 搜索树节点数
    intersection_count: int        # 交集计算次数
    embedding_count: int           # 已找到的嵌入数
    is_timeout: bool               # 是否超时
```

**C++ 端状态获取**：
```cpp
// 伪代码
struct SearchState {
    int current_depth;
    std::vector<std::pair<VertexID, VertexID>> matched_nodes;
    std::map<VertexID, std::vector<VertexID>> candidate_pools;
    size_t search_tree_size;
    size_t intersection_count;
    size_t embedding_count;
    bool is_timeout;
};

SearchState get_current_state() {
    // 从 EvaluateQuery 中获取当前状态
    SearchState state;
    state.current_depth = depth;
    state.matched_nodes = get_matched_nodes(embedding, depth);
    state.candidate_pools = get_candidate_pools(candidates, candidates_count);
    // ...
    return state;
}
```

#### 3.2.3 单步执行接口

**C++ 端需要支持单步执行**：
```cpp
// 伪代码
// 执行单个操作（gen(u) 或 exp(u,v)）
bool execute_single_operation(
    OperationType op_type,  // GEN 或 EXP
    VertexID u,             // 查询节点
    VertexID v = -1,        // 数据节点（仅 exp 需要）
    SearchState& state      // 返回当前状态
);

// 或者更细粒度
bool execute_gen(VertexID u, SearchState& state);
bool execute_exp(VertexID u, VertexID v, SearchState& state);
```

**Python 端决策循环**：
```python
# 伪代码
def dynamic_search(query_graph, data_graph):
    state = initialize_state()
    
    while not is_search_complete(state):
        # 1. 获取当前状态
        state = cpp_engine.get_current_state()
        
        # 2. RL 模型决策下一步操作
        action = rl_model.decide_next_action(state, query_graph)
        # action = {'type': 'gen', 'u': u} 或 {'type': 'exp', 'u': u, 'v': v}
        
        # 3. 执行操作
        success = cpp_engine.execute_operation(action, state)
        
        # 4. 更新状态（C++ 端已更新，这里只是获取）
        state = cpp_engine.get_current_state()
        
        # 5. 如果失败或需要调整，可以改变策略
        if not success:
            # 可以回溯或改变策略
            handle_failure(state)
    
    return state.embedding_count
```

### 3.3 具体实现步骤

#### 步骤 1：修改 C++ 接口支持单步执行

**当前**：
```cpp
// 一次性执行完整计划
size_t QuickSIStyleForRL(..., ui *order, ...);
```

**改进后**：
```cpp
// 初始化搜索
void init_search(Graph *data_graph, Graph *query_graph, ...);

// 执行单个操作
bool execute_gen(VertexID u, SearchState& state);
bool execute_exp(VertexID u, VertexID v, SearchState& state);

// 获取当前状态
SearchState get_current_state();

// 检查是否完成
bool is_search_complete();
```

#### 步骤 2：创建 Python C++ 扩展

**使用 pybind11**：
```cpp
// python_binding.cpp
#include <pybind11/pybind11.h>
#include "EvaluateQuery.h"

class SearchEngine {
public:
    void init_search(...);
    bool execute_gen(int u);
    bool execute_exp(int u, int v);
    py::dict get_current_state();
    bool is_complete();
};

PYBIND11_MODULE(search_engine, m) {
    py::class_<SearchEngine>(m, "SearchEngine")
        .def(py::init<>())
        .def("init_search", &SearchEngine::init_search)
        .def("execute_gen", &SearchEngine::execute_gen)
        .def("execute_exp", &SearchEngine::execute_exp)
        .def("get_current_state", &SearchEngine::get_current_state)
        .def("is_complete", &SearchEngine::is_complete);
}
```

#### 步骤 3：修改 Python 端训练循环

**当前**：
```python
# 生成完整计划
matching_plan = generate_complete_plan(...)
# 执行
result = execute_subgraph_matching_cpp(..., matching_plan)
```

**改进后**：
```python
# 初始化搜索引擎
engine = SearchEngine()
engine.init_search(data_graph, query_graph, ...)

# 动态决策循环
while not engine.is_complete():
    state = engine.get_current_state()
    action = rl_model.decide(state)
    
    if action['type'] == 'gen':
        engine.execute_gen(action['u'])
    elif action['type'] == 'exp':
        engine.execute_exp(action['u'], action['v'])
    
    # 收集训练数据
    collect_training_data(state, action)

result = engine.get_final_result()
```

## 四、exp(u) → exp(u,v) 与动态计划的关系

### 4.1 两个独立的改进维度

| 维度 | exp(u) → exp(u,v) | 固定计划 → 动态计划 |
|------|------------------|-------------------|
| **改进内容** | 操作粒度细化 | 决策时机改变 |
| **当前状态** | ❌ 未实现 | ❌ 未实现 |
| **可以独立实现** | ✅ 是 | ✅ 是 |
| **组合效果** | 更细粒度的动态计划 | 最佳性能 |

### 4.2 组合实现的效果

**固定计划 + exp(u)**：
```
搜索前：生成计划 [gen(u1), exp(u1), gen(u2), exp(u2)]
搜索中：按计划执行，exp(u1) 遍历所有 v
```

**固定计划 + exp(u,v)**：
```
搜索前：生成计划 [gen(u1), exp(u1,v1), exp(u1,v2), gen(u2), exp(u2,v1)]
搜索中：按计划执行，但可以跳过某些 v
```

**动态计划 + exp(u)**：
```
搜索中：每步决策
  → 状态：当前深度、已匹配节点...
  → 决策：gen(u) 或 exp(u)
  → 执行：exp(u) 时遍历所有 v
```

**动态计划 + exp(u,v)**（最佳）：
```
搜索中：每步决策
  → 状态：当前深度、已匹配节点、候选池...
  → 决策：gen(u) 或 exp(u,v)（选择具体的 v）
  → 执行：只尝试选定的 v
  → 根据结果调整下一步策略
```

## 五、实现优先级建议

### 阶段 1：实现动态计划框架（exp(u) 粒度）
1. 修改 C++ 支持单步执行（gen/exp）
2. 创建 Python C++ 扩展
3. 实现 Python 端动态决策循环
4. **验证动态计划的有效性**

### 阶段 2：细化操作粒度（exp(u,v)）
1. 在动态计划基础上，将 exp(u) 细化为 exp(u,v)
2. 实现候选池获取接口
3. 实现 (u,v) 对的状态编码
4. **验证细粒度动态计划的效果**

### 阶段 3：优化与加速
1. 批量决策（一次选择多个操作）
2. 预过滤候选池
3. 缓存状态信息

## 六、关键技术挑战

### 6.1 Python-C++ 交互开销
- **问题**：频繁交互可能带来性能开销
- **缓解**：
  - 使用高效的 IPC 机制（共享内存）
  - 批量传递状态信息
  - 减少不必要的状态查询

### 6.2 状态同步
- **问题**：Python 和 C++ 的状态需要保持一致
- **缓解**：
  - C++ 作为单一数据源
  - Python 只读状态，不修改
  - 使用版本号或时间戳验证一致性

### 6.3 训练稳定性
- **问题**：动态决策增加了训练复杂度
- **缓解**：
  - 使用课程学习
  - 设计合适的奖励函数
  - 使用经验回放

## 七、结论

### 7.1 当前状态
- ❌ **未实现动态计划**：当前是固定计划生成
- ❌ **未实现 exp(u,v)**：当前是 exp(u) 粒度
- ✅ **架构支持**：可以在此基础上实现

### 7.2 实现建议
1. **先实现动态计划框架**（使用 exp(u) 粒度）
2. **再细化操作粒度**（exp(u) → exp(u,v)）
3. **两者结合**：细粒度的动态计划 = 最佳性能

### 7.3 预期收益
- **动态计划**：可以自适应调整，避免无效搜索
- **exp(u,v)**：可以跳过无价值的候选，缩小搜索空间
- **组合**：预期 1-2 个数量级的性能提升

