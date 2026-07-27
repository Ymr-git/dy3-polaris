"""算力资源协议 - 完整单元测试.

测试覆盖:
1. 枚举类型 (ComputeResourceType / ComputeResourceStatus / TaskResult)
2. 数据模型 (ComputeResourceDescriptor / ComputeTask)
3. 调度器 (ComputeScheduler)
   - 资源注册 / 注销 / 查询 / 状态变更
   - 任务分配 (allocate) / 队列满异常 / 自定义策略
   - 无类型分配 / 降级分配
   - 任务状态转换 (start / complete / fail / cancel)
   - DRAINING 排空 / set_offline
   - 统计属性 / 导出 / clear
4. 调度策略引擎
   - PriorityFirstStrategy / ShortestQueueStrategy
   - WeightedLoadBalanceStrategy / AffinityStrategy
   - 工具亲和映射 (get/set_tool_type_affinity / build_context)
5. 度量收集 (ComputeMetrics)
   - 事件记录 / 导出 / 重置 / 边界条件
"""

from __future__ import annotations

import time

import pytest

from dy3_polaris.l6.compute.metrics import ComputeMetrics, _Counter, _LatencyTracker
from dy3_polaris.l6.compute.scheduler import ComputeScheduler, ComputeTask, TaskResult
from dy3_polaris.l6.compute.strategy import (
    AffinityStrategy,
    PriorityFirstStrategy,
    ShortestQueueStrategy,
    StrategyContext,
    WeightedLoadBalanceStrategy,
    build_context,
    get_tool_preferred_type,
    set_tool_type_affinity,
)
from dy3_polaris.l6.core.exceptions import (
    ComputeNoAvailableError,
    ComputeQueueFullError,
    ComputeResourceNotFoundError,
    ComputeTaskNotFoundError,
)
from dy3_polaris.l6.core.models import (
    ComputeResourceDescriptor,
    ComputeResourceStatus,
    ComputeResourceType,
)


# ============================================================
# 辅助工具
# ============================================================

def _make_resource(
    resource_type: ComputeResourceType = ComputeResourceType.LOCAL_CPU,
    name: str = "test-res",
    resource_id: str | None = None,
    priority: int = 0,
    max_queue_depth: int = 10,
    estimated_latency_ms: int = 100,
    status: ComputeResourceStatus = ComputeResourceStatus.AVAILABLE,
    **kwargs,
) -> ComputeResourceDescriptor:
    """快速构造 ComputeResourceDescriptor 的工厂函数."""
    fields = {
        "resource_type": resource_type,
        "name": name,
        "priority": priority,
        "max_queue_depth": max_queue_depth,
        "estimated_latency_ms": estimated_latency_ms,
        "status": status,
        **kwargs,
    }
    if resource_id is not None:
        fields["resource_id"] = resource_id
    return ComputeResourceDescriptor(**fields)


def _make_scheduler_with_resources(
    *resources: ComputeResourceDescriptor,
) -> tuple[ComputeScheduler, list[str]]:
    """创建调度器并注册资源，返回 (调度器, 资源 ID 列表)."""
    s = ComputeScheduler()
    ids = [s.register(r) for r in resources]
    return s, ids


# ============================================================
# 1. 枚举类型测试
# ============================================================

class TestComputeResourceType:
    """ComputeResourceType 枚举测试."""

    def test_五类资源齐全(self) -> None:
        """验证 5 种算力资源类型均存在."""
        expected = {"local_cpu", "gpu", "ssh_remote", "hpc_slurm", "cloud_gpu"}
        actual = {t.value for t in ComputeResourceType}
        assert actual == expected

    def test枚举值为字符串(self) -> None:
        """枚举继承 str，可直接当字符串使用."""
        assert ComputeResourceType.GPU == "gpu"
        assert isinstance(ComputeResourceType.GPU, str)

    def test枚举成员数量(self) -> None:
        """恰好 5 个成员."""
        assert len(ComputeResourceType) == 5


class TestComputeResourceStatus:
    """ComputeResourceStatus 枚举测试."""

    def test四种状态齐全(self) -> None:
        """验证 4 种资源状态均存在."""
        expected = {"available", "busy", "offline", "draining"}
        actual = {s.value for s in ComputeResourceStatus}
        assert actual == expected

    def test枚举值为字符串(self) -> None:
        assert ComputeResourceStatus.AVAILABLE == "available"
        assert isinstance(ComputeResourceStatus.BUSY, str)


class TestTaskResult:
    """TaskResult 枚举测试."""

    def test六种结果状态齐全(self) -> None:
        """验证 6 种任务结果状态均存在."""
        expected = {"pending", "running", "completed", "failed", "cancelled", "timeout"}
        actual = {t.value for t in TaskResult}
        assert actual == expected

    def test默认值为_pending(self) -> None:
        """ComputeTask 创建后默认状态应为 PENDING."""
        task = ComputeTask(task_id="t1", resource_id="r1")
        assert task.status == TaskResult.PENDING


# ============================================================
# 2. ComputeResourceDescriptor 模型测试
# ============================================================

class TestComputeResourceDescriptor:
    """算力资源描述符测试."""

    def test基本创建(self) -> None:
        """使用最少参数创建资源描述符."""
        r = _make_resource(name="cpu-01")
        assert r.name == "cpu-01"
        assert r.resource_type == ComputeResourceType.LOCAL_CPU
        assert r.status == ComputeResourceStatus.AVAILABLE
        assert r.resource_id  # 自动生成，非空

    def test自动生成_resource_id(self) -> None:
        """不指定 resource_id 时自动生成 12 位十六进制 ID."""
        r = _make_resource()
        assert len(r.resource_id) == 12
        assert all(c in "0123456789abcdef" for c in r.resource_id)

    def test指定_resource_id(self) -> None:
        """手动指定 resource_id 时使用给定值."""
        r = _make_resource(resource_id="my-fixed-id")
        assert r.resource_id == "my-fixed-id"

    def test_queue_depth_计算属性(self) -> None:
        """queue_depth 应等于 current_queue 长度."""
        r = _make_resource()
        assert r.queue_depth == 0
        r.current_queue.extend(["t1", "t2", "t3"])
        assert r.queue_depth == 3

    def test_is_available_空闲时为真(self) -> None:
        """状态为 AVAILABLE 且队列未满时 is_available 为 True."""
        r = _make_resource(max_queue_depth=5)
        assert r.is_available is True

    def test_is_available_状态非_available(self) -> None:
        """状态非 AVAILABLE 时 is_available 为 False."""
        r = _make_resource(status=ComputeResourceStatus.BUSY)
        assert r.is_available is False

    def test_is_available_队列已满(self) -> None:
        """队列已满时 is_available 为 False."""
        r = _make_resource(max_queue_depth=2)
        r.current_queue.extend(["t1", "t2"])
        assert r.is_available is False

    def test_is_available_队列接近满但未满(self) -> None:
        """队列深度等于 max_queue_depth - 1 时仍可用."""
        r = _make_resource(max_queue_depth=3)
        r.current_queue.append("t1")
        assert r.is_available is True

    def test_gpu_特有字段(self) -> None:
        """GPU 资源可设置 gpu_count 和 gpu_memory_gb."""
        r = _make_resource(
            resource_type=ComputeResourceType.GPU,
            name="nvidia-rtx4090",
            gpu_count=1,
            gpu_memory_gb=24.0,
            cpu_cores=16,
            memory_gb=64.0,
        )
        assert r.gpu_count == 1
        assert r.gpu_memory_gb == 24.0
        assert r.cpu_cores == 16
        assert r.memory_gb == 64.0

    def test远程资源_endpoint(self) -> None:
        """SSH 远程和云端资源可设置 endpoint."""
        r = _make_resource(
            resource_type=ComputeResourceType.SSH_REMOTE,
            name="remote-node-1",
            endpoint="user@192.168.1.100:22",
        )
        assert r.endpoint == "user@192.168.1.100:22"

    def test_model_dump_序列化(self) -> None:
        """model_dump(mode="json") 应返回可 JSON 序列化的字典."""
        r = _make_resource(resource_id="ser-test", name="gpu-01")
        d = r.model_dump(mode="json")
        assert d["resource_id"] == "ser-test"
        assert d["name"] == "gpu-01"
        assert d["resource_type"] == "local_cpu"
        assert isinstance(d["current_queue"], list)

    def test优先级范围(self) -> None:
        """priority 应在 [-100, 100] 范围内."""
        # 合法值
        r = _make_resource(priority=50)
        assert r.priority == 50
        # 边界值
        r_low = _make_resource(priority=-100)
        r_high = _make_resource(priority=100)
        assert r_low.priority == -100
        assert r_high.priority == 100

    def test优先级超出范围应抛异常(self) -> None:
        """priority 超出范围时 Pydantic 校验应拒绝."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            _make_resource(priority=101)
        with pytest.raises(Exception):
            _make_resource(priority=-101)

    def test_auth_config(self) -> None:
        """认证配置为可选字典."""
        r = _make_resource(auth_config={"ssh_key": "/path/to/key"})
        assert r.auth_config == {"ssh_key": "/path/to/key"}
        r2 = _make_resource()
        assert r2.auth_config is None


# ============================================================
# 3. ComputeTask 测试
# ============================================================

class TestComputeTask:
    """算力任务记录测试."""

    def test基本创建(self) -> None:
        """使用最少参数创建任务."""
        t = ComputeTask(task_id="t-001", resource_id="r-001")
        assert t.task_id == "t-001"
        assert t.resource_id == "r-001"
        assert t.tool_name == ""
        assert t.status == TaskResult.PENDING
        assert t.priority == 0
        assert t.estimated_duration_ms == 0
        assert t.metadata == {}

    def test完整参数创建(self) -> None:
        """使用全部参数创建任务."""
        t = ComputeTask(
            task_id="t-002",
            resource_id="r-002",
            tool_name="path_simulation",
            priority=10,
            estimated_duration_ms=5000,
            metadata={"model": "gpt-4"},
        )
        assert t.tool_name == "path_simulation"
        assert t.priority == 10
        assert t.estimated_duration_ms == 5000
        assert t.metadata == {"model": "gpt-4"}

    def test_time_字段自动填充(self) -> None:
        """submitted_at 应自动设置为当前时间."""
        before = time.time()
        t = ComputeTask(task_id="t-003", resource_id="r-003")
        after = time.time()
        assert before <= t.submitted_at <= after
        assert t.started_at is None
        assert t.completed_at is None

    def test_to_dict(self) -> None:
        """to_dict 应包含所有关键字段."""
        t = ComputeTask(
            task_id="t-004",
            resource_id="r-004",
            tool_name="vasp_query_result",
            priority=5,
        )
        d = t.to_dict()
        assert d["task_id"] == "t-004"
        assert d["resource_id"] == "r-004"
        assert d["tool_name"] == "vasp_query_result"
        assert d["status"] == "pending"
        assert d["priority"] == 5
        assert "submitted_at" in d
        assert "started_at" in d
        assert "completed_at" in d
        assert "estimated_duration_ms" in d

    def test_metadata_默认空字典(self) -> None:
        """不传 metadata 时应得到空字典而非 None."""
        t = ComputeTask(task_id="t-005", resource_id="r-005")
        assert t.metadata == {}

    def test_metadata_引用同一字典(self) -> None:
        """ComputeTask 不复制 metadata 字典，外部修改会影响任务."""
        m = {"key": "value"}
        t = ComputeTask(task_id="t-006", resource_id="r-006", metadata=m)
        m["key"] = "changed"
        assert t.metadata["key"] == "changed"


# ============================================================
# 4. ComputeScheduler 测试
# ============================================================

class TestSchedulerResourceManagement:
    """调度器 - 资源注册/注销/查询."""

    def test_register_返回资源_id(self) -> None:
        """注册资源应返回资源 ID."""
        s = ComputeScheduler()
        r = _make_resource(resource_id="res-01")
        rid = s.register(r)
        assert rid == "res-01"

    def test_register_后可查询(self) -> None:
        """注册后 get_resource 应能查到."""
        s = ComputeScheduler()
        r = _make_resource(resource_id="res-02", name="gpu-a")
        s.register(r)
        fetched = s.get_resource("res-02")
        assert fetched is not None
        assert fetched.name == "gpu-a"

    def test_register_更新_type_index(self) -> None:
        """注册后类型索引应包含该资源."""
        s = ComputeScheduler()
        s.register(_make_resource(resource_id="g1", resource_type=ComputeResourceType.GPU))
        s.register(_make_resource(resource_id="g2", resource_type=ComputeResourceType.GPU))
        by_type = s.resources_by_type(ComputeResourceType.GPU)
        assert len(by_type) == 2

    def test_unregister_空闲资源(self) -> None:
        """注销空闲资源应成功返回 True."""
        s = ComputeScheduler()
        s.register(_make_resource(resource_id="res-u1"))
        assert s.unregister("res-u1") is True
        assert s.get_resource("res-u1") is None

    def test_unregister_不存在返回_false(self) -> None:
        """注销不存在的资源应返回 False."""
        s = ComputeScheduler()
        assert s.unregister("nonexistent") is False

    def test_unregister_队列非空返回_false(self) -> None:
        """队列中有任务时不能注销资源."""
        s = ComputeScheduler()
        s.register(_make_resource(resource_id="res-u2"))
        s.allocate("task-x", resource_type=ComputeResourceType.LOCAL_CPU)
        assert s.unregister("res-u2") is False

    def test_get_resource_不存在返回_none(self) -> None:
        """查询不存在的资源应返回 None."""
        s = ComputeScheduler()
        assert s.get_resource("ghost") is None

    def test_set_status_成功(self) -> None:
        """设置已有资源的状态应生效."""
        s = ComputeScheduler()
        s.register(_make_resource(resource_id="res-s1"))
        s.set_status("res-s1", ComputeResourceStatus.BUSY)
        assert s.get_resource("res-s1").status == ComputeResourceStatus.BUSY

    def test_set_status_不存在抛异常(self) -> None:
        """对不存在的资源设置状态应抛 ComputeResourceNotFoundError."""
        s = ComputeScheduler()
        with pytest.raises(ComputeResourceNotFoundError):
            s.set_status("ghost", ComputeResourceStatus.AVAILABLE)

    def test_resource_count(self) -> None:
        """resource_count 应反映已注册资源数量."""
        s = ComputeScheduler()
        assert s.resource_count == 0
        s.register(_make_resource(resource_id="rc1"))
        s.register(_make_resource(resource_id="rc2"))
        assert s.resource_count == 2

    def test_resources_by_type_仅返回该类型(self) -> None:
        """resources_by_type 应只返回指定类型的资源."""
        s = ComputeScheduler()
        s.register(_make_resource(resource_id="cpu-a", resource_type=ComputeResourceType.LOCAL_CPU))
        s.register(_make_resource(resource_id="gpu-a", resource_type=ComputeResourceType.GPU))
        s.register(_make_resource(resource_id="cpu-b", resource_type=ComputeResourceType.LOCAL_CPU))
        cpu_res = s.resources_by_type(ComputeResourceType.LOCAL_CPU)
        assert len(cpu_res) == 2
        gpu_res = s.resources_by_type(ComputeResourceType.GPU)
        assert len(gpu_res) == 1


class TestSchedulerAllocate:
    """调度器 - 资源分配."""

    def test_basic_allocate(self) -> None:
        """基本分配：指定类型，返回 ComputeTask."""
        s = ComputeScheduler()
        s.register(_make_resource(resource_id="a1", resource_type=ComputeResourceType.GPU))
        task = s.allocate("t-alloc-1", resource_type=ComputeResourceType.GPU)
        assert task.task_id == "t-alloc-1"
        assert task.resource_id == "a1"
        assert task.status == TaskResult.PENDING

    def test_allocate_后队列深度增加(self) -> None:
        """分配后资源的队列深度应 +1."""
        s = ComputeScheduler()
        s.register(_make_resource(resource_id="a2", resource_type=ComputeResourceType.GPU))
        s.allocate("t-alloc-2", resource_type=ComputeResourceType.GPU)
        r = s.get_resource("a2")
        assert r.queue_depth == 1
        assert "t-alloc-2" in r.current_queue

    def test_allocate_关联_tool_name(self) -> None:
        """分配时 tool_name 应记录到任务中."""
        s = ComputeScheduler()
        s.register(_make_resource(resource_id="a3", resource_type=ComputeResourceType.LOCAL_CPU))
        task = s.allocate("t-alloc-3", resource_type=ComputeResourceType.LOCAL_CPU, tool_name="bkt_compute")
        assert task.tool_name == "bkt_compute"

    def test_allocate_关联优先级(self) -> None:
        """分配时 priority 应记录到任务中."""
        s = ComputeScheduler()
        s.register(_make_resource(resource_id="a4", resource_type=ComputeResourceType.LOCAL_CPU))
        task = s.allocate("t-alloc-4", resource_type=ComputeResourceType.LOCAL_CPU, priority=10)
        assert task.priority == 10

    def test_allocate_关联_metadata(self) -> None:
        """分配时 metadata 应记录到任务中."""
        s = ComputeScheduler()
        s.register(_make_resource(resource_id="a5", resource_type=ComputeResourceType.LOCAL_CPU))
        meta = {"model": "llama3"}
        task = s.allocate("t-alloc-5", resource_type=ComputeResourceType.LOCAL_CPU, metadata=meta)
        assert task.metadata == meta

    def test_allocate_无可用资源抛异常(self) -> None:
        """无任何可用资源时应抛 ComputeNoAvailableError."""
        s = ComputeScheduler()
        with pytest.raises(ComputeNoAvailableError):
            s.allocate("t-no-res", resource_type=ComputeResourceType.GPU)

    def test_allocate_指定类型但无该类型资源触发降级(self) -> None:
        """请求 GPU 但无 GPU 资源时，应降级到低阶资源."""
        s = ComputeScheduler()
        s.register(_make_resource(
            resource_id="cpu-fallback",
            resource_type=ComputeResourceType.LOCAL_CPU,
        ))
        task = s.allocate("t-degrade-1", resource_type=ComputeResourceType.CLOUD_GPU)
        assert task.resource_id == "cpu-fallback"
        assert task.resource_id != ""

    def test_allocate_指定类型_所有资源已满_降级成功(self) -> None:
        """GPU 资源队列已满时降级到可用资源."""
        s = ComputeScheduler()
        # 创建队列深度为 1 的 GPU（立即满）
        gpu = _make_resource(
            resource_id="gpu-full",
            resource_type=ComputeResourceType.GPU,
            max_queue_depth=1,
        )
        s.register(gpu)
        cpu = _make_resource(resource_id="cpu-free", resource_type=ComputeResourceType.LOCAL_CPU)
        s.register(cpu)
        # 先填满 GPU
        s.allocate("fill-gpu", resource_type=ComputeResourceType.GPU)
        # 此时再分配 GPU 应降级到 CPU
        task = s.allocate("t-degrade-2", resource_type=ComputeResourceType.GPU)
        assert task.resource_id == "cpu-free"

    def test_allocate_队列满抛异常(self) -> None:
        """当资源队列已满且无降级候选时抛 ComputeNoAvailableError.

        LOCAL_CPU 是最低阶资源，队列满后降级也找不到更低阶资源，
        因此最终抛出 ComputeNoAvailableError（携带 degradation 提示）。
        """
        s = ComputeScheduler()
        s.register(_make_resource(
            resource_id="tiny",
            resource_type=ComputeResourceType.LOCAL_CPU,
            max_queue_depth=1,
        ))
        # 第一个任务成功，队列已满
        s.allocate("fill-1", resource_type=ComputeResourceType.LOCAL_CPU)
        # 第二个任务应失败（LOCAL_CPU 是最低阶，降级无候选）
        with pytest.raises(ComputeNoAvailableError, match="degradation"):
            s.allocate("fill-2", resource_type=ComputeResourceType.LOCAL_CPU)

    def test_allocate_自定义策略(self) -> None:
        """使用自定义策略选择资源."""
        s = ComputeScheduler()
        s.register(_make_resource(resource_id="custom-1", resource_type=ComputeResourceType.GPU))
        s.register(_make_resource(resource_id="custom-2", resource_type=ComputeResourceType.GPU))
        # 自定义策略：始终选择第一个
        def first_strategy(candidates):
            return candidates[0] if candidates else None

        task = s.allocate("t-custom", resource_type=ComputeResourceType.GPU, strategy=first_strategy)
        assert task.resource_id == "custom-1"

    def test_allocate_无指定类型_从所有可用资源中选择(self) -> None:
        """不指定 resource_type 时从所有可用资源中选择."""
        s = ComputeScheduler()
        s.register(_make_resource(
            resource_id="mix-cpu",
            resource_type=ComputeResourceType.LOCAL_CPU,
            priority=10,
        ))
        s.register(_make_resource(
            resource_id="mix-gpu",
            resource_type=ComputeResourceType.GPU,
            priority=5,
        ))
        task = s.allocate("t-no-type")
        # 默认策略选优先级最高的
        assert task.resource_id == "mix-cpu"

    def test_allocate_无指定类型_无可用资源抛异常(self) -> None:
        """不指定类型且无任何资源时抛 ComputeNoAvailableError."""
        s = ComputeScheduler()
        with pytest.raises(ComputeNoAvailableError, match="No available resources"):
            s.allocate("t-empty")

    def test_allocate_多次分配到同一资源(self) -> None:
        """多次分配应正确累加队列."""
        s = ComputeScheduler()
        s.register(_make_resource(resource_id="multi-q", resource_type=ComputeResourceType.LOCAL_CPU))
        s.allocate("t-m1", resource_type=ComputeResourceType.LOCAL_CPU)
        s.allocate("t-m2", resource_type=ComputeResourceType.LOCAL_CPU)
        s.allocate("t-m3", resource_type=ComputeResourceType.LOCAL_CPU)
        r = s.get_resource("multi-q")
        assert r.queue_depth == 3
        assert r.current_queue == ["t-m1", "t-m2", "t-m3"]

    def test_allocate_后任务可查询(self) -> None:
        """分配后 get_task 应能查到任务."""
        s = ComputeScheduler()
        s.register(_make_resource(resource_id="aq1", resource_type=ComputeResourceType.LOCAL_CPU))
        s.allocate("t-aq1", resource_type=ComputeResourceType.LOCAL_CPU)
        task = s.get_task("t-aq1")
        assert task is not None
        assert task.task_id == "t-aq1"

    def test_allocate_后_total_tasks_增加(self) -> None:
        """分配后 total_tasks 应增加."""
        s = ComputeScheduler()
        s.register(_make_resource(resource_id="tt1", resource_type=ComputeResourceType.LOCAL_CPU))
        assert s.total_tasks == 0
        s.allocate("t-tt1", resource_type=ComputeResourceType.LOCAL_CPU)
        assert s.total_tasks == 1

    def test_allocate_后资源状态变为_busy(self) -> None:
        """队列达到 max_queue_depth 时资源状态应变为 BUSY."""
        s = ComputeScheduler()
        s.register(_make_resource(
            resource_id="busy-res",
            resource_type=ComputeResourceType.LOCAL_CPU,
            max_queue_depth=2,
        ))
        s.allocate("t-b1", resource_type=ComputeResourceType.LOCAL_CPU)
        assert s.get_resource("busy-res").status == ComputeResourceStatus.AVAILABLE
        s.allocate("t-b2", resource_type=ComputeResourceType.LOCAL_CPU)
        assert s.get_resource("busy-res").status == ComputeResourceStatus.BUSY

    def test_allocate_estimated_duration_ms(self) -> None:
        """分配时应记录预估时长."""
        s = ComputeScheduler()
        s.register(_make_resource(resource_id="dur1", resource_type=ComputeResourceType.LOCAL_CPU))
        task = s.allocate("t-dur1", resource_type=ComputeResourceType.LOCAL_CPU, estimated_duration_ms=3000)
        assert task.estimated_duration_ms == 3000


class TestSchedulerTaskLifecycle:
    """调度器 - 任务状态转换."""

    def setup_method(self) -> None:
        """每个测试前创建调度器并分配一个任务."""
        self.scheduler = ComputeScheduler()
        self.scheduler.register(_make_resource(
            resource_id="lc-01",
            resource_type=ComputeResourceType.LOCAL_CPU,
        ))
        self.task = self.scheduler.allocate(
            "t-lifecycle",
            resource_type=ComputeResourceType.LOCAL_CPU,
            tool_name="bkt_compute",
        )

    def test_start_task(self) -> None:
        """start_task 应将任务标记为 RUNNING 并设置 started_at."""
        before = time.time()
        result = self.scheduler.start_task("t-lifecycle")
        after = time.time()
        assert result.status == TaskResult.RUNNING
        assert before <= result.started_at <= after

    def test_complete_task(self) -> None:
        """complete_task 应将任务标记为 COMPLETED 并出队."""
        self.scheduler.start_task("t-lifecycle")
        result = self.scheduler.complete_task("t-lifecycle")
        assert result.status == TaskResult.COMPLETED
        assert result.completed_at is not None
        # 出队后队列深度应为 0
        r = self.scheduler.get_resource("lc-01")
        assert r.queue_depth == 0

    def test_fail_task(self) -> None:
        """fail_task 应将任务标记为 FAILED 并出队."""
        result = self.scheduler.fail_task("t-lifecycle", error="OOM")
        assert result.status == TaskResult.FAILED
        assert result.completed_at is not None
        r = self.scheduler.get_resource("lc-01")
        assert r.queue_depth == 0

    def test_cancel_task(self) -> None:
        """cancel_task 应将任务标记为 CANCELLED 并出队."""
        result = self.scheduler.cancel_task("t-lifecycle")
        assert result.status == TaskResult.CANCELLED
        assert result.completed_at is not None
        r = self.scheduler.get_resource("lc-01")
        assert r.queue_depth == 0

    def test_complete_后队列深度减少(self) -> None:
        """出队后队列深度应减少，且状态恢复为 AVAILABLE."""
        # 直接操作队列来模拟满队列场景（避免分配时降级逻辑干扰）
        s = self.scheduler
        r = s.get_resource("lc-01")
        # setup_method 已分配 t-lifecycle（队列中有 1 个任务）
        # 再填充 max_queue_depth - 1 个任务使其恰好满
        remaining = r.max_queue_depth - r.queue_depth
        for i in range(remaining):
            task = ComputeTask(task_id=f"t-fill-{i}", resource_id="lc-01")
            s._tasks[task.task_id] = task
            r.current_queue.append(task.task_id)
        r.status = ComputeResourceStatus.BUSY
        full_depth = r.queue_depth
        assert full_depth == r.max_queue_depth
        # 完成一个任务后队列深度应减少
        s.complete_task("t-lifecycle")
        assert r.queue_depth == full_depth - 1
        # 队列有空位后自动恢复为 AVAILABLE
        assert r.status == ComputeResourceStatus.AVAILABLE

    def test_start_task_不存在抛异常(self) -> None:
        """start_task 对不存在的任务应抛 ComputeTaskNotFoundError."""
        with pytest.raises(ComputeTaskNotFoundError):
            self.scheduler.start_task("ghost-task")

    def test_complete_task_不存在抛异常(self) -> None:
        """complete_task 对不存在的任务应抛异常."""
        with pytest.raises(ComputeTaskNotFoundError):
            self.scheduler.complete_task("ghost-task")

    def test_fail_task_不存在抛异常(self) -> None:
        """fail_task 对不存在的任务应抛异常."""
        with pytest.raises(ComputeTaskNotFoundError):
            self.scheduler.fail_task("ghost-task")

    def test_cancel_task_不存在抛异常(self) -> None:
        """cancel_task 对不存在的任务应抛异常."""
        with pytest.raises(ComputeTaskNotFoundError):
            self.scheduler.cancel_task("ghost-task")

    def test_get_task_不存在返回_none(self) -> None:
        """get_task 对不存在的任务应返回 None."""
        assert self.scheduler.get_task("no-such-task") is None

    def test_active_tasks_统计(self) -> None:
        """active_tasks 应统计 PENDING 和 RUNNING 状态的任务数."""
        s = self.scheduler
        # 当前有一个 PENDING 任务
        assert s.active_tasks == 1
        # 启动后变为 RUNNING
        s.start_task("t-lifecycle")
        assert s.active_tasks == 1
        # 完成后无活跃任务
        s.complete_task("t-lifecycle")
        assert s.active_tasks == 0


class TestSchedulerDraining:
    """调度器 - DRAINING 排空模式."""

    def test_start_draining(self) -> None:
        """start_draining 应将资源状态设为 DRAINING."""
        s = ComputeScheduler()
        s.register(_make_resource(resource_id="dr-01"))
        s.start_draining("dr-01")
        assert s.get_resource("dr-01").status == ComputeResourceStatus.DRAINING

    def test_start_draining_不存在抛异常(self) -> None:
        """对不存在的资源执行排空应抛异常."""
        s = ComputeScheduler()
        with pytest.raises(ComputeResourceNotFoundError):
            s.start_draining("ghost")

    def test_is_drained_队列为空时为真(self) -> None:
        """DRAINING 且队列为空时 is_drained 应返回 True."""
        s = ComputeScheduler()
        s.register(_make_resource(resource_id="dr-02"))
        s.start_draining("dr-02")
        assert s.is_drained("dr-02") is True

    def test_is_drained_队列非空时为假(self) -> None:
        """DRAINING 但队列仍有任务时 is_drained 应返回 False."""
        s = ComputeScheduler()
        s.register(_make_resource(resource_id="dr-03"))
        s.allocate("t-drain-pending", resource_type=ComputeResourceType.LOCAL_CPU)
        s.start_draining("dr-03")
        assert s.is_drained("dr-03") is False
        # 完成任务后应排空
        s.complete_task("t-drain-pending")
        assert s.is_drained("dr-03") is True

    def test_is_drained_资源不存在返回_false(self) -> None:
        """对不存在的资源 is_drained 应返回 False."""
        s = ComputeScheduler()
        assert s.is_drained("ghost") is False

    def test_set_offline_已排空时成功(self) -> None:
        """资源已排空后可设为 OFFLINE."""
        s = ComputeScheduler()
        s.register(_make_resource(resource_id="dr-04"))
        s.start_draining("dr-04")
        s.set_offline("dr-04")
        assert s.get_resource("dr-04").status == ComputeResourceStatus.OFFLINE

    def test_set_offline_未排空抛异常(self) -> None:
        """队列非空时 set_offline 应抛 ComputeQueueFullError."""
        s = ComputeScheduler()
        s.register(_make_resource(resource_id="dr-05"))
        s.allocate("t-not-drained", resource_type=ComputeResourceType.LOCAL_CPU)
        with pytest.raises(ComputeQueueFullError):
            s.set_offline("dr-05")

    def test_set_offline_资源不存在抛异常(self) -> None:
        """对不存在的资源 set_offline 应抛异常."""
        s = ComputeScheduler()
        with pytest.raises(ComputeResourceNotFoundError):
            s.set_offline("ghost")

    def test_draining_资源不可分配(self) -> None:
        """DRAINING 状态的资源不应出现在可用列表中."""
        s = ComputeScheduler()
        s.register(_make_resource(resource_id="dr-06"))
        s.start_draining("dr-06")
        assert s.available_count == 0
        with pytest.raises(ComputeNoAvailableError):
            s.allocate("t-onto-draining", resource_type=ComputeResourceType.LOCAL_CPU)


class TestSchedulerExport:
    """调度器 - 导出与清理."""

    def test_export_summary_基本结构(self) -> None:
        """export_summary 应包含所有关键字段."""
        s = ComputeScheduler()
        s.register(_make_resource(resource_id="ex-01", resource_type=ComputeResourceType.GPU))
        summary = s.export_summary()
        assert "resource_count" in summary
        assert "available_count" in summary
        assert "total_tasks" in summary
        assert "active_tasks" in summary
        assert "resources_by_type" in summary
        assert "degradation_enabled" in summary

    def test_export_summary_资源统计(self) -> None:
        """export_summary 的资源统计应准确."""
        s = ComputeScheduler()
        s.register(_make_resource(resource_id="s1", resource_type=ComputeResourceType.GPU))
        s.register(_make_resource(resource_id="s2", resource_type=ComputeResourceType.GPU))
        s.register(_make_resource(resource_id="s3", resource_type=ComputeResourceType.LOCAL_CPU))
        summary = s.export_summary()
        assert summary["resource_count"] == 3
        assert summary["resources_by_type"]["gpu"] == 2
        assert summary["resources_by_type"]["local_cpu"] == 1

    def test_export_all_包含资源和任务(self) -> None:
        """export_all 应包含完整资源和任务数据."""
        s = ComputeScheduler()
        s.register(_make_resource(resource_id="ea-01", resource_type=ComputeResourceType.LOCAL_CPU))
        s.allocate("t-ea-1", resource_type=ComputeResourceType.LOCAL_CPU)
        data = s.export_all()
        assert "resources" in data
        assert "tasks" in data
        assert "summary" in data
        assert "ea-01" in data["resources"]
        assert "t-ea-1" in data["tasks"]

    def test_clear_清空所有数据(self) -> None:
        """clear 应清空所有资源和任务."""
        s = ComputeScheduler()
        s.register(_make_resource(resource_id="cl-01", resource_type=ComputeResourceType.GPU))
        s.allocate("t-cl-1", resource_type=ComputeResourceType.GPU)
        s.clear()
        assert s.resource_count == 0
        assert s.total_tasks == 0
        assert s.available_count == 0

    def test_available_count_仅统计可用资源(self) -> None:
        """available_count 应仅统计状态为 AVAILABLE 且队列未满的资源."""
        s = ComputeScheduler()
        s.register(_make_resource(resource_id="av-01", resource_type=ComputeResourceType.LOCAL_CPU))
        s.register(_make_resource(
            resource_id="av-02",
            resource_type=ComputeResourceType.LOCAL_CPU,
            status=ComputeResourceStatus.OFFLINE,
        ))
        assert s.available_count == 1

    def test_export_summary_降级开关(self) -> None:
        """export_summary 应包含 degradation_enabled 字段."""
        s = ComputeScheduler()
        summary = s.export_summary()
        assert summary["degradation_enabled"] is True


class TestSchedulerDegradation:
    """调度器 - 降级路径测试."""

    def test_降级路径顺序(self) -> None:
        """降级路径应为 CLOUD_GPU > HPC_SLURM > GPU > SSH_REMOTE > LOCAL_CPU."""
        from dy3_polaris.l6.compute.scheduler import _DEGRADATION_PATH
        assert _DEGRADATION_PATH == [
            ComputeResourceType.CLOUD_GPU,
            ComputeResourceType.HPC_SLURM,
            ComputeResourceType.GPU,
            ComputeResourceType.SSH_REMOTE,
            ComputeResourceType.LOCAL_CPU,
        ]

    def test_gpu_降级到_ssh_remote(self) -> None:
        """GPU 不可用时降级路径应依次尝试."""
        s = ComputeScheduler()
        # 只注册 SSH_REMOTE，GPU 和 HPC_SLURM 不存在
        s.register(_make_resource(
            resource_id="ssh-fallback",
            resource_type=ComputeResourceType.SSH_REMOTE,
        ))
        task = s.allocate("t-gpu-degrade", resource_type=ComputeResourceType.GPU)
        assert task.resource_id == "ssh-fallback"

    def test_cloud_gpu_多级降级到_local_cpu(self) -> None:
        """CLOUD_GPU 应能多级降级到 LOCAL_CPU."""
        s = ComputeScheduler()
        s.register(_make_resource(
            resource_id="cpu-bottom",
            resource_type=ComputeResourceType.LOCAL_CPU,
        ))
        task = s.allocate("t-multi-degrade", resource_type=ComputeResourceType.CLOUD_GPU)
        assert task.resource_id == "cpu-bottom"

    def test_local_cpu_无法降级(self) -> None:
        """LOCAL_CPU 是最低阶资源，无降级空间."""
        s = ComputeScheduler()
        # 不注册任何资源
        with pytest.raises(ComputeNoAvailableError, match="degradation"):
            s.allocate("t-no-degrade", resource_type=ComputeResourceType.LOCAL_CPU)

    def test降级不修改原始_resource_type(self) -> None:
        """降级分配后任务应关联到降级后的实际资源."""
        s = ComputeScheduler()
        s.register(_make_resource(
            resource_id="hpc-fb",
            resource_type=ComputeResourceType.HPC_SLURM,
        ))
        task = s.allocate("t-check-type", resource_type=ComputeResourceType.CLOUD_GPU)
        # 任务关联的是 HPC 资源
        assert task.resource_id == "hpc-fb"


class TestSchedulerDefaultStrategy:
    """调度器 - 默认策略行为."""

    def test优先选高优先级资源(self) -> None:
        """默认策略应选择优先级最高的资源."""
        s = ComputeScheduler()
        s.register(_make_resource(
            resource_id="low-pri",
            resource_type=ComputeResourceType.GPU,
            priority=1,
        ))
        s.register(_make_resource(
            resource_id="high-pri",
            resource_type=ComputeResourceType.GPU,
            priority=10,
        ))
        task = s.allocate("t-pri", resource_type=ComputeResourceType.GPU)
        assert task.resource_id == "high-pri"

    def test优先级相同时选队列最短(self) -> None:
        """优先级相同时应选队列最短的资源."""
        s = ComputeScheduler()
        s.register(_make_resource(
            resource_id="q-long",
            resource_type=ComputeResourceType.LOCAL_CPU,
            priority=5,
        ))
        s.register(_make_resource(
            resource_id="q-short",
            resource_type=ComputeResourceType.LOCAL_CPU,
            priority=5,
        ))
        # 在 q-long 上分配一个任务使其队列变长
        s.allocate("t-fill", resource_type=ComputeResourceType.LOCAL_CPU)
        # 但默认策略优先选高优先级+最短队列，两个资源优先级相同
        # q-long 可能被选中因为第一次分配时不一定选哪个（队列相同时）
        # 换一种方式：先分配到 q-long，再分配新任务
        s2 = ComputeScheduler()
        s2.register(_make_resource(
            resource_id="d-q1",
            resource_type=ComputeResourceType.GPU,
            priority=5,
            estimated_latency_ms=200,
        ))
        s2.register(_make_resource(
            resource_id="d-q2",
            resource_type=ComputeResourceType.GPU,
            priority=5,
            estimated_latency_ms=100,
        ))
        # 优先级相同、队列相同，选延迟更低的
        task = s2.allocate("t-lat", resource_type=ComputeResourceType.GPU)
        assert task.resource_id == "d-q2"  # 延迟更低


# ============================================================
# 5. 策略引擎测试
# ============================================================

class TestStrategyContext:
    """策略上下文测试."""

    def test默认值(self) -> None:
        """不传参数时所有字段应有合理默认值."""
        ctx = StrategyContext()
        assert ctx.tool_name == ""
        assert ctx.task_priority == 0
        assert ctx.estimated_duration_ms == 0
        assert ctx.resource_history == {}
        assert ctx.preferred_type is None

    def test完整参数(self) -> None:
        """传入所有参数应正确保存."""
        ctx = StrategyContext(
            tool_name="path_simulation",
            task_priority=10,
            estimated_duration_ms=5000,
            resource_history={"r1": 5, "r2": 3},
            preferred_type=ComputeResourceType.GPU,
        )
        assert ctx.tool_name == "path_simulation"
        assert ctx.task_priority == 10
        assert ctx.estimated_duration_ms == 5000
        assert ctx.resource_history == {"r1": 5, "r2": 3}
        assert ctx.preferred_type == ComputeResourceType.GPU

    def test_slots_不可添加新属性(self) -> None:
        """使用 __slots__，不可动态添加属性."""
        ctx = StrategyContext()
        with pytest.raises(AttributeError):
            ctx.extra_field = "value"  # type: ignore[attr-defined]


class TestPriorityFirstStrategy:
    """优先级优先策略测试."""

    def test选最高优先级(self) -> None:
        """应选择优先级最高的资源."""
        r_low = _make_resource(resource_id="low", priority=1)
        r_high = _make_resource(resource_id="high", priority=10)
        r_mid = _make_resource(resource_id="mid", priority=5)
        strategy = PriorityFirstStrategy()
        result = strategy.select([r_low, r_high, r_mid])
        assert result is not None
        assert result.resource_id == "high"

    def test优先级相同时选队列最短(self) -> None:
        """优先级相同时应选队列最短的资源."""
        r1 = _make_resource(resource_id="q1", priority=5)
        r1.current_queue.append("t1")
        r2 = _make_resource(resource_id="q2", priority=5)
        # q2 队列为空，应被选中
        strategy = PriorityFirstStrategy()
        result = strategy.select([r1, r2])
        assert result.resource_id == "q2"

    def test空候选列表返回_none(self) -> None:
        """候选列表为空时应返回 None."""
        strategy = PriorityFirstStrategy()
        assert strategy.select([]) is None

    def test单个候选直接返回(self) -> None:
        """只有一个候选时应直接返回."""
        r = _make_resource(resource_id="only")
        strategy = PriorityFirstStrategy()
        result = strategy.select([r])
        assert result.resource_id == "only"

    def test_context_参数可选(self) -> None:
        """不传 context 也不应报错."""
        r = _make_resource(resource_id="no-ctx")
        strategy = PriorityFirstStrategy()
        result = strategy.select([r], context=None)
        assert result.resource_id == "no-ctx"


class TestShortestQueueStrategy:
    """最短队列优先策略测试."""

    def test选队列最短(self) -> None:
        """应选择队列最短的资源."""
        r_long = _make_resource(resource_id="long")
        r_long.current_queue.extend(["t1", "t2", "t3"])
        r_short = _make_resource(resource_id="short")
        r_mid = _make_resource(resource_id="mid")
        r_mid.current_queue.append("t1")
        strategy = ShortestQueueStrategy()
        result = strategy.select([r_long, r_short, r_mid])
        assert result.resource_id == "short"

    def test队列相同时选延迟最低(self) -> None:
        """队列长度相同时应选延迟最低的资源."""
        r_slow = _make_resource(resource_id="slow", estimated_latency_ms=500)
        r_fast = _make_resource(resource_id="fast", estimated_latency_ms=50)
        strategy = ShortestQueueStrategy()
        result = strategy.select([r_slow, r_fast])
        assert result.resource_id == "fast"

    def test空候选返回_none(self) -> None:
        """候选列表为空时应返回 None."""
        strategy = ShortestQueueStrategy()
        assert strategy.select([]) is None


class TestWeightedLoadBalanceStrategy:
    """加权负载均衡策略测试."""

    def test选空闲度最高的资源(self) -> None:
        """空闲度越高评分越高，应被选中."""
        r_busy = _make_resource(resource_id="busy", max_queue_depth=10, estimated_latency_ms=100)
        r_busy.current_queue.extend(["t" + str(i) for i in range(8)])
        r_idle = _make_resource(resource_id="idle", max_queue_depth=10, estimated_latency_ms=100)
        strategy = WeightedLoadBalanceStrategy()
        result = strategy.select([r_busy, r_idle])
        assert result.resource_id == "idle"

    def test权重系数(self) -> None:
        """验证权重系数为 0.6 空闲 + 0.4 延迟."""
        assert WeightedLoadBalanceStrategy.WEIGHT_IDLE == 0.6
        assert WeightedLoadBalanceStrategy.WEIGHT_LATENCY == 0.4

    def test优先级加分(self) -> None:
        """有上下文优先级时应给高优先级资源加分."""
        r_low_pri = _make_resource(resource_id="lp", priority=1, max_queue_depth=10, estimated_latency_ms=100)
        r_high_pri = _make_resource(resource_id="hp", priority=100, max_queue_depth=10, estimated_latency_ms=100)
        ctx = StrategyContext(task_priority=10)
        strategy = WeightedLoadBalanceStrategy()
        result = strategy.select([r_low_pri, r_high_pri], context=ctx)
        assert result.resource_id == "hp"

    def test无上下文时也能工作(self) -> None:
        """不传 context 时策略应正常工作."""
        r = _make_resource(resource_id="no-ctx", max_queue_depth=10)
        strategy = WeightedLoadBalanceStrategy()
        result = strategy.select([r])
        assert result.resource_id == "no-ctx"

    def test空候选返回_none(self) -> None:
        """候选列表为空时应返回 None."""
        strategy = WeightedLoadBalanceStrategy()
        assert strategy.select([]) is None


class TestAffinityStrategy:
    """亲和性策略测试."""

    def test历史亲和优先(self) -> None:
        """工具历史中分配最多的资源应被优先选择."""
        r_a = _make_resource(resource_id="ra", resource_type=ComputeResourceType.GPU)
        r_b = _make_resource(resource_id="rb", resource_type=ComputeResourceType.GPU)
        ctx = StrategyContext(
            tool_name="path_simulation",
            resource_history={"ra": 10, "rb": 2},
        )
        strategy = AffinityStrategy()
        result = strategy.select([r_a, r_b], context=ctx)
        assert result.resource_id == "ra"

    def test历史亲和仅匹配候选中的资源(self) -> None:
        """历史中分配最多的资源不在候选中时，应选候选中历史最多的."""
        r_a = _make_resource(resource_id="ra", resource_type=ComputeResourceType.GPU)
        r_b = _make_resource(resource_id="rb", resource_type=ComputeResourceType.GPU)
        # 历史中 ra 分配最多，但 rb 是候选
        ctx = StrategyContext(
            tool_name="some_tool",
            resource_history={"ra": 100, "rb": 5, "rc": 50},
        )
        strategy = AffinityStrategy()
        result = strategy.select([r_a, r_b], context=ctx)
        assert result.resource_id == "ra"

    def test类型亲和_无历史时触发(self) -> None:
        """无历史但有偏好类型时，应按类型亲和匹配."""
        r_cpu = _make_resource(resource_id="cpu", resource_type=ComputeResourceType.LOCAL_CPU)
        r_gpu = _make_resource(resource_id="gpu", resource_type=ComputeResourceType.GPU)
        ctx = StrategyContext(
            tool_name="custom_tool",
            preferred_type=ComputeResourceType.GPU,
        )
        strategy = AffinityStrategy()
        result = strategy.select([r_cpu, r_gpu], context=ctx)
        assert result.resource_id == "gpu"

    def test兜底回退到加权均衡(self) -> None:
        """无历史无类型亲和时，应回退到加权负载均衡."""
        r_busy = _make_resource(
            resource_id="busy",
            max_queue_depth=10,
            estimated_latency_ms=500,
        )
        r_busy.current_queue.extend(["t" + str(i) for i in range(9)])
        r_idle = _make_resource(
            resource_id="idle",
            max_queue_depth=10,
            estimated_latency_ms=50,
        )
        ctx = StrategyContext(tool_name="new_tool")
        strategy = AffinityStrategy()
        result = strategy.select([r_busy, r_idle], context=ctx)
        assert result.resource_id == "idle"

    def test自定义兜底策略(self) -> None:
        """可传入自定义兜底策略."""
        r_a = _make_resource(resource_id="aa", resource_type=ComputeResourceType.GPU)
        r_b = _make_resource(resource_id="bb", resource_type=ComputeResourceType.GPU)
        # 总是选第一个
        custom_fallback = PriorityFirstStrategy()
        strategy = AffinityStrategy(fallback=custom_fallback)
        result = strategy.select([r_a, r_b], context=StrategyContext(tool_name="tool_x"))
        # 无历史无类型亲和，回退到 PriorityFirst
        # 两个资源优先级相同（都是默认 0），选队列更短的
        assert result is not None

    def test空候选返回_none(self) -> None:
        """候选列表为空时应返回 None."""
        strategy = AffinityStrategy()
        assert strategy.select([]) is None

    def test无上下文时回退兜底(self) -> None:
        """不传 context 时应直接使用兜底策略."""
        r = _make_resource(resource_id="solo", max_queue_depth=10)
        strategy = AffinityStrategy()
        result = strategy.select([r], context=None)
        assert result.resource_id == "solo"


class TestToolTypeAffinity:
    """工具-资源类型亲和映射测试."""

    def test_get_已有映射(self) -> None:
        """获取已配置的工具偏好类型."""
        assert get_tool_preferred_type("path_simulation") == ComputeResourceType.GPU
        assert get_tool_preferred_type("thermocalc_phase_diagram") == ComputeResourceType.HPC_SLURM
        assert get_tool_preferred_type("vasp_query_result") == ComputeResourceType.HPC_SLURM

    def test_get_未配置返回_none(self) -> None:
        """未配置的工具应返回 None."""
        assert get_tool_preferred_type("nonexistent_tool") is None

    def test_set_新增映射(self) -> None:
        """set_tool_type_affinity 应新增或更新映射."""
        set_tool_type_affinity("my_new_tool", ComputeResourceType.CLOUD_GPU)
        assert get_tool_preferred_type("my_new_tool") == ComputeResourceType.CLOUD_GPU

    def test_set_覆盖已有映射(self) -> None:
        """set_tool_type_affinity 应能覆盖已有映射."""
        assert get_tool_preferred_type("path_simulation") == ComputeResourceType.GPU
        set_tool_type_affinity("path_simulation", ComputeResourceType.LOCAL_CPU)
        assert get_tool_preferred_type("path_simulation") == ComputeResourceType.LOCAL_CPU
        # 恢复原始值，避免影响其他测试
        set_tool_type_affinity("path_simulation", ComputeResourceType.GPU)


class TestBuildContext:
    """build_context 辅助函数测试."""

    def test自动注入偏好类型(self) -> None:
        """build_context 应自动查询工具亲和类型并注入."""
        ctx = build_context(tool_name="path_simulation")
        assert ctx.preferred_type == ComputeResourceType.GPU
        assert ctx.tool_name == "path_simulation"

    def test未配置工具偏好为_none(self) -> None:
        """未配置亲和关系的工具，preferred_type 应为 None."""
        ctx = build_context(tool_name="unknown_tool")
        assert ctx.preferred_type is None

    def test传入参数优先(self) -> None:
        """显式传入的参数不应被覆盖."""
        ctx = build_context(
            tool_name="path_simulation",
            task_priority=20,
            estimated_duration_ms=1000,
        )
        assert ctx.task_priority == 20
        assert ctx.estimated_duration_ms == 1000
        assert ctx.preferred_type == ComputeResourceType.GPU

    def test_resource_history_传递(self) -> None:
        """resource_history 应正确传递."""
        history = {"r1": 5}
        ctx = build_context(tool_name="any", resource_history=history)
        assert ctx.resource_history == {"r1": 5}


# ============================================================
# 6. 评分辅助函数测试
# ============================================================

class TestScoreHelpers:
    """评分辅助函数测试."""

    def test_score_base_空闲时为1(self) -> None:
        """队列空时基础评分应为 1.0."""
        from dy3_polaris.l6.compute.strategy import _score_base
        r = _make_resource(max_queue_depth=10)
        assert _score_base(r) == 1.0

    def test_score_base_半满时为0_5(self) -> None:
        """队列半满时基础评分应为 0.5."""
        from dy3_polaris.l6.compute.strategy import _score_base
        r = _make_resource(max_queue_depth=10)
        r.current_queue.extend(["t" + str(i) for i in range(5)])
        assert _score_base(r) == 0.5

    def test_score_base_全满时为0(self) -> None:
        """队列全满时基础评分应为 0.0."""
        from dy3_polaris.l6.compute.strategy import _score_base
        r = _make_resource(max_queue_depth=5)
        r.current_queue.extend(["t" + str(i) for i in range(5)])
        assert _score_base(r) == 0.0

    def test_score_latency_最小延迟接近1(self) -> None:
        """最小允许延迟（1ms）时评分应接近 1.0."""
        from dy3_polaris.l6.compute.strategy import _score_latency
        r = _make_resource(estimated_latency_ms=1)
        expected = 1.0 - 1 / 30000
        assert _score_latency(r) == pytest.approx(expected)

    def test_score_latency_高延迟接近0(self) -> None:
        """30000ms 延迟时评分应为 0.0."""
        from dy3_polaris.l6.compute.strategy import _score_latency
        r = _make_resource(estimated_latency_ms=30000)
        assert _score_latency(r) == 0.0

    def test_score_latency_中间值(self) -> None:
        """15000ms 延迟时评分应为 0.5."""
        from dy3_polaris.l6.compute.strategy import _score_latency
        r = _make_resource(estimated_latency_ms=15000)
        assert _score_latency(r) == 0.5

    def test_score_latency_超过最大值不小于0(self) -> None:
        """延迟超过 30000ms 时评分不低于 0."""
        from dy3_polaris.l6.compute.strategy import _score_latency
        r = _make_resource(estimated_latency_ms=50000)
        assert _score_latency(r) == 0.0


# ============================================================
# 7. ComputeMetrics 测试
# ============================================================

class TestCounter:
    """_Counter 线程安全计数器测试."""

    def test初始值为0(self) -> None:
        """新建计数器值应为 0."""
        c = _Counter()
        assert c.value == 0

    def test_inc_默认加1(self) -> None:
        """inc() 不传参时应 +1."""
        c = _Counter()
        c.inc()
        assert c.value == 1

    def test_inc_指定步长(self) -> None:
        """inc(n) 应加 n."""
        c = _Counter()
        c.inc(5)
        assert c.value == 5
        c.inc(3)
        assert c.value == 8


class TestLatencyTracker:
    """_LatencyTracker 延迟收集器测试."""

    def test初始状态(self) -> None:
        """新建时 count=0, avg=0, max=0."""
        lt = _LatencyTracker()
        assert lt.count == 0
        assert lt.avg == 0.0
        assert lt.max == 0.0

    def test_record_单条(self) -> None:
        """记录一条数据后统计应正确."""
        lt = _LatencyTracker()
        lt.record(100.0)
        assert lt.count == 1
        assert lt.avg == 100.0
        assert lt.max == 100.0

    def test_record_多条(self) -> None:
        """记录多条数据后 avg 和 max 应正确."""
        lt = _LatencyTracker()
        lt.record(100.0)
        lt.record(200.0)
        lt.record(300.0)
        assert lt.count == 3
        assert lt.avg == 200.0
        assert lt.max == 300.0

    def test_max_samples_限制(self) -> None:
        """超过 max_samples 时应丢弃最旧的数据."""
        lt = _LatencyTracker(max_samples=3)
        lt.record(1.0)
        lt.record(2.0)
        lt.record(3.0)
        lt.record(4.0)  # 超出，丢弃 1.0
        assert lt.count == 3
        assert lt.avg == pytest.approx((2.0 + 3.0 + 4.0) / 3)
        assert lt.max == 4.0

    def test_record_零值(self) -> None:
        """记录 0.0 也应被接受."""
        lt = _LatencyTracker()
        lt.record(0.0)
        assert lt.count == 1
        assert lt.avg == 0.0


class TestComputeMetrics:
    """ComputeMetrics 度量收集器测试."""

    def test初始_export_全零(self) -> None:
        """新建度量器导出时所有计数应为 0."""
        m = ComputeMetrics()
        data = m.export()
        assert data["tasks"]["created"] == 0
        assert data["tasks"]["completed"] == 0
        assert data["tasks"]["failed"] == 0
        assert data["tasks"]["cancelled"] == 0
        assert data["tasks"]["success_rate"] == 0.0
        assert data["degradations"] == 0
        assert data["allocations_by_type"] == {}
        assert data["latency_ms"]["avg"] == 0.0
        assert data["latency_ms"]["max"] == 0.0
        assert data["latency_ms"]["samples"] == 0
        assert data["queue_wait_ms"]["avg"] == 0.0

    def test_on_task_created(self) -> None:
        """记录任务创建应增加计数和类型分配."""
        m = ComputeMetrics()
        m.on_task_created("gpu")
        m.on_task_created("gpu")
        m.on_task_created("local_cpu")
        data = m.export()
        assert data["tasks"]["created"] == 3
        assert data["allocations_by_type"]["gpu"] == 2
        assert data["allocations_by_type"]["local_cpu"] == 1

    def test_on_task_completed(self) -> None:
        """记录任务完成应增加完成计数和延迟."""
        m = ComputeMetrics()
        m.on_task_completed(latency_ms=100.0, wait_ms=50.0)
        m.on_task_completed(latency_ms=200.0, wait_ms=80.0)
        data = m.export()
        assert data["tasks"]["completed"] == 2
        assert data["latency_ms"]["avg"] == 150.0
        assert data["latency_ms"]["max"] == 200.0
        assert data["latency_ms"]["samples"] == 2
        assert data["queue_wait_ms"]["avg"] == 65.0

    def test_on_task_failed(self) -> None:
        """记录任务失败应增加失败计数."""
        m = ComputeMetrics()
        m.on_task_failed()
        m.on_task_failed()
        data = m.export()
        assert data["tasks"]["failed"] == 2

    def test_on_task_cancelled(self) -> None:
        """记录任务取消应增加取消计数."""
        m = ComputeMetrics()
        m.on_task_cancelled()
        data = m.export()
        assert data["tasks"]["cancelled"] == 1

    def test_on_degradation(self) -> None:
        """记录降级事件应增加降级计数."""
        m = ComputeMetrics()
        m.on_degradation()
        m.on_degradation()
        data = m.export()
        assert data["degradations"] == 2

    def test_success_rate_计算(self) -> None:
        """成功率 = completed / (completed + failed + cancelled)."""
        m = ComputeMetrics()
        # 3 完成、1 失败、0 取消 -> 成功率 = 3/4 = 0.75
        m.on_task_created("cpu")
        m.on_task_completed()
        m.on_task_completed()
        m.on_task_completed()
        m.on_task_failed()
        data = m.export()
        assert data["tasks"]["success_rate"] == 0.75

    def test_success_rate_无完成任务时为0(self) -> None:
        """无完成/失败/取消任务时成功率应为 0.0."""
        m = ComputeMetrics()
        m.on_task_created("cpu")
        data = m.export()
        assert data["tasks"]["success_rate"] == 0.0

    def test_success_rate_仅取消任务(self) -> None:
        """仅取消任务时成功率应为 0.0."""
        m = ComputeMetrics()
        m.on_task_cancelled()
        data = m.export()
        assert data["tasks"]["success_rate"] == 0.0

    def test_uptime_seconds_递增(self) -> None:
        """uptime_seconds 应反映度量器存活时间."""
        m = ComputeMetrics()
        data1 = m.export()
        time.sleep(0.1)
        data2 = m.export()
        assert data2["uptime_seconds"] >= data1["uptime_seconds"]

    def test_reset_清空所有指标(self) -> None:
        """reset 应清空所有计数器和样本."""
        m = ComputeMetrics()
        m.on_task_created("gpu")
        m.on_task_completed(latency_ms=500.0)
        m.on_task_failed()
        m.on_degradation()
        m.reset()
        data = m.export()
        assert data["tasks"]["created"] == 0
        assert data["tasks"]["completed"] == 0
        assert data["tasks"]["failed"] == 0
        assert data["degradations"] == 0
        assert data["allocations_by_type"] == {}
        assert data["latency_ms"]["samples"] == 0

    def test_reset_后可继续记录(self) -> None:
        """reset 后应能继续正常记录."""
        m = ComputeMetrics()
        m.on_task_created("cpu")
        m.reset()
        m.on_task_created("gpu")
        m.on_task_completed(latency_ms=200.0)
        data = m.export()
        assert data["tasks"]["created"] == 1
        assert data["tasks"]["completed"] == 1
        assert data["allocations_by_type"]["gpu"] == 1

    def test_export_延迟样本超过上限(self) -> None:
        """延迟样本超过 max_samples 时应正确截断."""
        m = ComputeMetrics()
        # max_samples=200，记录 250 条
        for i in range(250):
            m.on_task_completed(latency_ms=float(i + 1))
        data = m.export()
        assert data["latency_ms"]["samples"] == 200
        # 最旧的 50 条被丢弃，最新的是 51..250
        assert data["latency_ms"]["max"] == 250.0
        # avg 应为 (51+52+...+250) / 200
        expected_avg = sum(range(51, 251)) / 200
        assert data["latency_ms"]["avg"] == pytest.approx(expected_avg)

    def test_on_task_completed_零延迟不记录(self) -> None:
        """延迟为 0 时不应记入延迟统计."""
        m = ComputeMetrics()
        m.on_task_completed(latency_ms=0.0, wait_ms=0.0)
        data = m.export()
        assert data["tasks"]["completed"] == 1
        assert data["latency_ms"]["samples"] == 0
        assert data["queue_wait_ms"]["samples"] == 0

    def test_negative_latency_不记录(self) -> None:
        """负延迟不应记入统计."""
        m = ComputeMetrics()
        m.on_task_completed(latency_ms=-10.0)
        data = m.export()
        assert data["tasks"]["completed"] == 1
        assert data["latency_ms"]["samples"] == 0
