"""L3 领域知识层 — 领域本体定义.

融合世界先进本体方案:
- W3C OWL: 类层级 (subClassOf)、属性约束 (cardinality, domain, range)
- Schema.org: 单根继承类型树、标准属性
- ChemOnt: 化学实体分类体系
- Materials Ontology: 四子本体组合 (Substance/Process/Environment/Property)
- Dublin Core: 15 个核心元数据属性

提供三个预构建领域本体 + 通用本体，支持自定义扩展。
"""

from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, Field

from .models import EntityType, RelationType, PropertyDataType, InferenceRuleType


# ============================================================
# 本体核心结构
# ============================================================


class OntologyProperty(BaseModel):
    """本体属性定义 (借鉴 OWL DatatypeProperty + SHACL 约束).

    Attributes:
        name: 属性名称
        display_name: 显示名称 (中文)
        description: 属性描述
        property_type: 属性类型 ("datatype" 或 "object")
        domain: 定义域 (适用的实体类型列表)
        range: 值域 (数据类型或实体类型)
        required: 是否必需
        cardinality: 基数约束 (0=任意, 1=单值, n=最多n个)
        default_value: 默认值
        enum_values: 枚举可选值
        data_type: 数据类型约束 (SHACL sh:datatype)
        min_count: 最小基数 (SHACL sh:minCount)
        max_count: 最大基数 (SHACL sh:maxCount)
        min_value: 数值下界 (SHACL sh:minInclusive)
        max_value: 数值上界 (SHACL sh:maxInclusive)
        min_length: 字符串最小长度 (SHACL sh:minLength)
        max_length: 字符串最大长度 (SHACL sh:maxLength)
        pattern: 正则模式 (SHACL sh:pattern)
    """

    name: str = Field(..., description="属性名称")
    display_name: str = Field(default="", description="显示名称 (中文)")
    description: str = Field(default="", description="属性描述")
    property_type: str = Field(default="datatype", description="属性类型 (datatype/object)")
    domain: list[EntityType] = Field(default_factory=list, description="定义域 (适用的实体类型)")
    range: str = Field(default="string", description="值域 (数据类型或实体类型)")
    required: bool = Field(default=False, description="是否必需")
    cardinality: int = Field(default=0, ge=0, description="基数约束 (0=任意)")
    default_value: Any = Field(default=None, description="默认值")
    enum_values: list[str] = Field(default_factory=list, description="枚举可选值")
    # ---- 增强字段: SHACL 风格约束 ----
    data_type: PropertyDataType | None = Field(default=None, description="数据类型约束 (SHACL sh:datatype)")
    min_count: int | None = Field(default=None, ge=0, description="最小基数 (SHACL sh:minCount)")
    max_count: int | None = Field(default=None, ge=0, description="最大基数 (SHACL sh:maxCount)")
    min_value: float | None = Field(default=None, description="数值下界 (SHACL sh:minInclusive)")
    max_value: float | None = Field(default=None, description="数值上界 (SHACL sh:maxInclusive)")
    min_length: int | None = Field(default=None, ge=0, description="字符串最小长度 (SHACL sh:minLength)")
    max_length: int | None = Field(default=None, ge=0, description="字符串最大长度 (SHACL sh:maxLength)")
    pattern: str | None = Field(default=None, description="正则模式 (SHACL sh:pattern)")

    def validate_value(self, value: Any) -> list[str]:
        """验证属性值是否符合 SHACL 约束.

        Returns:
            违规信息列表 (空列表表示通过)
        """
        violations: list[str] = []

        # 枚举值检查
        if self.enum_values and str(value) not in self.enum_values:
            violations.append(f"值 '{value}' 不在枚举 {self.enum_values} 中")

        # 数据类型检查
        if self.data_type is not None:
            type_violations = self._check_data_type(value)
            violations.extend(type_violations)

        # 数值范围检查
        if self.min_value is not None or self.max_value is not None:
            try:
                num_val = float(value)
                if self.min_value is not None and num_val < self.min_value:
                    violations.append(f"值 {num_val} 小于最小值 {self.min_value}")
                if self.max_value is not None and num_val > self.max_value:
                    violations.append(f"值 {num_val} 大于最大值 {self.max_value}")
            except (TypeError, ValueError):
                violations.append(f"值 '{value}' 不是有效数值，无法进行范围检查")

        # 字符串长度检查
        if self.min_length is not None or self.max_length is not None:
            str_val = str(value)
            if self.min_length is not None and len(str_val) < self.min_length:
                violations.append(f"字符串长度 {len(str_val)} 小于最小长度 {self.min_length}")
            if self.max_length is not None and len(str_val) > self.max_length:
                violations.append(f"字符串长度 {len(str_val)} 大于最大长度 {self.max_length}")

        # 正则模式检查
        if self.pattern is not None:
            import re
            try:
                if not re.search(self.pattern, str(value)):
                    violations.append(f"值 '{value}' 不匹配模式 '{self.pattern}'")
            except re.error:
                violations.append(f"正则模式 '{self.pattern}' 无效")

        return violations

    def _check_data_type(self, value: Any) -> list[str]:
        """检查值的数据类型."""
        violations: list[str] = []
        if self.data_type == PropertyDataType.STRING:
            if not isinstance(value, str):
                violations.append(f"期望 string 类型，实际 {type(value).__name__}")
        elif self.data_type == PropertyDataType.INTEGER:
            if not isinstance(value, int) or isinstance(value, bool):
                violations.append(f"期望 integer 类型，实际 {type(value).__name__}")
        elif self.data_type == PropertyDataType.FLOAT:
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                violations.append(f"期望 float 类型，实际 {type(value).__name__}")
        elif self.data_type == PropertyDataType.BOOLEAN:
            if not isinstance(value, bool):
                violations.append(f"期望 boolean 类型，实际 {type(value).__name__}")
        elif self.data_type == PropertyDataType.DATETIME:
            if not isinstance(value, (int, float)) and not isinstance(value, str):
                violations.append(f"期望 datetime 类型 (时间戳或 ISO 字符串)，实际 {type(value).__name__}")
        return violations


class OntologyRelation(BaseModel):
    """本体关系定义 (借鉴 OWL ObjectProperty).

    Attributes:
        name: 关系名称 (对应 RelationType 值)
        display_name: 显示名称 (中文)
        description: 关系描述
        domain: 定义域 (主语实体类型列表)
        range: 值域 (宾语实体类型列表)
        inverse_of: 逆关系名称
        transitive: 是否传递
        symmetric: 是否对称
        functional: 是否函数性 (一个主语最多对应一个宾语)
    """

    name: str = Field(..., description="关系名称")
    display_name: str = Field(default="", description="显示名称 (中文)")
    description: str = Field(default="", description="关系描述")
    domain: list[EntityType] = Field(default_factory=list, description="定义域 (主语实体类型)")
    range: list[EntityType] = Field(default_factory=list, description="值域 (宾语实体类型)")
    inverse_of: str = Field(default="", description="逆关系名称")
    transitive: bool = Field(default=False, description="是否传递")
    symmetric: bool = Field(default=False, description="是否对称")
    functional: bool = Field(default=False, description="是否函数性")


class OntologyClass(BaseModel):
    """本体类定义 (借鉴 OWL Class + Schema.org Type).

    Attributes:
        class_id: 类唯一标识
        entity_type: 对应的 EntityType 枚举值
        display_name: 显示名称 (中文)
        description: 类描述
        parent_type: 父类 EntityType (继承层级)
        properties: 类的属性定义列表
        allowed_relations: 允许的关系类型列表
        icon: 图标标识 (用于 UI 展示)
        metadata: 扩展元数据
    """

    class_id: str = Field(..., description="类唯一标识")
    entity_type: EntityType = Field(..., description="对应的 EntityType 枚举值")
    display_name: str = Field(default="", description="显示名称 (中文)")
    description: str = Field(default="", description="类描述")
    parent_type: EntityType | None = Field(default=None, description="父类 EntityType")
    properties: list[OntologyProperty] = Field(default_factory=list, description="属性定义列表")
    allowed_relations: list[RelationType] = Field(default_factory=list, description="允许的关系类型")
    icon: str = Field(default="", description="图标标识")
    metadata: dict[str, Any] = Field(default_factory=dict, description="扩展元数据")

    def has_property(self, prop_name: str) -> bool:
        """是否拥有指定属性."""
        return any(p.name == prop_name for p in self.properties)

    def get_property(self, prop_name: str) -> OntologyProperty | None:
        """获取指定属性定义."""
        for p in self.properties:
            if p.name == prop_name:
                return p
        return None

    def required_properties(self) -> list[OntologyProperty]:
        """获取所有必需属性."""
        return [p for p in self.properties if p.required]

    def is_subclass_of(self, parent: EntityType) -> bool:
        """是否是某个父类的子类."""
        return self.parent_type == parent

    def get_all_properties(self, ontology: DomainOntology | None = None) -> list[OntologyProperty]:
        """获取所有属性 (含从父类继承的属性) (借鉴 RDFS subClassOf 继承).

        Args:
            ontology: 所属领域本体 (用于查找父类定义)

        Returns:
            合并了父类属性的完整属性列表 (子类属性优先)
        """
        if ontology is None or self.parent_type is None:
            return list(self.properties)

        parent_cls = ontology.get_class(self.parent_type)
        if parent_cls is None:
            return list(self.properties)

        # 递归获取父类属性
        parent_props = parent_cls.get_all_properties(ontology)
        # 合并: 父类属性 + 自身属性 (自身覆盖同名属性)
        merged: dict[str, OntologyProperty] = {p.name: p for p in parent_props}
        for p in self.properties:
            merged[p.name] = p
        return list(merged.values())

    def get_all_allowed_relations(self, ontology: DomainOntology | None = None) -> list[RelationType]:
        """获取所有允许的关系 (含从父类继承的) (借鉴 RDFS subClassOf 继承).

        Args:
            ontology: 所属领域本体 (用于查找父类定义)

        Returns:
            合并了父类关系的完整关系列表 (去重)
        """
        if ontology is None or self.parent_type is None:
            return list(self.allowed_relations)

        parent_cls = ontology.get_class(self.parent_type)
        if parent_cls is None:
            return list(self.allowed_relations)

        parent_rels = parent_cls.get_all_allowed_relations(ontology)
        # 合并去重
        seen: set[RelationType] = set()
        result: list[RelationType] = []
        for r in parent_rels + list(self.allowed_relations):
            if r not in seen:
                seen.add(r)
                result.append(r)
        return result

    def get_required_properties(self, ontology: DomainOntology | None = None) -> list[OntologyProperty]:
        """获取所有必需属性 (含继承的)."""
        all_props = self.get_all_properties(ontology)
        return [p for p in all_props if p.required]


# ============================================================
# 本体公理与推理规则 (借鉴 OWL Axiom + SWRL + HermiT/Pellet)
# ============================================================


class OntologyAxiom(BaseModel):
    """本体公理 (借鉴 OWL Axiom + SHACL 逻辑约束).

    定义本体中的逻辑约束，用于推理和验证。

    Attributes:
        axiom_id: 公理唯一标识
        axiom_type: 公理类型 ("disjoint"/"equivalent"/"subclass"/"functional")
        description: 公理描述
        subject: 主体 (类/属性/关系名称)
        object: 客体 (类/属性/关系名称)
        constraints: 约束参数
    """

    axiom_id: str = Field(default_factory=lambda: f"ax-{uuid.uuid4().hex[:8]}")
    axiom_type: str = Field(..., description="公理类型")
    description: str = Field(default="", description="公理描述")
    subject: str = Field(..., description="主体 (类/属性/关系名称)")
    object: str = Field(default="", description="客体 (类/属性/关系名称)")
    constraints: dict[str, Any] = Field(default_factory=dict, description="约束参数")


class OntologyRule(BaseModel):
    """本体推理规则 (借鉴 SWRL 规则 + HermiT/Pellet 推理器).

    定义推理规则，用于从已有知识推导新知识。

    Attributes:
        rule_id: 规则唯一标识
        rule_type: 推理规则类型 (InferenceRuleType)
        description: 规则描述
        applies_to_relation: 目标关系名称 (传递/对称/逆关系)
        inverse_relation: 逆关系名称 (仅 inverse_relation 类型)
        property_chain: 属性链定义 (仅 property_chain 类型)
        enabled: 是否启用
    """

    rule_id: str = Field(default_factory=lambda: f"rl-{uuid.uuid4().hex[:8]}")
    rule_type: InferenceRuleType = Field(..., description="推理规则类型")
    description: str = Field(default="", description="规则描述")
    applies_to_relation: str = Field(default="", description="目标关系名称")
    inverse_relation: str = Field(default="", description="逆关系名称 (仅 inverse_relation)")
    property_chain: list[str] = Field(default_factory=list, description="属性链 (仅 property_chain)")
    enabled: bool = Field(default=True, description="是否启用")


class OntologyMapping(BaseModel):
    """跨本体映射 (借鉴 OWL equivalentClass/equivalentProperty + Ontology Alignment).

    定义不同本体之间的类/属性/关系映射关系。

    Attributes:
        mapping_id: 映射唯一标识
        source_domain: 源领域
        target_domain: 目标领域
        mapping_type: 映射类型 ("equivalent"/"subsumed"/"related")
        source_entity_type: 源实体类型
        target_entity_type: 目标实体类型
        source_property: 源属性名 (可选)
        target_property: 目标属性名 (可选)
        confidence: 映射置信度
    """

    mapping_id: str = Field(default_factory=lambda: f"mp-{uuid.uuid4().hex[:8]}")
    source_domain: str = Field(..., description="源领域")
    target_domain: str = Field(..., description="目标领域")
    mapping_type: str = Field(default="equivalent", description="映射类型")
    source_entity_type: EntityType = Field(..., description="源实体类型")
    target_entity_type: EntityType = Field(..., description="目标实体类型")
    source_property: str = Field(default="", description="源属性名")
    target_property: str = Field(default="", description="目标属性名")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="映射置信度")


# ============================================================
# 领域类型常量
# ============================================================


class DomainType(str):
    """领域类型常量."""

    CHEMISTRY = "chemistry"
    MATERIALS = "materials"
    EDUCATION = "education"
    GENERAL = "general"


class DomainOntology(BaseModel):
    """领域本体 (借鉴 Materials Ontology 多子本体组合模式).

    一个领域本体包含多个类定义、关系定义和属性定义，
    描述该领域内的实体类型、关系约束和属性规范。

    Attributes:
        ontology_id: 本体唯一标识
        domain: 领域标识
        display_name: 显示名称 (中文)
        description: 本体描述
        version: 本体版本
        classes: 类定义列表
        relations: 关系定义列表
        global_properties: 全局属性 (所有类共享)
    """

    ontology_id: str = Field(..., description="本体唯一标识")
    domain: str = Field(..., description="领域标识")
    display_name: str = Field(default="", description="显示名称 (中文)")
    description: str = Field(default="", description="本体描述")
    version: str = Field(default="1.0.0", description="本体版本")
    classes: list[OntologyClass] = Field(default_factory=list, description="类定义列表")
    relations: list[OntologyRelation] = Field(default_factory=list, description="关系定义列表")
    global_properties: list[OntologyProperty] = Field(default_factory=list, description="全局属性")
    # ---- 增强字段: 公理、推理规则、跨本体映射 ----
    axioms: list[OntologyAxiom] = Field(default_factory=list, description="本体公理列表")
    inference_rules: list[OntologyRule] = Field(default_factory=list, description="推理规则列表")
    mappings: list[OntologyMapping] = Field(default_factory=list, description="跨本体映射列表")

    def get_class(self, entity_type: EntityType) -> OntologyClass | None:
        """获取指定实体类型的类定义."""
        for c in self.classes:
            if c.entity_type == entity_type:
                return c
        return None

    def get_relation(self, name: str) -> OntologyRelation | None:
        """获取指定关系定义."""
        for r in self.relations:
            if r.name == name:
                return r
        return None

    def validate_entity_type(self, entity_type: EntityType) -> bool:
        """验证实体类型是否在本体中定义."""
        return any(c.entity_type == entity_type for c in self.classes)

    def validate_relation(
        self, relation: str, subject_type: EntityType, object_type: EntityType
    ) -> bool:
        """验证关系是否合法 (主语/宾语类型在定义域/值域内).

        Args:
            relation: 关系名称
            subject_type: 主语实体类型
            object_type: 宾语实体类型

        Returns:
            是否合法
        """
        rel = self.get_relation(relation)
        if rel is None:
            return False
        if rel.domain and subject_type not in rel.domain:
            return False
        if rel.range and object_type not in rel.range:
            return False
        return True

    def validate_properties(
        self, entity_type: EntityType, properties: dict[str, Any]
    ) -> list[str]:
        """验证实体的属性是否符合本体约束.

        Returns:
            违规信息列表 (空列表表示通过)
        """
        violations: list[str] = []
        cls = self.get_class(entity_type)
        if cls is None:
            violations.append(f"实体类型 {entity_type.value} 未在本体中定义")
            return violations

        # 检查必需属性
        for prop in cls.required_properties():
            if prop.name not in properties:
                violations.append(f"缺少必需属性: {prop.name}")

        # 检查属性值类型和枚举
        for prop_name, prop_value in properties.items():
            prop = cls.get_property(prop_name)
            if prop is None:
                # 检查全局属性
                prop = next((p for p in self.global_properties if p.name == prop_name), None)
            if prop is None:
                # 未定义的属性，允许但不验证
                continue
            if prop.enum_values and str(prop_value) not in prop.enum_values:
                violations.append(
                    f"属性 {prop_name} 的值 '{prop_value}' 不在枚举 {prop.enum_values} 中"
                )

        return violations

    def class_count(self) -> int:
        """类数量."""
        return len(self.classes)

    def relation_count(self) -> int:
        """关系数量."""
        return len(self.relations)

    # ---- 增强方法: SHACL 风格验证 ----

    def validate_data_types(
        self, entity_type: EntityType, properties: dict[str, Any]
    ) -> list[str]:
        """验证属性值的数据类型 (SHACL sh:datatype 风格).

        Returns:
            违规信息列表 (空列表表示通过)
        """
        violations: list[str] = []
        cls = self.get_class(entity_type)
        if cls is None:
            return [f"实体类型 {entity_type.value} 未在本体中定义"]

        all_props = cls.get_all_properties(self)
        prop_map: dict[str, OntologyProperty] = {p.name: p for p in all_props}

        # 检查全局属性
        for gp in self.global_properties:
            if gp.name not in prop_map:
                prop_map[gp.name] = gp

        for prop_name, prop_value in properties.items():
            prop = prop_map.get(prop_name)
            if prop is not None:
                violations.extend(prop.validate_value(prop_value))

        return violations

    def validate_cardinality(
        self, entity_type: EntityType, properties: dict[str, Any]
    ) -> list[str]:
        """验证属性基数约束 (SHACL sh:minCount/sh:maxCount 风格).

        对于列表类型的属性值，检查最小/最大出现次数。

        Returns:
            违规信息列表 (空列表表示通过)
        """
        violations: list[str] = []
        cls = self.get_class(entity_type)
        if cls is None:
            return [f"实体类型 {entity_type.value} 未在本体中定义"]

        all_props = cls.get_all_properties(self)
        for prop in all_props:
            if prop.min_count is not None or prop.max_count is not None:
                value = properties.get(prop.name)
                if value is None:
                    if prop.min_count is not None and prop.min_count > 0:
                        violations.append(f"属性 {prop.name} 最少需要 {prop.min_count} 个值")
                    continue
                count = len(value) if isinstance(value, (list, set, tuple)) else 1
                if prop.min_count is not None and count < prop.min_count:
                    violations.append(f"属性 {prop.name} 值数量 {count} 小于最小基数 {prop.min_count}")
                if prop.max_count is not None and count > prop.max_count:
                    violations.append(f"属性 {prop.name} 值数量 {count} 大于最大基数 {prop.max_count}")

        return violations

    def validate_full(
        self, entity_type: EntityType, properties: dict[str, Any]
    ) -> list[str]:
        """完整验证 (属性存在性 + 枚举值 + 数据类型 + 基数约束).

        Returns:
            所有违规信息列表 (空列表表示通过)
        """
        violations: list[str] = []
        violations.extend(self.validate_properties(entity_type, properties))
        violations.extend(self.validate_data_types(entity_type, properties))
        violations.extend(self.validate_cardinality(entity_type, properties))
        return violations

    # ---- 增强方法: 类层级管理 ----

    def get_class_hierarchy(self, entity_type: EntityType) -> list[EntityType]:
        """获取实体类型的完整类层级 (从根类到当前类).

        Returns:
            类层级列表 (从根到当前)，如果类型不存在返回空列表
        """
        hierarchy: list[EntityType] = []
        current = self.get_class(entity_type)
        if current is None:
            return hierarchy

        # 反向构建: 从当前类向上追溯到根
        chain: list[EntityType] = [entity_type]
        visited: set[EntityType] = {entity_type}
        while current.parent_type is not None and current.parent_type not in visited:
            chain.append(current.parent_type)
            visited.add(current.parent_type)
            current = self.get_class(current.parent_type)
            if current is None:
                break

        # 反转为从根到当前
        return list(reversed(chain))

    def get_subclasses(self, entity_type: EntityType) -> list[EntityType]:
        """获取某类型的所有直接子类.

        Returns:
            直接子类列表
        """
        return [
            c.entity_type for c in self.classes
            if c.parent_type == entity_type
        ]

    def get_all_subclasses(self, entity_type: EntityType) -> list[EntityType]:
        """获取某类型的所有子类 (递归).

        Returns:
            所有子类列表 (含间接子类)
        """
        result: list[EntityType] = []
        direct = self.get_subclasses(entity_type)
        for sub in direct:
            result.append(sub)
            result.extend(self.get_all_subclasses(sub))
        return result

    def is_subclass_of(
        self, child: EntityType, parent: EntityType
    ) -> bool:
        """检查 child 是否是 parent 的子类 (含间接继承).

        Returns:
            True 如果 child 是 parent 或其子类
        """
        hierarchy = self.get_class_hierarchy(child)
        return parent in hierarchy

    # ---- 增强方法: 本体推理 ----

    def infer_transitive_closure(
        self, triples: list[tuple[str, str, str]], relation: str
    ) -> list[tuple[str, str, str]]:
        """推理传递闭包 (借鉴 OWL transitive property + HermiT 推理).

        对指定关系应用传递性规则: A→B, B→C ⟹ A→C

        Args:
            triples: 已知三元组列表 [(subject, predicate, object)]
            relation: 要推理的传递关系名称

        Returns:
            推理出的新三元组列表 (不含原始三元组)
        """
        # 检查关系是否为传递关系
        rel_def = self.get_relation(relation)
        if rel_def is None or not rel_def.transitive:
            return []

        # 构建 A→B 邻接表
        adjacency: dict[str, set[str]] = {}
        for s, p, o in triples:
            if p == relation:
                if s not in adjacency:
                    adjacency[s] = set()
                adjacency[s].add(o)

        # 计算传递闭包 (Floyd-Warshall 变体)
        inferred: set[tuple[str, str]] = set()
        changed = True
        while changed:
            changed = False
            for s, targets in list(adjacency.items()):
                for t in list(targets):
                    if t in adjacency:
                        for tt in adjacency[t]:
                            if tt not in targets and (s, tt) not in inferred:
                                targets.add(tt)
                                inferred.add((s, tt))
                                changed = True

        return [(s, relation, o) for s, o in inferred]

    def infer_inverse_relations(
        self, triples: list[tuple[str, str, str]], relation: str
    ) -> list[tuple[str, str, str]]:
        """推理逆关系 (借鉴 OWL owl:inverseOf + Pellet 推理).

        对指定关系应用逆关系规则: A→B (R), inverse(R)=R' ⟹ B→A (R')

        Args:
            triples: 已知三元组列表 [(subject, predicate, object)]
            relation: 要推理的关系名称

        Returns:
            推理出的新三元组列表
        """
        rel_def = self.get_relation(relation)
        if rel_def is None or not rel_def.inverse_of:
            return []

        inverse_rel = rel_def.inverse_of
        inferred: list[tuple[str, str, str]] = []
        for s, p, o in triples:
            if p == relation:
                inferred.append((o, inverse_rel, s))
        return inferred

    def infer_symmetric_closure(
        self, triples: list[tuple[str, str, str]], relation: str
    ) -> list[tuple[str, str, str]]:
        """推理对称闭包 (借鉴 OWL symmetric property).

        对指定关系应用对称性规则: A→B (R) ⟹ B→A (R)

        Args:
            triples: 已知三元组列表
            relation: 要推理的关系名称

        Returns:
            推理出的新三元组列表
        """
        rel_def = self.get_relation(relation)
        if rel_def is None or not rel_def.symmetric:
            return []

        existing = {(s, o) for s, p, o in triples if p == relation}
        inferred: list[tuple[str, str, str]] = []
        for s, p, o in triples:
            if p == relation and (o, s) not in existing:
                inferred.append((o, relation, s))
        return inferred

    def infer_property_chain(
        self,
        triples: list[tuple[str, str, str]],
        chain: list[str],
    ) -> list[tuple[str, str, str]]:
        """推理属性链 (借鉴 OWL property chain axiom + Pellet 推理).

        对属性链 [R1, R2, ..., Rn] 应用推理:
        A→B (R1), B→C (R2), ..., → 新关系 A→C (合成关系)

        Args:
            triples: 已知三元组列表
            chain: 属性链 [R1, R2, ...] (至少 2 个元素)

        Returns:
            推理出的新三元组列表
        """
        if len(chain) < 2:
            return []

        # 逐跳遍历: 从第一个关系开始，逐步扩展
        current_pairs: set[tuple[str, str]] = set()
        for s, p, o in triples:
            if p == chain[0]:
                current_pairs.add((s, o))

        for rel in chain[1:]:
            next_pairs: set[tuple[str, str]] = set()
            # 构建当前关系的邻接表
            adjacency: dict[str, set[str]] = {}
            for s, p, o in triples:
                if p == rel:
                    if s not in adjacency:
                        adjacency[s] = set()
                    adjacency[s].add(o)

            for s, mid in current_pairs:
                if mid in adjacency:
                    for target in adjacency[mid]:
                        next_pairs.add((s, target))
            current_pairs = next_pairs

        # 合成关系名称: chain[0] + "_" + chain[-1] (或自定义)
        chain_name = "_".join(chain)
        existing = {(s, p, o) for s, p, o in triples}
        inferred: list[tuple[str, str, str]] = []
        for s, o in current_pairs:
            triple = (s, chain_name, o)
            if triple not in existing:
                inferred.append(triple)

        return inferred

    def infer_subclass_inheritance(
        self,
        instances: list[tuple[str, EntityType]],
    ) -> list[tuple[str, EntityType]]:
        """推理子类继承 (借鉴 OWL reasoning: rdfs:subClassOf).

        如果 x 是 B 的实例，且 B subClassOf A，则 x 也是 A 的实例。

        Args:
            instances: 实例列表 [(entity_id, entity_type)]

        Returns:
            推理出的新实例关系列表 [(entity_id, inherited_type)]
        """
        inferred: list[tuple[str, EntityType]] = []
        existing = set(instances)

        for entity_id, entity_type in instances:
            hierarchy = self.get_class_hierarchy(entity_type)
            for parent_type in hierarchy:
                if parent_type != entity_type:
                    new_pair = (entity_id, parent_type)
                    if new_pair not in existing:
                        inferred.append(new_pair)
                        existing.add(new_pair)

        return inferred

    def apply_inference_rules(
        self, triples: list[tuple[str, str, str]]
    ) -> list[tuple[str, str, str]]:
        """应用所有启用的推理规则 (借鉴 HermiT 推理引擎批量推理).

        依次应用所有启用的推理规则，返回所有新推理出的三元组。
        推理顺序: 对称闭包 → 逆关系 → 传递闭包 → 属性链

        Args:
            triples: 已知三元组列表

        Returns:
            所有新推理出的三元组列表
        """
        all_inferred: list[tuple[str, str, str]] = []
        existing = set(triples)
        working_set = list(triples)

        for rule in self.inference_rules:
            if not rule.enabled:
                continue
            if not rule.applies_to_relation and rule.rule_type != InferenceRuleType.PROPERTY_CHAIN:
                continue

            if rule.rule_type == InferenceRuleType.TRANSITIVE_CLOSURE:
                new = self.infer_transitive_closure(working_set, rule.applies_to_relation)
            elif rule.rule_type == InferenceRuleType.INVERSE_RELATION:
                new = self.infer_inverse_relations(working_set, rule.applies_to_relation)
            elif rule.rule_type == InferenceRuleType.SYMMETRIC_CLOSURE:
                new = self.infer_symmetric_closure(working_set, rule.applies_to_relation)
            elif rule.rule_type == InferenceRuleType.PROPERTY_CHAIN and rule.property_chain:
                new = self.infer_property_chain(working_set, rule.property_chain)
            else:
                continue

            for t in new:
                if t not in existing:
                    all_inferred.append(t)
                    existing.add(t)
                    working_set.append(t)

        return all_inferred

    def get_enabled_rules(self) -> list[OntologyRule]:
        """获取所有启用的推理规则."""
        return [r for r in self.inference_rules if r.enabled]

    def axiom_count(self) -> int:
        """公理数量."""
        return len(self.axioms)

    def mapping_count(self) -> int:
        """映射数量."""
        return len(self.mappings)

    # ---- 增强方法: 公理验证 ----

    def validate_disjoint(
        self, entity_types: list[EntityType]
    ) -> list[str]:
        """验证不相交公理 (借鉴 OWL owl:disjointWith).

        检查实体类型列表中是否存在被声明为不相交的类型对。

        Args:
            entity_types: 实体类型列表

        Returns:
            违规信息列表 (空列表表示通过)
        """
        violations: list[str] = []
        disjoint_pairs: list[tuple[str, str]] = []

        for ax in self.axioms:
            if ax.axiom_type == "disjoint":
                disjoint_pairs.append((ax.subject, ax.object))

        for i, t1 in enumerate(entity_types):
            for t2 in entity_types[i + 1:]:
                t1_str = t1.value if hasattr(t1, 'value') else str(t1)
                t2_str = t2.value if hasattr(t2, 'value') else str(t2)
                for pair in disjoint_pairs:
                    if (pair[0] == t1_str and pair[1] == t2_str) or \
                       (pair[0] == t2_str and pair[1] == t1_str):
                        violations.append(
                            f"类型 {t1_str} 和 {t2_str} 被声明为不相交，但实体同时具有这两种类型"
                        )

        return violations

    def validate_axioms(
        self,
        entity_type: EntityType,
        properties: dict[str, Any],
    ) -> list[str]:
        """验证所有适用的公理约束.

        Args:
            entity_type: 实体类型
            properties: 实体属性

        Returns:
            违规信息列表 (空列表表示通过)
        """
        violations: list[str] = []
        type_str = entity_type.value if hasattr(entity_type, 'value') else str(entity_type)

        for ax in self.axioms:
            if ax.axiom_type == "functional" and ax.subject in properties:
                # 函数性公理: 属性值必须是单值
                val = properties.get(ax.subject)
                if isinstance(val, (list, set)) and len(val) > 1:
                    violations.append(
                        f"属性 '{ax.subject}' 是函数性属性，但提供了 {len(val)} 个值"
                    )
            elif ax.axiom_type == "equivalent" and ax.subject == type_str:
                # 等价类公理: 需要同时满足等价类的约束
                eq_type_str = ax.object
                for et in EntityType:
                    if et.value == eq_type_str:
                        eq_violations = self.validate_properties(et, properties)
                        violations.extend(eq_violations)
                        break

        return violations

    def validate_all(
        self, entity_type: EntityType, properties: dict[str, Any]
    ) -> list[str]:
        """全量验证 (属性存在性 + 枚举 + 数据类型 + 基数 + 公理).

        Args:
            entity_type: 实体类型
            properties: 实体属性

        Returns:
            所有违规信息列表 (空列表表示通过)
        """
        violations: list[str] = []
        violations.extend(self.validate_full(entity_type, properties))
        violations.extend(self.validate_axioms(entity_type, properties))
        return violations

    # ---- 增强方法: 跨本体映射 ----

    def find_mappings(
        self,
        entity_type: EntityType | None = None,
        target_domain: str | None = None,
    ) -> list[OntologyMapping]:
        """查找符合条件的跨本体映射.

        Args:
            entity_type: 源实体类型 (None=所有)
            target_domain: 目标领域 (None=所有)

        Returns:
            匹配的映射列表
        """
        result: list[OntologyMapping] = []
        for m in self.mappings:
            if entity_type is not None and m.source_entity_type != entity_type:
                continue
            if target_domain is not None and m.target_domain != target_domain:
                continue
            result.append(m)
        return result

    def get_equivalent_type(
        self, entity_type: EntityType, target_domain: str
    ) -> EntityType | None:
        """获取在目标领域中的等价实体类型.

        Args:
            entity_type: 源实体类型
            target_domain: 目标领域

        Returns:
            等价的实体类型，或 None
        """
        for m in self.mappings:
            if (
                m.source_entity_type == entity_type
                and m.target_domain == target_domain
                and m.mapping_type == "equivalent"
            ):
                return m.target_entity_type
        return None


# ============================================================
# 预构建本体: 通用本体
# ============================================================


def _build_general_ontology() -> DomainOntology:
    """构建通用本体 (所有领域共享的基础类型)."""
    return DomainOntology(
        ontology_id="onto-general",
        domain=DomainType.GENERAL,
        display_name="通用本体",
        description="所有领域共享的基础本体，定义通用实体类型和关系",
        version="1.0.0",
        classes=[
            OntologyClass(
                class_id="cls-concept",
                entity_type=EntityType.CONCEPT,
                display_name="概念",
                description="抽象概念或知识点",
                properties=[
                    OntologyProperty(
                        name="definition",
                        display_name="定义",
                        description="概念的定义",
                        required=True,
                    ),
                    OntologyProperty(
                        name="category",
                        display_name="分类",
                        description="概念所属分类",
                    ),
                ],
                allowed_relations=[
                    RelationType.RELATED_TO,
                    RelationType.PART_OF,
                    RelationType.EQUIVALENT_TO,
                    RelationType.DEPENDS_ON,
                ],
            ),
            OntologyClass(
                class_id="cls-person",
                entity_type=EntityType.PERSON,
                display_name="人物",
                description="人物实体 (作者、研究者等)",
                properties=[
                    OntologyProperty(name="full_name", display_name="姓名", required=True),
                    OntologyProperty(name="affiliation", display_name="所属机构"),
                    OntologyProperty(name="email", display_name="邮箱"),
                    OntologyProperty(name="orcid", display_name="ORCID"),
                ],
                allowed_relations=[
                    RelationType.AUTHORED_BY,
                    RelationType.RELATED_TO,
                ],
            ),
            OntologyClass(
                class_id="cls-organization",
                entity_type=EntityType.ORGANIZATION,
                display_name="组织",
                description="组织机构 (大学、研究机构、出版社等)",
                properties=[
                    OntologyProperty(name="full_name", display_name="全称", required=True),
                    OntologyProperty(name="short_name", display_name="简称"),
                    OntologyProperty(name="country", display_name="国家"),
                    OntologyProperty(name="website", display_name="网站"),
                ],
                allowed_relations=[
                    RelationType.RELATED_TO,
                    RelationType.PART_OF,
                ],
            ),
            OntologyClass(
                class_id="cls-document-chunk",
                entity_type=EntityType.DOCUMENT_CHUNK,
                display_name="文档切片",
                description="文档切片实体，对应一个知识片段",
                properties=[
                    OntologyProperty(name="content", display_name="内容", required=True),
                    OntologyProperty(name="source", display_name="来源"),
                    OntologyProperty(name="section", display_name="章节"),
                    OntologyProperty(name="page", display_name="页码", range="integer"),
                ],
                allowed_relations=[
                    RelationType.DERIVED_FROM,
                    RelationType.REFERENCES,
                    RelationType.RELATED_TO,
                ],
            ),
        ],
        relations=[
            OntologyRelation(
                name=RelationType.RELATED_TO.value,
                display_name="相关",
                description="通用关联关系",
                symmetric=True,
            ),
            OntologyRelation(
                name=RelationType.PART_OF.value,
                display_name="属于",
                description="部分-整体关系",
                transitive=True,
            ),
            OntologyRelation(
                name=RelationType.EQUIVALENT_TO.value,
                display_name="等价",
                description="等价关系",
                symmetric=True,
                transitive=True,
            ),
            OntologyRelation(
                name=RelationType.DEPENDS_ON.value,
                display_name="依赖",
                description="依赖关系",
                transitive=True,
            ),
            OntologyRelation(
                name=RelationType.AUTHORED_BY.value,
                display_name="作者",
                description="创作关系",
                domain=[EntityType.PAPER, EntityType.TEXTBOOK],
                range=[EntityType.PERSON],
            ),
            OntologyRelation(
                name=RelationType.REFERENCES.value,
                display_name="引用",
                description="引用关系",
                domain=[EntityType.DOCUMENT_CHUNK, EntityType.PAPER],
            ),
            OntologyRelation(
                name=RelationType.DERIVED_FROM.value,
                display_name="派生自",
                description="派生关系",
                transitive=True,
            ),
        ],
        global_properties=[
            OntologyProperty(name="name", display_name="名称", required=True),
            OntologyProperty(name="description", display_name="描述"),
            OntologyProperty(name="created_at", display_name="创建时间", range="number"),
            OntologyProperty(name="updated_at", display_name="更新时间", range="number"),
            OntologyProperty(name="tags", display_name="标签", cardinality=0),
        ],
    )


# ============================================================
# 预构建本体: 化学本体
# ============================================================


def _build_chemistry_ontology() -> DomainOntology:
    """构建化学领域本体 (借鉴 ChemOnt 分类体系)."""
    return DomainOntology(
        ontology_id="onto-chemistry",
        domain=DomainType.CHEMISTRY,
        display_name="化学本体",
        description="化学领域本体，定义化合物、光谱、反应等实体类型和关系",
        version="1.0.0",
        classes=[
            OntologyClass(
                class_id="cls-chemical-compound",
                entity_type=EntityType.CHEMICAL_COMPOUND,
                display_name="化学化合物",
                description="化学化合物实体",
                parent_type=EntityType.CONCEPT,
                properties=[
                    OntologyProperty(name="formula", display_name="分子式", required=True),
                    OntologyProperty(name="cas_number", display_name="CAS 号"),
                    OntologyProperty(name="molecular_weight", display_name="分子量", range="number"),
                    OntologyProperty(name="smiles", display_name="SMILES"),
                    OntologyProperty(name="inchi", display_name="InChI"),
                    OntologyProperty(
                        name="state",
                        display_name="状态",
                        enum_values=["solid", "liquid", "gas", "unknown"],
                    ),
                    OntologyProperty(name="melting_point", display_name="熔点", range="number"),
                    OntologyProperty(name="boiling_point", display_name="沸点", range="number"),
                    OntologyProperty(name="density", display_name="密度", range="number"),
                    OntologyProperty(name="solubility", display_name="溶解度"),
                ],
                allowed_relations=[
                    RelationType.HAS_PROPERTY,
                    RelationType.RELATED_TO,
                    RelationType.EQUIVALENT_TO,
                    RelationType.PART_OF,
                ],
            ),
            OntologyClass(
                class_id="cls-method",
                entity_type=EntityType.METHOD,
                display_name="方法",
                description="化学分析方法 (光谱、色谱、量热等)",
                parent_type=EntityType.CONCEPT,
                properties=[
                    OntologyProperty(name="method_type", display_name="方法类型", required=True),
                    OntologyProperty(name="instrument", display_name="仪器"),
                    OntologyProperty(name="precision", display_name="精度", range="number"),
                ],
                allowed_relations=[
                    RelationType.RELATED_TO,
                    RelationType.DEPENDS_ON,
                ],
            ),
            OntologyClass(
                class_id="cls-experiment",
                entity_type=EntityType.EXPERIMENT,
                display_name="实验",
                description="化学实验记录",
                properties=[
                    OntologyProperty(name="procedure", display_name="实验步骤", required=True),
                    OntologyProperty(name="conditions", display_name="实验条件"),
                    OntologyProperty(name="result", display_name="实验结果"),
                    OntologyProperty(name="reproducibility", display_name="可重复性", range="number"),
                ],
                allowed_relations=[
                    RelationType.DERIVED_FROM,
                    RelationType.RELATED_TO,
                    RelationType.DEPENDS_ON,
                ],
            ),
        ],
        relations=[
            OntologyRelation(
                name=RelationType.HAS_PROPERTY.value,
                display_name="具有属性",
                description="化合物具有某种物理化学属性",
                domain=[EntityType.CHEMICAL_COMPOUND],
            ),
            OntologyRelation(
                name=RelationType.EQUIVALENT_TO.value,
                display_name="等价",
                description="化学等价 (同一物质不同标识)",
                domain=[EntityType.CHEMICAL_COMPOUND],
                range=[EntityType.CHEMICAL_COMPOUND],
                symmetric=True,
                transitive=True,
            ),
            OntologyRelation(
                name=RelationType.CONTRADICTS.value,
                display_name="矛盾",
                description="实验结果相互矛盾",
                domain=[EntityType.EXPERIMENT],
                range=[EntityType.EXPERIMENT],
                symmetric=True,
            ),
            OntologyRelation(
                name=RelationType.SUPPORTS.value,
                display_name="支持",
                description="实验结果相互支持",
                domain=[EntityType.EXPERIMENT],
                range=[EntityType.EXPERIMENT],
            ),
        ],
        global_properties=[
            OntologyProperty(name="name", display_name="名称", required=True),
            OntologyProperty(name="description", display_name="描述"),
            OntologyProperty(name="safety_level", display_name="安全等级",
                            enum_values=["low", "medium", "high", "extreme"]),
        ],
    )


# ============================================================
# 预构建本体: 材料本体
# ============================================================


def _build_materials_ontology() -> DomainOntology:
    """构建材料科学本体 (借鉴 Materials Ontology 四子本体模式)."""
    return DomainOntology(
        ontology_id="onto-materials",
        domain=DomainType.MATERIALS,
        display_name="材料科学本体",
        description="材料科学领域本体，覆盖物质/过程/环境/性质四维度",
        version="1.0.0",
        classes=[
            OntologyClass(
                class_id="cls-material",
                entity_type=EntityType.MATERIAL,
                display_name="材料",
                description="材料实体 (金属、陶瓷、聚合物、复合材料等)",
                parent_type=EntityType.CONCEPT,
                properties=[
                    OntologyProperty(name="composition", display_name="组成", required=True),
                    OntologyProperty(name="crystal_structure", display_name="晶体结构"),
                    OntologyProperty(
                        name="material_class",
                        display_name="材料类别",
                        enum_values=["metal", "ceramic", "polymer", "composite", "semiconductor", "other"],
                        required=True,
                    ),
                    OntologyProperty(name="band_gap", display_name="带隙", range="number"),
                    OntologyProperty(name="conductivity", display_name="电导率", range="number"),
                    OntologyProperty(name="thermal_conductivity", display_name="热导率", range="number"),
                    OntologyProperty(name="youngs_modulus", display_name="杨氏模量", range="number"),
                    OntologyProperty(name="yield_strength", display_name="屈服强度", range="number"),
                ],
                allowed_relations=[
                    RelationType.HAS_PROPERTY,
                    RelationType.RELATED_TO,
                    RelationType.PART_OF,
                    RelationType.DERIVED_FROM,
                ],
            ),
            OntologyClass(
                class_id="cls-dataset",
                entity_type=EntityType.DATASET,
                display_name="数据集",
                description="材料计算/实验数据集 (DFT/MD/实验数据)",
                properties=[
                    OntologyProperty(name="data_type", display_name="数据类型", required=True),
                    OntologyProperty(name="format", display_name="数据格式"),
                    OntologyProperty(name="size", display_name="数据量", range="number"),
                    OntologyProperty(name="license", display_name="许可证"),
                ],
                allowed_relations=[
                    RelationType.DERIVED_FROM,
                    RelationType.RELATED_TO,
                    RelationType.REFERENCES,
                ],
            ),
            OntologyClass(
                class_id="cls-experiment-mat",
                entity_type=EntityType.EXPERIMENT,
                display_name="材料实验",
                description="材料实验/计算记录",
                properties=[
                    OntologyProperty(name="method", display_name="方法", required=True),
                    OntologyProperty(name="temperature", display_name="温度", range="number"),
                    OntologyProperty(name="pressure", display_name="压力", range="number"),
                    OntologyProperty(name="software", display_name="计算软件"),
                    OntologyProperty(name="convergence", display_name="收敛标准", range="number"),
                ],
                allowed_relations=[
                    RelationType.DERIVED_FROM,
                    RelationType.RELATED_TO,
                    RelationType.SUPPORTS,
                    RelationType.CONTRADICTS,
                ],
            ),
        ],
        relations=[
            OntologyRelation(
                name=RelationType.HAS_PROPERTY.value,
                display_name="具有性质",
                description="材料具有某种物理性质",
                domain=[EntityType.MATERIAL],
            ),
            OntologyRelation(
                name=RelationType.DERIVED_FROM.value,
                display_name="派生自",
                description="数据/结果派生自某材料或实验",
                transitive=True,
            ),
            OntologyRelation(
                name=RelationType.SUPERSEDES.value,
                display_name="替代",
                description="新版本替代旧版本",
                domain=[EntityType.DATASET, EntityType.METHOD],
            ),
            OntologyRelation(
                name=RelationType.INSTANTIATES.value,
                display_name="实例化",
                description="实验实例化某种方法",
                domain=[EntityType.EXPERIMENT],
                range=[EntityType.METHOD],
            ),
        ],
        global_properties=[
            OntologyProperty(name="name", display_name="名称", required=True),
            OntologyProperty(name="description", display_name="描述"),
            OntologyProperty(name="reference", display_name="参考文献"),
            OntologyProperty(name="computed", display_name="是否计算值", range="boolean"),
        ],
    )


# ============================================================
# 预构建本体: 教育本体
# ============================================================


def _build_education_ontology() -> DomainOntology:
    """构建教育领域本体 (借鉴 Dublin Core + FOAF)."""
    return DomainOntology(
        ontology_id="onto-education",
        domain=DomainType.EDUCATION,
        display_name="教育本体",
        description="教育领域本体，定义教材、课程、知识点等实体类型和关系",
        version="1.0.0",
        classes=[
            OntologyClass(
                class_id="cls-textbook",
                entity_type=EntityType.TEXTBOOK,
                display_name="教材",
                description="教材/教科书实体",
                parent_type=EntityType.CONCEPT,
                properties=[
                    OntologyProperty(name="isbn", display_name="ISBN"),
                    OntologyProperty(name="author", display_name="作者", required=True),
                    OntologyProperty(name="publisher", display_name="出版社"),
                    OntologyProperty(name="edition", display_name="版次", range="integer"),
                    OntologyProperty(name="year", display_name="出版年份", range="integer"),
                    OntologyProperty(name="subject", display_name="学科"),
                    OntologyProperty(name="level", display_name="难度等级",
                                    enum_values=["introductory", "intermediate", "advanced"]),
                ],
                allowed_relations=[
                    RelationType.AUTHORED_BY,
                    RelationType.PUBLISHED_IN,
                    RelationType.PART_OF,
                    RelationType.RELATED_TO,
                    RelationType.REFERENCES,
                ],
            ),
            OntologyClass(
                class_id="cls-paper",
                entity_type=EntityType.PAPER,
                display_name="论文",
                description="学术论文实体",
                parent_type=EntityType.CONCEPT,
                properties=[
                    OntologyProperty(name="doi", display_name="DOI"),
                    OntologyProperty(name="title", display_name="标题", required=True),
                    OntologyProperty(name="abstract", display_name="摘要"),
                    OntologyProperty(name="keywords", display_name="关键词", cardinality=0),
                    OntologyProperty(name="venue", display_name="发表期刊/会议"),
                    OntologyProperty(name="year", display_name="发表年份", range="integer"),
                    OntologyProperty(name="citation_count", display_name="引用数", range="integer"),
                ],
                allowed_relations=[
                    RelationType.AUTHORED_BY,
                    RelationType.PUBLISHED_IN,
                    RelationType.CITES,
                    RelationType.REFERENCES,
                    RelationType.RELATED_TO,
                    RelationType.SUPPORTS,
                    RelationType.CONTRADICTS,
                ],
            ),
            OntologyClass(
                class_id="cls-course",
                entity_type=EntityType.COURSE,
                display_name="课程",
                description="课程实体",
                properties=[
                    OntologyProperty(name="course_code", display_name="课程代码"),
                    OntologyProperty(name="credits", display_name="学分", range="number"),
                    OntologyProperty(name="prerequisites", display_name="先修课程", cardinality=0),
                    OntologyProperty(name="level", display_name="难度等级",
                                    enum_values=["undergraduate", "graduate", "phd"]),
                ],
                allowed_relations=[
                    RelationType.PART_OF,
                    RelationType.DEPENDS_ON,
                    RelationType.RELATED_TO,
                ],
            ),
        ],
        relations=[
            OntologyRelation(
                name=RelationType.CITES.value,
                display_name="引用",
                description="论文引用关系",
                domain=[EntityType.PAPER],
                range=[EntityType.PAPER],
                transitive=False,
            ),
            OntologyRelation(
                name=RelationType.AUTHORED_BY.value,
                display_name="作者",
                description="创作关系",
                domain=[EntityType.PAPER, EntityType.TEXTBOOK],
                range=[EntityType.PERSON],
                functional=False,
            ),
            OntologyRelation(
                name=RelationType.PUBLISHED_IN.value,
                display_name="发表于",
                description="发表关系",
                domain=[EntityType.PAPER, EntityType.TEXTBOOK],
                range=[EntityType.ORGANIZATION],
            ),
            OntologyRelation(
                name=RelationType.CITES.value,
                display_name="引用",
                description="论文间的引用关系",
                domain=[EntityType.PAPER],
                range=[EntityType.PAPER],
            ),
        ],
        global_properties=[
            OntologyProperty(name="name", display_name="名称", required=True),
            OntologyProperty(name="description", display_name="描述"),
            OntologyProperty(name="language", display_name="语言", default_value="zh"),
            OntologyProperty(name="license", display_name="许可证"),
        ],
    )


# ============================================================
# 本体注册中心
# ============================================================


class OntologyRegistry:
    """本体注册中心.

    管理所有领域本体的注册、查询和验证。
    支持运行时注册自定义本体。

    Usage::

        registry = OntologyRegistry()
        chemistry = registry.get_ontology("chemistry")
        ok = registry.validate_entity("chemistry", EntityType.CHEMICAL_COMPOUND, {...})
    """

    def __init__(self) -> None:
        self._ontologies: dict[str, DomainOntology] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        """注册预构建本体."""
        self.register(_build_general_ontology())
        self.register(_build_chemistry_ontology())
        self.register(_build_materials_ontology())
        self.register(_build_education_ontology())

    def register(self, ontology: DomainOntology) -> None:
        """注册本体."""
        self._ontologies[ontology.domain] = ontology

    def get_ontology(self, domain: str) -> DomainOntology | None:
        """获取指定领域的本体."""
        return self._ontologies.get(domain)

    def list_domains(self) -> list[str]:
        """列出所有已注册的领域."""
        return list(self._ontologies.keys())

    def validate_entity_type(self, domain: str, entity_type: EntityType) -> bool:
        """验证实体类型在指定领域本体中是否定义."""
        onto = self.get_ontology(domain)
        if onto is None:
            return False
        return onto.validate_entity_type(entity_type)

    def validate_properties(
        self, domain: str, entity_type: EntityType, properties: dict[str, Any]
    ) -> list[str]:
        """验证实体的属性是否符合本体约束."""
        onto = self.get_ontology(domain)
        if onto is None:
            return [f"领域本体未找到: {domain}"]
        return onto.validate_properties(entity_type, properties)

    def validate_relation(
        self,
        domain: str,
        relation: str,
        subject_type: EntityType,
        object_type: EntityType,
    ) -> bool:
        """验证关系是否合法."""
        onto = self.get_ontology(domain)
        if onto is None:
            return False
        return onto.validate_relation(relation, subject_type, object_type)

    def get_class(self, domain: str, entity_type: EntityType) -> OntologyClass | None:
        """获取指定领域的类定义."""
        onto = self.get_ontology(domain)
        if onto is None:
            return None
        return onto.get_class(entity_type)

    # ---- 增强方法: SHACL 验证封装 ----

    def validate_data_types(
        self, domain: str, entity_type: EntityType, properties: dict[str, Any]
    ) -> list[str]:
        """验证属性值的数据类型."""
        onto = self.get_ontology(domain)
        if onto is None:
            return [f"领域本体未找到: {domain}"]
        return onto.validate_data_types(entity_type, properties)

    def validate_cardinality(
        self, domain: str, entity_type: EntityType, properties: dict[str, Any]
    ) -> list[str]:
        """验证属性基数约束."""
        onto = self.get_ontology(domain)
        if onto is None:
            return [f"领域本体未找到: {domain}"]
        return onto.validate_cardinality(entity_type, properties)

    def validate_full(
        self, domain: str, entity_type: EntityType, properties: dict[str, Any]
    ) -> list[str]:
        """完整验证 (属性存在性 + 枚举 + 数据类型 + 基数)."""
        onto = self.get_ontology(domain)
        if onto is None:
            return [f"领域本体未找到: {domain}"]
        return onto.validate_full(entity_type, properties)

    def validate_all(
        self, domain: str, entity_type: EntityType, properties: dict[str, Any]
    ) -> list[str]:
        """全量验证 (含公理约束)."""
        onto = self.get_ontology(domain)
        if onto is None:
            return [f"领域本体未找到: {domain}"]
        return onto.validate_all(entity_type, properties)

    # ---- 增强方法: 类层级封装 ----

    def get_class_hierarchy(
        self, domain: str, entity_type: EntityType
    ) -> list[EntityType]:
        """获取实体类型的完整类层级."""
        onto = self.get_ontology(domain)
        if onto is None:
            return []
        return onto.get_class_hierarchy(entity_type)

    def get_subclasses(
        self, domain: str, entity_type: EntityType
    ) -> list[EntityType]:
        """获取某类型的所有直接子类."""
        onto = self.get_ontology(domain)
        if onto is None:
            return []
        return onto.get_subclasses(entity_type)

    def get_all_subclasses(
        self, domain: str, entity_type: EntityType
    ) -> list[EntityType]:
        """获取某类型的所有子类 (递归)."""
        onto = self.get_ontology(domain)
        if onto is None:
            return []
        return onto.get_all_subclasses(entity_type)

    def is_subclass_of(
        self, domain: str, child: EntityType, parent: EntityType
    ) -> bool:
        """检查 child 是否是 parent 的子类."""
        onto = self.get_ontology(domain)
        if onto is None:
            return False
        return onto.is_subclass_of(child, parent)

    # ---- 增强方法: 推理封装 ----

    def apply_inference_rules(
        self, domain: str, triples: list[tuple[str, str, str]]
    ) -> list[tuple[str, str, str]]:
        """应用指定领域的推理规则."""
        onto = self.get_ontology(domain)
        if onto is None:
            return []
        return onto.apply_inference_rules(triples)

    def infer_transitive_closure(
        self, domain: str, triples: list[tuple[str, str, str]], relation: str
    ) -> list[tuple[str, str, str]]:
        """推理传递闭包."""
        onto = self.get_ontology(domain)
        if onto is None:
            return []
        return onto.infer_transitive_closure(triples, relation)

    def infer_inverse_relations(
        self, domain: str, triples: list[tuple[str, str, str]], relation: str
    ) -> list[tuple[str, str, str]]:
        """推理逆关系."""
        onto = self.get_ontology(domain)
        if onto is None:
            return []
        return onto.infer_inverse_relations(triples, relation)

    def infer_symmetric_closure(
        self, domain: str, triples: list[tuple[str, str, str]], relation: str
    ) -> list[tuple[str, str, str]]:
        """推理对称闭包."""
        onto = self.get_ontology(domain)
        if onto is None:
            return []
        return onto.infer_symmetric_closure(triples, relation)

    def infer_property_chain(
        self, domain: str, triples: list[tuple[str, str, str]], chain: list[str]
    ) -> list[tuple[str, str, str]]:
        """推理属性链."""
        onto = self.get_ontology(domain)
        if onto is None:
            return []
        return onto.infer_property_chain(triples, chain)

    def infer_subclass_inheritance(
        self, domain: str, instances: list[tuple[str, EntityType]]
    ) -> list[tuple[str, EntityType]]:
        """推理子类继承."""
        onto = self.get_ontology(domain)
        if onto is None:
            return []
        return onto.infer_subclass_inheritance(instances)

    # ---- 增强方法: 跨本体映射封装 ----

    def find_mappings(
        self,
        domain: str,
        entity_type: EntityType | None = None,
        target_domain: str | None = None,
    ) -> list[OntologyMapping]:
        """查找跨本体映射."""
        onto = self.get_ontology(domain)
        if onto is None:
            return []
        return onto.find_mappings(entity_type, target_domain)

    def get_equivalent_type(
        self, domain: str, entity_type: EntityType, target_domain: str
    ) -> EntityType | None:
        """获取在目标领域中的等价实体类型."""
        onto = self.get_ontology(domain)
        if onto is None:
            return None
        return onto.get_equivalent_type(entity_type, target_domain)

    # ---- 统计 ----

    def total_classes(self) -> int:
        """所有领域的类总数."""
        return sum(o.class_count() for o in self._ontologies.values())

    def total_relations(self) -> int:
        """所有领域的关系总数."""
        return sum(o.relation_count() for o in self._ontologies.values())

    def total_axioms(self) -> int:
        """所有领域的公理总数."""
        return sum(o.axiom_count() for o in self._ontologies.values())

    def total_mappings(self) -> int:
        """所有领域的映射总数."""
        return sum(o.mapping_count() for o in self._ontologies.values())


__all__ = [
    "OntologyProperty",
    "OntologyRelation",
    "OntologyClass",
    "OntologyAxiom",
    "OntologyRule",
    "OntologyMapping",
    "DomainType",
    "DomainOntology",
    "OntologyRegistry",
]
