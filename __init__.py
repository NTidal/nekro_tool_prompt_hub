"""工具调用提示中枢（Tool Prompt Hub）——把「何时用工具、怎么用工具」的提示词从人格预设里剥离，统一管理。

要解决的问题
--------------------
实践发现：NA 里 LLM 是否主动调用插件工具，很大程度上取决于**人格预设（preset）里写没写
"遇到 XX 就调用 XX 工具"这类指引**。于是每加/改一个工具都得去改人格文案，换个人格又得抄一遍，
人格文件越来越臃肿，且「角色塑造」和「工具用法」两种关注点耦合在一起。

原理（NA 的 prompt 组装链路，读源码确认）
--------------------
每个 agent 回合，system prompt = policy_kernel + persona(preset.content) + runtime_contract
+ 各插件能力块（由**沙盒方法的函数名 + docstring** 渲染，`render_sandbox_methods_prompt`）；
此外，启用了 ``mount_prompt_inject_method`` 的常驻/激活插件，其返回文本会被框架包成
``<plugin_runtime_context module_name="...">...`` 注入**历史消息头部**（`_render_plugin_runtime_prompt`），
且 runtime_contract 里已向模型声明这类内容是「权威的系统级上下文信息」。

因此本插件不碰人格、不改其他插件，而是通过 ``mount_prompt_inject_method`` 在**每个回合**
注入一块「工具调用规范」：

- **全局准则**（GLOBAL_RULES）：跨工具的通用行动规范，如「判断权在你，不用等口令」
  「工具返回的是事实，用人格口吻转述，别念技术数据」；
- **场景规则**（RULES）：每条规则对应一类工具/场景，可配关键词——只在最近 N 条消息
  命中关键词时才注入，控制 token 开销；关键词留空则每回合注入。

人格回归纯粹的角色塑造；工具用法的迭代全部收敛到本插件的配置页（WebUI 插件详情里改，
保存即生效，**下一回合**就按新规范注入，无需重启）。

设计要点
--------------------
- ``allow_sleep=False``：prompt 注入只对 always_awake/active 状态的插件渲染，必须常驻；
- 注入函数**全量兜底异常**：`render_inject_prompt` 不捕获异常，若抛错会让整个 agent 回合
  组装上下文失败，所以任何意外都降级为「本回合不注入」；
- 注入总量受 ``MAX_INJECT_CHARS`` 保护，超限按「全局准则 → 规则顺序」截断并记日志；
- 只注入"事实与规范"，不注入元指令式的角色扮演要求——表达永远留给人格。

已知框架行为（已在代码中兼容）
--------------------
NA WebUI 保存插件配置走 ``POST /api/config/batch``，该链路把嵌套字段写成
**裸 dict** 后直接更新内存配置，**不经过 Pydantic 校验**。因此 ``config.RULES``
的元素可能在保存后从 ``ToolPromptRule`` 退化为 ``dict``。本插件统一用
``_rule_field()`` 取值以兼容两种形态；若你改了配置却"注入没生效"，先看日志
有没有 ``'dict' object has no attribute`` —— 那说明取值处漏了兼容。

工具搜索（v1.1.0 新增）
--------------------
规范再全也覆盖不了"模型临时想确认某个能力是否存在"的场景。本插件提供 AGENT 沙盒方法
``search_plugin_tools(keyword)``：按关键词搜索**全部已启用插件**的沙盒方法
（方法名/显示名/描述/docstring/插件名），结果回灌后模型继续对话。要点：

- 数据源：`plugin_collector.get_all_active_plugins()` + 各插件
  `collect_available_methods(ctx)`（含动态方法收集，如 basic.py 的按上下文暴露）；
- 遵守适配器过滤（``support_adapter``）——当前平台用不了的工具不出现；
- 结果复用 system prompt 里的既有标记 ``**[AGENT METHOD - STOP AFTER CALL]**``
  （模型无需学习新约定），休眠插件标注 ``[休眠]``（该插件的工具块当前只有摘要，
  需先 ``activate_plugin`` 唤醒——机制本身已由 runtime_contract 教给模型）；
- 搜索与注入相互独立，``ENABLE_INJECT=False`` 时搜索仍可用；
- 同一能力对人类开放：插件路由 ``GET /``（搜索页）与 ``GET /api/tools``（JSON），
  挂在 ``/plugins/NTidal.nekro_tool_prompt_hub`` 下（本机插件惯例无鉴权，只读）。

安装与使用
--------------------
- 把本目录放到 ``{NEKRO_DATA_DIR}/plugins/workdir/nekro_tool_prompt_hub/``（文件为 ``__init__.py``）；
  数据目录是 bind mount，容器重建不会丢
- 启停：WebUI 插件列表开关，或 ``POST /api/plugins/toggle/NTidal.nekro_tool_prompt_hub``
- 改配置：WebUI 插件详情配置页（保存即写入内存配置，**下一回合**生效，无需重启）
- 搜索页：``/plugins/NTidal.nekro_tool_prompt_hub``（只读）
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from nekro_agent.api import i18n
from nekro_agent.api.plugin import ConfigBase, ExtraField, NekroPlugin, SandboxMethodType
from nekro_agent.api.schemas import AgentCtx

plugin = NekroPlugin(
    name="工具调用提示中枢",
    module_name="nekro_tool_prompt_hub",
    description="把「何时调用工具、如何使用工具」的提示词从人格预设中剥离，以系统级规范按回合统一注入",
    version="1.1.1",
    author="NTidal",
    url="https://github.com/NTidal/nekro_tool_prompt_hub",
    i18n_name=i18n.i18n_text(
        zh_CN="工具调用提示中枢",
        en_US="Tool Prompt Hub",
    ),
    i18n_description=i18n.i18n_text(
        zh_CN="把「何时调用工具、如何使用工具」的提示词从人格预设中剥离，以系统级规范按回合统一注入",
        en_US="Manages tool-usage guidance as system-level rules injected each turn, instead of stuffing them into persona presets",
    ),
    allow_sleep=False,  # prompt 注入只对常驻/激活插件渲染，必须常驻
)


class ToolPromptRule(BaseModel):
    """单条工具调用规则。"""

    name: str = Field(
        default="新规则",
        title="规则名称",
        description="规则的简短标识，仅用于配置管理展示，也会作为注入分节的标题",
    )
    content: str = Field(
        default="",
        title="规则内容",
        description="注入给 LLM 的规范正文，写「什么场景用、怎么用、注意什么」，不要写角色扮演要求",
        json_schema_extra=ExtraField(is_textarea=True).model_dump(),
    )
    keywords: List[str] = Field(
        default=[],
        title="触发关键词（可选）",
        description="留空 = 每回合都注入；填写后 = 仅当最近若干条消息命中任一关键词时才注入（省 token）",
        json_schema_extra=ExtraField(sub_item_name="关键词").model_dump(),
    )
    enabled: bool = Field(
        default=True,
        title="启用",
        description="关闭后该规则不参与注入",
    )


DEFAULT_PREAMBLE = (
    "以下是本会话生效的工具使用规范（系统级指引）：\n"
    "- 是否调用工具、调用哪个工具，由你根据当前场景自主判断，无需等待用户下达明确指令或说出特定口令；\n"
    "- 用户消息中哪怕只是模糊地表达了相关意图，也应主动评估是否调用对应工具；\n"
    "- 本规范约束的是「怎么用工具」，与你的角色人格无关：人格决定你怎么说话，规范决定你怎么行动，二者不冲突；\n"
    "- 工具返回的是事实与结果，请用自己的角色口吻自然转述，不要照念技术数据或运维信息。"
)

DEFAULT_EXAMPLE_RULE = ToolPromptRule(
    name="示例：天气查询（演示写法，默认停用）",
    content=(
        "当用户询问天气、出行是否需要带伞等与天气相关的话题时，调用天气查询工具获取真实数据，"
        "基于返回结果回答；拿不到数据时如实说明，不要凭感觉编造天气。"
    ),
    keywords=["天气", "下雨", "气温", "带伞"],
    enabled=False,
)


@plugin.mount_config()
class ToolPromptHubConfig(ConfigBase):
    """工具调用提示中枢配置。"""

    ENABLE_INJECT: bool = Field(
        default=True,
        title="启用注入",
        description="总开关；关闭后本插件不注入任何内容",
        json_schema_extra=ExtraField(
            i18n_title=i18n.i18n_text(zh_CN="启用注入", en_US="Enable Injection"),
            i18n_description=i18n.i18n_text(
                zh_CN="总开关；关闭后本插件不注入任何内容",
                en_US="Master switch; when off, nothing is injected",
            ),
        ).model_dump(),
    )
    PREAMBLE: str = Field(
        default=DEFAULT_PREAMBLE,
        title="注入导语",
        description="放在规范块开头、向模型说明这套规范的定位与使用方式",
        json_schema_extra=ExtraField(is_textarea=True).model_dump(),
    )
    GLOBAL_RULES: str = Field(
        default="",
        title="全局准则",
        description="跨工具的通用规范，每个回合都注入；留空则跳过这一节",
        json_schema_extra=ExtraField(is_textarea=True).model_dump(),
    )
    RULES: List[ToolPromptRule] = Field(
        default=[DEFAULT_EXAMPLE_RULE],
        title="场景规则",
        description="按工具/场景拆分的规则列表；「触发关键词」留空的规则每回合注入，填了关键词的仅在最近消息命中时注入",
        json_schema_extra=ExtraField(sub_item_name="规则").model_dump(),
    )
    KEYWORD_SCAN_COUNT: int = Field(
        default=10,
        ge=1,
        le=50,
        title="关键词回看消息数",
        description="关键词规则匹配时，回看最近多少条消息（含触发本回合的消息）",
    )
    SEARCH_MAX_RESULTS: int = Field(
        default=20,
        ge=1,
        le=100,
        title="工具搜索结果上限",
        description="search_plugin_tools 单次返回的最大条数，超出部分提示模型换更具体的关键词",
    )
    MAX_INJECT_CHARS: int = Field(
        default=4000,
        ge=500,
        le=20000,
        title="注入总字符上限",
        description="注入块超过此长度时按「导语 → 全局准则 → 规则顺序」截断，保护上下文预算",
    )
    EXCLUDE_CHAT_KEYS: List[str] = Field(
        default=[],
        title="排除的频道",
        description="这些 chat_key 不注入任何规范",
        json_schema_extra=ExtraField(sub_item_name="频道").model_dump(),
    )
    LOG_INJECT: bool = Field(
        default=True,
        title="记录注入日志",
        description="每次注入写一条 INFO 日志（注入了多少条规则/字符），确认生效后可关闭",
    )


config = plugin.get_config(ToolPromptHubConfig)


def _safe_chat_key(ctx: AgentCtx) -> Optional[str]:
    """读取 chat_key；上下文异常时返回 None（放弃注入）。"""
    try:
        return ctx.chat_key
    except Exception:  # noqa: BLE001
        return None


async def _recent_text_blob(chat_key: str, limit: int) -> str:
    """取最近 limit 条消息的纯文本，拼成一个大字符串供关键词匹配。

    注意：注入发生在消息落库之后（run_agent 组装上下文阶段），
    所以最近消息已包含触发本回合的那条。

    ⚠️ `values_list()` 返回的是惰性查询对象，**必须 await** 才能取值；
    忘记 await 会得到 `'ValuesListQuery' object is not iterable`，
    异常被下面的兜底吞掉后表现为「关键词规则永远不生效」。
    """
    from nekro_agent.models.db_chat_message import DBChatMessage

    rows = await (
        DBChatMessage.filter(chat_key=chat_key)
        .order_by("-id")
        .limit(limit)
        .values_list("content_text", flat=True)
    )
    return "\n".join(str(t) for t in rows).lower()


def _rule_field(rule, field: str, default=None):
    """读取规则字段，兼容两种形态。

    ⚠️ 为什么需要它：NA 的 WebUI 保存配置走 `POST /api/config/batch`，
    该链路把嵌套字段写成**裸 dict** 后就更新了内存配置，不再经过 Pydantic
    校验。于是 config.RULES 的元素会从 ToolPromptRule 退化成 dict，
    此时 `rule.enabled` 直接抛 `'dict' object has no attribute 'enabled'`，
    导致之后每个回合的注入都失败（直到重启才恢复）。
    这里统一按「模型对象 / dict」两种形态取值。
    """
    if isinstance(rule, dict):
        return rule.get(field, default)
    return getattr(rule, field, default)


def _split_rules() -> Tuple[List[Any], List[Any]]:
    """把启用规则分成（每回合注入, 需关键词命中）。

    用 _rule_field 取值，兼容 config.RULES 里混入裸 dict 的情况。
    """
    always: List[Any] = []
    conditional: List[Any] = []
    for rule in config.RULES or []:
        if not _rule_field(rule, "enabled", True):
            continue
        content = (_rule_field(rule, "content", "") or "").strip()
        if not content:
            continue
        keywords = [str(k) for k in (_rule_field(rule, "keywords") or []) if str(k).strip()]
        if keywords:
            conditional.append(rule)
        else:
            always.append(rule)
    return always, conditional


@plugin.mount_prompt_inject_method(
    name="工具调用规范注入",
    description="每个回合把统一管理的工具调用规范注入到模型上下文",
)
async def inject_tool_prompts(_ctx: AgentCtx) -> str:
    """组装并返回本回合的工具调用规范块（框架会包上 plugin_runtime_context 标签）。

    任何异常都降级为返回空串——绝不能让注入失败打断 agent 上下文组装。
    """
    try:
        if not config.ENABLE_INJECT:
            return ""

        chat_key = _safe_chat_key(_ctx)
        if chat_key is None or chat_key in (config.EXCLUDE_CHAT_KEYS or []):
            return ""

        always_rules, conditional_rules = _split_rules()

        matched_conditional: List[Any] = []
        if conditional_rules:
            blob = await _recent_text_blob(chat_key, config.KEYWORD_SCAN_COUNT)
            for rule in conditional_rules:
                kws = [
                    str(kw).strip().lower()
                    for kw in (_rule_field(rule, "keywords") or [])
                    if str(kw).strip()
                ]
                if any(kw in blob for kw in kws):
                    matched_conditional.append(rule)

        sections: List[str] = []
        if config.PREAMBLE.strip():
            sections.append(f"## 工具调用规范（系统级指引）\n{config.PREAMBLE.strip()}")
        if config.GLOBAL_RULES.strip():
            sections.append(f"### 全局准则\n{config.GLOBAL_RULES.strip()}")
        for rule in [*always_rules, *matched_conditional]:
            _name = (_rule_field(rule, "name", "") or "").strip() or "未命名规则"
            _content = (_rule_field(rule, "content", "") or "").strip()
            sections.append(f"### 规则：{_name}\n{_content}")

        if len(sections) <= (1 if config.PREAMBLE.strip() else 0):
            # 只有导语、没有任何实际规则时不注入，省 token
            return ""

        injected = "\n\n".join(sections)
        if len(injected) > config.MAX_INJECT_CHARS:
            plugin.logger.warning(
                f"[tool_prompt_hub] 注入内容 {len(injected)} 字符超上限 {config.MAX_INJECT_CHARS}，已截断"
                f"（可增大的配置：MAX_INJECT_CHARS）",
            )
            injected = injected[: config.MAX_INJECT_CHARS] + "\n（规范过长已截断）"

        if config.LOG_INJECT:
            rule_names = [
                (_rule_field(r, "name", "") or "").strip() or "未命名规则"
                for r in [*always_rules, *matched_conditional]
            ]
            plugin.logger.info(
                f"[tool_prompt_hub] 已注入工具调用规范 | chat={chat_key} "
                f"| 规则 {len(rule_names)} 条 {rule_names} | {len(injected)} 字符",
            )
        return injected
    except Exception as e:  # noqa: BLE001 - 注入绝不能打断 agent 回合
        plugin.logger.error(f"[tool_prompt_hub] 注入失败，本回合跳过：{e}")
        return ""


# ---------------------------------------------------------------------------
# 工具搜索：LLM 沙盒方法 + 人类 WebUI，共用同一份核心逻辑
# ---------------------------------------------------------------------------

_AGENT_TYPES = (SandboxMethodType.AGENT.value, SandboxMethodType.MULTIMODAL_AGENT.value)


async def _sleeping_modules(chat_key: str) -> set:
    """对当前会话处于休眠态的插件 module_name 集合。

    判定与 `build_prompt_disclosure_view` 一致：可休眠插件（is_sleep_effective）
    且激活轮次 <= 0 即为休眠。休眠插件的沙盒方法仍可被搜到（方法注册与休眠无关），
    但其工具块在 prompt 里只有摘要——这层事实要如实告诉模型。
    """
    from nekro_agent.services.plugin.collector import plugin_collector
    from nekro_agent.services.plugin.prompt_activation import get_activation_state, is_sleep_effective

    state = await get_activation_state(chat_key)
    return {
        pl.module_name
        for pl in plugin_collector.get_all_active_plugins()
        if is_sleep_effective(pl) and state.module_rounds.get(pl.module_name, 0) <= 0
    }


async def _collect_search_entries(ctx: Optional[AgentCtx]) -> Tuple[List[Dict[str, Any]], int]:
    """收集已启用插件的全部沙盒方法条目。返回 (条目列表, 插件数)。

    - ctx 非空：走 `collect_available_methods(ctx)`（含动态方法收集 + 适配器过滤），
      并计算每条所属插件对当前会话的休眠状态；
    - ctx 为空（WebUI 调用）：退回静态 `sandbox_methods`，不做会话级休眠标注。
    """
    from nekro_agent.services.plugin.collector import plugin_collector

    adapter_key = ctx.adapter_key if ctx else None
    sleeping: set = await _sleeping_modules(ctx.chat_key) if ctx else set()

    entries: List[Dict[str, Any]] = []
    plugins = [
        pl
        for pl in plugin_collector.get_all_active_plugins()
        if not (adapter_key and pl.support_adapter and adapter_key not in pl.support_adapter)
    ]
    for pl in plugins:
        try:
            methods = await pl.collect_available_methods(ctx) if ctx else pl.sandbox_methods
        except Exception as e:  # noqa: BLE001 - 单个插件收集失败不拖垮搜索
            plugin.logger.warning(f"[tool_prompt_hub] 收集插件 {pl.module_name} 方法失败，回退静态列表：{e}")
            methods = pl.sandbox_methods
        for m in methods:
            entries.append(
                {
                    "func_name": m.func.__name__,
                    "method_type": getattr(m.method_type, "value", str(m.method_type)),
                    "name": (m.name or "").strip(),
                    "description": (m.description or "").strip(),
                    "docstring": (m.func.__doc__ or "").strip(),
                    "plugin_name": pl.name,
                    "module_name": pl.module_name,
                    "plugin_sleeping": pl.module_name in sleeping,
                },
            )
    return entries, len(plugins)


def _match_entries(entries: List[Dict[str, Any]], keyword: str) -> List[Dict[str, Any]]:
    """按关键词过滤：方法名/显示名命中排前，其余（描述/docstring/插件名）命中排后。"""
    kw = keyword.strip().lower()
    if not kw:
        return list(entries)
    hits: List[Tuple[Dict[str, Any], bool]] = []
    for e in entries:
        name_hit = kw in e["func_name"].lower() or kw in e["name"].lower()
        text_hit = (
            kw in f'{e["description"]}\n{e["docstring"]}'.lower()
            or kw in e["plugin_name"].lower()
            or kw in e["module_name"].lower()
        )
        if name_hit or text_hit:
            hits.append((e, name_hit))
    hits.sort(key=lambda t: (not t[1], t[0]["module_name"], t[0]["func_name"]))
    return [e for e, _ in hits]


async def _search_tools_payload(keyword: str, ctx: Optional[AgentCtx] = None, limit: Optional[int] = None) -> Dict[str, Any]:
    """核心搜索，供沙盒方法与 WebUI API 共用。异常全兜底，绝不抛出。

    limit：本次返回的条数上限；缺省用 `SEARCH_MAX_RESULTS`（给 LLM 的保守值），
    WebUI 可显式传更大的值（绝对上限 100）浏览全量。
    """
    try:
        entries, plugin_count = await _collect_search_entries(ctx)
        hits = _match_entries(entries, keyword)
        effective_limit = max(1, min(limit if limit is not None else config.SEARCH_MAX_RESULTS, 100))
        return {
            "keyword": keyword.strip(),
            "plugin_count": plugin_count,
            "total_methods": len(entries),
            "matched": len(hits),
            "truncated": len(hits) > effective_limit,
            "results": hits[:effective_limit],
        }
    except Exception as e:  # noqa: BLE001 - 搜索失败返回空结果而非异常
        plugin.logger.error(f"[tool_prompt_hub] 工具搜索失败：{e}")
        return {
            "keyword": keyword.strip(),
            "plugin_count": 0,
            "total_methods": 0,
            "matched": 0,
            "truncated": False,
            "results": [],
            "error": str(e),
        }


def _format_for_llm(payload: Dict[str, Any]) -> str:
    """把搜索结果格式化为给 LLM 的事实性文本（不含指令与运维信息）。"""
    matched = payload["matched"]
    kw = payload["keyword"]
    if payload.get("error"):
        return f"工具搜索暂不可用：{payload['error']}"
    if not kw:
        return (
            f"当前已启用 {payload['plugin_count']} 个插件，共注册 {payload['total_methods']} 个工具方法。"
            "传入关键词可按方法名、说明或插件名搜索。"
        )
    if matched == 0:
        return f"没有找到与「{kw}」相关的工具方法（已搜索 {payload['total_methods']} 个方法）。"

    lines = [f"共找到 {matched} 个与「{kw}」相关的工具方法："]
    for e in payload["results"]:
        agent_mark = " **[AGENT METHOD - STOP AFTER CALL]**" if e["method_type"] in _AGENT_TYPES else ""
        sleep_mark = " [休眠]" if e["plugin_sleeping"] else ""
        lines.append(f"* {e['func_name']}{agent_mark} — 插件「{e['plugin_name']}」({e['module_name']}){sleep_mark}")
        desc = e["docstring"] or e["description"] or e["name"]
        if desc:
            lines.append(f"  {desc[:200]}")
    if payload["truncated"]:
        lines.append(f"（仅显示前 {len(payload['results'])} 条，其余未列出，可用更具体的关键词缩小范围）")
    return "\n".join(lines)


@plugin.mount_sandbox_method(
    SandboxMethodType.AGENT,
    name="搜索插件工具",
    description="按关键词搜索当前已启用插件提供的全部工具方法",
)
async def search_plugin_tools(_ctx: AgentCtx, keyword: str = "") -> str:
    """搜索系统里已启用的插件工具方法。

    当你不确定系统里有没有能完成某件事的工具、想确认某类功能的工具名称与用法、
    或需要的功能没出现在当前可用方法里时，调用本方法搜索；判断权在你。
    keyword 支持中文或英文关键词，会匹配方法名、说明和插件名；留空则返回工具总量概览。
    返回列表中带 **[AGENT METHOD - STOP AFTER CALL]** 标记的方法，调用后必须立刻停止
    生成代码；标注 [休眠] 的插件需先 activate_plugin 唤醒，其工具才会完整可见。
    """
    payload = await _search_tools_payload(keyword, _ctx)
    plugin.logger.info(
        f"[tool_prompt_hub] 工具搜索 | chat={_ctx.chat_key} keyword={keyword!r} "
        f"| 命中 {payload['matched']}/{payload['total_methods']}",
    )
    return _format_for_llm(payload)


_TOOLS_PAGE_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>工具搜索 · 工具调用提示中枢</title>
<style>
 body{font-family:system-ui,"Segoe UI","Microsoft YaHei",sans-serif;margin:24px auto;max-width:1100px;color:#1f2328;padding:0 16px}
 h1{font-size:20px}
 .muted{color:#656d76}
 .bar{display:flex;gap:8px;margin-bottom:12px}
 input{flex:1;padding:8px 10px;border:1px solid #d0d7de;border-radius:6px;font-size:14px}
 button{padding:8px 16px;border:0;border-radius:6px;background:#0969da;color:#fff;cursor:pointer}
 table{border-collapse:collapse;width:100%;font-size:13px}
 th,td{border:1px solid #d8dee4;padding:6px 8px;text-align:left;vertical-align:top}
 th{background:#f6f8fa}
 code{background:#f6f8fa;padding:1px 5px;border-radius:4px;font-size:12px}
 .tag{display:inline-block;padding:0 6px;border-radius:10px;font-size:11px;margin-left:4px;white-space:nowrap}
 .tag.agent{background:#fff1c9;color:#7a4d00}
 .tag.sleep{background:#ffebe9;color:#a40e26}
 #stat{margin:8px 0;color:#656d76;font-size:13px}
</style>
</head>
<body>
<h1>已启用插件的工具搜索 <span class="muted" style="font-size:13px">工具调用提示中枢</span></h1>
<div class="bar">
 <input id="kw" placeholder="按方法名 / 说明 / 插件名搜索，留空列出全部" onkeydown="if(event.key==='Enter')doSearch()">
 <button onclick="doSearch()">搜索</button>
</div>
<div id="stat">加载中…</div>
<table><thead><tr><th style="width:220px">方法</th><th style="width:240px">插件</th><th>说明</th></tr></thead>
<tbody id="rows"></tbody></table>
<script>
const esc = s => (s ?? '').toString().replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
async function doSearch(){
  const kw = document.getElementById('kw').value;
  try {
    const r = await fetch('api/tools?limit=100&keyword=' + encodeURIComponent(kw));
    const d = await r.json();
    document.getElementById('stat').textContent =
      `已启用插件 ${d.plugin_count} 个，工具方法共 ${d.total_methods} 个；命中 ${d.matched} 条` +
      (d.truncated ? `（仅显示前 ${d.results.length} 条）` : '');
    document.getElementById('rows').innerHTML = d.results.map(e => `<tr>
      <td><code>${esc(e.func_name)}</code>${(e.method_type==='agent'||e.method_type==='multimodal_agent')?'<span class="tag agent">AGENT</span>':''}</td>
      <td>${esc(e.plugin_name)}<br><span class="muted">${esc(e.module_name)}</span>${e.plugin_sleeping?'<span class="tag sleep">休眠</span>':''}</td>
      <td>${esc(e.docstring || e.description || e.name || '')}</td>
    </tr>`).join('') || '<tr><td colspan="3" class="muted">无匹配结果</td></tr>';
  } catch (err) {
    document.getElementById('stat').textContent = '搜索失败：' + err;
  }
}
doSearch();
</script>
</body>
</html>
"""


@plugin.mount_router()
def _router() -> APIRouter:
    """人类侧工具搜索：只读页面 + JSON API。"""
    router = APIRouter()

    @router.get("/", response_class=HTMLResponse, summary="工具搜索页")
    async def tools_page() -> str:
        return _TOOLS_PAGE_HTML

    @router.get("/api/tools", summary="搜索已启用插件的工具方法")
    async def api_tools(
        keyword: str = Query("", description="关键词，匹配方法名/说明/插件名；留空返回全部"),
        limit: int = Query(0, ge=0, le=100, description="返回条数上限，0 表示用插件配置的默认值"),
    ) -> Dict[str, Any]:
        return await _search_tools_payload(keyword, limit=limit or None)

    return router


@plugin.mount_init_method()
async def on_init() -> None:
    """启动时自我说明，便于在日志里确认插件已加载与配置规模。"""
    always, conditional = _split_rules()
    plugin.logger.info(
        f"[tool_prompt_hub] 已加载 | enabled={config.ENABLE_INJECT} "
        f"| 全局准则 {len(config.GLOBAL_RULES or '')} 字符 "
        f"| 规则：每回合 {len(always)} 条 / 关键词 {len(conditional)} 条 "
        f"| 上限 {config.MAX_INJECT_CHARS} 字符 | 搜索结果上限 {config.SEARCH_MAX_RESULTS}",
    )
