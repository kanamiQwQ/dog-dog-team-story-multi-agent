import json
import logging
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd

from langchain_openai import ChatOpenAI


FALLBACK_LORE = {
    "莱德": {
        "identity": "汪汪队队长、人类男孩",
        "traits": "聪明、热心、具备出色的领导力和科技头脑",
        "equipment": ["全地形越野车", "狗狗巡逻板(平板电脑)"],
        "skills": ["战术指挥", "载具驾驶", "机械维修"],
        "locations": ["冒险湾全域", "汪汪队总部"]
    },
    "阿奇": {
        "identity": "警犬 / 特工犬",
        "traits": "勇敢、守纪律、对猫毛和羽毛过敏",
        "equipment": ["警车", "绞盘", "网球发射器", "扩音器", "无人机", "特工滑索"],
        "skills": ["追踪搜寻", "指挥交通", "维持秩序", "特工潜入"],
        "locations": ["公路", "城镇街道", "迷宫或密室"]
    },
    "毛毛": {
        "identity": "消防犬 / 急救犬",
        "traits": "平时笨手笨脚(经常撞倒大家)，但执行任务时非常可靠",
        "equipment": ["消防车(带云梯)", "水炮", "急救包(X光屏幕/绷带)"],
        "skills": ["灭火", "医疗急救", "高处救援(使用云梯)"],
        "locations": ["火灾现场", "高树或高楼", "医院"]
    },
    "天天": {
        "identity": "飞行犬",
        "traits": "优雅、喜欢翻跟头、害怕老鹰",
        "equipment": ["直升机(带救援吊带)", "粉色飞行背包", "飞行护目镜"],
        "skills": ["空中侦察", "飞行吊挂救援", "空投物资"],
        "locations": ["天空", "高山之巅", "悬崖"]
    },
    "灰灰": {
        "identity": "环保维修犬",
        "traits": "极度怕水(除了当人鱼狗时)、充满创意、什么破烂都能变废为宝",
        "equipment": ["绿色垃圾回收车", "多功能机械臂", "工具箱", "胶布与胶水"],
        "skills": ["废物利用", "机械修理", "临时建筑搭设"],
        "locations": ["废品回收站", "损坏的设施旁", "断桥"]
    },
    "路马": {
        "identity": "水上救援犬",
        "traits": "活力四射、爱玩水、冲浪高手",
        "equipment": ["橙色气垫船(可变潜水艇)", "潜水服", "救生圈"],
        "skills": ["潜水", "水上拖拽", "水下搜救"],
        "locations": ["海洋", "河流", "沙滩", "水下"]
    },
    "小砾": {
        "identity": "工程犬",
        "traits": "贪吃、喜欢洗泡泡浴、喜欢滑雪、害怕蜘蛛",
        "equipment": ["黄色工程推土机", "电钻", "机械抓斗", "铲子"],
        "skills": ["挖掘碎石", "搬运重物", "清理路障"],
        "locations": ["建筑工地", "矿洞", "被落石堵住的山路"]
    }
}
class Config:
    LLM_MODEL_NAME = "Qwen/Qwen3-14B-FP8"
    OPENAI_API_KEY = "openai"
    OPENAI_BASE_URL = "https://u968412-80b9-4a53f296.westx.seetacloud.com:8443/v1"
    DEFAULT_TEMPERATURE = 0.3
    CSV_FILE_PATH = Path(__file__).with_name("role1.1.3.csv")


def get_character_info(df: pd.DataFrame, requested_names: list) -> str:
    """
    为 DEC 模块获取狗狗设定档案。
    如果 requested_names 为空，则默认返回所有核心狗狗的档案，供大模型挑选。
    """
    # 1. 核心候选人池逻辑：如果用户没点名，DEC 需要看到所有主力的简历才能挑人
    core_team = ["莱德", "阿奇", "毛毛", "天天", "灰灰", "路马", "小砾"] # 可以加上珠珠、小克等
    
    # 决定要查询的名单
    target_names = requested_names if requested_names else core_team
    
    # 防止用户点名了 "天天"，但还需要大模型补位，所以最好把核心团队也带上
    # （可选）建议将用户点名的和核心团队合并去重：
    target_names = list(set(target_names + core_team))

    info_blocks = []
    
    for name in target_names:
        rows = df[df["角色名"] == name]
        if len(rows) == 0:
            continue
            
        row = rows.iloc[0]
        
        # 辅助函数：判断值是否有效（非 nan 且不为 '无'）
        def is_valid(val):
            if pd.isna(val) or str(val).strip() in ['无', '空', 'none', '']:
                return False
            return True

        # 2. 使用 Markdown 缩进格式组装（大模型最容易读懂的格式）
        lines = [f"### [{name}] ({row['职务/身份']})"]
        
        if is_valid(row.get('性格/特点')):
            lines.append(f"- 性格/特点: {row['性格/特点']}")
            
        # 载具与装备（合并为一个清晰的条目，防止大模型搞混）
        equipments = []
        if is_valid(row.get('载具/店铺')): equipments.append(row['载具/店铺'])
        if is_valid(row.get('载具功能')): equipments.append(f"({row['载具功能']})")
        if is_valid(row.get('装备')): equipments.append(f"随身装备: {row['装备']}")
        if equipments:
            lines.append(f"- 核心装备: {' '.join(equipments)}")
            
        if is_valid(row.get('技能/能力')):
            lines.append(f"- 技能能力: {row['技能/能力']}")
            
        # 升级形态单独列出，防止大模型在普通任务中乱用
        if is_valid(row.get('升级名称')):
            upgrade_info = f"- 特殊形态 ({row['升级名称']}): {row.get('升级后形态', '')} - {row.get('升级后的技能', '')}"
            lines.append(upgrade_info)
            
        info_blocks.append("\n".join(lines))
        
    # 用分隔符将不同角色的档案分开
    return "\n\n".join(info_blocks)

PROMPT_DEC_DISPATCHER = """
# Role: 汪汪队紧急调度中心大脑 (PAW Patrol Dispatch Center)

## Core Objective
你负责在汪汪队出动前，制定绝对严谨的战术简报。你需要根据当前的危机事件，以及用户已指定出场的狗狗，为其分配合理的装备和具体的任务分工。严禁生成故事正文，只输出结构化的 JSON 调度单。

## Context
- 危机事件 (Crisis): {crisis_event}
- 用户点名要求的角色 (Requested Roster): {requested_pups}
- 汪汪队官方角色设定集 (Lore Dictionary): 
{real_character_data}

## Dispatch Rules (严格调度原则)
1. **尊重用户选择**: 如果 `Requested Roster` 中有指定的狗狗，它们必须全员出战。
2. **智能补位与克制**: 如果指定狗狗不足以解决危机，请智能追加最合适的狗狗。**【注意】绝不为了让狗狗出场而强行捏造次生灾害！** 如果危机只是“暴风雨”，就只处理暴风雨，严禁脑补“燃油泄漏”、“起火”等原文未提及的灾难。
3. **莱德的定位**: 莱德队长永远作为指挥官存在，负责调度和骑乘全地形小车，他不使用狗狗的专属装备。

##绝对红线法则 (CRITICAL LORE BINDING)
1. **【跨界禁令】**: 狗狗只能且必须使用《汪汪队官方角色设定集》中属于它自己的装备！
   - 绝对禁止张冠李戴！（例如：严禁让小砾开直升机，严禁让阿奇用水炮）。
   - 如果当前危机需要飞行救援，但天天（Skye）没有出场，你必须将天天加入名单，而不是把直升机硬塞给其他狗狗！
2. **【地形常识禁令】**: 必须根据环境（海洋、天空、雪地、丛林）指派装备形态。
   - 水上任务中，普通轮式车辆（警车、消防车、工程推土机）完全失效，严禁下水。
   - 若非水上专精犬（如阿奇、毛毛）必须参与水上救援，必须为他们挂载【水上专属形态】（如水上警用快艇），否则只能在岸边待命。
3. **【物理逻辑校验】**: 装备必须符合物理常识。（例如：如果是处理高空断桥，使用路马的气垫船或毛毛的水炮是无效的，必须使用工程车辆或飞行器）。

## Output Format
必须且只能输出一个合法的 JSON 对象，不要包含 Markdown 代码块标记（如 ```json）。

{{
  "mission_manifest": {{
    "primary_objective": "本次救援的首要目标（一句话描述，严禁添加危机事件中未提及的灾难）",
    "final_roster": [
      {{
        "name": "角色名称（如：莱德，阿奇）",
        "equipment": "严格依据设定集分配的专属装备",
        "task_assignment": "具体职责"
      }}
    ]
  }}
}}
"""

def get_lore_dict() -> Dict[str, Dict[str, Any]]:
    return FALLBACK_LORE


def get_character_info(requested_names: List[str]) -> str:
    lore = get_lore_dict()
    core_team = ["莱德", "阿奇", "毛毛", "天天", "灰灰", "路马", "小砾"]
    names = requested_names or core_team
    merged_names: List[str] = []
    for name in names + core_team:
        if name in lore and name not in merged_names:
            merged_names.append(name)

    info_blocks: List[str] = []
    for name in merged_names:
        item = lore[name]
        lines = [
            f"### [{name}] ({item['identity']})",
            f"- 性格/特点: {item['traits']}",
            f"- 核心装备: {'、'.join(item['equipment'])}",
            f"- 技能能力: {'、'.join(item['skills'])}",
            f"- 常见场景: {'、'.join(item['locations'])}",
        ]
        info_blocks.append("\n".join(lines))
    return "\n\n".join(info_blocks)


def prepare_lore_context(si_result: Dict[str, Any]) -> str:
    requested_pups = si_result.get("entities", {}).get("pups", []) or []
    return get_character_info(requested_pups)


def _default_manifest(si_result: Dict[str, Any]) -> Dict[str, Any]:
    lore = get_lore_dict()
    entities = si_result.get("entities", {})
    instruction = si_result.get("instruction") or "执行常规巡逻救援任务"
    requested_pups = [name for name in entities.get("pups", []) if name in lore]
    if "莱德" not in requested_pups:
        requested_pups.insert(0, "莱德")
    if len(requested_pups) == 1:
        requested_pups.extend(["阿奇", "毛毛"])

    roster = []
    for name in requested_pups:
        item = lore.get(name, {})
        equipment = item.get("equipment", ["标准救援装备"])[0]
        task = "现场总指挥" if name == "莱德" else f"使用{equipment}执行救援"
        roster.append(
            {
                "name": name,
                "equipment": equipment,
                "task_assignment": task,
            }
        )

    return {
        "primary_objective": instruction,
        "final_roster": roster,
    }


def process_dec_module(si_result: Dict[str, Any], llm: ChatOpenAI) -> Dict[str, Any]:
    entities = si_result.get("entities", {})
    instruction = si_result.get("instruction") or "未知危机，需要巡逻发现"
    requested_pups = entities.get("pups", []) or []
    if "莱德" not in requested_pups:
        requested_pups.insert(0, "莱德")

    lore_context = prepare_lore_context(si_result)
    prompt_text = "\n".join(
        [
            PROMPT_DEC_DISPATCHER,
            f"用户指令: {instruction}",
            f"用户点名角色: {', '.join(requested_pups) or '无'}",
            "角色设定:",
            lore_context,
        ]
    )

    try:
        response = llm.invoke(
            [
                ("system", prompt_text),
                ("human", "请输出本次行动的调度单。"),
            ]
        )
        payload = json.loads(response.content)
        manifest = payload["mission_manifest"]
        if "莱德" not in [item.get("name") for item in manifest.get("final_roster", [])]:
            manifest = _default_manifest(si_result)
        logging.info("DEC 调度完成。")
        return manifest
    except Exception as exc:
        logging.warning("DEC 调度失败，启用兜底调度: %s", exc)
        return _default_manifest(si_result)


def answer_lore_query(user_input: str, entities: Dict[str, Any]) -> str:
    lore = get_lore_dict()
    targets: List[str] = []
    for name in entities.get("pups", []) or []:
        if name in lore and name not in targets:
            targets.append(name)

    if not targets:
        for name in lore:
            if name in user_input and name not in targets:
                targets.append(name)

    if not targets:
        targets = ["莱德", "阿奇", "毛毛"]

    lines: List[str] = []
    for name in targets:
        item = lore[name]
        lines.append(
            "\n".join(
                [
                    f"{name}：{item['identity']}",
                    f"性格：{item['traits']}",
                    f"装备：{'、'.join(item['equipment'])}",
                    f"技能：{'、'.join(item['skills'])}",
                    f"常见场景：{'、'.join(item['locations'])}",
                ]
            )
        )
    return "\n\n".join(lines)
