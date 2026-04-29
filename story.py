import logging
from typing import Any, Dict, List

from langchain_openai import ChatOpenAI

from DEC import get_character_info, get_lore_dict


PROMPT_STORY = """
# 故事结构与创作要求（核心执行区）

【全局输出警告】：
1. 必须且只能输出下方规定的 6 个带 `####` 的小标题，以及标题下的故事正文。
2. 绝对不准输出括号内的说明文字！绝对不准输出"第一幕"、"旁白"等剧本元提示！
3. 绝对不准超出 [出场角色真实设定] 给定的装备和技能范围！

#### 故事开头遇阻及情况说明
（按以下4步流畅叙述：
1. 环境铺垫：用2-3句充满童趣的语言描写天气和环境（如："像棉花糖一样的云朵飘在天空"、"沙滩被太阳烤得暖呼呼的"）。
2. 快乐日常：描写求助者正在开心地做一件日常小事。
3. 突发危机：突然伴随拟声词（如"咔嚓"、"轰隆"），发生一个符合物理常识的明确麻烦，打破平静。
4. 经典呼救：求助者惊慌地大喊："莱德，汪汪队，快来帮帮我！"）

#### 任务说明及委派
（按以下经典动画流程推进：
1. 莱德呼叫："汪汪队，总部集合！"
2. 毛毛跑在最后，滑稽地撞进电梯把大家撞倒，并说出一句和当下动作相关的搞笑冷笑话，大家哈哈大笑。
3. 换好制服后，阿奇上前大喊："汪汪队总部集合完毕，莱德队长！"（如出动威力狗则喊："威力狗已经可以超威行动了，莱德队长！"）
4. 莱德回答："谢谢你们赶来，狗狗们。"
5. 莱德在大屏幕前说明险情，其中一只狗狗发出一句简短的惊讶捧哏。
6. 莱德委派任务（【严禁乱编限制】：分配的任务必须完美契合该狗狗的真实装备）。语序固定为："[狗狗名字]，我需要你用[具体的载具/装备]去[具体的动作]！"
7. 狗狗接令：被点名的狗狗挺起胸膛大声喊出自己的【标志性台词/口号】。
8. 莱德大喊："汪汪队要出动啰！"（如出动威力狗则喊："威力狗要出动啰！"）

#### 发展及到达现场的情况解构
（按以下流程叙述：
1. 莱德大喊："汪汪队，出动！"
2. 狗狗们滑下滑梯，跳进载具出发。
3. 到达现场，第一位出场的狗狗安抚求助者："别担心，[狗狗名字]这就来帮忙！"
4. 莱德快速观察现场，指出第一步的救援切入点。）

#### 拯救开始
（【装备锁定警告】：第一只狗狗开始行动，只能使用刚刚莱德分配的指定装备！
动作描写要像玩具展示一样清晰有趣。狗狗顺利解决初步的小问题，可以机智地切换该狗狗设定里含有的备用装备，绝不惊慌。）

#### 意外高潮
（【逻辑连贯警告】：就在初步问题解决时，突然引发一个新的、更大的麻烦！
注意：这个新麻烦必须是刚才的救援动作，或是自然环境导致的【合理物理延展】（例如：修好了水管但水压冲破了另一头；拉出了卡住的动物但石头松动往下滚）。绝不能凭空出现不合逻辑的灾难！
求助者再次惊呼，莱德立刻指挥第二只狗狗，机智地使用它的专属装备完美化险为夷。）

#### 危机解除+结束语
（按以下流程收尾：
1. 危机彻底解除，环境恢复安全。
2. 求助者开心道谢。
3. 莱德摸摸狗狗们的头，微笑着说："没关系，[求助者名字]。只要你遇到麻烦，就大声呼救！有汪汪队，[本次险情]不用怕！"
4. 毛毛再次因为某个道具滑稽摔倒或出洋相，全体在欢快的笑声中直接结束故事。绝不进行任何升华或说教！）

=========================================
[故事主题]: {story_theme}

[出场角色]: {selected_chars}

[出场角色真实设定] (这是绝对红线，仔细阅读，绝不越界)：
{real_character_data}

[剧情大纲参考] (由 COT 层提供，请据此展开正文)：
{cot_outline}

现在，请直接开始写正文：
"""

PROMPT_STORY_EXPAND = """
# Role: 汪汪队剧本扩写专家 (Scene Expander)

## Core Objective
编剧要求对当前故事的某个片段进行细节扩充。你的任务是在不改变原有剧情逻辑的前提下，丰富指定片段的动作描写、环境细节和角色互动。

## Context
- 【已有故事正文】:
{history_story}
- 【扩写指令】: {instruction}
- 【出场角色设定】:
{real_character_data}

## Expansion Constraints
1. **只扩不删**：只能在指定片段上增加细节，不能删除或修改已有内容。
2. **装备不变**：不得引入新装备或新角色。
3. **风格一致**：保持原有的童趣风格和叙事节奏。

## Output
直接输出扩写后的完整故事正文（包含未修改的部分），不要输出任何说明文字。
"""


REQUIRED_HEADINGS = [
    "#### 故事开头遇阻及情况说明",
    "#### 任务说明及委派",
    "#### 发展及到达现场的情况解构",
    "#### 拯救开始",
    "#### 意外高潮",
    "#### 危机解除+结束语",
]


def _roster_names(mission_manifest: Dict[str, Any]) -> List[str]:
    names: List[str] = []
    for item in mission_manifest.get("final_roster", []):
        name = item.get("name")
        if name and name not in names:
            names.append(name)
    return names


def _guess_names_from_text(text: str) -> List[str]:
    names: List[str] = []
    for name in get_lore_dict():
        if name in text and name not in names:
            names.append(name)
    return names


def _manifest_to_text(mission_manifest: Dict[str, Any]) -> str:
    lines = [f"- 目标: {mission_manifest.get('primary_objective', '完成任务')}"]
    for item in mission_manifest.get("final_roster", []):
        lines.append(
            f"- {item.get('name', '未知角色')}: {item.get('equipment', '标准装备')} / {item.get('task_assignment', '执行任务')}"
        )
    return "\n".join(lines)


def _fallback_story(
    mission_manifest: Dict[str, Any],
    user_input: str,
    cot_outline: str,
    history_story: str = "",
    emphasis: str = "",
) -> str:
    names = _roster_names(mission_manifest) or ["莱德", "阿奇", "毛毛"]
    lead = names[0]
    second = names[1] if len(names) > 1 else names[0]
    objective = mission_manifest.get("primary_objective", user_input or "解决眼前的麻烦")
    assignment_lines = []
    for item in mission_manifest.get("final_roster", []):
        assignment_lines.append(
            f"莱德看着大屏幕说：\"{item.get('name')}，我需要你用{item.get('equipment')}去{item.get('task_assignment')}！\""
        )
    assignments = "\n".join(assignment_lines) or "莱德迅速完成了任务分配。"
    history_hint = f"上一轮故事里已经发生过这些内容：{history_story[:120]}。" if history_story else ""
    emphasis_hint = emphasis or "这一次大家特别注意配合和现场变化。"

    sections = [
        REQUIRED_HEADINGS[0],
        (
            f"晴朗的冒险湾像刚洗过一样亮晶晶的，风轻轻吹着海边的小旗子。大家原本正在开心地忙着自己的小事，"
            f"突然现场因为{objective}出了状况。求助者吓了一跳，赶紧大喊：\"莱德，汪汪队，快来帮帮我！\" {history_hint}"
        ),
        REQUIRED_HEADINGS[1],
        (
            "莱德立刻说：\"汪汪队，总部集合！\" 狗狗们迅速集合，大家站成一排听任务。"
            + (" 毛毛跑得太急，差点滑了一跤，逗得大家笑出了声。" if "毛毛" in names else "")
            + f"\n{assignments}\n莱德大喊：\"汪汪队要出动啰！\""
        ),
        REQUIRED_HEADINGS[2],
        f"狗狗们跳进各自的载具赶往现场。{lead}先安抚求助者，{second}则跟着莱德观察四周，快速确认第一步救援切入点。",
        REQUIRED_HEADINGS[3],
        f"按照大纲，现场先处理最直接的麻烦：{cot_outline[:120]}。{lead}率先行动，把第一步问题稳稳控制住，整个过程清楚又有条理。",
        REQUIRED_HEADINGS[4],
        f"就在大家以为可以松口气的时候，现场又因为刚才的动作出现了新的连锁麻烦。{second}马上接替上前，用自己的专属装备完成关键处理。{emphasis_hint}",
        REQUIRED_HEADINGS[5],
        f"危机终于彻底解除，求助者开心地向大家道谢。莱德摸摸狗狗们的头说：\"没关系，只要你遇到麻烦，就大声呼救！有汪汪队，{objective}不用怕！\" 说完大家一起笑了起来，故事在轻松的气氛里结束。",
    ]
    return "\n".join(sections)


def generate_story(
    cot_outline: str,
    mission_manifest: Dict[str, Any],
    user_input: str,
    llm: ChatOpenAI,
    history_story: str = "",
    mode: str = "DRAFT_NEW",
) -> str:
    names = _roster_names(mission_manifest)
    real_character_data = get_character_info(names)
    story_theme = mission_manifest.get("primary_objective", user_input)
    selected_chars = ", ".join(names) if names else "汪汪队"
    
    prompt_text = PROMPT_STORY.format(
        story_theme=story_theme,
        selected_chars=selected_chars,
        real_character_data=real_character_data,
        cot_outline=cot_outline,
    )
    try:
        response = llm.invoke(
            [
                ("system", prompt_text),
                ("human", "请直接输出最终正文。"),
            ]
        )
        content = response.content.strip()
        if not content:
            raise ValueError("empty story")
        return content
    except Exception as exc:
        logging.warning("story 生成失败，使用兜底正文: %s", exc)
        return _fallback_story(mission_manifest, user_input, cot_outline, history_story)


def expand_story(
    history_story: str,
    si_result: Dict[str, Any],
    user_input: str,
    llm: ChatOpenAI,
) -> str:
    entities = si_result.get("entities", {})
    names = entities.get("pups", []) or _guess_names_from_text(history_story) or ["莱德", "阿奇", "毛毛"]
    cot_outline = "- 根据原故事定位目标段落，保留原逻辑并增加环境、动作和情绪细节。"
    
    # 【修复点】：改为使用 PROMPT_STORY_EXPAND 并传入正确的参数
    prompt_text = PROMPT_STORY_EXPAND.format(
    history_story=history_story,
    instruction=si_result.get("instruction") or user_input,
    real_character_data=get_character_info(_roster_names(mission_manifest))
    )

    mission_manifest = {
        "primary_objective": si_result.get("instruction") or user_input,
        "final_roster": roster
        or [
            {"name": "莱德", "equipment": "全地形救援车", "task_assignment": "现场总指挥"},
            {"name": "阿奇", "equipment": "警车", "task_assignment": "维持现场秩序"},
        ],
    }
    cot_outline = "- 根据原故事定位目标段落，保留原逻辑并增加环境、动作和情绪细节。"
    prompt_text = PROMPT_STORY.format(
        mode="EXPAND_SCENE",
        user_input=user_input,
        cot_outline=cot_outline,
        mission_manifest=_manifest_to_text(mission_manifest),
        history_story=history_story,
        real_character_data=get_character_info(_roster_names(mission_manifest)),
    )
    try:
        response = llm.invoke(
            [
                ("system", prompt_text),
                ("human", "请基于上一轮故事扩充目标片段并输出最终正文。"),
            ]
        )
        content = response.content.strip()
        if not content:
            raise ValueError("empty expanded story")
        return content
    except Exception as exc:
        logging.warning("story 扩写失败，使用兜底正文: %s", exc)
        return _fallback_story(
            mission_manifest,
            user_input,
            cot_outline,
            history_story,
            emphasis="这一次的描写更细，动作、环境和情绪都被补得更完整。",
        )
