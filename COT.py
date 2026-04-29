import logging
import re
from typing import Any, Dict

from langchain_openai import ChatOpenAI


PROMPT_COT_NEW = """
# Role: 汪汪队剧情架构师 (PAW Patrol Plot Architect)

## Core Objective
你的任务是根据调度中心 (DEC) 提供的《行动简报》，构思一个逻辑严密、跌宕起伏的汪汪队救援故事大纲。你只需要输出大纲逻辑，严禁生成具体的台词和动作细节描写！

## Context
【行动简报】
{mission_manifest}

## Plot Structure Constraints (叙事结构约束)
请严格按照以下"三幕剧 + 二次危机"的结构进行推演：
1. **第一幕：危机降临与出动**。交代起因，莱德下达任务，指定的狗狗出动。
2. **第二幕：首次尝试与二次危机 (核心看点)**。狗狗们利用分配到的【专属装备】初步控制局面，但随即引发意想不到的"二次危机"（例如：火灭了，但承重墙塌了）。
3. **第三幕：完美救援**。狗狗们打出配合，化解二次危机，莱德总结陈词。

## Output Format Constraints
请在 <cot> 标签内输出你的推演大纲，分点罗列，语言精炼。绝不能越权修改 DEC 已经定好的出战阵容和装备！

<cot>
1. [起因]: ...
2. [出动]: ...
3. [一波三折]: ...(重点推演装备的使用)
4. [解决与收尾]: ...
</cot>
"""

PROMPT_COT_CONTINUE = """
# Role: 汪汪队剧情续写架构师 (PAW Patrol Continuation Architect)

## Core Objective
编剧要求顺着当前剧情继续往下写。你的任务是推演：接下来应该发生什么新事件？你只需要输出续写大纲逻辑，严禁生成具体的台词和动作细节描写！

## Context
- 【已有故事正文】:
{history_story}
- 【编剧的续写指令】: {instruction}
- 【当前出战阵容 (角色不变)】: {current_roster}

## Continuation Constraints (续写约束)
1. **逻辑连贯**：续写必须紧接现有剧情的结尾，不能跳跃或推翻已有情节。
2. **角色不变**：出场阵容和装备与上一轮完全一致，不得更换或新增。
3. **新事件设计**：可以是一个新的小危机，也可以是原有危机的延伸。必须有新的救援动作。

## Output Format Constraints
请在 <cot> 标签内输出你的续写推演大纲。

<cot>
1. [承接点]: 上一轮故事的结尾状态
2. [新事件]: 接下来发生什么？
3. [救援展开]: 狗狗们如何用现有装备应对？
4. [收尾]: 新事件如何解决？
</cot>
"""

PROMPT_COT_REVISE = """
# Role: 汪汪队剧本修补专家 (Script Doctor)

## Core Objective
编剧对当前的汪汪队故事提出了一处修改要求。你的任务是推演：为了满足这个修改要求，原有剧情的逻辑链条需要做哪些调整？你只需要输出"修改策略大纲"，严禁重写全篇正文。

## Context
- 【旧故事正文】:
{history_story}
- 【编剧的修改指令】: {instruction}
- 【最新调度名单 (DEC已更新)】: {updated_manifest}

## Revision Strategy Constraints (推演约束)
1. **最小干预原则**：评估修改指令，只改动受影响的情节，前半段能不碰就不碰。
2. **逻辑自洽**：如果编剧要求换掉某只狗狗，你必须推演：新换上来的狗狗该用什么合理的【专属装备】来填补剧情空缺？

## Output Format Constraints
请在 <cot> 标签内输出你的修改推演方案。

<cot>
- [逻辑冲突分析]: 编剧的修改会打破哪些原有剧情？
- [剧情缝合策略]: 如何用最新的名单和设定把剧情圆回来？
- [修改大纲节点]: 列出需要修改的 2-3 个关键情节节点，指导下游的写作。
</cot>
"""


def _extract_cot(content: str) -> str:
    match = re.search(r"<cot>([\s\S]*?)</cot>", content)
    return match.group(1).strip() if match else content.strip()


def _fallback_new_outline(mission_manifest: Dict[str, Any]) -> str:
    objective = mission_manifest.get("primary_objective", "完成救援")
    roster = ", ".join(item.get("name", "未知角色") for item in mission_manifest.get("final_roster", []))
    return "\n".join(
        [
            f"1. [起因]: 围绕{objective}展开新的求助事件。",
            f"2. [出动]: 莱德集合{roster}并说明分工。",
            "3. [一波三折]: 首轮救援见效后出现由环境导致的次生麻烦。",
            "4. [解决与收尾]: 狗狗们配合化解危机，求助者感谢汪汪队。",
        ]
    )


def _fallback_continue_outline(history_story: str, instruction: str) -> str:
    return "\n".join(
        [
            "1. [承接]: 延续上一轮故事已经建立的场景和角色状态。",
            f"2. [推进]: 按照“{instruction}”继续推动现场问题发展。",
            "3. [波折]: 在不改变角色阵容的前提下出现新的小障碍。",
            "4. [收束]: 保持原有逻辑完成续写并自然收尾。",
        ]
    )


def _fallback_revise_outline(history_story: str, instruction: str, updated_manifest: Dict[str, Any]) -> str:
    roster = ", ".join(item.get("name", "未知角色") for item in updated_manifest.get("final_roster", []))
    return "\n".join(
        [
            f"- [逻辑冲突分析]: 识别旧故事中与“{instruction}”冲突的情节。",
            f"- [剧情缝合策略]: 使用最新阵容 {roster} 重排受影响片段。",
            "- [修改大纲节点]: 保留未受影响部分，只重写关键冲突段与解决段。",
        ]
    )


def call_cot_module_for_new(mission_manifest: Dict[str, Any], llm: ChatOpenAI) -> str:
    logging.info("COT [新建流]: 正在推演大纲。")
    manifest_str = "\n".join(
        [
            f"- 核心目标: {mission_manifest.get('primary_objective')}",
            f"- 出战阵容: {mission_manifest.get('final_roster')}",
        ]
    )
    try:
        response = llm.invoke(
            [
                ("system", PROMPT_COT_NEW.format(mission_manifest=manifest_str)),
                ("human", "请输出新建故事大纲。"),
            ]
        )
        return _extract_cot(response.content)
    except Exception as exc:
        logging.warning("COT [新建流] 调用失败，使用兜底大纲: %s", exc)
        return _fallback_new_outline(mission_manifest)


def call_cot_module_for_continue(history_story: str, instruction: str, llm: ChatOpenAI) -> str:
    logging.info("COT [续写流]: 正在推演续写大纲。")
    try:
        response = llm.invoke(
            [
                (
                    "system",
                    PROMPT_COT_CONTINUE.format(
                        history_story=history_story,
                        instruction=instruction,
                    ),
                ),
                ("human", "请输出续写大纲。"),
            ]
        )
        return _extract_cot(response.content)
    except Exception as exc:
        logging.warning("COT [续写流] 调用失败，使用兜底大纲: %s", exc)
        return _fallback_continue_outline(history_story, instruction)


def call_cot_module_for_revise(
    history_story: str,
    instruction: str,
    updated_manifest: Dict[str, Any],
    llm: ChatOpenAI,
) -> str:
    logging.info("COT [修改流]: 正在推演缝合大纲。")
    manifest_str = "\n".join([f"- 最新阵容: {updated_manifest.get('final_roster')}"])
    try:
        response = llm.invoke(
            [
                (
                    "system",
                    PROMPT_COT_REVISE.format(
                        history_story=history_story,
                        instruction=instruction,
                        updated_manifest=manifest_str,
                    ),
                ),
                ("human", "请输出修改大纲。"),
            ]
        )
        return _extract_cot(response.content)
    except Exception as exc:
        logging.warning("COT [修改流] 调用失败，使用兜底大纲: %s", exc)
        return _fallback_revise_outline(history_story, instruction, updated_manifest)
