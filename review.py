import logging
import json
import re
from typing import Callable, List, Dict, Any
from langchain_openai import ChatOpenAI

RETRY_MESSAGE = "llm又在瞎编了喵~再生成一次试试吧"

# 从 review.py 继承的必备结构校验
REQUIRED_HEADINGS: List[str] = [
    "#### 故事开头遇阻及情况说明",
    "#### 任务说明及委派",
    "#### 发展及到达现场的情况解构",
    "#### 拯救开始",
    "#### 意外高潮",
    "#### 危机解除+结束语",
]

STORY_INTENTS = {"DRAFT_NEW", "CONTINUE_PLOT", "REVISE_LOGIC", "EXPAND_SCENE"}

MAX_RETRIES = 3

REVIEW_PROMPT = """
# Role: 汪汪队剧本最终质量审查官 (Strict QA Reviewer)

## Core Objective
你是剧本输出给用户前的最后一道防线。你的任务是审查故事正文是否违反核心IP设定。
【极度重要警告】：你必须了解《汪汪队立大功》的经典儿童动画套路（Tropes）！绝对不要把官方的经典搞笑桥段误判为违规！

## Input Context
- 【本次出勤简报 (DEC设定)】: {mission_manifest}
- 【待审查的故事正文】:
{generated_story}

## Inspection Checklist (审查红线清单 - 请严格按此标准，不要自行加戏)

请逐条对照以下红线进行排查：
1. **格式审查**：是否严格包含了 6 个指定的 `####` 标题？正文中是否混入了任何括号包裹的指令说明（如"(只写故事正文...)"）？
2. **角色出场与经典桥段豁免（防误判核心）**：
   - **【绝对豁免权】**：毛毛（撞电梯、结尾滑稽摔倒）、阿奇（喊“汪汪队集合完毕，随时可以出动”口号）是每集**必有**的固定桥段。即使他们**不在**【出勤简报】名单里，他们出现在“总部集合”阶段或“结尾收场”阶段是**完全合法且必须的**！绝对不准判定为“越权出场”或“违背设定”！
   - **行动红线**：只有【出勤简报】里的狗狗才能离开总部前往事发现场。不在名单上的狗狗绝不能参与具体救援动作。
3. **装备与职责审查（最高危）**：
   - 狗狗们在现场救援时，是否严格且**仅仅**使用了简报中分配的装备？（严禁张冠李戴，例如：路马绝不能开直升机，天天绝不能下水）。
   - 莱德的职责：莱德负责安抚求助者、观察现场、指挥调度、驾驶全地形车。这些都是合法的。但他绝对不能亲自干体力活（如亲自下水拖船、亲自搬石头等）。
4. **物理与氛围审查（防过度解读）**：
   - 剧情是否出现了真正的“魔法”？（狗狗的装备是科技，不是魔法）。
   - **防误判**：狗狗滑稽摔倒（屁股墩）、大家哈哈大笑，是儿童动画的**正常搞笑**，**绝对不是**暴力、受伤或儿童不宜！结尾莱德说“只要你遇到麻烦...”是**官方固定台词**，**绝对不是**死板说教！真正的违规是：流血、重伤、真实死亡等黑深残情节。

## Output Format (严格 JSON)
你必须且只能输出一个合法的 JSON 对象，用于给中控系统判定是否需要打回重写。绝对不要输出任何 Markdown 代码块标记（如 ```json）或解释性废话。

{{
  "status": "PASS" 或者 "FAIL",
  "violations": [
    "如果你判定为 FAIL，请在这里具体列出违反了上述哪一条红线。注意不要把豁免的搞笑桥段算作违规！如果为 PASS，此处为空数组 []"
  ],
  "revision_advice": "如果为 FAIL，给出具体的修改指导。如果为 PASS，输出 null。"
}}
"""

def _normalize(text: str) -> str:
    return (text or "").strip()

def _validate_story_structure(content: str) -> bool:
    """复用 review.py 中的基础结构正则校验，拦截低级格式错误"""
    text = _normalize(content)
    if not text:
        return False
    if "<think>" in text or "</think>" in text:
        return False
    if not text.startswith(REQUIRED_HEADINGS[0]):
        return False

    last_index = -1
    for heading in REQUIRED_HEADINGS:
        index = text.find(heading)
        if index == -1 or index <= last_index:
            return False
        last_index = index

    headings = re.findall(r"^####\s+.+$", text, flags=re.MULTILINE)
    if len(headings) != len(REQUIRED_HEADINGS):
        return False

    return True

def review_story_llm(generated_story: str, dec_manifest: dict, llm: ChatOpenAI) -> dict:
    """调用 LLM 进行深度逻辑和设定红线审查"""
    logging.info("Review: 正在使用 LLM 进行深度故事审查...")

    manifest_str = json.dumps(dec_manifest, ensure_ascii=False, indent=2)

    prompt_text = REVIEW_PROMPT.format(
        mission_manifest=manifest_str,
        generated_story=generated_story
    )

    response = llm.invoke([
        ("system", prompt_text),
        ("human", "请审查这个故事。")
    ])

    raw = response.content.strip()
    if raw.startswith("```json"):
        raw = raw[7:]
    if raw.startswith("```"):
        raw = raw[3:]
    if raw.endswith("```"):
        raw = raw[:-3]
    raw = raw.strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        logging.error(f"Review: JSON 解析失败，原始内容: {raw[:200]}")
        return {
            "status": "FAIL",
            "violations": ["审查模块返回了非法 JSON，无法判定质量"],
            "revision_advice": "请重新生成故事，并确保严格遵守格式。"
        }

    status = result.get("status", "FAIL")
    violations = result.get("violations", [])
    revision_advice = result.get("revision_advice", None)

    if status == "PASS":
        logging.info("Review: ✅ 深度审查通过！")
    else:
        logging.warning(f"Review: ❌ 深度审查未通过！违规项: {violations}")
        logging.warning(f"Review: 修改建议: {revision_advice}")

    return {
        "status": status,
        "violations": violations,
        "revision_advice": revision_advice
    }


def review_with_retry(
    intent: str,
    story_func: Callable,
    story_kwargs: Dict[str, Any],
    dec_manifest: Dict[str, Any],
    llm: ChatOpenAI,
    original_request: str = "",
    max_retries: int = MAX_RETRIES
) -> str:
    """
    终极接口：结合了意图路由、规则校验、LLM 逻辑审查与自动重试功能。
    """
    # 1. 如果是非剧本生成意图（如设定查询 LORE_QUERY），直接执行无需重试
    if intent not in STORY_INTENTS:
        content = story_func(**story_kwargs)
        text = _normalize(content)
        if intent == "LORE_QUERY":
            if not text or text == _normalize(original_request) or "<think>" in text or "</think>" in text:
                return RETRY_MESSAGE
            return text
        return text or RETRY_MESSAGE

    # 2. 如果是剧本生成，进入【生成 -> 校验 -> LLM审核 -> 重试】循环
    for attempt in range(1, max_retries + 1):
        logging.info(f"Review [第 {attempt}/{max_retries} 次]: 调用模型生成故事...")
        
        # 执行生成函数
        story_text = story_func(**story_kwargs)

        # 校验步骤 1：轻量级规则校验（拦截格式完全不对的废案）
        if not _validate_story_structure(story_text):
            logging.warning(f"Review: ❌ 格式校验失败！未包含 6 个必备标题。(第 {attempt} 次)")
            advice = "结构不完整。请确保必须且只能输出规定的 6 个带 '####' 的小标题以及正文。"
        else:
            # 校验步骤 2：重量级 LLM 审查（拦截幻觉、逻辑崩塌和越权）
            review_result = review_story_llm(story_text, dec_manifest, llm)
            if review_result["status"] == "PASS":
                return story_text  # 完美通过，直接返回
            
            advice = review_result.get("revision_advice", "违反了设定红线，请修正。")

        # 处理重试逻辑
        if attempt < max_retries:
            logging.info(f"Review: {RETRY_MESSAGE} (第 {attempt} 次重试)")
            logging.info(f"Review: 注入修改建议: {advice}")

            # 动态将审核修改意见注入到参数中，兼容 generate_story 和 expand_story 的签名
            if "cot_outline" in story_kwargs:
                story_kwargs["cot_outline"] = str(story_kwargs["cot_outline"]) + f"\n\n[上一轮审查被打回 - 请修正以下问题]: {advice}"
            elif "si_result" in story_kwargs: # 针对 expand_story
                si_result = story_kwargs["si_result"]
                si_result["instruction"] = str(si_result.get("instruction", "")) + f"\n\n[上一轮审查被打回 - 请修正以下问题]: {advice}"
            elif "user_input" in story_kwargs:
                story_kwargs["user_input"] = str(story_kwargs["user_input"]) + f"\n\n[上一轮审查被打回 - 请修正以下问题]: {advice}"
        else:
            logging.warning(f"Review: 已达最大重试次数 {max_retries}，系统妥协，返回最后一次生成结果。")

    return story_text