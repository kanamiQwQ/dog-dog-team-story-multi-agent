import logging
import json
from langchain_openai import ChatOpenAI

REVIEW_PROMPT = """
# Role: 汪汪队剧本最终质量审查官 (Strict QA Reviewer)

## Core Objective
你是剧本输出给用户前的最后一道防线。你的任务是极其严苛地审查由下游模型生成的《汪汪队》故事正文，确保它绝对没有违反任何IP红线设定和物理逻辑。你必须像一个无情的判官，对任何"大模型幻觉"零容忍。

## Input Context
- 【本次出勤简报 (DEC设定)】: {mission_manifest}
- 【待审查的故事正文】:
{generated_story}

## Inspection Checklist (审查红线清单)

请逐条对照以下红线进行排查：
1. **格式审查**：是否严格包含了 6 个指定的 `####` 标题？正文中是否混入了任何括号包裹的指令说明（如"(只写故事正文...)"）？
2. **装备与人设审查（最高危）**：狗狗们是否严格且仅仅使用了【出勤简报】中分配的装备？
   - 绝对禁止越权使用装备！（例如：阿奇绝不能用水炮/灭火，毛毛绝不能飞，小砾绝不能潜水）。
   - 莱德是否亲自参与了体力救援？（莱德只能指挥和使用平板/全地形车，绝对不能动手）。
3. **物理与逻辑审查**：剧情是否出现了"魔法"或"超自然现象"？（比如：急救包里的东西无故变成蝴蝶、石头突然变成食物等）。二次危机是否符合基本的因果逻辑？
4. **氛围审查**：是否出现了受伤流血、暴力、死亡等儿童不宜的内容？结尾是否出现了成人的死板说教？

## Output Format (严格 JSON)
你必须且只能输出一个合法的 JSON 对象，用于给中控系统判定是否需要打回重写。绝对不要输出任何 Markdown 代码块标记（如 ```json）或解释性废话。

{{
  "status": "PASS" 或者 "FAIL",
  "violations": [
    "如果你判定为 FAIL，请在这里具体列出故事违反了上述哪一条红线（例如：'严重错误：阿奇使用了无人机喷水，违背了装备限制'）。如果为 PASS，此处为空数组 []"
  ],
  "revision_advice": "如果为 FAIL，请给上游的故事生成模型提供具体的修改指导（例如：'请重写意外高潮部分，将阿奇灭火改为使用无人机寻找灭火器，由毛毛来灭火'）。如果为 PASS，输出 null。"
}}
"""

MAX_RETRIES = 3


def review_story(generated_story: str, dec_manifest: dict, llm: ChatOpenAI) -> dict:
    logging.info("Review: 正在审查故事质量...")

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
            "revision_advice": "请重新生成故事"
        }

    status = result.get("status", "FAIL")
    violations = result.get("violations", [])
    revision_advice = result.get("revision_advice", None)

    if status == "PASS":
        logging.info("Review: ✅ 审查通过！")
    else:
        logging.warning(f"Review: ❌ 审查未通过！违规项: {violations}")
        logging.warning(f"Review: 修改建议: {revision_advice}")

    return {
        "status": status,
        "violations": violations,
        "revision_advice": revision_advice
    }


def review_with_retry(
    story_func,
    story_kwargs: dict,
    dec_manifest: dict,
    llm: ChatOpenAI,
    max_retries: int = MAX_RETRIES
) -> str:
    """
    带重试的审查流程：生成故事 → 审查 → 不通过则重试
    story_func: 生成故事的函数（如 generate_story 或 expand_story）
    story_kwargs: 传给 story_func 的关键字参数
    """
    for attempt in range(1, max_retries + 1):
        logging.info(f"Review [第 {attempt}/{max_retries} 次]: 生成故事并审查...")

        story_text = story_func(**story_kwargs)

        review_result = review_story(story_text, dec_manifest, llm)

        if review_result["status"] == "PASS":
            return story_text

        if attempt < max_retries:
            advice = review_result.get("revision_advice", "")
            logging.info(f"Review: llm又在瞎编了喵~再生成一次试试吧 (第 {attempt} 次重试)")
            logging.info(f"Review: 修改建议: {advice}")

            if advice:
                if "cot_outline" in story_kwargs:
                    story_kwargs["cot_outline"] = (
                        story_kwargs.get("cot_outline", "") +
                        f"\n\n[审查反馈 - 请修正以下问题]: {advice}"
                    )
                elif "instruction" in story_kwargs:
                    story_kwargs["instruction"] = (
                        story_kwargs.get("instruction", "") +
                        f"\n\n[审查反馈 - 请修正以下问题]: {advice}"
                    )
        else:
            logging.warning(f"Review: 已达最大重试次数 {max_retries}，返回最后一次结果")

    return story_text
