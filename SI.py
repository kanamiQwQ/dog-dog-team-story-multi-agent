import json
import logging
import re
import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from fastapi.concurrency import run_in_threadpool

from COT import (
    call_cot_module_for_continue,
    call_cot_module_for_new,
    call_cot_module_for_revise,
)
from DEC import (
    Config as DecConfig,
    answer_lore_query,
    get_lore_dict,
    prepare_lore_context,
    process_dec_module,
)
from story import expand_story, generate_story
from review import review_with_retry


SYSTEM_PROMPT = """你是汪汪队剧本创作助手 - 语义解析协议 (SI Protocol)

## 核心任务
你是系统的意图分发中枢。你的任务是将编剧的自然语言指令转化为结构化的指令流。你只负责"理解"和"分类"，严禁进行任何剧本创作、续写或设定解释。

## 意图协议 (Intent Schema)
你必须将用户的输入归类为以下五个互斥的意图之一：

1. DRAFT_NEW: 从零开始创作一个全新的故事大纲或剧本。
2. EXPAND_SCENE: 对当前剧本的某个特定片段进行细节扩充、动作描写丰富。
3. CONTINUE_PLOT: 顺着当前剧情继续往下写，不改变现有逻辑。
4. REVISE_LOGIC: 对已有的剧情进行结构化修改或逻辑推翻。
5. LORE_QUERY: 查询汪汪队的世界观设定、角色装备或历史剧情。

## 强制约束
- JSON 唯一性: 你必须且只能输出一个合法的 JSON 对象。
- 严禁废话: 严禁输出任何解释性文字、寒暄或 Markdown 代码块标签。
- 指代消解: 如果输入中包含"它"、"他"、"那里"，必须结合上下文还原为具体的角色名或地点名。
- 提取 Entities 时必须绝对忠于用户的原话！用户没点名的狗狗、没提到的地点，必须输出为空表 [] 或 null。绝不允许自行补充!
- 思考控制: 绝对不能出现 `<think>` 或 `</think>` 等思维链标记，只输出最终 JSON 结果。

## 输出格式 (Schema)
{
  "intent": "DRAFT_NEW | EXPAND_SCENE | CONTINUE_PLOT | REVISE_LOGIC | LORE_QUERY",
  "entities": {
    "pups": ["角色列表"],
    "tools": ["装备列表"],
    "location": "地点"
  },
  "instruction": "核心指令字符串"
}
"""


class Config:
    LLM_MODEL_NAME = DecConfig.LLM_MODEL_NAME
    OPENAI_API_KEY = DecConfig.OPENAI_API_KEY
    OPENAI_BASE_URL = DecConfig.OPENAI_BASE_URL
    DEFAULT_TEMPERATURE = 0.3


app = FastAPI(title="SI服务", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    stream: Optional[bool] = False


VALID_INTENTS = {
    "DRAFT_NEW",
    "EXPAND_SCENE",
    "CONTINUE_PLOT",
    "REVISE_LOGIC",
    "LORE_QUERY",
    "UNKNOWN",
}


def build_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=Config.LLM_MODEL_NAME,
        api_key=Config.OPENAI_API_KEY,
        base_url=Config.OPENAI_BASE_URL,
        temperature=Config.DEFAULT_TEMPERATURE,
    )


def _safe_json_loads(text: str) -> Dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        raise ValueError("empty json response")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", raw)
        if not match:
            raise
        return json.loads(match.group(0))


def _extract_last_story(messages: List[ChatMessage]) -> str:
    for message in reversed(messages):
        if message.role == "assistant" and message.content.strip():
            return message.content.strip()
    return ""


def _extract_entities(user_input: str) -> Dict[str, Any]:
    lore = get_lore_dict()
    pups = [name for name in lore if name in user_input]
    tools: List[str] = []
    for item in lore.values():
        for equipment in item.get("equipment", []):
            if equipment in user_input and equipment not in tools:
                tools.append(equipment)

    known_locations = [
        "冒险湾",
        "总部",
        "海边",
        "码头",
        "雪山",
        "丛林",
        "森林",
        "山谷",
        "工地",
        "桥上",
        "海上",
    ]
    location = None
    for candidate in known_locations:
        if candidate in user_input:
            location = candidate
            break

    return {
        "pups": pups,
        "tools": tools,
        "location": location,
    }


def _heuristic_intent(user_input: str) -> str:
    text = user_input.strip()
    if not text:
        return "UNKNOWN"
    if any(keyword in text for keyword in ["设定", "世界观", "角色", "装备", "技能", "资料", "介绍", "是谁"]):
        return "LORE_QUERY"
    if any(keyword in text for keyword in ["续写", "继续", "接着", "后面", "然后呢"]):
        return "CONTINUE_PLOT"
    if any(keyword in text for keyword in ["修改", "改成", "重写", "调整逻辑", "修正", "不合理"]):
        return "REVISE_LOGIC"
    if any(keyword in text for keyword in ["扩写", "扩充", "展开", "细化", "丰富描写"]):
        return "EXPAND_SCENE"
    return "DRAFT_NEW"


def _heuristic_si(user_input: str) -> Dict[str, Any]:
    return {
        "intent": _heuristic_intent(user_input),
        "entities": _extract_entities(user_input),
        "instruction": user_input.strip(),
    }


def _normalize_si_result(payload: Dict[str, Any], user_input: str) -> Dict[str, Any]:
    intent = payload.get("intent")
    if intent not in VALID_INTENTS:
        intent = "UNKNOWN"

    entities = payload.get("entities") or {}
    return {
        "intent": intent,
        "entities": {
            "pups": entities.get("pups") or [],
            "tools": entities.get("tools") or [],
            "location": entities.get("location"),
        },
        "instruction": payload.get("instruction") or user_input.strip(),
    }


def call_si(user_input: str, llm: ChatOpenAI, history_story: str = "") -> Dict[str, Any]:
    try:
        context_prompt = (
            SYSTEM_PROMPT
            + "\n\n对话上下文（如果没有则为空）：\n"
            + (history_story[:1000] or "无")
        )
        response = llm.invoke(
            [
                ("system", context_prompt),
                ("human", user_input),
            ]
        )
        return _normalize_si_result(_safe_json_loads(response.content), user_input)
    except Exception as exc:
        logging.warning("SI 解析失败，使用规则兜底: %s", exc)
        return _normalize_si_result(_heuristic_si(user_input), user_input)


def _build_manifest_from_history(si_result: Dict[str, Any], history_story: str) -> Dict[str, Any]:
    lore = get_lore_dict()
    names = si_result.get("entities", {}).get("pups", []) or [name for name in lore if name in history_story]
    if not names:
        names = ["莱德", "阿奇", "毛毛"]
    if "莱德" not in names:
        names.insert(0, "莱德")

    roster = []
    for name in names:
        item = lore.get(name)
        if not item:
            continue
        roster.append(
            {
                "name": name,
                "equipment": item["equipment"][0],
                "task_assignment": "延续上一轮任务继续救援" if name != "莱德" else "现场总指挥",
            }
        )

    return {
        "primary_objective": si_result.get("instruction") or "延续上一轮剧情",
        "final_roster": roster,
    }


def run_pipeline(request: ChatRequest) -> str:
    user_input = request.messages[-1].content.strip()
    history_story = _extract_last_story(request.messages[:-1])
    llm = build_llm()
    si_result = call_si(user_input, llm, history_story)
    intent = si_result["intent"]

    if intent == "UNKNOWN":
        return "暂时无法分析你的意图，请换个说法试试喵~"

    if intent == "DRAFT_NEW":
        logging.info("命中 DRAFT_NEW 路由。")
        logging.info("角色知识查询结果：%s", prepare_lore_context(si_result))
        mission_manifest = process_dec_module(si_result, llm)
        cot_outline = call_cot_module_for_new(mission_manifest, llm)
        story_text = generate_story(cot_outline, mission_manifest, user_input, llm, mode="DRAFT_NEW")
# ... 
        if intent == "DRAFT_NEW":
        # ... 前置代码不变
            return review_with_retry(
                intent=intent,
                story_func=generate_story,
                story_kwargs={
                    "cot_outline": cot_outline, 
                    "mission_manifest": mission_manifest, 
                    "user_input": user_input, 
                    "llm": llm, 
                    "mode": "DRAFT_NEW"
                    },
                    dec_manifest=mission_manifest,
                    llm=llm,
                    original_request=user_input
                )

    if intent == "CONTINUE_PLOT":
        logging.info("命中 CONTINUE_PLOT 路由。")
        if not history_story:
            return "没有找到上一轮故事，暂时无法续写喵~"
        mission_manifest = _build_manifest_from_history(si_result, history_story)
        cot_outline = call_cot_module_for_continue(history_story, si_result["instruction"], llm)
        story_text = generate_story(
            cot_outline,
            mission_manifest,
            user_input,
            llm,
            history_story=history_story,
            mode="CONTINUE_PLOT",
        )
        return review_with_retry(
                intent=intent,
                story_func=generate_story,
                story_kwargs={
                    "cot_outline": cot_outline, 
                    "mission_manifest": mission_manifest, 
                    "user_input": user_input, 
                    "llm": llm, 
                    "mode": "CONTINUE_PLOT",
                    "history_story": history_story
                    },
                    dec_manifest=mission_manifest,
                    llm=llm,
                    original_request=user_input
                )       

    if intent == "REVISE_LOGIC":
        logging.info("命中 REVISE_LOGIC 路由。")
        if not history_story:
            return "没有找到上一轮故事，暂时无法修改逻辑喵~"
        mission_manifest = process_dec_module(si_result, llm)
        cot_outline = call_cot_module_for_revise(
            history_story,
            si_result["instruction"],
            mission_manifest,
            llm,
        )
        story_text = generate_story(
            cot_outline,
            mission_manifest,
            user_input,
            llm,
            history_story=history_story,
            mode="REVISE_LOGIC",
        )
        return review_with_retry(
                intent=intent,
                story_func=generate_story,
                story_kwargs={
                    "cot_outline": cot_outline, 
                    "mission_manifest": mission_manifest, 
                    "user_input": user_input, 
                    "llm": llm, 
                    "mode": "REVISE_LOGIC",
                    "history_story": history_story
                    },
                    dec_manifest=mission_manifest,
                    llm=llm,
                    original_request=user_input
                )

    if intent == "EXPAND_SCENE":
        logging.info("命中 EXPAND_SCENE 路由。")
        if not history_story:
            return "没有找到上一轮故事，暂时无法扩写场景喵~"
        story_text = expand_story(history_story, si_result, user_input, llm)
        return review_with_retry(
                intent=intent,
                story_func=expand_story,
                story_kwargs={
                    "history_story": history_story, 
                    "si_result": si_result, 
                    "user_input": user_input, 
                    "llm": llm
                    },
                    dec_manifest=mission_manifest,
                    llm=llm,
                    original_request=user_input
                )

    if intent == "LORE_QUERY":
        logging.info("命中 LORE_QUERY 路由。")
        lore_answer = answer_lore_query(user_input, si_result["entities"])
        return review_with_retry(
                intent=intent,
                story_func=answer_lore_query,
                story_kwargs={      
                    "user_input": user_input, 
                    "entities": si_result["entities"]
                    },
                    dec_manifest=mission_manifest,
                    llm=llm,
                    original_request=user_input
                )

    return "暂时无法听懂你说话喵，换个说法试试喵~"


def _build_completion_response(content: str, model_name: str) -> Dict[str, Any]:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


@app.get("/v1/models")
def list_models() -> Dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": Config.LLM_MODEL_NAME,
                "object": "model",
                "created": 1713680000,
                "owned_by": "local-server",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    if not request.messages:
        return _build_completion_response("请输入内容", request.model or Config.LLM_MODEL_NAME)

    user_input = request.messages[-1].content.strip()
    if not user_input:
        return _build_completion_response("请输入内容", request.model or Config.LLM_MODEL_NAME)

    # 【关键修改】使用线程池执行阻塞任务
    final_text = await run_in_threadpool(run_pipeline, request)
    model_name = request.model or Config.LLM_MODEL_NAME

    if request.stream:
        async def generate_stream():
            chat_id = f"chatcmpl-{uuid.uuid4().hex}"
            created_time = int(time.time())
            stream_data = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": model_name,
                "choices": [{"index": 0, "delta": {"content": final_text}}],
            }
            yield f"data: {json.dumps(stream_data, ensure_ascii=False)}\n\n"
            stop_data = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": model_name,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(stop_data, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(generate_stream(), media_type="text/event-stream")

    return _build_completion_response(final_text, model_name)


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    uvicorn.run(app, host="0.0.0.0", port=7707)
