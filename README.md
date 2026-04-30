# 苏州某小公司实习项目重构

# 项目介绍
- 一个multi-agent汪汪队故事生成器，用于生成特定场景/角色/事件的故事。
- 项目采用python3.11框架，使用模型为Qwen-3-14b-Instruct。
# 项目架构
- 本multi-agent分为5个agent，每个agent负责独立任务。
- SI（意图识别） -> DEC（角色选择） -> COT（思维炼生成） -> story（故事生成） -> review（故事审查）
- agent之间使用mcp协议进行通信。除review获取全部json参数，其余agent只获取最小参数及前一个agent的输出。
# 实现功能
- 对故事的全新生成/修改/润色/扩写。
- 在Qwen-3-14b-Instruct上进行3次review即为超时进行熔断，返回第三次生成的故事以及review给出的修改建议。
- 对于用户输入的文本，先进行意图识别，再根据意图选择对应的agent进行任务分配。若意图识别出参数未在系统中则返回"暂时无法听懂你说话喵，换个说法试试喵~"
# 项目接口
- 项目接口采用fast api，用于与前端进行交互。
- 默认启动运行在7707端口。如需修改可在SI.py中底部对"port=7707"进行修改。
- 对于模型选择以及端口和token的配置，在DEC.py中class Config中进行配置。
# 使用方法
- 当前目录下运行以下命令启动：
  ```bash
  python Sl.py
- 对于前端调用，设置接口地址为http://localhost:7707/，默认token为openai，如需修改可在DEC.py中class Config中进行配置。
# 项目对比
- 对比单agent项目，本项目在完整逻辑闭环上以及功能上都有较大幅提升。且额外增加熔断机制防止token爆炸。
- 对于token消耗比单agent项目仅增加5000token左右。触发3次review的情况下，额外消耗的token为10000token左右。
# 📄 许可证
- 本项目采用MIT许可证，您可以在遵守许可证条款的前提下自由使用、修改和分发本项目。
# 🤝 贡献
欢迎 Star / Fork / PR
# 后记
- 这个项目是我第一个multi-agent项目，如有不足以及问题或建议，欢迎提交issue或pull request。
- 沟槽的公司真黑奴啊，3000块还想拿到这套multi-agent。
