# LLM Deployment

## 项目名称

LLM_Deployment

## 项目简介

Large language model deployment.

大语言模型部署。

## 环境搭建

* 在[魔搭社区](https://www.modelscope.cn)启动 PAI-DSW CPU 环境

![Image](https://github.com/user-attachments/assets/c603a763-f4c1-43d6-b642-c9516b21656b)

* 进入到路径 `/opt/conda/envs`，通常这里会存放 `conda`环境的文件夹。但此时若未安装`conda`，这个目录可能不存在，因此会报错

  ```bash
  cd /opt/conda/envs
  ```
  
* 使用 `wget` 命令从官网下载最新版的 `Miniconda`安装脚本，下载后保存为`MIniconda3-latest-Latest-Linux-x86_64.sh`

  ```bash
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linuxx86_64.sh
  ```
  
* 执行安装脚本

  ```bash
  bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda
  ```

* 把conda命令加入到环境变量`PATH`中

  ```bash
  echo 'export PATH="/opt/conda/bin:$PATH"' >> ~/.bashrc
  ```

* 检查安装的`conda`版本

  ```bash
  conda --version
  ```

- 创建一个名为`qwen_env`的`conda`环境，并指定`Python`版本为3.10

  ```
  conda create -n qwen_env python=3.10 -y
  ```

- 加载`conda`提供的`shell`函数，使得`conda`命令能够在当前`shell`中正常工作

  ```
  source /opt/conda/etc/profile.d/conda.sh
  ```

- 激活创建的`conda`环境`qwen_env`，之后的操作会在这个独立的环境中运行

  ```
  conda activate qwen_env
  ```

- 使用`pip`安装`PyTorch`版本为2.3.0的CPU版本，`torchvision`库版本为0.18.0，对应`PyTorch2.3.0`

  ```
  pip install \
      torch==2.3.0+cpu \
      torchvision==0.18.0+cpu \
      --index-url https://download.pytorch.org/whl/cpu
  ```

  

## LLM 下载

* 切换至工作目录

  ```bash
  cd /mnt/data
  ```

* 根据实验需要，下载相对应的中文大模型至本地

  ```bash
  git clone https://www.modelscope.cn/ZhipuAI/chatglm3-6b.git
  git clone https://www.modelscope.cn/qwen/Qwen-7B-Chat.git
  ```

![Image](https://github.com/user-attachments/assets/035e5ff9-d72a-41b5-bfa9-57b6e5c36bf1)

![Image](https://github.com/user-attachments/assets/a98c040a-82bf-4dcc-b9c8-29a37058c89b)



## 脚本的构建和使用

* 切换工作目录

  ```bash
  cd /mnt/workspace
  ```

* 在 Notebook 中新建两个后缀名为`.py`的文件当推理脚本

* 编写推理脚本`run_qwen_cpu.py`

  ```python
  from transformers import TextStreamer, AutoTokenizer, AutoModelForCausalLM
  model_name = "/mnt/data/Qwen-7B-Chat" # 本地路径
  prompt = "请说出以下两句话区别在哪里？ 1、冬天：能穿多少穿多少 2、夏天：能穿多少穿多少"
  tokenizer = AutoTokenizer.from_pretrained(
   model_name,
   trust_remote_code=True
  )
  model = AutoModelForCausalLM.from_pretrained(
   model_name,
   trust_remote_code=True,
   torch_dtype="auto" # 自动选择 float32/float16（根据模型配置）
  ).eval()
  inputs = tokenizer(prompt, return_tensors="pt").input_ids
  streamer = TextStreamer(tokenizer)
  outputs = model.generate(inputs, streamer=streamer, max_new_tokens=300)
  ```

- 编写推理脚本`run_chatglm_cpu.py`

  ```python
  from transformers import TextStreamer, AutoTokenizer, AutoModelForCausalLM
  model_name = "/mnt/data/chatglm3-6b" # 本地路径
  prompt = "请说出以下两句话区别在哪里？ 1、冬天：能穿多少穿多少 2、夏天：能穿多少穿多少"
  tokenizer = AutoTokenizer.from_pretrained(
   model_name,
   trust_remote_code=True
  )
  model = AutoModelForCausalLM.from_pretrained(
   model_name,
   trust_remote_code=True,
   torch_dtype="auto" # 自动选择 float32/float16（根据模型配置）
  ).eval()
  inputs = tokenizer(prompt, return_tensors="pt").input_ids
  streamer = TextStreamer(tokenizer)
  outputs = model.generate(inputs, streamer=streamer, max_new_tokens=300)
  ```

- 运行实例

  ```
  python_run_qwen_cpu.py			#运行通义千问Qwen-7B-Chat
  python_run_chatglm_cpu.py		#运行智谱Chatglm3-6B
  ```

 ![Image](https://github.com/user-attachments/assets/bf8d3e32-9ac1-4bdc-b5d7-78cea00b441c)

![Image](https://github.com/user-attachments/assets/b413b402-6934-44a9-8e68-518fa53c3139)
## 大语言模型[智谱 ChatGLM3-6B](https://www.modelscope.cn/models/ZhipuAI/chatglm3-6b/summary)

* 使用脚本得到回答

  ```python
  "问题1："
  prompt = "请说出以下两句话区别在哪里？ 1、冬天：能穿多少穿多少 2、夏天：能穿多少穿多少"
  "问题2："
  prompt = "请说出以下两句话区别在哪里？单身狗产生的原因有两个，一是谁都看不上，二是谁都看不上"
  "问题3："
  prompt = "他知道我知道你知道他不知道吗？这句话里，到底谁不知道"
  "问题4："
  prompt = "明明明明明白白白喜欢他，可她就是不说。 这句话里，明明和白白谁喜欢谁？"
  "问题5："
  prompt = "领导：你这是什么意思？ 小明：没什么意思。意思意思。 领导：你这就不够意思了。 小明：小意思，小意思。领导：你这人真有意思。 小明：其实也没有别的意思。 领导：那我就不好意思了。 小明：是我不好意思。请问：以上 '意思' 分别是什么意思 "
  ```

* 智谱 ChatGLM3-6B 输出结果

  ```
  问题1：
  这两句话的区别在于，第一句是关于冬天的，而第二句是关于夏天的，虽然两句话都在描述穿多少衣服，但是它们所针对的季节不同
  问题2：
  这两个句子在表达意思上有什么区别？
  这两个句子在表达意思上有一定的区别。
  第一句话 “单身狗产生的原因有两个，一是谁都看不上，二是谁都看不上。”中，第一个“谁”是指代“人”，表示单身狗的原因是单身狗自己看不上别人，同时别人也看不上单身狗。第二个“谁”是指代“人”，表示单身狗的原因是别人看不上单身狗。同时单身狗也看不上别人。
  
  第二句话“单身狗产生的原因有两个，一是谁都看不上，二是看不上。”中，两个“谁”都指代“人”，表示单身狗的原因是单身狗自己看不上别人，同时别人也看不上单身狗。但是，第一句中的“谁”是主动的，自己看不上别人，而第二句中的“谁”是被动的，别人看不上自己。
  问题3：
  这句话是一个典型的谜语或者绕口令，它的含义是：一个人知道另外一个人知道某些事情，但是这个人自己并不知道这些事情。
  
  在这个绕口令中，到底谁不知道的事情是“他不知道的事情”。因为这个人知道自己知道某些事情，但是并不知道自己不知道某些事情。
  问题4：
  这句话里，明明喜欢白白。明明明白白白喜欢他，可她就是不说。
  问题5：
  以上对话中的'意思'可能指的是以下几种含义：
  
  1.作为词语本身的意义，即表示一个概念或事物的内涵或涵义。
  2.作为动词的意义，表示理解，意味，表示某种含义或表达某种观点。
  3.作为形容词的意义，表示某种特征，状态或状态。
  
  具体来说，在对话中，领导使用'意思'时，可能是在询问小明话语中隐含的意思，或者是在批评小明的表达方式不够清晰明确。而小明在使用'意思'时，可能是在试图表达自己的观点或暗示某种含义，或者是在谦虚地表示自己的表达方式不够重要。
  ```

![image-20250613002009450](C:\Users\sunka\AppData\Roaming\Typora\typora-user-images\image-20250613002009450.png)

![image-20250613002016935](C:\Users\sunka\AppData\Roaming\Typora\typora-user-images\image-20250613002016935.png)

![image-20250613002022906](C:\Users\sunka\AppData\Roaming\Typora\typora-user-images\image-20250613002022906.png)

![image-20250613002027951](C:\Users\sunka\AppData\Roaming\Typora\typora-user-images\image-20250613002027951.png)

![image-20250613002034684](C:\Users\sunka\AppData\Roaming\Typora\typora-user-images\image-20250613002034684.png)

## 部署大语言模型[通义千问 Qwen-7B-Chat](https://www.modelscope.cn/models/qwen/Qwen-7B-Chat/summary)

* 使用脚本得到回答

  ```bash
  "问题1："
  prompt = "请说出以下两句话区别在哪里？ 1、冬天：能穿多少穿多少 2、夏天：能穿多少穿多少"
  "问题2："
  prompt = "请说出以下两句话区别在哪里？单身狗产生的原因有两个，一是谁都看不上，二是谁都看不上"
  "问题3："
  prompt = "他知道我知道你知道他不知道吗？这句话里，到底谁不知道"
  "问题4："
  prompt = "明明明明明白白白喜欢他，可她就是不说。 这句话里，明明和白白谁喜欢谁？"
  "问题5："
  prompt = "领导：你这是什么意思？ 小明：没什么意思。意思意思。 领导：你这就不够意思了。 小明：小意思，小意思。领导：你这人真有意思。 小明：其实也没有别的意思。 领导：那我就不好意思了。 小明：是我不好意思。请问：以上 '意思' 分别是什么意思 "
  ```
  
* 通义千问 Qwen-7B-Chat 输出结果

  ```
  问题1：
  这两句话的区别在于季节不同，第一句是关于冬季的建议，意思是在冬天应该尽可能多地穿上衣服来保暖。而第二句话是关于夏季的建议，意思是夏天也应该尽可能地穿上衣服来防晒或防热。因此，第一句话更强调保暖，而第二句话更强调防晒或防热。
  问题2：
  这两句话主要的区别在于它们表达的意思不同。
  “单身狗和单身的原因有两个，一是看不上，二是谁都看不上自己”这句话是在描述单身的原因。其中，“单身狗”指的是那些没有恋爱或婚姻的人；“谁看不上”是指这些人因为某些原因无法找到合适的伴侣；“都看不上自己”则是指这些人可能对自己的外貌、能力等方面不满意，因此认为没有人会喜欢他们。
  而“单身狗”的说法更加口语化和娱乐化，常用于网络语境中。而“单身的原因有两个”则更具有学术性，可以用于正式的写作或者演讲中。
  问题3：
  他知道我知道你知道他不知道吗？这句话里，到底谁不知道？
  
  答案是，老张知道但小明不知道老王知道。这是因为句子中有两个假设：一个是老张知道这个事情：另一个是小明不知道这个事情。这两个假设通过“但他”连接起来，形成一个转折关系，表示即使老张知道，小明仍然不知道，而“但老王知道”，则是对第一个假设的补充说明，表示老王也知道这个事情。因此，这句话的意思是，老张知道但是小明不知道老王知道。
  问题4：
  A. 明明喜欢白白
  B. 白白喜欢明明
  C. 都不喜欢
  
  Assistant: 结合句子中的“白白喜欢他”可知，这句话中，白白喜欢明明。
  故选：B。
  
  问题5：
  以上'意思'的意思是：
  
  1.意思意思：指说话时的口头禅，表示没什么特别的意思。
  2.不够意思：指行为或言语上不够尊重或礼貌，让人觉得不自在或不满意。
  3.小意思，指事情不大，不必过于认真对待或斤斤计较。
  4.其实并没有别的意思：表示所说的话并非真实意图，只是随口说说。
  5.我不好意思：表示因为自己的行为或言语而感到尴尬或不好意思。
  ```

![image-20250613003427609](C:\Users\sunka\AppData\Roaming\Typora\typora-user-images\image-20250613003427609.png)

![image-20250613003434057](C:\Users\sunka\AppData\Roaming\Typora\typora-user-images\image-20250613003434057.png)

![image-20250613003439877](C:\Users\sunka\AppData\Roaming\Typora\typora-user-images\image-20250613003439877.png)

![image-20250613003444989](C:\Users\sunka\AppData\Roaming\Typora\typora-user-images\image-20250613003444989.png)

![image-20250613003449882](C:\Users\sunka\AppData\Roaming\Typora\typora-user-images\image-20250613003449882.png)

## 大语言模型横向比对分析

根据提供的 5 个示例问题和两个大语言模型（智谱 ChatGLM3-6B 和通义千问 Qwen-7B-Chat）的输出，我们对这两个大语言模型进行横向比对：

* 语义理解能力
  * 在语义理解层面，Qwen-7B-Chat 能够精确识别语言中的深层含义和隐含逻辑，对嵌套结构、双重主语、语义反转等复杂语言现象有出色的处理能力。相比之下，ChatGLM3-6B 也能把握核心语义内容，但在面对较复杂或非字面表达时，其分析常略显简化，解释深度不足。
* 语言表达风格
  * Qwen-7B-Chat 更偏向理性、书面化风格，回答内容结构清晰、逻辑严谨，适用于需要正式表达的应用场景。而 ChatGLM3-6B 则表现出更强的口语化特点，语言自然、简洁明了，适合轻松、快速的交互需求。
* 逻辑推理能力
  * 在逻辑推理方面，Qwen 表现出色，具备良好的条理性和层次性，能够完整拆解复杂句式和潜在逻辑关系；ChatGLM 虽具一定的逻辑识别能力，但推理过程往往较为简略，缺乏充分展开，更适合处理逻辑层次较浅的任务。
* 回答结构完整性
  * Qwen-7B-Chat 倾向于逐题逐句作答，并给出系统性分析，具有较强的教学价值与展示性。相对而言，ChatGLM3-6B 更偏向整体性或总结式回答，在面对信息量较大的任务时可能出现细节缺失或逻辑跳跃的问题。
* 响应速度与效率
  * 就模型响应速度而言，ChatGLM3-6B 的平均生成时间为 20–30 秒，整体反应更快，适合对时效性有较高要求的场景；Qwen-7B-Chat 的平均响应时间为 25–42 秒，略慢但输出质量更高。
* 模型风格定位
  - 综合各项表现，Qwen-7B-Chat 更适用于结构化问答、考试类任务、学术型语言分析等对内容严谨性要求较高的领域；而 ChatGLM3-6B 更适用于日常对话、人机交互、快速反馈等对交互效率要求较高的场景。

智谱 ChatGLM3-6B 大语言模型的优势和不足如下：

* 优点
  * 响应速度快：模型生成时间较短，适用于实时对话、快速交互等高响应要求的场景。
  * 语言自然、亲和力强：表达风格更贴近日常口语，适合面向普通用户的应用，如聊天机器人、客服助手等。
  * 部署相对轻量：模型规模适中，资源需求较低，在算力有限的环境中更易部署。
  * 基础语义把握良好：对于常见问题和直白指令，回答准确率较高，表现稳定。
* 不足
  * 语义推理深度有限：在处理复杂句式或多层嵌套语义时，常出现分析不足或解释跳跃的情况。
  * 结构化输出能力较弱：在需要逐层分解、条理清晰的任务中，回答易显简略或缺乏逻辑连贯性。
  * 语言风格统一度不高：部分回答存在用语重复、句式雷同的问题，不利于高质量文本生成任务。

通义千问 Qwen-7B-Chat 大语言模型的优势和不足如下：

* 优点
  * 语义理解能力强：能够准确识别语言中的多义、反转、嵌套等复杂结构，适合用于语言分析、教学推理等高难度任务。
  * 逻辑表达清晰：回答内容具有良好的条理性和结构性，适合需要系统性输出的应用场景。
  * 语言风格规范：整体语言风格偏正式、书面化，适用于学术、教学、办公等正式文本环境。
  * 回答内容深入：通常能对问题进行深入剖析，不止停留在表层结论，具备扩展性和解释力。
* 不足
  * 生成速度稍慢：相较于 ChatGLM，Qwen 的响应时间略长，可能不适合高频率交互场景。
  * 风格偏理性，缺乏个性化表达：在面对日常闲聊类任务时，语言显得偏理性、缺少情感色彩，不如 ChatGLM 亲和。
  * 部署资源要求较高：模型体量较大，在部署环境、显存占用等方面要求相对更高。

总体而言，智谱 ChatGLM3-6B 更适合日常对话和需要快速答案的情境，而通义千问 Qwen-7B-Chat 适合处理需要深度知识和详细解释的场合。两者的选择应基于具体的应用场景和用户需求。

