import type { ChatCompletionMessageParam } from '@fastgpt/global/core/ai/type.d';
import { getAIApi } from '../config';
import { countGptMessagesTokens } from '../../../common/string/tiktoken/index';
import { loadRequestMessages } from '../../chat/utils';
import { llmCompletionsBodyFormat } from '../utils';

export const Prompt_QuestionGuide = `你是一名人工智能助手，负责根据对话历史记录预测用户的下一个问题。你的目标是生成 3 个问题来引导用户继续对话。生成这些问题时，请遵循以下规则：
1. 使用中文。
2. 每个问题的长度不要超过 20 个字符。
分析提供给你的对话历史记录，并将其用作上下文来生成相关且引人入胜的后续问题。你的预测应该是用户有兴趣进一步探索的相关领域的逻辑扩展。
请记住与现有对话保持语气和风格的一致性，同时提供多种选项供用户选择。你的目标是保持对话自然流畅，并帮助用户更深入地探索相关主题。

以 JSON 格式返回问题：["question1", "question2", "question3"]，不要输出其他内容。`;

export async function createQuestionGuide({
  messages,
  model
}: {
  messages: ChatCompletionMessageParam[];
  model: string;
}) {
  const concatMessages: ChatCompletionMessageParam[] = [
    ...messages,
    {
      role: 'user',
      content: Prompt_QuestionGuide
    }
  ];

  const ai = getAIApi({
    timeout: 480000
  });
  const data = await ai.chat.completions.create(
    llmCompletionsBodyFormat(
      {
        model,
        temperature: 0.1,
        max_tokens: 200,
        messages: await loadRequestMessages({
          messages: concatMessages,
          useVision: false
        }),
        stream: false
      },
      model
    )
  );

  const answer = data.choices?.[0]?.message?.content || '';

  const start = answer.indexOf('[');
  const end = answer.lastIndexOf(']');

  const tokens = await countGptMessagesTokens(concatMessages);

  if (start === -1 || end === -1) {
    return {
      result: [],
      tokens: 0
    };
  }

  const jsonStr = answer
    .substring(start, end + 1)
    .replace(/(\\n|\\)/g, '')
    .replace(/  /g, '');

  try {
    return {
      result: JSON.parse(jsonStr),
      tokens
    };
  } catch (error) {
    return {
      result: [],
      tokens: 0
    };
  }
}
