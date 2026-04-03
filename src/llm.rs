//! OpenAI-compatible chat client via `async_openai`.

use std::sync::Arc;

use anyhow::{Context, Result};
use async_openai::config::OpenAIConfig;
use async_openai::types::{
    ChatCompletionRequestAssistantMessage, ChatCompletionRequestAssistantMessageContent,
    ChatCompletionRequestMessage, ChatCompletionRequestSystemMessageArgs,
    ChatCompletionRequestToolMessageArgs, ChatCompletionRequestUserMessageArgs, ChatCompletionTool,
    ChatCompletionToolChoiceOption, CreateChatCompletionRequestArgs, FinishReason,
};
use async_openai::Client;

/// Resolved LLM settings from environment and CLI overrides.
#[derive(Debug, Clone)]
pub struct LlmConfig {
    pub api_key: String,
    pub api_base: Option<String>,
    pub model: String,
}

/// Limits for the tool-calling loop.
#[derive(Debug, Clone)]
pub struct ToolLoopConfig {
    pub max_rounds: u32,
    pub max_tool_calls: u32,
    pub max_tool_output_chars: usize,
    pub max_context_tool_results: usize,
}

impl Default for ToolLoopConfig {
    fn default() -> Self {
        Self {
            max_rounds: 256,
            max_tool_calls: 512,
            max_tool_output_chars: 128_000,
            max_context_tool_results: 16,
        }
    }
}

impl LlmConfig {
    /// Build config from optional CLI values and environment defaults.
    pub fn from_cli(
        api_key: Option<String>,
        api_base: Option<String>,
        model: Option<String>,
    ) -> Result<Self> {
        let api_key = api_key
            .or_else(|| std::env::var("OPENAI_API_KEY").ok())
            .context("OPENAI_API_KEY is required (or pass --api-key)")?;

        let model = model
            .or_else(|| std::env::var("OPENAI_MODEL").ok())
            .unwrap_or_else(|| "gpt-5-nano".to_string());

        let api_base = api_base.or_else(|| std::env::var("OPENAI_API_BASE").ok());

        Ok(Self {
            api_key,
            api_base,
            model,
        })
    }

    fn openai_config(&self) -> OpenAIConfig {
        let mut cfg = OpenAIConfig::new().with_api_key(self.api_key.clone());
        if let Some(ref base) = self.api_base {
            cfg = cfg.with_api_base(base);
        }
        cfg
    }

    /// Create an async_openai client for this configuration.
    pub fn client(&self) -> Client<OpenAIConfig> {
        Client::with_config(self.openai_config())
    }

    /// Run a chat completion with the given system and user messages.
    pub async fn complete(&self, system: &str, user: &str) -> Result<String> {
        let client = self.client();
        let request = CreateChatCompletionRequestArgs::default()
            .model(&self.model)
            .messages([
                ChatCompletionRequestSystemMessageArgs::default()
                    .content(system)
                    .build()?
                    .into(),
                ChatCompletionRequestUserMessageArgs::default()
                    .content(user)
                    .build()?
                    .into(),
            ])
            .build()?;

        let response = client
            .chat()
            .create(request)
            .await
            .context("chat completion request failed")?;

        let text = response
            .choices
            .first()
            .and_then(|c| c.message.content.clone())
            .filter(|s| !s.is_empty())
            .context("empty completion response")?;

        Ok(text)
    }

    /// Multi-turn chat with function tools. `run_tool` returns JSON text for each call.
    ///
    /// `tools` is wrapped in [`Arc`] so definitions can be shared; each API round still builds an
    /// owned [`Vec`] for the request body. The conversation `messages` vector is cloned each round
    /// because [`CreateChatCompletionRequest`](async_openai::types::CreateChatCompletionRequest)
    /// requires an owned history snapshot.
    pub async fn chat_with_tools(
        &self,
        messages: Vec<ChatCompletionRequestMessage>,
        tools: Arc<Vec<ChatCompletionTool>>,
        mut run_tool: impl FnMut(&str, &str) -> String,
        config: ToolLoopConfig,
    ) -> Result<String> {
        let client = self.client();
        let mut msgs = messages;
        let mut rounds: u32 = 0;
        let mut tool_calls_used: u32 = 0;
        loop {
            if rounds >= config.max_rounds {
                anyhow::bail!("tool loop exceeded max_rounds ({})", config.max_rounds);
            }
            rounds += 1;

            evict_older_tool_results(&mut msgs, config.max_context_tool_results);

            // Tool list is small; clone from Arc for the request struct (same cost as a Vec param).
            let request = CreateChatCompletionRequestArgs::default()
                .model(&self.model)
                .messages(msgs.clone())
                .tools(tools.as_ref().clone())
                .tool_choice(ChatCompletionToolChoiceOption::Auto)
                .build()?;

            let response = client
                .chat()
                .create(request)
                .await
                .context("chat completion (tools) failed")?;

            let choice = response
                .choices
                .first()
                .context("no choices in completion response")?;
            let msg = &choice.message;

            if let Some(tool_calls) = &msg.tool_calls {
                if tool_calls.is_empty() {
                    let text = msg.content.clone().unwrap_or_default();
                    if !text.is_empty() {
                        return Ok(text);
                    }
                    anyhow::bail!("empty tool_calls from model");
                }

                let assistant = ChatCompletionRequestMessage::Assistant(
                    ChatCompletionRequestAssistantMessage {
                        content: msg
                            .content
                            .clone()
                            .map(ChatCompletionRequestAssistantMessageContent::Text),
                        tool_calls: Some(tool_calls.clone()),
                        ..Default::default()
                    },
                );
                msgs.push(assistant);

                for call in tool_calls {
                    if tool_calls_used >= config.max_tool_calls {
                        anyhow::bail!("exceeded max_tool_calls ({})", config.max_tool_calls);
                    }
                    tool_calls_used += 1;
                    let name = &call.function.name;
                    let args = &call.function.arguments;
                    let mut out = run_tool(name, args);
                    if out.len() > config.max_tool_output_chars {
                        out = out
                            .chars()
                            .take(config.max_tool_output_chars)
                            .collect::<String>();
                        out.push_str("\n[tool output truncated]\n");
                    }
                    let tool_msg = ChatCompletionRequestToolMessageArgs::default()
                        .tool_call_id(call.id.clone())
                        .content(out)
                        .build()?
                        .into();
                    msgs.push(tool_msg);
                }
                continue;
            }

            let finish = choice.finish_reason;
            let text = msg.content.clone().unwrap_or_default();
            if finish == Some(FinishReason::ToolCalls) && text.is_empty() {
                anyhow::bail!("finish_reason tool_calls but no tool_calls in message");
            }
            if !text.is_empty() {
                return Ok(text);
            }
            anyhow::bail!("empty assistant message without tool calls");
        }
    }
}

pub(crate) fn evict_older_tool_results(msgs: &mut [ChatCompletionRequestMessage], max_keep: usize) {
    let mut kept_tools = 0;
    for msg in msgs.iter_mut().rev() {
        if let ChatCompletionRequestMessage::Tool(t) = msg {
            kept_tools += 1;
            if kept_tools > max_keep {
                if let Ok(new_t) = ChatCompletionRequestToolMessageArgs::default()
                    .tool_call_id(t.tool_call_id.clone())
                    .content("[Result evicted to save context]")
                    .build()
                {
                    *msg = new_t.into();
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eviction_keeps_recent_tools_and_replaces_old() {
        let mut msgs = vec![
            ChatCompletionRequestToolMessageArgs::default()
                .tool_call_id("call_1")
                .content("first tool result")
                .build()
                .unwrap()
                .into(),
            ChatCompletionRequestToolMessageArgs::default()
                .tool_call_id("call_2")
                .content("second tool result")
                .build()
                .unwrap()
                .into(),
            ChatCompletionRequestToolMessageArgs::default()
                .tool_call_id("call_3")
                .content("third tool result")
                .build()
                .unwrap()
                .into(),
        ];

        evict_older_tool_results(&mut msgs, 2);

        // Third and Second should be kept (most recent)
        // First should be evicted
        if let ChatCompletionRequestMessage::Tool(t) = &msgs[0] {
            if let async_openai::types::ChatCompletionRequestToolMessageContent::Text(text) =
                &t.content
            {
                assert_eq!(text, "[Result evicted to save context]");
            } else {
                panic!("Expected context text");
            }
        } else {
            panic!("Expected Tool message");
        }

        if let ChatCompletionRequestMessage::Tool(t) = &msgs[1] {
            if let async_openai::types::ChatCompletionRequestToolMessageContent::Text(text) =
                &t.content
            {
                assert_eq!(text, "second tool result");
            } else {
                panic!("Expected context text");
            }
        }

        if let ChatCompletionRequestMessage::Tool(t) = &msgs[2] {
            if let async_openai::types::ChatCompletionRequestToolMessageContent::Text(text) =
                &t.content
            {
                assert_eq!(text, "third tool result");
            } else {
                panic!("Expected context text");
            }
        }
    }
}
