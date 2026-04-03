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
    pub max_context_chars: usize,
}

impl Default for ToolLoopConfig {
    fn default() -> Self {
        Self {
            max_rounds: 128,
            max_tool_calls: 512,
            max_tool_output_chars: 128_000,
            max_context_tool_results: 16,
            max_context_chars: 128_000,
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
            .unwrap_or_else(|| "gpt-5.4-nano".to_string());

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

            if estimate_context_size(&msgs) > (config.max_context_chars * 3 / 4) {
                tracing::info!("Context threshold reached; summarizing older history...");
                self.summarize_history(&mut msgs).await?;
            }

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

    pub async fn summarize_history(
        &self,
        msgs: &mut Vec<ChatCompletionRequestMessage>,
    ) -> Result<()> {
        // We need enough messages to make summarization worthwhile
        if msgs.len() < 10 {
            return Ok(());
        }

        // Keep System (0), Initial User (1), and the last 4 turns.
        let keep_front = 2;
        let keep_back = 4;
        let summarize_slice = &msgs[keep_front..msgs.len() - keep_back];

        let summary_text = self.summarize_messages(summarize_slice).await?;

        let summary_msg = ChatCompletionRequestSystemMessageArgs::default()
            .content(format!(
                "[SUMMARY of previous interactions to save space]:\n\n{summary_text}"
            ))
            .build()?
            .into();

        let mut new_msgs = Vec::with_capacity(keep_front + 1 + keep_back);
        new_msgs.push(msgs[0].clone());
        new_msgs.push(msgs[1].clone());
        new_msgs.push(summary_msg);
        new_msgs.extend(msgs[msgs.len() - keep_back..].iter().cloned());

        *msgs = new_msgs;
        Ok(())
    }

    async fn summarize_messages(
        &self,
        messages: &[ChatCompletionRequestMessage],
    ) -> Result<String> {
        let mut history_text = String::new();
        for m in messages {
            match m {
                ChatCompletionRequestMessage::Assistant(a) => {
                    history_text.push_str("Assistant: ");
                    if let Some(content) = &a.content {
                        match content {
                            ChatCompletionRequestAssistantMessageContent::Text(t) => {
                                history_text.push_str(t)
                            }
                            _ => {
                                history_text.push_str("[Non-text content]");
                            }
                        }
                    }
                    if let Some(calls) = &a.tool_calls {
                        for call in calls {
                            history_text.push_str(&format!(
                                "\nTool call: {} with args: {}\n",
                                call.function.name, call.function.arguments
                            ));
                        }
                    }
                    history_text.push('\n');
                }
                ChatCompletionRequestMessage::Tool(t) => {
                    history_text.push_str("Tool Output: ");
                    match &t.content {
                        async_openai::types::ChatCompletionRequestToolMessageContent::Text(
                            text,
                        ) => history_text.push_str(text),
                        async_openai::types::ChatCompletionRequestToolMessageContent::Array(
                            parts,
                        ) => {
                            for part in parts {
                                let async_openai::types::ChatCompletionRequestToolMessageContentPart::Text(tp) = part;
                                history_text.push_str(&tp.text);
                            }
                        }
                    }
                    history_text.push('\n');
                }
                ChatCompletionRequestMessage::User(u) => {
                    history_text.push_str("User: ");
                    match &u.content {
                        async_openai::types::ChatCompletionRequestUserMessageContent::Text(t) => {
                            history_text.push_str(t)
                        }
                        async_openai::types::ChatCompletionRequestUserMessageContent::Array(a) => {
                            for item in a {
                                if let async_openai::types::ChatCompletionRequestUserMessageContentPart::Text(t) = item {
                                    history_text.push_str(&t.text);
                                }
                            }
                        }
                    }
                    history_text.push('\n');
                }
                _ => {}
            }
        }

        let prompt = "The following is part of a code review agent's turn history. \
                      Summarize the key findings, code contents seen, and relevant grep results into a concise technical brief. \
                      Exclude meta-conversation but preserve details needed for the rest of the review.";

        self.complete(prompt, &history_text).await
    }
}

pub(crate) fn estimate_context_size(msgs: &[ChatCompletionRequestMessage]) -> usize {
    let mut total = 0;
    for m in msgs {
        match m {
            ChatCompletionRequestMessage::System(s) => match &s.content {
                async_openai::types::ChatCompletionRequestSystemMessageContent::Text(t) => {
                    total += t.len();
                }
                _ => {
                    // Ignore other content types for size estimation
                }
            },
            ChatCompletionRequestMessage::User(u) => match &u.content {
                async_openai::types::ChatCompletionRequestUserMessageContent::Text(t) => {
                    total += t.len()
                }
                async_openai::types::ChatCompletionRequestUserMessageContent::Array(parts) => {
                    for part in parts {
                        if let async_openai::types::ChatCompletionRequestUserMessageContentPart::Text(
                            tp,
                        ) = part
                        {
                            total += tp.text.len();
                        }
                    }
                }
            },
            ChatCompletionRequestMessage::Assistant(a) => {
                if let Some(ChatCompletionRequestAssistantMessageContent::Text(t)) = &a.content {
                    total += t.len();
                }
                if let Some(calls) = &a.tool_calls {
                    for call in calls {
                        total += call.function.name.len();
                        total += call.function.arguments.len();
                    }
                }
            }
            ChatCompletionRequestMessage::Tool(t) => match &t.content {
                async_openai::types::ChatCompletionRequestToolMessageContent::Text(text) => {
                    total += text.len()
                }
                async_openai::types::ChatCompletionRequestToolMessageContent::Array(parts) => {
                    for part in parts {
                        let async_openai::types::ChatCompletionRequestToolMessageContentPart::Text(
                            tp,
                        ) = part;
                        total += tp.text.len();
                    }
                }
            },
            _ => {}
        }
    }
    total
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

    #[test]
    fn estimate_size_counts_all_message_types() {
        let msgs = vec![
            ChatCompletionRequestSystemMessageArgs::default()
                .content("system")
                .build()
                .unwrap()
                .into(),
            ChatCompletionRequestUserMessageArgs::default()
                .content("user")
                .build()
                .unwrap()
                .into(),
            ChatCompletionRequestAssistantMessage {
                content: Some(ChatCompletionRequestAssistantMessageContent::Text(
                    "assistant".into(),
                )),
                ..Default::default()
            }
            .into(),
            ChatCompletionRequestToolMessageArgs::default()
                .tool_call_id("id")
                .content("tool")
                .build()
                .unwrap()
                .into(),
        ];
        // 6 + 4 + 9 + 4 = 23
        assert_eq!(estimate_context_size(&msgs), 23);
    }
}
