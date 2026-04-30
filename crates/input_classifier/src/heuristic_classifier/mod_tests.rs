use warp_completer::util::parse_current_commands_and_tokens;
use warp_completer::{ParsedTokenData, ParsedTokensSnapshot, meta::SpannedItem};

use crate::{Context, test_utils::CompletionContext};

use super::*;

/// Builds a minimal parsed snapshot for helper-level classifier tests.
fn mock_single_token_input(buffer_text: &str) -> ParsedTokensSnapshot {
    ParsedTokensSnapshot {
        buffer_text: buffer_text.to_string(),
        parsed_tokens: vec![ParsedTokenData {
            token: buffer_text.to_string().spanned((0, buffer_text.len())),
            token_index: 0,
            token_description: None,
        }],
    }
}

async fn mock_parsed_input_token(buffer_text: String) -> ParsedTokensSnapshot {
    let completion_context = CompletionContext::new();
    parse_current_commands_and_tokens(buffer_text, &completion_context).await
}

#[test]
fn test_single_non_ascii_token_is_treated_as_natural_language() {
    let input = mock_single_token_input("帮我看看这个报错");
    assert!(is_single_non_ascii_natural_language_token(
        &["帮我看看这个报错".to_string()],
        &input,
    ));
}

#[test]
fn test_path_like_non_ascii_token_is_not_treated_as_natural_language() {
    let input = mock_single_token_input("C:/用户/项目");
    assert!(!is_single_non_ascii_natural_language_token(
        &["C:/用户/项目".to_string()],
        &input,
    ));
}

#[test]
fn test_input_detection() {
    futures::executor::block_on(async move {
        let classifier = HeuristicClassifier;

        let mut context = Context {
            current_input_type: InputType::AI,
            is_agent_follow_up: false,
        };

        let token = mock_parsed_input_token("cargo --version".to_string()).await;
        assert_eq!(
            classifier.detect_input_type(token, &context).await,
            InputType::Shell
        );

        // We have to override the first token description here given the mocked completion
        // parser will parse the first token always as commands.
        //
        // Mock the case where cargo is not installed. We should still parse this as Shell input.
        let mut token = mock_parsed_input_token("cargo --version".to_string()).await;
        token.parsed_tokens[0].token_description = None;
        assert_eq!(
            classifier.detect_input_type(token, &context).await,
            InputType::Shell
        );

        let mut token = mock_parsed_input_token("rvm install 3.3".to_string()).await;
        token.parsed_tokens[0].token_description = None;
        assert_eq!(
            classifier.detect_input_type(token, &context).await,
            InputType::Shell
        );

        // Short queries with NL should be parsed as AI input when already in AI input.
        let mut token = mock_parsed_input_token("Explain this".to_string()).await;
        token.parsed_tokens[0].token_description = None;
        assert_eq!(
            classifier.detect_input_type(token.clone(), &context).await,
            InputType::AI
        );

        context.current_input_type = InputType::Shell;

        // Typing "fix this" after an error block is a common use case.
        let mut token = mock_parsed_input_token("fix this".to_string()).await;
        token.parsed_tokens[0].token_description = None;
        assert_eq!(
            classifier.detect_input_type(token, &context).await,
            InputType::AI,
        );

        // Single-token Chinese prompts should stay in AI mode.
        let mut token = mock_parsed_input_token("帮我看看这个报错".to_string()).await;
        token.parsed_tokens[0].token_description = None;
        assert_eq!(
            classifier.detect_input_type(token, &context).await,
            InputType::AI,
        );

        // Inputs with explicit path syntax should still be treated as shell-y.
        let mut token = mock_parsed_input_token("C:/用户/项目".to_string()).await;
        token.parsed_tokens[0].token_description = None;
        assert_eq!(
            classifier.detect_input_type(token, &context).await,
            InputType::Shell,
        );

        // Short queries with punctuation should be parsed as AI input.
        let token = mock_parsed_input_token("What went wrong?".to_string()).await;
        assert_eq!(
            classifier.detect_input_type(token, &context).await,
            InputType::AI
        );
        // Short queries with contractions should be parsed as AI input.
        let mut token = mock_parsed_input_token("What's the reason".to_string()).await;
        token.parsed_tokens[0].token_description = None;
        assert_eq!(
            classifier.detect_input_type(token, &context).await,
            InputType::AI
        );

        // Short queries with quotations should be parsed as AI input.
        let mut token =
            mock_parsed_input_token("The message is \"utils::future ... ok\"".to_string()).await;
        token.parsed_tokens[0].token_description = None;
        assert_eq!(
            classifier.detect_input_type(token, &context).await,
            InputType::AI
        );

        // String tokens with special shell syntax should not be treated as negative NL signal.
        let mut token = mock_parsed_input_token("The type is \"<>\"".to_string()).await;
        token.parsed_tokens[0].token_description = None;
        assert_eq!(
            classifier.detect_input_type(token, &context).await,
            InputType::AI
        );
    });
}
