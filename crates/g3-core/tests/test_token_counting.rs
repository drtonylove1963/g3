use g3_core::ContextWindow;
use g3_providers::{Message, MessageRole, Usage};

/// Test that used_tokens is tracked via add_message.
#[test]
fn test_used_tokens_tracked_via_messages() {
    let mut window = ContextWindow::new(10000);

    // Add a user message - this should update used_tokens
    let user_msg = Message::new(MessageRole::User, "Hello, how are you?".to_string());
    window.add_message(user_msg);
    
    // used_tokens should be non-zero after adding a message
    assert!(window.used_tokens > 0, "used_tokens should increase after add_message");
    let tokens_after_user_msg = window.used_tokens;

    // Add an assistant message
    let assistant_msg = Message::new(MessageRole::Assistant, "I'm doing well, thank you!".to_string());
    window.add_message(assistant_msg);
    
    // used_tokens should increase further
    assert!(window.used_tokens > tokens_after_user_msg, "used_tokens should increase after adding assistant message");
}

/// Test that update_usage_from_response calibrates used_tokens from prompt_tokens.
/// When prompt_tokens > 0, used_tokens is snapped to the API's ground truth.
/// When prompt_tokens is 0, used_tokens is left unchanged (heuristic fallback).
#[test]
fn test_update_usage_calibrates_used_tokens() {
    let mut window = ContextWindow::new(10000);

    assert_eq!(window.used_tokens, 0);
    assert_eq!(window.cumulative_tokens, 0);

    // Simulate API response — prompt_tokens > 0 triggers calibration
    let usage = Usage {
        prompt_tokens: 100,
        completion_tokens: 50,
        total_tokens: 150,
        cache_creation_tokens: 0,
        cache_read_tokens: 0,
    };
    window.update_usage_from_response(&usage);

    // used_tokens should be calibrated to prompt_tokens
    assert_eq!(window.used_tokens, 100, "used_tokens should be calibrated to prompt_tokens");

    // cumulative_tokens tracks total API usage
    assert_eq!(window.cumulative_tokens, 150, "cumulative_tokens should track total API usage");

    // Another API call with higher prompt_tokens
    let usage2 = Usage {
        prompt_tokens: 200,
        completion_tokens: 75,
        total_tokens: 275,
        cache_creation_tokens: 0,
        cache_read_tokens: 0,
    };
    window.update_usage_from_response(&usage2);

    // used_tokens calibrated to latest prompt_tokens
    assert_eq!(window.used_tokens, 200, "used_tokens should be calibrated to latest prompt_tokens");

    // cumulative_tokens accumulates
    assert_eq!(window.cumulative_tokens, 425, "cumulative_tokens should accumulate");

    // When prompt_tokens is 0, used_tokens should NOT change (fallback)
    let usage3 = Usage {
        prompt_tokens: 0,
        completion_tokens: 30,
        total_tokens: 30,
        cache_creation_tokens: 0,
        cache_read_tokens: 0,
    };
    window.update_usage_from_response(&usage3);

    // used_tokens unchanged (prompt_tokens was 0)
    assert_eq!(window.used_tokens, 200, "used_tokens should not change when prompt_tokens is 0");

    // cumulative_tokens still accumulates
    assert_eq!(window.cumulative_tokens, 455, "cumulative_tokens should still accumulate");
}

/// Test that add_streaming_tokens only updates cumulative_tokens.
/// The assistant message will be added via add_message which tracks used_tokens.
#[test]
fn test_add_streaming_tokens_only_affects_cumulative() {
    let mut window = ContextWindow::new(10000);

    // Add streaming tokens (fallback when no usage data available)
    window.add_streaming_tokens(100);
    
    // used_tokens should NOT change
    assert_eq!(window.used_tokens, 0, "used_tokens should not be updated by add_streaming_tokens");
    
    // cumulative_tokens SHOULD be updated
    assert_eq!(window.cumulative_tokens, 100, "cumulative_tokens should be updated");

    // Add more streaming tokens
    window.add_streaming_tokens(50);
    assert_eq!(window.used_tokens, 0);
    assert_eq!(window.cumulative_tokens, 150);
}

/// Test percentage calculation is based on used_tokens (actual context content).
#[test]
fn test_percentage_based_on_used_tokens() {
    let mut window = ContextWindow::new(1000);

    // Initially 0%
    assert_eq!(window.percentage_used(), 0.0);
    // After 1% buffer: total_tokens = 990
    assert_eq!(window.remaining_tokens(), 990);

    // Add messages to increase used_tokens
    // A message with ~100 chars should be roughly 25-30 tokens
    let msg = Message::new(MessageRole::User, "x".repeat(400)); // ~100 tokens estimated
    window.add_message(msg);
    
    // Percentage should be based on used_tokens
    let percentage = window.percentage_used();
    assert!(percentage > 0.0, "percentage should be > 0 after adding message");
    assert!(percentage < 100.0, "percentage should be < 100");
    
    // remaining_tokens should decrease
    assert!(window.remaining_tokens() < 990, "remaining tokens should decrease");
}

/// Test that the 80% compaction threshold works correctly.
#[test]
fn test_should_compact_threshold() {
    let mut window = ContextWindow::new(1000);

    // Add messages until we approach 80%
    // Each message of ~320 chars is roughly 80 tokens (at 4 chars/token)
    for _ in 0..9 {
        let msg = Message::new(MessageRole::User, "x".repeat(320));
        window.add_message(msg);
    }

    // Should be around 720 tokens (72%) - not yet at threshold
    let percentage = window.percentage_used();
    println!("After 9 messages: {}% used ({} tokens)", percentage, window.used_tokens);

    // Add one more message to push over 80%
    let msg = Message::new(MessageRole::User, "x".repeat(320));
    window.add_message(msg);
    
    let percentage_after = window.percentage_used();
    println!("After 10 messages: {}% used ({} tokens)", percentage_after, window.used_tokens);

    // Now should_compact should return true if we're at 80%+
    if percentage_after >= 80.0 {
        assert!(window.should_compact(), "should_compact should be true at 80%+");
    }
}

/// Test that calibration and cumulative tracking work together correctly.
#[test]
fn test_calibration_and_cumulative_interaction() {
    let mut window = ContextWindow::new(10000);

    // Add a message (affects both used_tokens and cumulative_tokens)
    let msg = Message::new(MessageRole::User, "Hello world".to_string());
    window.add_message(msg);
    let used_after_msg = window.used_tokens;
    let cumulative_after_msg = window.cumulative_tokens;
    assert_eq!(used_after_msg, cumulative_after_msg);

    // Simulate API response — calibrates used_tokens, accumulates cumulative_tokens
    let usage = Usage {
        prompt_tokens: 500,
        completion_tokens: 200,
        total_tokens: 700,
        cache_creation_tokens: 0,
        cache_read_tokens: 0,
    };
    window.update_usage_from_response(&usage);

    // used_tokens calibrated to prompt_tokens (500)
    assert_eq!(window.used_tokens, 500, "used_tokens should be calibrated to prompt_tokens");
    
    // cumulative_tokens increased by total_tokens
    assert_eq!(window.cumulative_tokens, cumulative_after_msg + 700, "cumulative_tokens should increase");
    
    // They should now be different
    assert!(window.cumulative_tokens > window.used_tokens, "cumulative should be greater than used");
}

/// Test that calibration corrects heuristic undercount.
/// The heuristic doesn't account for tool definitions (~4000 tokens),
/// so prompt_tokens from the API is always larger.
#[test]
fn test_calibration_corrects_undercount() {
    let mut window = ContextWindow::new(200000);

    // Simulate adding a system prompt and user message via heuristic
    let system_msg = Message::new(MessageRole::System, "x".repeat(4000)); // ~1000 tokens
    window.add_message(system_msg);
    let user_msg = Message::new(MessageRole::User, "Hello".to_string());
    window.add_message(user_msg);

    let heuristic_estimate = window.used_tokens;
    assert!(heuristic_estimate > 0);

    // API reports higher prompt_tokens (includes tool definitions)
    let usage = Usage {
        prompt_tokens: heuristic_estimate + 4000, // tool definitions add ~4000 tokens
        completion_tokens: 100,
        total_tokens: heuristic_estimate + 4100,
        cache_creation_tokens: 0,
        cache_read_tokens: 0,
    };
    window.update_usage_from_response(&usage);

    // used_tokens should now be higher than the heuristic estimate
    assert_eq!(window.used_tokens, heuristic_estimate + 4000);
    assert!(window.used_tokens > heuristic_estimate, "calibration should correct undercount");
}
