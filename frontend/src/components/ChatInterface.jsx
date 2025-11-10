import React, { useState, useRef, useEffect } from 'react';
import { Send, Loader2, Brain, Zap, Target } from 'lucide-react';
import { generateText, createStreamingConnection } from '../services/api';

const ChatInterface = ({ modelLoaded }) => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [reasoningEffort, setReasoningEffort] = useState('medium');
  const [temperature, setTemperature] = useState(0.7);
  const [maxTokens, setMaxTokens] = useState(512);
  const [useStreaming, setUseStreaming] = useState(true);
  const messagesEndRef = useRef(null);
  const wsRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() || !modelLoaded || isGenerating) return;

    const userMessage = { role: 'user', content: input };
    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsGenerating(true);

    if (useStreaming) {
      handleStreamingGeneration(input);
    } else {
      handleNonStreamingGeneration(input);
    }
  };

  const handleStreamingGeneration = (prompt) => {
    const assistantMessage = { role: 'assistant', content: '' };
    setMessages((prev) => [...prev, assistantMessage]);

    let fullContent = '';

    const onMessage = (token) => {
      fullContent += token;
      setMessages((prev) => {
        const newMessages = [...prev];
        newMessages[newMessages.length - 1] = {
          role: 'assistant',
          content: fullContent,
        };
        return newMessages;
      });
    };

    const onError = (error) => {
      setMessages((prev) => {
        const newMessages = [...prev];
        newMessages[newMessages.length - 1] = {
          role: 'assistant',
          content: `Error: ${error}`,
          isError: true,
        };
        return newMessages;
      });
      setIsGenerating(false);
    };

    const onComplete = () => {
      setIsGenerating(false);
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };

    wsRef.current = createStreamingConnection(onMessage, onError, onComplete);

    wsRef.current.onopen = () => {
      wsRef.current.send(
        JSON.stringify({
          prompt,
          reasoning_effort: reasoningEffort,
          max_new_tokens: maxTokens,
          temperature,
          top_p: 0.9,
        })
      );
    };
  };

  const handleNonStreamingGeneration = async (prompt) => {
    try {
      const response = await generateText({
        prompt,
        reasoning_effort: reasoningEffort,
        max_new_tokens: maxTokens,
        temperature,
        top_p: 0.9,
      });

      const assistantMessage = {
        role: 'assistant',
        content: response.generated_text || 'No response generated.',
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      const errorMessage = {
        role: 'assistant',
        content: `Error: ${error.message}`,
        isError: true,
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsGenerating(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const clearChat = () => {
    setMessages([]);
  };

  const getReasoningIcon = () => {
    switch (reasoningEffort) {
      case 'low':
        return <Zap className="w-4 h-4" />;
      case 'high':
        return <Brain className="w-4 h-4" />;
      default:
        return <Target className="w-4 h-4" />;
    }
  };

  return (
    <div className="flex flex-col h-full bg-white rounded-lg shadow-lg">
      {/* Header */}
      <div className="px-6 py-4 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <h2 className="text-xl font-semibold text-gray-800">Chat Interface</h2>
          <div className="flex items-center gap-4">
            <span className="text-sm text-gray-600">
              {modelLoaded ? (
                <span className="flex items-center gap-2 text-green-600">
                  <span className="w-2 h-2 bg-green-600 rounded-full"></span>
                  Model Ready
                </span>
              ) : (
                <span className="flex items-center gap-2 text-red-600">
                  <span className="w-2 h-2 bg-red-600 rounded-full"></span>
                  No Model
                </span>
              )}
            </span>
            <button
              onClick={clearChat}
              className="text-sm text-blue-600 hover:text-blue-800"
            >
              Clear Chat
            </button>
          </div>
        </div>

        {/* Settings */}
        <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-4">
          <div>
            <label className="block text-xs font-medium text-gray-700 mb-1">
              Reasoning Effort
            </label>
            <div className="flex items-center gap-2">
              {getReasoningIcon()}
              <select
                value={reasoningEffort}
                onChange={(e) => setReasoningEffort(e.target.value)}
                className="flex-1 text-sm border border-gray-300 rounded-md px-2 py-1 focus:outline-none focus:ring-2 focus:ring-blue-500"
                disabled={isGenerating}
              >
                <option value="low">Low</option>
                <option value="medium">Medium</option>
                <option value="high">High</option>
              </select>
            </div>
          </div>

          <div>
            <label className="block text-xs font-medium text-gray-700 mb-1">
              Temperature: {temperature}
            </label>
            <input
              type="range"
              min="0"
              max="2"
              step="0.1"
              value={temperature}
              onChange={(e) => setTemperature(parseFloat(e.target.value))}
              className="w-full"
              disabled={isGenerating}
            />
          </div>

          <div>
            <label className="block text-xs font-medium text-gray-700 mb-1">
              Max Tokens: {maxTokens}
            </label>
            <input
              type="range"
              min="128"
              max="2048"
              step="128"
              value={maxTokens}
              onChange={(e) => setMaxTokens(parseInt(e.target.value))}
              className="w-full"
              disabled={isGenerating}
            />
          </div>

          <div>
            <label className="block text-xs font-medium text-gray-700 mb-1">
              Streaming
            </label>
            <label className="flex items-center cursor-pointer">
              <input
                type="checkbox"
                checked={useStreaming}
                onChange={(e) => setUseStreaming(e.target.checked)}
                className="mr-2"
                disabled={isGenerating}
              />
              <span className="text-sm">Enabled</span>
            </label>
          </div>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-6 space-y-4">
        {messages.length === 0 ? (
          <div className="flex items-center justify-center h-full text-gray-400">
            <div className="text-center">
              <Brain className="w-16 h-16 mx-auto mb-4 opacity-50" />
              <p className="text-lg">Start a conversation with the model</p>
              <p className="text-sm mt-2">
                Ask questions, solve problems, or explore multilingual reasoning
              </p>
            </div>
          </div>
        ) : (
          messages.map((message, index) => (
            <div
              key={index}
              className={`flex ${
                message.role === 'user' ? 'justify-end' : 'justify-start'
              }`}
            >
              <div
                className={`max-w-[80%] rounded-lg px-4 py-3 ${
                  message.role === 'user'
                    ? 'bg-blue-600 text-white'
                    : message.isError
                    ? 'bg-red-100 text-red-800 border border-red-300'
                    : 'bg-gray-100 text-gray-800'
                }`}
              >
                <div className="text-xs font-medium mb-1 opacity-75">
                  {message.role === 'user' ? 'You' : 'Assistant'}
                </div>
                <div className="whitespace-pre-wrap">{message.content}</div>
              </div>
            </div>
          ))
        )}
        {isGenerating && useStreaming && messages[messages.length - 1]?.role === 'assistant' && messages[messages.length - 1]?.content === '' && (
          <div className="flex justify-start">
            <div className="bg-gray-100 rounded-lg px-4 py-3">
              <Loader2 className="w-5 h-5 animate-spin text-gray-600" />
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="px-6 py-4 border-t border-gray-200">
        <div className="flex gap-3">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder={
              modelLoaded
                ? 'Type your message... (Shift+Enter for new line)'
                : 'Please load a model first'
            }
            className="flex-1 resize-none border border-gray-300 rounded-lg px-4 py-3 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-100 disabled:cursor-not-allowed"
            rows="3"
            disabled={!modelLoaded || isGenerating}
          />
          <button
            onClick={handleSend}
            disabled={!modelLoaded || isGenerating || !input.trim()}
            className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
          >
            {isGenerating ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                <span>Generating...</span>
              </>
            ) : (
              <>
                <Send className="w-5 h-5" />
                <span>Send</span>
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
};

export default ChatInterface;
