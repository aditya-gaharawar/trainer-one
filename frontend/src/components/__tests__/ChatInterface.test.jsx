/**
 * Tests for ChatInterface component
 */

import { describe, it, expect, beforeEach, vi } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import ChatInterface from '../ChatInterface'
import * as api from '../../services/api'

// Mock API
vi.mock('../../services/api')

describe('ChatInterface Component', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('should render component', () => {
    render(<ChatInterface modelLoaded={false} />)

    expect(screen.getByText('Chat Interface')).toBeInTheDocument()
  })

  it('should show no model message when model not loaded', () => {
    render(<ChatInterface modelLoaded={false} />)

    expect(screen.getByText(/No Model/i)).toBeInTheDocument()
  })

  it('should show model ready when model is loaded', () => {
    render(<ChatInterface modelLoaded={true} />)

    expect(screen.getByText(/Model Ready/i)).toBeInTheDocument()
  })

  it('should display empty state message', () => {
    render(<ChatInterface modelLoaded={true} />)

    expect(
      screen.getByText(/Start a conversation with the model/i)
    ).toBeInTheDocument()
  })

  it('should disable input when model not loaded', () => {
    render(<ChatInterface modelLoaded={false} />)

    const textarea = screen.getByPlaceholderText(/Please load a model first/i)
    expect(textarea).toBeDisabled()
  })

  it('should enable input when model is loaded', () => {
    render(<ChatInterface modelLoaded={true} />)

    const textarea = screen.getByPlaceholderText(/Type your message/i)
    expect(textarea).not.toBeDisabled()
  })

  it('should send message on button click', async () => {
    api.generateText.mockResolvedValue({
      status: 'success',
      generated_text: 'Test response',
    })

    const user = userEvent.setup()
    render(<ChatInterface modelLoaded={true} />)

    const textarea = screen.getByPlaceholderText(/Type your message/i)
    await user.type(textarea, 'Hello')

    const sendButton = screen.getByText('Send')
    await user.click(sendButton)

    await waitFor(() => {
      expect(screen.getByText('Hello')).toBeInTheDocument()
      expect(screen.getByText('Test response')).toBeInTheDocument()
    })
  })

  it('should send message on Enter key', async () => {
    api.generateText.mockResolvedValue({
      status: 'success',
      generated_text: 'Test response',
    })

    const user = userEvent.setup()
    render(<ChatInterface modelLoaded={true} />)

    const textarea = screen.getByPlaceholderText(/Type your message/i)
    await user.type(textarea, 'Hello{Enter}')

    await waitFor(() => {
      expect(screen.getByText('Hello')).toBeInTheDocument()
    })
  })

  it('should not send empty message', async () => {
    const user = userEvent.setup()
    render(<ChatInterface modelLoaded={true} />)

    const sendButton = screen.getByText('Send')
    await user.click(sendButton)

    expect(api.generateText).not.toHaveBeenCalled()
  })

  it('should clear chat', async () => {
    api.generateText.mockResolvedValue({
      status: 'success',
      generated_text: 'Response',
    })

    const user = userEvent.setup()
    render(<ChatInterface modelLoaded={true} />)

    // Send a message
    const textarea = screen.getByPlaceholderText(/Type your message/i)
    await user.type(textarea, 'Hello')
    const sendButton = screen.getByText('Send')
    await user.click(sendButton)

    await waitFor(() => {
      expect(screen.getByText('Hello')).toBeInTheDocument()
    })

    // Clear chat
    const clearButton = screen.getByText('Clear Chat')
    await user.click(clearButton)

    expect(screen.queryByText('Hello')).not.toBeInTheDocument()
  })

  it('should adjust reasoning effort', async () => {
    const user = userEvent.setup()
    render(<ChatInterface modelLoaded={true} />)

    const select = screen.getByDisplayValue('Medium')
    await user.selectOptions(select, 'high')

    expect(select.value).toBe('high')
  })

  it('should adjust temperature', () => {
    render(<ChatInterface modelLoaded={true} />)

    const slider = screen.getByLabelText(/Temperature: 0.7/i)
    fireEvent.change(slider, { target: { value: '1.5' } })

    expect(screen.getByLabelText(/Temperature: 1.5/i)).toBeInTheDocument()
  })

  it('should adjust max tokens', () => {
    render(<ChatInterface modelLoaded={true} />)

    const slider = screen.getByLabelText(/Max Tokens: 512/i)
    fireEvent.change(slider, { target: { value: '1024' } })

    expect(screen.getByLabelText(/Max Tokens: 1024/i)).toBeInTheDocument()
  })

  it('should toggle streaming', async () => {
    const user = userEvent.setup()
    render(<ChatInterface modelLoaded={true} />)

    const checkbox = screen.getByRole('checkbox', { name: /Enabled/i })
    expect(checkbox).toBeChecked()

    await user.click(checkbox)
    expect(checkbox).not.toBeChecked()
  })

  it('should use streaming when enabled', async () => {
    const mockWs = {
      onopen: null,
      onmessage: null,
      send: vi.fn(),
      close: vi.fn(),
    }
    api.createStreamingConnection.mockReturnValue(mockWs)

    const user = userEvent.setup()
    render(<ChatInterface modelLoaded={true} />)

    const textarea = screen.getByPlaceholderText(/Type your message/i)
    await user.type(textarea, 'Test')

    const sendButton = screen.getByText('Send')
    await user.click(sendButton)

    await waitFor(() => {
      expect(api.createStreamingConnection).toHaveBeenCalled()
    })
  })

  it('should use non-streaming when disabled', async () => {
    api.generateText.mockResolvedValue({
      status: 'success',
      generated_text: 'Response',
    })

    const user = userEvent.setup()
    render(<ChatInterface modelLoaded={true} />)

    // Disable streaming
    const checkbox = screen.getByRole('checkbox', { name: /Enabled/i })
    await user.click(checkbox)

    const textarea = screen.getByPlaceholderText(/Type your message/i)
    await user.type(textarea, 'Test')

    const sendButton = screen.getByText('Send')
    await user.click(sendButton)

    await waitFor(() => {
      expect(api.generateText).toHaveBeenCalled()
    })
  })

  it('should disable controls while generating', async () => {
    api.generateText.mockImplementation(
      () => new Promise((resolve) => setTimeout(resolve, 1000))
    )

    const user = userEvent.setup()
    render(<ChatInterface modelLoaded={true} />)

    const textarea = screen.getByPlaceholderText(/Type your message/i)
    await user.type(textarea, 'Test')

    const sendButton = screen.getByText('Send')
    await user.click(sendButton)

    // Controls should be disabled during generation
    expect(textarea).toBeDisabled()
    expect(sendButton).toBeDisabled()
  })

  it('should display user and assistant roles correctly', async () => {
    api.generateText.mockResolvedValue({
      status: 'success',
      generated_text: 'Assistant response',
    })

    const user = userEvent.setup()
    render(<ChatInterface modelLoaded={true} />)

    const textarea = screen.getByPlaceholderText(/Type your message/i)
    await user.type(textarea, 'User message')

    const sendButton = screen.getByText('Send')
    await user.click(sendButton)

    await waitFor(() => {
      expect(screen.getByText('You')).toBeInTheDocument()
      expect(screen.getByText('Assistant')).toBeInTheDocument()
    })
  })

  it('should handle generation errors', async () => {
    api.generateText.mockRejectedValue(new Error('Generation failed'))

    const user = userEvent.setup()
    render(<ChatInterface modelLoaded={true} />)

    const textarea = screen.getByPlaceholderText(/Type your message/i)
    await user.type(textarea, 'Test')

    const sendButton = screen.getByText('Send')
    await user.click(sendButton)

    await waitFor(() => {
      expect(screen.getByText(/Error: Generation failed/i)).toBeInTheDocument()
    })
  })

  it('should scroll to bottom when new messages arrive', async () => {
    api.generateText.mockResolvedValue({
      status: 'success',
      generated_text: 'Response',
    })

    // Mock scrollIntoView
    Element.prototype.scrollIntoView = vi.fn()

    const user = userEvent.setup()
    render(<ChatInterface modelLoaded={true} />)

    const textarea = screen.getByPlaceholderText(/Type your message/i)
    await user.type(textarea, 'Test')

    const sendButton = screen.getByText('Send')
    await user.click(sendButton)

    await waitFor(() => {
      expect(Element.prototype.scrollIntoView).toHaveBeenCalled()
    })
  })
})
