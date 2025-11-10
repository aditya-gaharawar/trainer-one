/**
 * Tests for ModelManager component
 */

import { describe, it, expect, beforeEach, vi } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import ModelManager from '../ModelManager'
import * as api from '../../services/api'

// Mock API
vi.mock('../../services/api')

describe('ModelManager Component', () => {
  const mockOnModelStatusChange = vi.fn()

  beforeEach(() => {
    vi.clearAllMocks()

    // Default mock implementations
    api.getModelStatus.mockResolvedValue({
      loaded: false,
      model_name: null,
      cuda_available: true,
      cuda_memory_allocated: null,
    })
  })

  it('should render component', async () => {
    render(<ModelManager onModelStatusChange={mockOnModelStatusChange} />)

    await waitFor(() => {
      expect(screen.getByText('Model Manager')).toBeInTheDocument()
    })
  })

  it('should display current status', async () => {
    api.getModelStatus.mockResolvedValue({
      loaded: true,
      model_name: 'test-model',
      cuda_available: true,
      cuda_memory_allocated: 1024 * 1024 * 1024,
    })

    render(<ModelManager onModelStatusChange={mockOnModelStatusChange} />)

    await waitFor(() => {
      expect(screen.getByText('Model Loaded')).toBeInTheDocument()
    })
  })

  it('should show no model loaded when status is false', async () => {
    render(<ModelManager onModelStatusChange={mockOnModelStatusChange} />)

    await waitFor(() => {
      expect(screen.getByText('No Model Loaded')).toBeInTheDocument()
    })
  })

  it('should load model on button click', async () => {
    api.loadModel.mockResolvedValue({
      status: 'success',
      loaded: true,
    })

    const user = userEvent.setup()
    render(<ModelManager onModelStatusChange={mockOnModelStatusChange} />)

    await waitFor(() => {
      expect(screen.getByText('Load Model')).toBeInTheDocument()
    })

    const loadButton = screen.getByText('Load Model')
    await user.click(loadButton)

    await waitFor(() => {
      expect(api.loadModel).toHaveBeenCalled()
    })
  })

  it('should unload model on button click', async () => {
    api.getModelStatus.mockResolvedValue({
      loaded: true,
      model_name: 'test-model',
      cuda_available: true,
    })

    api.unloadModel.mockResolvedValue({
      status: 'success',
      loaded: false,
    })

    const user = userEvent.setup()
    render(<ModelManager onModelStatusChange={mockOnModelStatusChange} />)

    await waitFor(() => {
      expect(screen.getByText('Unload Model')).toBeInTheDocument()
    })

    const unloadButton = screen.getByText('Unload Model')
    await user.click(unloadButton)

    await waitFor(() => {
      expect(api.unloadModel).toHaveBeenCalled()
    })
  })

  it('should display CUDA availability', async () => {
    api.getModelStatus.mockResolvedValue({
      loaded: false,
      cuda_available: true,
    })

    render(<ModelManager onModelStatusChange={mockOnModelStatusChange} />)

    await waitFor(() => {
      expect(screen.getByText(/CUDA: Available/i)).toBeInTheDocument()
    })
  })

  it('should update model configuration', async () => {
    const user = userEvent.setup()
    render(<ModelManager onModelStatusChange={mockOnModelStatusChange} />)

    await waitFor(() => {
      expect(screen.getByLabelText(/Max Sequence Length/i)).toBeInTheDocument()
    })

    const slider = screen.getByLabelText(/Max Sequence Length/i)
    fireEvent.change(slider, { target: { value: '2048' } })

    expect(slider.value).toBe('2048')
  })

  it('should toggle 4-bit loading', async () => {
    const user = userEvent.setup()
    render(<ModelManager onModelStatusChange={mockOnModelStatusChange} />)

    await waitFor(() => {
      const checkbox = screen.getByLabelText(/Load in 4-bit/i)
      expect(checkbox).toBeInTheDocument()
      expect(checkbox).toBeChecked()
    })

    const checkbox = screen.getByLabelText(/Load in 4-bit/i)
    await user.click(checkbox)

    expect(checkbox).not.toBeChecked()
  })

  it('should disable controls when model is loaded', async () => {
    api.getModelStatus.mockResolvedValue({
      loaded: true,
      model_name: 'test-model',
    })

    render(<ModelManager onModelStatusChange={mockOnModelStatusChange} />)

    await waitFor(() => {
      const select = screen.getByRole('combobox')
      expect(select).toBeDisabled()
    })
  })

  it('should call onModelStatusChange callback', async () => {
    api.getModelStatus.mockResolvedValue({
      loaded: true,
      model_name: 'test-model',
    })

    render(<ModelManager onModelStatusChange={mockOnModelStatusChange} />)

    await waitFor(() => {
      expect(mockOnModelStatusChange).toHaveBeenCalledWith(true)
    })
  })

  it('should handle load model error', async () => {
    api.loadModel.mockRejectedValue(new Error('Loading failed'))

    // Mock window.alert
    const alertMock = vi.spyOn(window, 'alert').mockImplementation(() => {})

    const user = userEvent.setup()
    render(<ModelManager onModelStatusChange={mockOnModelStatusChange} />)

    await waitFor(() => {
      expect(screen.getByText('Load Model')).toBeInTheDocument()
    })

    const loadButton = screen.getByText('Load Model')
    await user.click(loadButton)

    await waitFor(() => {
      expect(alertMock).toHaveBeenCalled()
    })

    alertMock.mockRestore()
  })

  it('should format memory correctly', async () => {
    api.getModelStatus.mockResolvedValue({
      loaded: true,
      model_name: 'test-model',
      cuda_available: true,
      cuda_memory_allocated: 2 * 1024 * 1024 * 1024, // 2 GB
    })

    render(<ModelManager onModelStatusChange={mockOnModelStatusChange} />)

    await waitFor(() => {
      expect(screen.getByText(/Memory: 2\.00 GB/i)).toBeInTheDocument()
    })
  })

  it('should refresh status periodically', async () => {
    vi.useFakeTimers()

    render(<ModelManager onModelStatusChange={mockOnModelStatusChange} />)

    await waitFor(() => {
      expect(api.getModelStatus).toHaveBeenCalledTimes(1)
    })

    // Advance time by 5 seconds (refresh interval)
    vi.advanceTimersByTime(5000)

    await waitFor(() => {
      expect(api.getModelStatus).toHaveBeenCalledTimes(2)
    })

    vi.useRealTimers()
  })
})
