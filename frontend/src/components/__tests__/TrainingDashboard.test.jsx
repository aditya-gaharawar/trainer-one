/**
 * Tests for TrainingDashboard component
 */

import { describe, it, expect, beforeEach, vi } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import TrainingDashboard from '../TrainingDashboard'
import * as api from '../../services/api'

// Mock API
vi.mock('../../services/api')

describe('TrainingDashboard Component', () => {
  const mockDefaultConfig = {
    model_name: 'test-model',
    dataset_name: 'test-dataset',
    max_steps: 30,
    learning_rate: 0.0002,
    batch_size: 1,
    lora_r: 8,
  }

  beforeEach(() => {
    vi.clearAllMocks()

    api.getDefaultConfig.mockResolvedValue(mockDefaultConfig)
    api.listTrainingJobs.mockResolvedValue({ jobs: [] })
  })

  it('should render component', async () => {
    render(<TrainingDashboard />)

    await waitFor(() => {
      expect(screen.getByText('Training Dashboard')).toBeInTheDocument()
    })
  })

  it('should load default configuration', async () => {
    render(<TrainingDashboard />)

    await waitFor(() => {
      expect(api.getDefaultConfig).toHaveBeenCalled()
    })
  })

  it('should display configuration panel when shown', async () => {
    const user = userEvent.setup()
    render(<TrainingDashboard />)

    await waitFor(() => {
      expect(screen.getByText('Show Config')).toBeInTheDocument()
    })

    const showConfigButton = screen.getByText('Show Config')
    await user.click(showConfigButton)

    expect(screen.getByText('Training Configuration')).toBeInTheDocument()
    expect(screen.getByText('Hide Config')).toBeInTheDocument()
  })

  it('should update configuration values', async () => {
    const user = userEvent.setup()
    render(<TrainingDashboard />)

    await waitFor(() => {
      const showConfigButton = screen.getByText('Show Config')
      return user.click(showConfigButton)
    })

    await waitFor(() => {
      expect(screen.getByLabelText('Max Steps')).toBeInTheDocument()
    })

    const input = screen.getByLabelText('Max Steps')
    await user.clear(input)
    await user.type(input, '50')

    expect(input.value).toBe('50')
  })

  it('should create and start training job', async () => {
    const mockJob = {
      job_id: 'test_job_001',
      status: 'pending',
      config: mockDefaultConfig,
    }

    api.createTrainingJob.mockResolvedValue({
      status: 'success',
      job: mockJob,
    })

    api.startTrainingJob.mockResolvedValue({
      status: 'success',
      job_id: 'test_job_001',
    })

    api.getTrainingJob.mockResolvedValue({
      ...mockJob,
      status: 'running',
      current_step: 0,
      total_steps: 30,
      progress: 0,
    })

    const user = userEvent.setup()
    render(<TrainingDashboard />)

    await waitFor(() => {
      expect(screen.getByText('Start New Training Job')).toBeInTheDocument()
    })

    const startButton = screen.getByText('Start New Training Job')
    await user.click(startButton)

    await waitFor(() => {
      expect(api.createTrainingJob).toHaveBeenCalled()
      expect(api.startTrainingJob).toHaveBeenCalledWith('test_job_001')
    })
  })

  it('should display running job', async () => {
    const runningJob = {
      job_id: 'test_job_001',
      status: 'running',
      current_step: 15,
      total_steps: 30,
      progress: 50,
      metrics: {
        loss: 1.5,
        learning_rate: 0.0002,
        epoch: 0.5,
      },
    }

    api.listTrainingJobs.mockResolvedValue({
      jobs: [runningJob],
    })

    render(<TrainingDashboard />)

    await waitFor(() => {
      expect(screen.getByText('Training in Progress')).toBeInTheDocument()
      expect(screen.getByText(/Step 15 \/ 30/i)).toBeInTheDocument()
    })
  })

  it('should display training metrics', async () => {
    const runningJob = {
      job_id: 'test_job_001',
      status: 'running',
      current_step: 15,
      total_steps: 30,
      progress: 50,
      metrics: {
        loss: 1.5234,
        learning_rate: 0.0002,
        epoch: 0.5,
      },
    }

    api.listTrainingJobs.mockResolvedValue({
      jobs: [runningJob],
    })

    render(<TrainingDashboard />)

    await waitFor(() => {
      expect(screen.getByText(/1\.5234/i)).toBeInTheDocument()
    })
  })

  it('should cancel running job', async () => {
    const runningJob = {
      job_id: 'test_job_001',
      status: 'running',
      current_step: 15,
      total_steps: 30,
      progress: 50,
    }

    api.listTrainingJobs.mockResolvedValue({
      jobs: [runningJob],
    })

    api.cancelTrainingJob.mockResolvedValue({
      status: 'success',
      message: 'Job cancelled',
    })

    api.getTrainingJob.mockResolvedValue({
      ...runningJob,
      status: 'cancelled',
    })

    const user = userEvent.setup()
    render(<TrainingDashboard />)

    await waitFor(() => {
      expect(screen.getByText('Cancel')).toBeInTheDocument()
    })

    const cancelButton = screen.getByText('Cancel')
    await user.click(cancelButton)

    await waitFor(() => {
      expect(api.cancelTrainingJob).toHaveBeenCalledWith('test_job_001')
    })
  })

  it('should list training history', async () => {
    const jobs = [
      {
        job_id: 'job_001',
        status: 'completed',
        current_step: 30,
        total_steps: 30,
        progress: 100,
      },
      {
        job_id: 'job_002',
        status: 'failed',
        current_step: 10,
        total_steps: 30,
        progress: 33.3,
      },
    ]

    api.listTrainingJobs.mockResolvedValue({ jobs })

    render(<TrainingDashboard />)

    await waitFor(() => {
      expect(screen.getByText('Training History')).toBeInTheDocument()
      expect(screen.getByText('job_001')).toBeInTheDocument()
      expect(screen.getByText('job_002')).toBeInTheDocument()
    })
  })

  it('should show empty state when no jobs', async () => {
    render(<TrainingDashboard />)

    await waitFor(() => {
      expect(screen.getByText(/No training jobs yet/i)).toBeInTheDocument()
    })
  })

  it('should refresh jobs', async () => {
    const user = userEvent.setup()
    render(<TrainingDashboard />)

    await waitFor(() => {
      expect(screen.getByText('Refresh')).toBeInTheDocument()
    })

    const refreshButton = screen.getByText('Refresh')
    await user.click(refreshButton)

    await waitFor(() => {
      expect(api.listTrainingJobs).toHaveBeenCalledTimes(2) // Initial load + refresh
    })
  })

  it('should display progress bar', async () => {
    const runningJob = {
      job_id: 'test_job_001',
      status: 'running',
      current_step: 15,
      total_steps: 30,
      progress: 50,
    }

    api.listTrainingJobs.mockResolvedValue({
      jobs: [runningJob],
    })

    render(<TrainingDashboard />)

    await waitFor(() => {
      expect(screen.getByText('50.0%')).toBeInTheDocument()
    })
  })

  it('should refresh job status periodically when running', async () => {
    vi.useFakeTimers()

    const runningJob = {
      job_id: 'test_job_001',
      status: 'running',
      current_step: 15,
      total_steps: 30,
      progress: 50,
    }

    api.listTrainingJobs.mockResolvedValue({
      jobs: [runningJob],
    })

    api.getTrainingJob.mockResolvedValue(runningJob)

    render(<TrainingDashboard />)

    await waitFor(() => {
      expect(screen.getByText('Training in Progress')).toBeInTheDocument()
    })

    // Advance time by 2 seconds (refresh interval)
    vi.advanceTimersByTime(2000)

    await waitFor(() => {
      expect(api.getTrainingJob).toHaveBeenCalled()
    })

    vi.useRealTimers()
  })

  it('should handle job creation error', async () => {
    api.createTrainingJob.mockRejectedValue(
      new Error('Job creation failed')
    )

    // Mock window.alert
    const alertMock = vi.spyOn(window, 'alert').mockImplementation(() => {})

    const user = userEvent.setup()
    render(<TrainingDashboard />)

    await waitFor(() => {
      expect(screen.getByText('Start New Training Job')).toBeInTheDocument()
    })

    const startButton = screen.getByText('Start New Training Job')
    await user.click(startButton)

    await waitFor(() => {
      expect(alertMock).toHaveBeenCalled()
    })

    alertMock.mockRestore()
  })

  it('should display job status icons correctly', async () => {
    const jobs = [
      { job_id: 'job_001', status: 'completed', current_step: 30, total_steps: 30 },
      { job_id: 'job_002', status: 'running', current_step: 15, total_steps: 30 },
      { job_id: 'job_003', status: 'failed', current_step: 5, total_steps: 30 },
      { job_id: 'job_004', status: 'cancelled', current_step: 10, total_steps: 30 },
    ]

    api.listTrainingJobs.mockResolvedValue({ jobs })

    render(<TrainingDashboard />)

    await waitFor(() => {
      expect(screen.getByText('Completed')).toBeInTheDocument()
      expect(screen.getByText('Running')).toBeInTheDocument()
      expect(screen.getByText('Failed')).toBeInTheDocument()
      expect(screen.getByText('Cancelled')).toBeInTheDocument()
    })
  })

  it('should disable start button while loading', async () => {
    api.createTrainingJob.mockImplementation(
      () => new Promise((resolve) => setTimeout(resolve, 1000))
    )

    const user = userEvent.setup()
    render(<TrainingDashboard />)

    await waitFor(() => {
      expect(screen.getByText('Start New Training Job')).toBeInTheDocument()
    })

    const startButton = screen.getByText('Start New Training Job')
    await user.click(startButton)

    expect(startButton).toBeDisabled()
    expect(screen.getByText('Starting Training...')).toBeInTheDocument()
  })
})
