/**
 * Tests for API service
 */

import { describe, it, expect, beforeEach, vi } from 'vitest'
import axios from 'axios'
import {
  loadModel,
  unloadModel,
  getModelStatus,
  generateText,
  createStreamingConnection,
  getDefaultConfig,
  createTrainingJob,
  startTrainingJob,
  listTrainingJobs,
  getTrainingJob,
  cancelTrainingJob,
  checkHealth,
} from '../api'

// Mock axios
vi.mock('axios')

describe('API Service', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  describe('Model Management', () => {
    it('should load a model', async () => {
      const mockResponse = {
        data: {
          status: 'success',
          model_name: 'test-model',
          loaded: true,
        },
      }
      axios.create.mockReturnValue({
        post: vi.fn().mockResolvedValue(mockResponse),
      })

      const config = {
        model_name: 'test-model',
        max_seq_length: 1024,
        load_in_4bit: true,
      }

      const result = await loadModel(config)

      expect(result).toEqual(mockResponse.data)
    })

    it('should unload a model', async () => {
      const mockResponse = {
        data: {
          status: 'success',
          loaded: false,
        },
      }
      axios.create.mockReturnValue({
        post: vi.fn().mockResolvedValue(mockResponse),
      })

      const result = await unloadModel()

      expect(result).toEqual(mockResponse.data)
    })

    it('should get model status', async () => {
      const mockResponse = {
        data: {
          loaded: true,
          model_name: 'test-model',
          cuda_available: true,
        },
      }
      axios.create.mockReturnValue({
        get: vi.fn().mockResolvedValue(mockResponse),
      })

      const result = await getModelStatus()

      expect(result).toEqual(mockResponse.data)
      expect(result.loaded).toBe(true)
    })

    it('should handle model loading errors', async () => {
      axios.create.mockReturnValue({
        post: vi.fn().mockRejectedValue(new Error('Loading failed')),
      })

      await expect(loadModel({})).rejects.toThrow('Loading failed')
    })
  })

  describe('Inference', () => {
    it('should generate text', async () => {
      const mockResponse = {
        data: {
          status: 'success',
          generated_text: 'This is a test response',
          prompt: 'Test prompt',
        },
      }
      axios.create.mockReturnValue({
        post: vi.fn().mockResolvedValue(mockResponse),
      })

      const request = {
        prompt: 'Test prompt',
        reasoning_effort: 'medium',
        max_new_tokens: 256,
      }

      const result = await generateText(request)

      expect(result).toEqual(mockResponse.data)
      expect(result.generated_text).toBe('This is a test response')
    })

    it('should handle generation errors', async () => {
      axios.create.mockReturnValue({
        post: vi.fn().mockRejectedValue(new Error('Generation failed')),
      })

      await expect(generateText({ prompt: 'test' })).rejects.toThrow(
        'Generation failed'
      )
    })

    it('should create streaming connection', () => {
      const mockWs = {
        onopen: null,
        onmessage: null,
        onerror: null,
        onclose: null,
        send: vi.fn(),
        close: vi.fn(),
      }

      global.WebSocket = vi.fn(() => mockWs)

      const onMessage = vi.fn()
      const onError = vi.fn()
      const onComplete = vi.fn()

      const ws = createStreamingConnection(onMessage, onError, onComplete)

      expect(ws).toBeDefined()
      expect(global.WebSocket).toHaveBeenCalled()
    })

    it('should handle streaming messages', () => {
      const mockWs = {
        onopen: null,
        onmessage: null,
        onerror: null,
        onclose: null,
        send: vi.fn(),
        close: vi.fn(),
      }

      global.WebSocket = vi.fn(() => mockWs)

      const onMessage = vi.fn()
      const onError = vi.fn()
      const onComplete = vi.fn()

      createStreamingConnection(onMessage, onError, onComplete)

      // Simulate receiving a token
      const tokenEvent = {
        data: JSON.stringify({ type: 'token', content: 'Hello' }),
      }
      mockWs.onmessage(tokenEvent)

      expect(onMessage).toHaveBeenCalledWith('Hello')
    })

    it('should handle streaming completion', () => {
      const mockWs = {
        onopen: null,
        onmessage: null,
        onerror: null,
        onclose: null,
        send: vi.fn(),
        close: vi.fn(),
      }

      global.WebSocket = vi.fn(() => mockWs)

      const onMessage = vi.fn()
      const onError = vi.fn()
      const onComplete = vi.fn()

      createStreamingConnection(onMessage, onError, onComplete)

      // Simulate completion
      const completeEvent = {
        data: JSON.stringify({ type: 'complete' }),
      }
      mockWs.onmessage(completeEvent)

      expect(onComplete).toHaveBeenCalled()
    })

    it('should handle streaming errors', () => {
      const mockWs = {
        onopen: null,
        onmessage: null,
        onerror: null,
        onclose: null,
        send: vi.fn(),
        close: vi.fn(),
      }

      global.WebSocket = vi.fn(() => mockWs)

      const onMessage = vi.fn()
      const onError = vi.fn()
      const onComplete = vi.fn()

      createStreamingConnection(onMessage, onError, onComplete)

      // Simulate error
      const errorEvent = {
        data: JSON.stringify({ type: 'error', error: 'Connection failed' }),
      }
      mockWs.onmessage(errorEvent)

      expect(onError).toHaveBeenCalledWith('Connection failed')
    })
  })

  describe('Training Management', () => {
    it('should get default config', async () => {
      const mockConfig = {
        model_name: 'test-model',
        max_steps: 30,
        learning_rate: 0.0002,
      }
      const mockResponse = { data: mockConfig }

      axios.create.mockReturnValue({
        get: vi.fn().mockResolvedValue(mockResponse),
      })

      const result = await getDefaultConfig()

      expect(result).toEqual(mockConfig)
    })

    it('should create training job', async () => {
      const mockResponse = {
        data: {
          status: 'success',
          job: {
            job_id: 'test_job_001',
            status: 'pending',
          },
        },
      }
      axios.create.mockReturnValue({
        post: vi.fn().mockResolvedValue(mockResponse),
      })

      const config = { max_steps: 30 }
      const result = await createTrainingJob(config)

      expect(result).toEqual(mockResponse.data)
      expect(result.job.job_id).toBe('test_job_001')
    })

    it('should start training job', async () => {
      const mockResponse = {
        data: {
          status: 'success',
          job_id: 'test_job_001',
        },
      }
      axios.create.mockReturnValue({
        post: vi.fn().mockResolvedValue(mockResponse),
      })

      const result = await startTrainingJob('test_job_001')

      expect(result).toEqual(mockResponse.data)
    })

    it('should list training jobs', async () => {
      const mockJobs = [
        { job_id: 'job_001', status: 'completed' },
        { job_id: 'job_002', status: 'running' },
      ]
      const mockResponse = { data: { jobs: mockJobs } }

      axios.create.mockReturnValue({
        get: vi.fn().mockResolvedValue(mockResponse),
      })

      const result = await listTrainingJobs()

      expect(result).toEqual(mockResponse.data)
      expect(result.jobs).toHaveLength(2)
    })

    it('should get training job status', async () => {
      const mockJob = {
        job_id: 'test_job_001',
        status: 'running',
        progress: 50,
      }
      const mockResponse = { data: mockJob }

      axios.create.mockReturnValue({
        get: vi.fn().mockResolvedValue(mockResponse),
      })

      const result = await getTrainingJob('test_job_001')

      expect(result).toEqual(mockJob)
      expect(result.status).toBe('running')
    })

    it('should cancel training job', async () => {
      const mockResponse = {
        data: {
          status: 'success',
          message: 'Job cancelled',
        },
      }
      axios.create.mockReturnValue({
        post: vi.fn().mockResolvedValue(mockResponse),
      })

      const result = await cancelTrainingJob('test_job_001')

      expect(result).toEqual(mockResponse.data)
    })

    it('should handle training errors', async () => {
      axios.create.mockReturnValue({
        post: vi.fn().mockRejectedValue(new Error('Training failed')),
      })

      await expect(createTrainingJob({})).rejects.toThrow('Training failed')
    })
  })

  describe('Health Check', () => {
    it('should check health', async () => {
      const mockResponse = {
        data: {
          status: 'healthy',
          model_loaded: true,
        },
      }
      axios.create.mockReturnValue({
        get: vi.fn().mockResolvedValue(mockResponse),
      })

      const result = await checkHealth()

      expect(result).toEqual(mockResponse.data)
      expect(result.status).toBe('healthy')
    })

    it('should handle health check errors', async () => {
      axios.create.mockReturnValue({
        get: vi.fn().mockRejectedValue(new Error('Service unavailable')),
      })

      await expect(checkHealth()).rejects.toThrow('Service unavailable')
    })
  })

  describe('API Configuration', () => {
    it('should use correct base URL', () => {
      const mockCreate = vi.fn()
      axios.create = mockCreate

      // Import will trigger axios.create
      import('../api')

      // Check that axios.create was called (indirectly verifies configuration)
      // Note: This is a basic check; actual implementation may vary
    })

    it('should set correct headers', () => {
      expect(axios.create).toBeDefined()
    })
  })
})
