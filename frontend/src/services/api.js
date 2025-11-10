/**
 * API Service for communicating with the FastAPI backend
 */

import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// ============================================================================
// Model Management
// ============================================================================

export const loadModel = async (modelConfig) => {
  const response = await api.post('/model/load', modelConfig);
  return response.data;
};

export const unloadModel = async () => {
  const response = await api.post('/model/unload');
  return response.data;
};

export const getModelStatus = async () => {
  const response = await api.get('/model/status');
  return response.data;
};

// ============================================================================
// Inference
// ============================================================================

export const generateText = async (inferenceRequest) => {
  const response = await api.post('/inference', inferenceRequest);
  return response.data;
};

export const createStreamingConnection = (onMessage, onError, onComplete) => {
  const wsUrl = API_BASE_URL.replace('http', 'ws') + '/inference/stream';
  const ws = new WebSocket(wsUrl);

  ws.onopen = () => {
    console.log('WebSocket connected');
  };

  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);

    if (data.type === 'start') {
      // Generation started
    } else if (data.type === 'token') {
      onMessage(data.content);
    } else if (data.type === 'complete') {
      onComplete();
    } else if (data.type === 'error' || data.error) {
      onError(data.error);
    }
  };

  ws.onerror = (error) => {
    console.error('WebSocket error:', error);
    onError('WebSocket connection error');
  };

  ws.onclose = () => {
    console.log('WebSocket disconnected');
  };

  return ws;
};

// ============================================================================
// Training Management
// ============================================================================

export const getDefaultConfig = async () => {
  const response = await api.get('/training/config/default');
  return response.data;
};

export const createTrainingJob = async (config) => {
  const response = await api.post('/training/create', config);
  return response.data;
};

export const startTrainingJob = async (jobId) => {
  const response = await api.post(`/training/start/${jobId}`);
  return response.data;
};

export const listTrainingJobs = async () => {
  const response = await api.get('/training/jobs');
  return response.data;
};

export const getTrainingJob = async (jobId) => {
  const response = await api.get(`/training/job/${jobId}`);
  return response.data;
};

export const cancelTrainingJob = async (jobId) => {
  const response = await api.post(`/training/cancel/${jobId}`);
  return response.data;
};

// ============================================================================
// Health Check
// ============================================================================

export const checkHealth = async () => {
  const response = await api.get('/health');
  return response.data;
};

export default api;
