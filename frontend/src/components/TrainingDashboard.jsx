import React, { useState, useEffect } from 'react';
import {
  Play,
  Square,
  RefreshCw,
  Settings,
  TrendingUp,
  Clock,
  CheckCircle,
  XCircle,
  Loader2,
} from 'lucide-react';
import {
  getDefaultConfig,
  createTrainingJob,
  startTrainingJob,
  listTrainingJobs,
  getTrainingJob,
  cancelTrainingJob,
} from '../services/api';

const TrainingDashboard = () => {
  const [config, setConfig] = useState(null);
  const [jobs, setJobs] = useState([]);
  const [currentJob, setCurrentJob] = useState(null);
  const [loading, setLoading] = useState(false);
  const [showConfig, setShowConfig] = useState(false);

  useEffect(() => {
    loadDefaultConfig();
    loadJobs();
  }, []);

  useEffect(() => {
    let interval;
    if (currentJob && currentJob.status === 'running') {
      interval = setInterval(() => {
        refreshJobStatus(currentJob.job_id);
      }, 2000);
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [currentJob]);

  const loadDefaultConfig = async () => {
    try {
      const defaultConfig = await getDefaultConfig();
      setConfig(defaultConfig);
    } catch (error) {
      console.error('Error loading default config:', error);
    }
  };

  const loadJobs = async () => {
    try {
      const response = await listTrainingJobs();
      setJobs(response.jobs || []);

      // Find running job
      const running = response.jobs?.find((job) => job.status === 'running');
      if (running) {
        setCurrentJob(running);
      }
    } catch (error) {
      console.error('Error loading jobs:', error);
    }
  };

  const refreshJobStatus = async (jobId) => {
    try {
      const job = await getTrainingJob(jobId);
      setCurrentJob(job);

      // Update in jobs list
      setJobs((prev) =>
        prev.map((j) => (j.job_id === jobId ? job : j))
      );
    } catch (error) {
      console.error('Error refreshing job status:', error);
    }
  };

  const handleCreateAndStartJob = async () => {
    if (!config) return;

    setLoading(true);
    try {
      // Create job
      const createResponse = await createTrainingJob(config);
      const newJob = createResponse.job;

      // Start job
      await startTrainingJob(newJob.job_id);

      // Refresh job to get running status
      const updatedJob = await getTrainingJob(newJob.job_id);
      setCurrentJob(updatedJob);

      // Reload jobs list
      await loadJobs();
    } catch (error) {
      console.error('Error creating/starting job:', error);
      alert(`Error: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleCancelJob = async (jobId) => {
    try {
      await cancelTrainingJob(jobId);
      await refreshJobStatus(jobId);
      await loadJobs();
    } catch (error) {
      console.error('Error cancelling job:', error);
      alert(`Error: ${error.message}`);
    }
  };

  const updateConfigValue = (key, value) => {
    setConfig((prev) => ({
      ...prev,
      [key]: value,
    }));
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'running':
        return <Loader2 className="w-5 h-5 text-blue-600 animate-spin" />;
      case 'completed':
        return <CheckCircle className="w-5 h-5 text-green-600" />;
      case 'failed':
        return <XCircle className="w-5 h-5 text-red-600" />;
      case 'cancelled':
        return <Square className="w-5 h-5 text-gray-600" />;
      default:
        return <Clock className="w-5 h-5 text-gray-400" />;
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'running':
        return 'bg-blue-100 text-blue-800 border-blue-300';
      case 'completed':
        return 'bg-green-100 text-green-800 border-green-300';
      case 'failed':
        return 'bg-red-100 text-red-800 border-red-300';
      case 'cancelled':
        return 'bg-gray-100 text-gray-800 border-gray-300';
      default:
        return 'bg-yellow-100 text-yellow-800 border-yellow-300';
    }
  };

  if (!config) {
    return (
      <div className="flex items-center justify-center h-full">
        <Loader2 className="w-8 h-8 animate-spin text-blue-600" />
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full bg-white rounded-lg shadow-lg">
      {/* Header */}
      <div className="px-6 py-4 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <h2 className="text-xl font-semibold text-gray-800">Training Dashboard</h2>
          <div className="flex gap-2">
            <button
              onClick={() => setShowConfig(!showConfig)}
              className="px-4 py-2 text-sm bg-gray-100 text-gray-700 rounded-md hover:bg-gray-200 transition-colors flex items-center gap-2"
            >
              <Settings className="w-4 h-4" />
              {showConfig ? 'Hide' : 'Show'} Config
            </button>
            <button
              onClick={loadJobs}
              className="px-4 py-2 text-sm bg-gray-100 text-gray-700 rounded-md hover:bg-gray-200 transition-colors flex items-center gap-2"
            >
              <RefreshCw className="w-4 h-4" />
              Refresh
            </button>
          </div>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {/* Configuration Panel */}
        {showConfig && (
          <div className="bg-gray-50 border border-gray-200 rounded-lg p-4">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <Settings className="w-5 h-5" />
              Training Configuration
            </h3>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Model Name
                </label>
                <input
                  type="text"
                  value={config.model_name}
                  onChange={(e) => updateConfigValue('model_name', e.target.value)}
                  className="w-full text-sm border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Dataset Name
                </label>
                <input
                  type="text"
                  value={config.dataset_name}
                  onChange={(e) => updateConfigValue('dataset_name', e.target.value)}
                  className="w-full text-sm border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Max Steps
                </label>
                <input
                  type="number"
                  value={config.max_steps}
                  onChange={(e) => updateConfigValue('max_steps', parseInt(e.target.value))}
                  className="w-full text-sm border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Learning Rate
                </label>
                <input
                  type="number"
                  step="0.0001"
                  value={config.learning_rate}
                  onChange={(e) => updateConfigValue('learning_rate', parseFloat(e.target.value))}
                  className="w-full text-sm border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Batch Size
                </label>
                <input
                  type="number"
                  value={config.batch_size}
                  onChange={(e) => updateConfigValue('batch_size', parseInt(e.target.value))}
                  className="w-full text-sm border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  LoRA Rank
                </label>
                <input
                  type="number"
                  value={config.lora_r}
                  onChange={(e) => updateConfigValue('lora_r', parseInt(e.target.value))}
                  className="w-full text-sm border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>
            </div>
          </div>
        )}

        {/* Current Job Status */}
        {currentJob && currentJob.status === 'running' && (
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <div className="flex items-start justify-between mb-4">
              <div>
                <h3 className="text-lg font-semibold flex items-center gap-2">
                  <TrendingUp className="w-5 h-5 text-blue-600" />
                  Training in Progress
                </h3>
                <p className="text-sm text-gray-600 mt-1">Job ID: {currentJob.job_id}</p>
              </div>
              <button
                onClick={() => handleCancelJob(currentJob.job_id)}
                className="px-4 py-2 bg-red-600 text-white text-sm rounded-md hover:bg-red-700 transition-colors flex items-center gap-2"
              >
                <Square className="w-4 h-4" />
                Cancel
              </button>
            </div>

            {/* Progress Bar */}
            <div className="mb-4">
              <div className="flex justify-between text-sm text-gray-600 mb-2">
                <span>
                  Step {currentJob.current_step} / {currentJob.total_steps}
                </span>
                <span>{currentJob.progress?.toFixed(1)}%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-3">
                <div
                  className="bg-blue-600 h-3 rounded-full transition-all duration-300"
                  style={{ width: `${currentJob.progress}%` }}
                />
              </div>
            </div>

            {/* Metrics */}
            {currentJob.metrics && (
              <div className="grid grid-cols-3 gap-4 text-sm">
                <div className="bg-white rounded-md p-3">
                  <div className="text-gray-600">Loss</div>
                  <div className="text-lg font-semibold text-gray-800">
                    {currentJob.metrics.loss?.toFixed(4) || 'N/A'}
                  </div>
                </div>
                <div className="bg-white rounded-md p-3">
                  <div className="text-gray-600">Learning Rate</div>
                  <div className="text-lg font-semibold text-gray-800">
                    {currentJob.metrics.learning_rate?.toExponential(2) || 'N/A'}
                  </div>
                </div>
                <div className="bg-white rounded-md p-3">
                  <div className="text-gray-600">Epoch</div>
                  <div className="text-lg font-semibold text-gray-800">
                    {currentJob.metrics.epoch?.toFixed(2) || 'N/A'}
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Start Training Button */}
        {(!currentJob || currentJob.status !== 'running') && (
          <button
            onClick={handleCreateAndStartJob}
            disabled={loading}
            className="w-full py-4 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors flex items-center justify-center gap-3 text-lg font-semibold"
          >
            {loading ? (
              <>
                <Loader2 className="w-6 h-6 animate-spin" />
                Starting Training...
              </>
            ) : (
              <>
                <Play className="w-6 h-6" />
                Start New Training Job
              </>
            )}
          </button>
        )}

        {/* Training Jobs History */}
        <div>
          <h3 className="text-lg font-semibold mb-4">Training History</h3>
          {jobs.length === 0 ? (
            <div className="text-center text-gray-400 py-8">
              <TrendingUp className="w-12 h-12 mx-auto mb-2 opacity-50" />
              <p>No training jobs yet</p>
            </div>
          ) : (
            <div className="space-y-3">
              {jobs.slice().reverse().map((job) => (
                <div
                  key={job.job_id}
                  className={`border rounded-lg p-4 ${getStatusColor(job.status)}`}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      {getStatusIcon(job.status)}
                      <div>
                        <div className="font-medium">{job.job_id}</div>
                        <div className="text-sm opacity-75">
                          {job.status.charAt(0).toUpperCase() + job.status.slice(1)}
                        </div>
                      </div>
                    </div>
                    <div className="text-sm">
                      <div>
                        Steps: {job.current_step} / {job.total_steps}
                      </div>
                      {job.progress !== undefined && (
                        <div className="text-right">{job.progress.toFixed(1)}%</div>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default TrainingDashboard;
