import React, { useState, useEffect } from 'react';
import { Download, Upload, Database, Cpu, MemoryStick, Loader2 } from 'lucide-react';
import { loadModel, unloadModel, getModelStatus } from '../services/api';

const ModelManager = ({ onModelStatusChange }) => {
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(false);
  const [modelName, setModelName] = useState('unsloth/gpt-oss-20b');
  const [maxSeqLength, setMaxSeqLength] = useState(1024);
  const [loadIn4Bit, setLoadIn4Bit] = useState(true);

  useEffect(() => {
    refreshStatus();
    const interval = setInterval(refreshStatus, 5000);
    return () => clearInterval(interval);
  }, []);

  const refreshStatus = async () => {
    try {
      const statusData = await getModelStatus();
      setStatus(statusData);
      if (onModelStatusChange) {
        onModelStatusChange(statusData.loaded);
      }
    } catch (error) {
      console.error('Error fetching model status:', error);
    }
  };

  const handleLoadModel = async () => {
    setLoading(true);
    try {
      const result = await loadModel({
        model_name: modelName,
        max_seq_length: maxSeqLength,
        load_in_4bit: loadIn4Bit,
      });

      if (result.status === 'success') {
        await refreshStatus();
        alert('Model loaded successfully!');
      } else {
        alert(`Error: ${result.error}`);
      }
    } catch (error) {
      console.error('Error loading model:', error);
      alert(`Error: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleUnloadModel = async () => {
    setLoading(true);
    try {
      const result = await unloadModel();

      if (result.status === 'success') {
        await refreshStatus();
        alert('Model unloaded successfully!');
      } else {
        alert(`Error: ${result.error}`);
      }
    } catch (error) {
      console.error('Error unloading model:', error);
      alert(`Error: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const formatMemory = (bytes) => {
    if (!bytes) return 'N/A';
    const mb = bytes / (1024 * 1024);
    const gb = mb / 1024;
    return gb > 1 ? `${gb.toFixed(2)} GB` : `${mb.toFixed(2)} MB`;
  };

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <h2 className="text-xl font-semibold text-gray-800 mb-4 flex items-center gap-2">
        <Database className="w-6 h-6" />
        Model Manager
      </h2>

      {/* Status Display */}
      <div className="mb-6 p-4 bg-gray-50 rounded-lg border border-gray-200">
        <h3 className="text-sm font-semibold text-gray-700 mb-3">Current Status</h3>
        <div className="grid grid-cols-2 gap-4">
          <div className="flex items-center gap-2">
            <div
              className={`w-3 h-3 rounded-full ${
                status?.loaded ? 'bg-green-500' : 'bg-red-500'
              }`}
            />
            <span className="text-sm font-medium">
              {status?.loaded ? 'Model Loaded' : 'No Model Loaded'}
            </span>
          </div>

          {status?.model_name && (
            <div className="flex items-center gap-2">
              <Database className="w-4 h-4 text-gray-600" />
              <span className="text-sm text-gray-600 truncate" title={status.model_name}>
                {status.model_name}
              </span>
            </div>
          )}

          <div className="flex items-center gap-2">
            <Cpu className="w-4 h-4 text-gray-600" />
            <span className="text-sm text-gray-600">
              CUDA: {status?.cuda_available ? 'Available' : 'Not Available'}
            </span>
          </div>

          {status?.cuda_memory_allocated !== null && (
            <div className="flex items-center gap-2">
              <MemoryStick className="w-4 h-4 text-gray-600" />
              <span className="text-sm text-gray-600">
                Memory: {formatMemory(status.cuda_memory_allocated)}
              </span>
            </div>
          )}
        </div>
      </div>

      {/* Model Configuration */}
      <div className="mb-6">
        <h3 className="text-sm font-semibold text-gray-700 mb-3">Model Configuration</h3>
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Model Name
            </label>
            <select
              value={modelName}
              onChange={(e) => setModelName(e.target.value)}
              disabled={status?.loaded || loading}
              className="w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-100 disabled:cursor-not-allowed"
            >
              <option value="unsloth/gpt-oss-20b">GPT-OSS 20B</option>
              <option value="unsloth/gpt-oss-20b-unsloth-bnb-4bit">
                GPT-OSS 20B (4-bit)
              </option>
              <option value="unsloth/gpt-oss-120b">GPT-OSS 120B</option>
              <option value="unsloth/gpt-oss-120b-unsloth-bnb-4bit">
                GPT-OSS 120B (4-bit)
              </option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Max Sequence Length: {maxSeqLength}
            </label>
            <input
              type="range"
              min="512"
              max="4096"
              step="512"
              value={maxSeqLength}
              onChange={(e) => setMaxSeqLength(parseInt(e.target.value))}
              disabled={status?.loaded || loading}
              className="w-full disabled:opacity-50 disabled:cursor-not-allowed"
            />
            <div className="flex justify-between text-xs text-gray-500 mt-1">
              <span>512</span>
              <span>2048</span>
              <span>4096</span>
            </div>
          </div>

          <div className="flex items-center">
            <input
              type="checkbox"
              id="load4bit"
              checked={loadIn4Bit}
              onChange={(e) => setLoadIn4Bit(e.target.checked)}
              disabled={status?.loaded || loading}
              className="mr-2 disabled:cursor-not-allowed"
            />
            <label
              htmlFor="load4bit"
              className="text-sm font-medium text-gray-700 cursor-pointer"
            >
              Load in 4-bit (Reduces memory usage by ~75%)
            </label>
          </div>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="flex gap-3">
        {!status?.loaded ? (
          <button
            onClick={handleLoadModel}
            disabled={loading}
            className="flex-1 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors flex items-center justify-center gap-2 font-semibold"
          >
            {loading ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                Loading Model...
              </>
            ) : (
              <>
                <Download className="w-5 h-5" />
                Load Model
              </>
            )}
          </button>
        ) : (
          <button
            onClick={handleUnloadModel}
            disabled={loading}
            className="flex-1 py-3 bg-red-600 text-white rounded-lg hover:bg-red-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors flex items-center justify-center gap-2 font-semibold"
          >
            {loading ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                Unloading Model...
              </>
            ) : (
              <>
                <Upload className="w-5 h-5" />
                Unload Model
              </>
            )}
          </button>
        )}
      </div>

      {/* Info Box */}
      <div className="mt-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
        <h4 className="text-sm font-semibold text-blue-900 mb-2">Model Information</h4>
        <ul className="text-xs text-blue-800 space-y-1">
          <li>• GPT-OSS is a 20B/120B parameter multilingual reasoning model</li>
          <li>• Supports adjustable reasoning effort (low/medium/high)</li>
          <li>• 4-bit quantization recommended for consumer GPUs</li>
          <li>• Loading may take several minutes depending on your hardware</li>
        </ul>
      </div>
    </div>
  );
};

export default ModelManager;
