import React, { useState } from 'react';
import { Brain, MessageSquare, TrendingUp, Github } from 'lucide-react';
import ChatInterface from './components/ChatInterface';
import TrainingDashboard from './components/TrainingDashboard';
import ModelManager from './components/ModelManager';

function App() {
  const [activeTab, setActiveTab] = useState('chat');
  const [modelLoaded, setModelLoaded] = useState(false);

  const tabs = [
    { id: 'chat', name: 'Chat', icon: MessageSquare },
    { id: 'training', name: 'Training', icon: TrendingUp },
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="bg-gradient-to-br from-blue-600 to-purple-600 p-2 rounded-lg">
                <Brain className="w-8 h-8 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-gray-900">
                  GPT-OSS Fine-tuning Platform
                </h1>
                <p className="text-sm text-gray-600">
                  Train and interact with large language models
                </p>
              </div>
            </div>
            <a
              href="https://github.com/unslothai/unsloth"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 px-4 py-2 bg-gray-900 text-white rounded-lg hover:bg-gray-800 transition-colors"
            >
              <Github className="w-5 h-5" />
              <span className="hidden sm:inline">Unsloth</span>
            </a>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-6 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Sidebar - Model Manager */}
          <div className="lg:col-span-1">
            <ModelManager onModelStatusChange={setModelLoaded} />
          </div>

          {/* Main Content Area */}
          <div className="lg:col-span-2">
            {/* Tab Navigation */}
            <div className="bg-white rounded-lg shadow-lg mb-6">
              <div className="flex border-b border-gray-200">
                {tabs.map((tab) => {
                  const Icon = tab.icon;
                  return (
                    <button
                      key={tab.id}
                      onClick={() => setActiveTab(tab.id)}
                      className={`flex-1 px-6 py-4 text-sm font-medium transition-colors flex items-center justify-center gap-2 ${
                        activeTab === tab.id
                          ? 'text-blue-600 border-b-2 border-blue-600 bg-blue-50'
                          : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
                      }`}
                    >
                      <Icon className="w-5 h-5" />
                      {tab.name}
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Tab Content */}
            <div className="h-[calc(100vh-280px)]">
              {activeTab === 'chat' && <ChatInterface modelLoaded={modelLoaded} />}
              {activeTab === 'training' && <TrainingDashboard />}
            </div>
          </div>
        </div>

        {/* Footer */}
        <footer className="mt-8 text-center text-sm text-gray-600">
          <p>
            Built with{' '}
            <a
              href="https://docs.unsloth.ai/"
              target="_blank"
              rel="noopener noreferrer"
              className="text-blue-600 hover:underline"
            >
              Unsloth
            </a>
            {' • '}
            Fast fine-tuning for large language models
          </p>
        </footer>
      </main>
    </div>
  );
}

export default App;
