import React, { useState } from 'react';
import { Brain, MessageSquare, TrendingUp, Github, Zap } from 'lucide-react';
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
              {/* Trainer One Logo */}
              <div className="relative">
                <div className="bg-gradient-to-br from-blue-600 via-purple-600 to-indigo-700 p-2.5 rounded-xl shadow-lg">
                  <Brain className="w-8 h-8 text-white" />
                </div>
                <div className="absolute -bottom-1 -right-1 bg-yellow-400 rounded-full p-1">
                  <Zap className="w-3 h-3 text-gray-900" />
                </div>
              </div>
              <div>
                <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                  Trainer One
                </h1>
                <p className="text-sm text-gray-600">
                  Professional AI Model Fine-tuning Platform
                </p>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <a
                href="https://github.com/aditya-gaharawar/trainer-one"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-lg hover:from-blue-700 hover:to-purple-700 transition-all shadow-md hover:shadow-lg"
              >
                <Github className="w-5 h-5" />
                <span className="hidden sm:inline">View on GitHub</span>
              </a>
            </div>
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
          <div className="flex items-center justify-center gap-2 mb-2">
            <Brain className="w-4 h-4 text-purple-600" />
            <span className="font-semibold text-gray-800">Trainer One</span>
          </div>
          <p>
            Powered by{' '}
            <a
              href="https://docs.unsloth.ai/"
              target="_blank"
              rel="noopener noreferrer"
              className="text-blue-600 hover:underline font-medium"
            >
              Unsloth
            </a>
            {' • '}
            Professional-grade LLM fine-tuning made simple
          </p>
          <p className="mt-1 text-xs text-gray-500">
            © 2024 Trainer One. All rights reserved.
          </p>
        </footer>
      </main>
    </div>
  );
}

export default App;
