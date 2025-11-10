# Testing Guide

This document provides comprehensive information about testing the GPT-OSS Fine-tuning Platform.

## Table of Contents

- [Overview](#overview)
- [Backend Testing](#backend-testing)
- [Frontend Testing](#frontend-testing)
- [Running Tests](#running-tests)
- [Test Coverage](#test-coverage)
- [Writing Tests](#writing-tests)
- [CI/CD Integration](#cicd-integration)

## Overview

The project uses comprehensive testing to ensure code quality and reliability:

- **Backend**: pytest for Python tests
- **Frontend**: Vitest + React Testing Library for JavaScript/React tests
- **Coverage**: Both backend and frontend track test coverage

## Backend Testing

### Technology Stack

- **pytest**: Testing framework
- **pytest-asyncio**: Async test support
- **pytest-cov**: Coverage reporting
- **pytest-mock**: Mocking utilities
- **httpx**: HTTP client for API testing

### Test Structure

```
backend/
├── tests/
│   ├── __init__.py
│   ├── conftest.py              # Shared fixtures
│   ├── test_model_manager.py    # Model management tests
│   ├── test_training_manager.py # Training management tests
│   └── test_api.py              # API integration tests
├── pytest.ini                   # Pytest configuration
└── requirements-dev.txt         # Testing dependencies
```

### Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### Running Backend Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_model_manager.py

# Run specific test
pytest tests/test_model_manager.py::TestModelManager::test_load_model

# Run with coverage
pytest --cov=. --cov-report=html

# Run in parallel (faster)
pytest -n auto
```

### Backend Test Categories

#### 1. Model Manager Tests (`test_model_manager.py`)

Tests for model loading, inference, and management:

- Model initialization
- Loading models with Unsloth
- Fallback to transformers
- Model unloading
- Text generation (streaming and non-streaming)
- Error handling
- Status reporting

**Example:**
```python
def test_load_model_with_unsloth(model_manager_instance):
    """Test loading model with Unsloth"""
    result = model_manager_instance.load_model(
        model_name="test-model",
        max_seq_length=512,
        load_in_4bit=True
    )

    assert result["status"] == "success"
    assert result["loaded"] is True
```

#### 2. Training Manager Tests (`test_training_manager.py`)

Tests for training job management:

- Job creation
- Job lifecycle (pending → running → completed/failed/cancelled)
- Job status tracking
- Progress monitoring
- Error handling
- Metrics tracking

**Example:**
```python
def test_create_job(training_manager_instance, sample_training_config):
    """Test creating a training job"""
    job = training_manager_instance.create_job(sample_training_config)

    assert job.status == "pending"
    assert job.config == sample_training_config
```

#### 3. API Integration Tests (`test_api.py`)

Tests for FastAPI endpoints:

- Health checks
- Model management endpoints
- Inference endpoints
- Training endpoints
- WebSocket streaming
- Request validation
- CORS headers

**Example:**
```python
def test_inference_success(test_client, sample_inference_request):
    """Test successful inference"""
    response = test_client.post("/inference", json=sample_inference_request)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
```

### Backend Fixtures

Available fixtures in `conftest.py`:

- `mock_torch_cuda` - Mock CUDA functions
- `mock_model` - Mock language model
- `mock_tokenizer` - Mock tokenizer
- `model_manager_instance` - Fresh ModelManager instance
- `training_manager_instance` - Fresh TrainingManager instance
- `test_client` - FastAPI test client
- `sample_training_config` - Sample training configuration
- `sample_inference_request` - Sample inference request

## Frontend Testing

### Technology Stack

- **Vitest**: Fast unit testing framework
- **React Testing Library**: React component testing
- **@testing-library/jest-dom**: Custom matchers
- **@testing-library/user-event**: User interaction simulation
- **jsdom**: DOM environment for tests

### Test Structure

```
frontend/
├── src/
│   ├── tests/
│   │   └── setup.js                        # Test setup
│   ├── components/
│   │   └── __tests__/
│   │       ├── ChatInterface.test.jsx
│   │       ├── ModelManager.test.jsx
│   │       └── TrainingDashboard.test.jsx
│   └── services/
│       └── __tests__/
│           └── api.test.js
├── vitest.config.js                        # Vitest configuration
└── package.json
```

### Setup

```bash
cd frontend

# Install dependencies
npm install
```

### Running Frontend Tests

```bash
# Run all tests
npm test

# Run in watch mode
npm test -- --watch

# Run with UI
npm run test:ui

# Run with coverage
npm run test:coverage

# Run specific test file
npm test -- ChatInterface.test.jsx

# Run specific test
npm test -- -t "should send message"
```

### Frontend Test Categories

#### 1. Component Tests

##### ChatInterface Tests (`ChatInterface.test.jsx`)

Tests for chat functionality:

- Component rendering
- Message sending (click and Enter key)
- Streaming vs non-streaming modes
- Parameter adjustments (temperature, reasoning effort, max tokens)
- Chat clearing
- Error handling
- Model status awareness

**Example:**
```javascript
it('should send message on button click', async () => {
  const user = userEvent.setup()
  render(<ChatInterface modelLoaded={true} />)

  const textarea = screen.getByPlaceholderText(/Type your message/i)
  await user.type(textarea, 'Hello')

  const sendButton = screen.getByText('Send')
  await user.click(sendButton)

  await waitFor(() => {
    expect(screen.getByText('Hello')).toBeInTheDocument()
  })
})
```

##### ModelManager Tests (`ModelManager.test.jsx`)

Tests for model management:

- Status display
- Model loading/unloading
- Configuration updates
- Control state management
- Error handling
- Periodic status refresh

**Example:**
```javascript
it('should load model on button click', async () => {
  const user = userEvent.setup()
  render(<ModelManager onModelStatusChange={mockCallback} />)

  const loadButton = screen.getByText('Load Model')
  await user.click(loadButton)

  await waitFor(() => {
    expect(api.loadModel).toHaveBeenCalled()
  })
})
```

##### TrainingDashboard Tests (`TrainingDashboard.test.jsx`)

Tests for training management:

- Configuration display and editing
- Job creation and starting
- Progress monitoring
- Job cancellation
- Training history
- Metrics display

**Example:**
```javascript
it('should create and start training job', async () => {
  const user = userEvent.setup()
  render(<TrainingDashboard />)

  const startButton = screen.getByText('Start New Training Job')
  await user.click(startButton)

  await waitFor(() => {
    expect(api.createTrainingJob).toHaveBeenCalled()
    expect(api.startTrainingJob).toHaveBeenCalled()
  })
})
```

#### 2. Service Tests

##### API Service Tests (`api.test.js`)

Tests for API client:

- Model management API calls
- Inference requests
- WebSocket streaming
- Training job management
- Error handling
- Request/response formatting

**Example:**
```javascript
it('should load a model', async () => {
  const config = {
    model_name: 'test-model',
    max_seq_length: 1024,
  }

  const result = await loadModel(config)

  expect(result.status).toBe('success')
  expect(result.loaded).toBe(true)
})
```

### Mocking

#### API Mocking

```javascript
import { vi } from 'vitest'
import * as api from '../../services/api'

vi.mock('../../services/api')

// Mock implementation
api.generateText.mockResolvedValue({
  status: 'success',
  generated_text: 'Response'
})
```

#### WebSocket Mocking

```javascript
const mockWs = {
  send: vi.fn(),
  close: vi.fn(),
  onopen: null,
  onmessage: null,
}

global.WebSocket = vi.fn(() => mockWs)
```

## Test Coverage

### Viewing Coverage Reports

#### Backend

After running `pytest --cov=. --cov-report=html`:

```bash
# Open coverage report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

#### Frontend

After running `npm run test:coverage`:

```bash
# Open coverage report
open coverage/index.html  # macOS
xdg-open coverage/index.html  # Linux
start coverage/index.html  # Windows
```

### Coverage Goals

- **Overall**: > 80%
- **Critical paths** (model loading, inference, training): > 90%
- **UI components**: > 70%

## Writing Tests

### Best Practices

#### 1. Test Structure

Follow the **Arrange-Act-Assert** pattern:

```python
def test_example():
    # Arrange: Set up test data
    config = {"model_name": "test"}

    # Act: Perform action
    result = load_model(config)

    # Assert: Verify results
    assert result["status"] == "success"
```

#### 2. Test Names

Use descriptive names:

```python
# Good
def test_load_model_with_4bit_quantization()

# Bad
def test_load_model()
```

#### 3. Mock External Dependencies

```python
@patch('module.external_function')
def test_feature(mock_external):
    mock_external.return_value = "mocked"
    # Test code
```

#### 4. Test Error Cases

```python
def test_load_model_error_handling():
    with pytest.raises(ValueError):
        load_model(invalid_config)
```

#### 5. Use Fixtures

```python
@pytest.fixture
def sample_data():
    return {"key": "value"}

def test_with_fixture(sample_data):
    assert sample_data["key"] == "value"
```

### Frontend Testing Tips

#### 1. User-Centric Testing

Test from the user's perspective:

```javascript
// Good - tests what user sees
expect(screen.getByText('Load Model')).toBeInTheDocument()

// Bad - tests implementation details
expect(component.state.loading).toBe(false)
```

#### 2. Async Operations

Use `waitFor` for async operations:

```javascript
await waitFor(() => {
  expect(screen.getByText('Success')).toBeInTheDocument()
})
```

#### 3. User Events

Use `@testing-library/user-event`:

```javascript
const user = userEvent.setup()
await user.click(button)
await user.type(input, 'text')
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  backend-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          cd backend
          pip install -r requirements.txt
          pip install -r requirements-dev.txt
      - name: Run tests
        run: |
          cd backend
          pytest --cov=. --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v2

  frontend-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Node.js
        uses: actions/setup-node@v2
        with:
          node-version: '18'
      - name: Install dependencies
        run: |
          cd frontend
          npm install
      - name: Run tests
        run: |
          cd frontend
          npm run test:coverage
      - name: Upload coverage
        uses: codecov/codecov-action@v2
```

## Troubleshooting

### Common Issues

#### Backend

**Issue**: Import errors in tests
```bash
# Solution: Install package in editable mode
pip install -e .
```

**Issue**: CUDA-related test failures
```bash
# Solution: Tests should mock CUDA (already configured in conftest.py)
```

#### Frontend

**Issue**: Module not found errors
```bash
# Solution: Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install
```

**Issue**: Tests hanging
```bash
# Solution: Check for missing await statements or unresolved promises
```

## Additional Resources

- [pytest Documentation](https://docs.pytest.org/)
- [Vitest Documentation](https://vitest.dev/)
- [React Testing Library](https://testing-library.com/react)
- [Testing Best Practices](https://testingjavascript.com/)

## Summary

- **Backend**: 35+ tests covering model management, training, and API
- **Frontend**: 40+ tests covering components and services
- **Coverage**: Comprehensive coverage of critical functionality
- **Integration**: Ready for CI/CD pipelines

Run tests before every commit to ensure code quality! 🚀
