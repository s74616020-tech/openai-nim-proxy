const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json({ limit: '50mb' }));

const OPENROUTER_API_BASE = 'https://openrouter.ai/api/v1';
const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY;

const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'meta-llama/llama-3.1-8b-instruct:free',
  'gpt-4': 'meta-llama/llama-3.3-70b-instruct:free',
  'gpt-4-turbo': 'meta-llama/llama-3.3-70b-instruct:free',
  'gpt-4o': 'openai/gpt-oss-120b:free',
  'claude-3-opus': 'nvidia/nemotron-3-ultra-550b-a55b:free',
  'claude-3-sonnet': 'meta-llama/llama-3.3-70b-instruct:free',
  'gemini-pro': 'google/gemma-4-31b-it:free',

  // Type these directly in Janitor AI
  'meta-llama/llama-3.3-70b-instruct:free': 'meta-llama/llama-3.3-70b-instruct:free',
  'meta-llama/llama-3.1-8b-instruct:free': 'meta-llama/llama-3.1-8b-instruct:free',
  'openai/gpt-oss-120b:free': 'openai/gpt-oss-120b:free',
  'nvidia/llama-3.1-nemotron-ultra-253b-v1:free': 'nvidia/llama-3.1-nemotron-ultra-253b-v1:free',
  'google/gemma-4-31b-it:free': 'google/gemma-4-31b-it:free',
  'deepseek/deepseek-r1:free': 'deepseek/deepseek-r1:free',
  'mistralai/mistral-small-3.2-24b-instruct:free': 'mistralai/mistral-small-3.2-24b-instruct:free'
};

const DEFAULT_MODEL = 'meta-llama/llama-3.3-70b-instruct:free';

app.get('/health', (req, res) => {
  res.json({ status: 'ok', service: 'OpenRouter Proxy' });
});

app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(id => ({
    id,
    object: 'model',
    created: Date.now(),
    owned_by: 'openrouter'
  }));
  res.json({ object: 'list', data: models });
});

app.post(['/chat/completions', '/v1/chat/completions'], async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens } = req.body;

    const orModel = MODEL_MAPPING[model] || DEFAULT_MODEL;

    console.log(`Request for model: ${model} → ${orModel}`);

    const orRequest = {
      model: orModel,
      messages: messages,
      temperature: temperature || 0.7,
      max_tokens: max_tokens || 2048,
      stream: false
    };

    const response = await axios.post(
      `${OPENROUTER_API_BASE}/chat/completions`,
      orRequest,
      {
        headers: {
          'Authorization': `Bearer ${OPENROUTER_API_KEY}`,
          'Content-Type': 'application/json',
          'HTTP-Referer': 'https://janitorai.com',
          'X-Title': 'JanitorAI Proxy'
        },
        timeout: 60000
      }
    );

    res.json({
      id: `chatcmpl-${Date.now()}`,
      object: 'chat.completion',
      created: Math.floor(Date.now() / 1000),
      model: model,
      choices: response.data.choices.map(choice => ({
        index: choice.index,
        message: {
          role: choice.message.role,
          content: choice.message.content || ''
        },
        finish_reason: choice.finish_reason
      })),
      usage: response.data.usage || {
        prompt_tokens: 0,
        completion_tokens: 0,
        total_tokens: 0
      }
    });

  } catch (error) {
    const status = error.response?.status || 500;
    const body = error.response?.data;
    console.error(`Error ${status}:`, typeof body === 'object' ? JSON.stringify(body) : body || error.message);
    res.status(status).json({
      error: {
        message: error.response?.data?.error?.message || error.message || 'Internal server error',
        type: 'invalid_request_error',
        code: status
      }
    });
  }
});

app.all('*', (req, res) => {
  res.status(404).json({
    error: {
      message: `Endpoint ${req.path} not found`,
      type: 'invalid_request_error',
      code: 404
    }
  });
});

app.listen(PORT, () => {
  console.log(`Proxy running on port ${PORT}`);
});
