// server.js - OpenAI to NVIDIA NIM API Proxy (with OpenRouter fallback)
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// Prevent one bad request (e.g. an error thrown inside a streaming callback)
// from crashing the whole process and dropping every other in-flight request.
process.on('uncaughtException', (err) => {
  console.error('UNCAUGHT EXCEPTION (process kept alive):', err);
});
process.on('unhandledRejection', (reason) => {
  console.error('UNHANDLED REJECTION (process kept alive):', reason);
});

// Middleware
app.use(cors());
app.use(express.json());

// NVIDIA NIM API configuration
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

// OpenRouter fallback configuration (used automatically if NIM fails)
const OPENROUTER_API_BASE = process.env.OPENROUTER_API_BASE || 'https://openrouter.ai/api/v1';
const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY;
// If a model isn't in OPENROUTER_MODEL_MAPPING below, this is used as the
// fallback default when NIM fails and we retry against OpenRouter.
const OPENROUTER_DEFAULT_MODEL = process.env.OPENROUTER_DEFAULT_MODEL || 'meta-llama/llama-3.1-8b-instruct';

// 🔥 REASONING DISPLAY TOGGLE - Shows/hides reasoning in output
const SHOW_REASONING = false; // Set to true to show reasoning with <think> tags

// 🔥 THINKING MODE TOGGLE - Enables thinking for specific models that support it
const ENABLE_THINKING_MODE = false; // Set to true to enable chat_template_kwargs thinking parameter

// Optional aliasing for NIM - if a requested model matches a key here, it's
// swapped for the mapped NIM slug. Anything else passes straight through.
const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'nvidia/llama-3.1-nemotron-ultra-253b-v1',
  'gpt-4': 'qwen/qwen3-coder-480b-a35b-instruct',
  'gpt-4-turbo': 'moonshotai/kimi-k2-instruct-0905',
  'gpt-4o': 'deepseek-ai/deepseek-v3.1',
  'claude-3-opus': 'openai/gpt-oss-120b',
  'claude-3-sonnet': 'openai/gpt-oss-20b',
  'gemini-pro': 'qwen/qwen3-next-80b-a3b-thinking'
};

// Optional aliasing for OpenRouter fallback - translate the same friendly
// names (or NIM slugs you commonly use) into their OpenRouter equivalents.
// Anything not listed here falls back to OPENROUTER_DEFAULT_MODEL.
const OPENROUTER_MODEL_MAPPING = {
  'gpt-3.5-turbo': 'meta-llama/llama-3.1-8b-instruct',
  'gpt-4': 'qwen/qwen3-coder',
  'gpt-4-turbo': 'moonshotai/kimi-k2',
  'gpt-4o': 'deepseek/deepseek-chat',
  'claude-3-opus': 'openai/gpt-oss-120b',
  'claude-3-sonnet': 'openai/gpt-oss-20b',
  'gemini-pro': 'qwen/qwen3-next-80b-a3b-thinking',
  'deepseek-ai/deepseek-v3.1': 'deepseek/deepseek-chat',
  'deepseek-ai/deepseek-v4-flash': 'deepseek/deepseek-chat',
  'meta/llama-3.1-8b-instruct': 'meta-llama/llama-3.1-8b-instruct',
  'meta/llama-3.1-70b-instruct': 'meta-llama/llama-3.1-70b-instruct',
  'meta/llama-3.3-70b-instruct': 'meta-llama/llama-3.3-70b-instruct'
};

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'OpenAI to NVIDIA NIM Proxy',
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE,
    openrouter_fallback_configured: Boolean(OPENROUTER_API_KEY)
  });
});

// List models endpoint (OpenAI compatible)
app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(model => ({
    id: model,
    object: 'model',
    created: Date.now(),
    owned_by: 'nvidia-nim-proxy'
  }));

  res.json({
    object: 'list',
    data: models
  });
});

// Reads a Node stream fully into a string. Used for safely reading error
// bodies from upstream when responseType was 'stream'.
function streamToString(stream) {
  return new Promise((resolve, reject) => {
    let raw = '';
    stream.on('data', (chunk) => { raw += chunk.toString(); });
    stream.on('end', () => resolve(raw));
    stream.on('error', reject);
  });
}

// Safely extract a readable error message/detail from an axios error,
// whether the upstream sent JSON, plain text, or (for stream requests) a
// raw stream that needs to be read first.
async function extractErrorDetail(error) {
  if (!error.response) {
    return error.message || 'Unknown error';
  }
  const data = error.response.data;
  if (typeof data === 'string') return data;
  if (data && (typeof data.pipe === 'function' || typeof data.on === 'function')) {
    try {
      const text = await streamToString(data);
      return text || 'Empty error response from upstream';
    } catch (streamErr) {
      return 'Failed to read upstream error stream: ' + streamErr.message;
    }
  }
  try {
    return JSON.stringify(data);
  } catch (e) {
    return '[Unserializable error response from upstream]';
  }
}

// Makes the actual call to a provider (NIM or OpenRouter). Returns the
// axios response. Throws on failure (network error or non-2xx status).
async function callProvider({ baseURL, apiKey, model, messages, temperature, max_tokens, stream, enableThinking }) {
  const requestBody = {
    model,
    messages,
    temperature: temperature || 0.6,
    max_tokens: max_tokens || 9024,
    stream: stream || false
  };
  if (enableThinking) {
    requestBody.extra_body = { chat_template_kwargs: { thinking: true } };
  }

  return axios.post(`${baseURL}/chat/completions`, requestBody, {
    headers: {
      'Authorization': `Bearer ${apiKey}`,
      'Content-Type': 'application/json'
    },
    responseType: stream ? 'stream' : 'json',
    timeout: 60000
  });
}

// Streams a provider's SSE response back to the client, translating
// reasoning_content into <think> tags if SHOW_REASONING is on.
function pipeStreamToClient(providerResponse, res) {
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');

  let buffer = '';
  let reasoningStarted = false;

  // Watchdog: if no data arrives for 30s mid-stream, kill the connection
  // instead of leaving the client spinning forever.
  let watchdog = setTimeout(() => {
    console.error('Stream stalled — no data for 30s, closing connection');
    res.end();
  }, 30000);

  providerResponse.data.on('data', (chunk) => {
    clearTimeout(watchdog);
    watchdog = setTimeout(() => {
      console.error('Stream stalled — no data for 30s, closing connection');
      res.end();
    }, 30000);

    buffer += chunk.toString();
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';

    lines.forEach(line => {
      if (line.startsWith('data: ')) {
        if (line.includes('[DONE]')) {
          res.write(line + '\n');
          return;
        }

        try {
          const data = JSON.parse(line.slice(6));
          if (data.choices?.[0]?.delta) {
            const reasoning = data.choices[0].delta.reasoning_content;
            const content = data.choices[0].delta.content;

            if (SHOW_REASONING) {
              let combinedContent = '';

              if (reasoning && !reasoningStarted) {
                combinedContent = '<think>\n' + reasoning;
                reasoningStarted = true;
              } else if (reasoning) {
                combinedContent = reasoning;
              }

              if (content && reasoningStarted) {
                combinedContent += '</think>\n\n' + content;
                reasoningStarted = false;
              } else if (content) {
                combinedContent += content;
              }

              if (combinedContent) {
                data.choices[0].delta.content = combinedContent;
                delete data.choices[0].delta.reasoning_content;
              }
            } else {
              data.choices[0].delta.content = content || '';
              delete data.choices[0].delta.reasoning_content;
            }
          }
          res.write(`data: ${JSON.stringify(data)}\n\n`);
        } catch (e) {
          res.write(line + '\n');
        }
      }
    });
  });

  providerResponse.data.on('end', () => {
    clearTimeout(watchdog);
    res.end();
  });
  providerResponse.data.on('error', (err) => {
    clearTimeout(watchdog);
    console.error('Stream error:', err);
    res.end();
  });
}

// Sends a non-streaming provider response back to the client in OpenAI format.
function sendJsonResponse(providerResponse, model, res) {
  const openaiResponse = {
    id: `chatcmpl-${Date.now()}`,
    object: 'chat.completion',
    created: Math.floor(Date.now() / 1000),
    model: model,
    choices: providerResponse.data.choices.map(choice => {
      let fullContent = choice.message?.content || '';

      if (SHOW_REASONING && choice.message?.reasoning_content) {
        fullContent = '<think>\n' + choice.message.reasoning_content + '\n</think>\n\n' + fullContent;
      }

      return {
        index: choice.index,
        message: {
          role: choice.message.role,
          content: fullContent
        },
        finish_reason: choice.finish_reason
      };
    }),
    usage: providerResponse.data.usage || {
      prompt_tokens: 0,
      completion_tokens: 0,
      total_tokens: 0
    }
  };

  res.json(openaiResponse);
}

// Chat completions endpoint (main proxy)
app.post(['/chat/completions', '/v1/chat/completions'], async (req, res) => {
  const { model, messages, temperature, max_tokens, stream } = req.body;

  if (!model) {
    return res.status(400).json({
      error: { message: 'No model specified in request', type: 'invalid_request_error', code: 400 }
    });
  }

  const nimModel = MODEL_MAPPING[model] || model;

  // --- Attempt 1: NVIDIA NIM ---
  try {
    const nimResponse = await callProvider({
      baseURL: NIM_API_BASE,
      apiKey: NIM_API_KEY,
      model: nimModel,
      messages, temperature, max_tokens, stream,
      enableThinking: ENABLE_THINKING_MODE
    });

    if (stream) {
      pipeStreamToClient(nimResponse, res);
    } else {
      sendJsonResponse(nimResponse, model, res);
    }
    return;
  } catch (nimError) {
    const nimErrorDetail = await extractErrorDetail(nimError);
    console.error('NIM attempt failed:', nimErrorDetail);

    if (!OPENROUTER_API_KEY) {
      // No fallback configured — return the NIM error as before.
      if (!res.headersSent) {
        res.status(nimError.response?.status || 500).json({
          error: { message: nimErrorDetail, type: 'invalid_request_error', code: nimError.response?.status || 500 }
        });
      } else {
        res.end();
      }
      return;
    }

    // --- Attempt 2: OpenRouter fallback ---
    const openrouterModel = OPENROUTER_MODEL_MAPPING[model] || OPENROUTER_DEFAULT_MODEL;
    console.log(`Falling back to OpenRouter with model: ${openrouterModel}`);

    try {
      const orResponse = await callProvider({
        baseURL: OPENROUTER_API_BASE,
        apiKey: OPENROUTER_API_KEY,
        model: openrouterModel,
        messages, temperature, max_tokens, stream,
        enableThinking: false // OpenRouter doesn't use NIM's chat_template_kwargs format
      });

      if (stream) {
        pipeStreamToClient(orResponse, res);
      } else {
        sendJsonResponse(orResponse, model, res);
      }
      return;
    } catch (orError) {
      const orErrorDetail = await extractErrorDetail(orError);
      console.error('OpenRouter fallback also failed:', orErrorDetail);

      if (!res.headersSent) {
        res.status(orError.response?.status || 500).json({
          error: {
            message: `Both providers failed. NIM: ${nimErrorDetail} | OpenRouter: ${orErrorDetail}`,
            type: 'invalid_request_error',
            code: orError.response?.status || 500
          }
        });
      } else {
        res.end();
      }
    }
  }
});

// Catch-all for unsupported endpoints
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
  console.log(`OpenAI to NVIDIA NIM Proxy running on port ${PORT}`);
  console.log(`Health check: http://localhost:${PORT}/health`);
  console.log(`Reasoning display: ${SHOW_REASONING ? 'ENABLED' : 'DISABLED'}`);
  console.log(`Thinking mode: ${ENABLE_THINKING_MODE ? 'ENABLED' : 'DISABLED'}`);
  console.log(`OpenRouter fallback: ${OPENROUTER_API_KEY ? 'CONFIGURED' : 'NOT CONFIGURED (set OPENROUTER_API_KEY env var)'}`);
});
