// server.js - OpenAI to NVIDIA NIM API Proxy
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());

// NVIDIA NIM API configuration
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

// 🔥 REASONING DISPLAY TOGGLE - Shows/hides reasoning in output
const SHOW_REASONING = false; // Set to true to show reasoning with <think> tags

// 🔥 THINKING MODE TOGGLE - Enables thinking for specific models that support it
const ENABLE_THINKING_MODE = false; // Set to true to enable chat_template_kwargs thinking parameter

// Fallback model used when the incoming `model` isn't one of our known aliases
const FALLBACK_MODEL = 'deepseek-ai/deepseek-v4-flash';

// Model mapping (adjust based on available NIM models)
const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'nvidia/llama-3.1-nemotron-ultra-253b-v1',
  'gpt-4': 'qwen/qwen3-coder-480b-a35b-instruct',
  'gpt-4-turbo': 'moonshotai/kimi-k2-instruct-0905',
  'gpt-4o': 'deepseek-ai/deepseek-v3.1',
  'claude-3-opus': 'openai/gpt-oss-120b',
  'claude-3-sonnet': 'openai/gpt-oss-20b',
  'gemini-pro': 'qwen/qwen3-next-80b-a3b-thinking'
};

// If the requested model is overloaded (ResourceExhausted / 429 / 503), we
// retry it a couple of times with backoff, then fall through to the next
// model in this chain. The originally-requested model is always tried first;
// these are just backup options if NIM's capacity for it is maxed out.
const FALLBACK_CHAIN = [
  'deepseek-ai/deepseek-v4-flash',
  'deepseek-ai/deepseek-v3.1',
  'meta/llama-3.3-70b-instruct'
];

const MAX_RETRIES_PER_MODEL = 2; // retries on the same model before moving on
const RETRY_BASE_DELAY_MS = 800; // backoff: 800ms, 1600ms, ...

function isCapacityError(err) {
  const msg = err.response?.data?.error?.message || err.message || '';
  return (
    err.response?.status === 429 ||
    err.response?.status === 503 ||
    /ResourceExhausted/i.test(msg) ||
    /rate.?limit/i.test(msg)
  );
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// Tries `primaryModel` first, retrying on capacity errors with backoff, then
// falls through the rest of FALLBACK_CHAIN if it stays overloaded. Non-capacity
// errors (bad request, auth, etc.) are thrown immediately with no retry/fallback.
async function postToNimWithFallback(baseRequest, axiosConfig, primaryModel) {
  const chain = [primaryModel, ...FALLBACK_CHAIN.filter((m) => m !== primaryModel)];
  let lastError;

  for (const candidateModel of chain) {
    const requestBody = { ...baseRequest, model: candidateModel };

    for (let attempt = 0; attempt <= MAX_RETRIES_PER_MODEL; attempt++) {
      try {
        const response = await axios.post(`${NIM_API_BASE}/chat/completions`, requestBody, axiosConfig);
        return { response, modelUsed: candidateModel };
      } catch (err) {
        lastError = err;

        if (!isCapacityError(err)) {
          throw err; // real error (bad request, auth, etc.) — don't retry or fall back
        }

        if (attempt < MAX_RETRIES_PER_MODEL) {
          console.warn(`${candidateModel} at capacity, retrying in ${RETRY_BASE_DELAY_MS * (attempt + 1)}ms (attempt ${attempt + 1}/${MAX_RETRIES_PER_MODEL})`);
          await sleep(RETRY_BASE_DELAY_MS * (attempt + 1));
        } else {
          console.warn(`${candidateModel} still at capacity after ${MAX_RETRIES_PER_MODEL} retries, falling back to next model`);
        }
      }
    }
  }

  throw lastError;
}

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'OpenAI to NVIDIA NIM Proxy',
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE,
    fallback_model: FALLBACK_MODEL
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

// Chat completions endpoint (main proxy)
app.post(['/chat/completions', '/v1/chat/completions'], async (req, res) => {
  let streamRequested = false;

  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;
    streamRequested = !!stream;

    // If the incoming model isn't one of our known OpenAI-style aliases,
    // pass it straight through (it's probably already a valid NIM model id
    // like "deepseek-ai/deepseek-v4-flash") instead of silently mapping it.
    let nimModel = MODEL_MAPPING[model] || model || FALLBACK_MODEL;

    const nimRequest = {
      model: nimModel,
      messages: messages,
      temperature: temperature || 0.6,
      max_tokens: max_tokens || 9024,
      stream: streamRequested
    };

    if (ENABLE_THINKING_MODE) {
      nimRequest.extra_body = { chat_template_kwargs: { thinking: true } };
    }

    // Make request to NVIDIA NIM API, retrying/falling back on capacity errors
    const { response, modelUsed } = await postToNimWithFallback(nimRequest, {
      headers: {
        'Authorization': `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json'
      },
      responseType: streamRequested ? 'stream' : 'json',
      timeout: 60000 // fail fast instead of hanging forever if NIM stalls
    }, nimModel);

    if (modelUsed !== nimModel) {
      console.log(`Requested ${nimModel} was unavailable, served by ${modelUsed} instead`);
    }
    res.setHeader('X-Model-Used', modelUsed); // harmless for clients, useful for your own debugging

    if (streamRequested) {
      // Handle streaming response with reasoning
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache, no-transform');
      res.setHeader('Connection', 'keep-alive');
      res.setHeader('X-Accel-Buffering', 'no'); // prevent Render/nginx from buffering the stream
      res.flushHeaders(); // send headers immediately instead of waiting for first chunk

      let buffer = '';
      let reasoningStarted = false;

      response.data.on('data', (chunk) => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        lines.forEach(line => {
          if (line.startsWith('data: ')) {
            if (line.includes('[DONE]')) {
              res.write(line + '\n\n'); // must match SSE event framing (blank line terminator)
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
              res.write(line + '\n\n');
            }
          }
        });
      });

      response.data.on('end', () => res.end());
      response.data.on('error', (err) => {
        console.error('Stream error:', err.message);
        if (!res.writableEnded) {
          // Let the client know the stream died instead of just going silent
          res.write(`data: ${JSON.stringify({ error: { message: 'Upstream stream error', detail: err.message } })}\n\n`);
          res.write('data: [DONE]\n\n');
          res.end();
        }
      });
    } else {
      // Transform NIM response to OpenAI format with reasoning
      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: model,
        choices: response.data.choices.map(choice => {
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
        usage: response.data.usage || {
          prompt_tokens: 0,
          completion_tokens: 0,
          total_tokens: 0
        }
      };

      res.json(openaiResponse);
    }

  } catch (error) {
    let errBody = error.message;

    // When responseType is 'stream', an error response's body is itself a
    // readable stream, not parsed JSON — read it so the real NIM error shows
    // up in logs instead of just "Request failed with status code XXX".
    if (streamRequested && error.response?.data?.readable) {
      try {
        const chunks = [];
        for await (const chunk of error.response.data) chunks.push(chunk);
        errBody = Buffer.concat(chunks).toString();
      } catch (readErr) {
        errBody = `${error.message} (also failed to read error body: ${readErr.message})`;
      }
    } else if (error.response?.data) {
      errBody = typeof error.response.data === 'string'
        ? error.response.data
        : JSON.stringify(error.response.data);
    }

    console.error('Proxy error:', errBody);

    if (!res.headersSent) {
      res.status(error.response?.status || 500).json({
        error: {
          message: errBody || 'Internal server error',
          type: 'invalid_request_error',
          code: error.response?.status || 500
        }
      });
    } else if (!res.writableEnded) {
      // Headers/stream already started — can't send a JSON error body now,
      // so end the stream cleanly instead of hanging.
      res.write(`data: ${JSON.stringify({ error: { message: errBody } })}\n\n`);
      res.write('data: [DONE]\n\n');
      res.end();
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
  console.log(`Fallback model: ${FALLBACK_MODEL}`);
});
