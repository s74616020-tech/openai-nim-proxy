const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());

const OPENROUTER_API_BASE = 'https://openrouter.ai/api/v1';
const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY;

const SHOW_REASONING = false;
const ENABLE_THINKING_MODE = false;

// Map fake OpenAI model names to real OpenRouter model names
// You can also type these real names directly in Janitor AI
// Browse available models at https://openrouter.ai/models
const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'meta-llama/llama-3.1-8b-instruct',
  'gpt-4': 'meta-llama/llama-3.1-70b-instruct',
  'gpt-4-turbo': 'meta-llama/llama-3.3-70b-instruct',
  'gpt-4o': 'openrouter/free',
  'claude-3-opus': 'meta-llama/llama-3.1-405b-instruct',
  'claude-3-sonnet': 'meta-llama/llama-3.3-70b-instruct',
  'gemini-pro': 'mistralai/mixtral-8x7b-instruct',

  // Real OpenRouter model names — type these directly in Janitor AI
  'openrouter/free': 'openrouter/free',
  'openrouter/owl-alpha': 'openrouter/owl-alpha',
  'deepseek/deepseek-chat': 'deepseek/deepseek-chat',
  'deepseek/deepseek-v3': 'deepseek/deepseek-v3',
  'meta-llama/llama-3.3-70b-instruct': 'meta-llama/llama-3.3-70b-instruct',
  'meta-llama/llama-3.1-405b-instruct': 'meta-llama/llama-3.1-405b-instruct',
  'mistralai/mistral-large': 'mistralai/mistral-large',
  'mistralai/mixtral-8x7b-instruct': 'mistralai/mixtral-8x7b-instruct'
};

app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'OpenAI to OpenRouter Proxy',
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE
  });
});

app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(model => ({
    id: model,
    object: 'model',
    created: Date.now(),
    owned_by: 'openrouter-proxy'
  }));
  res.json({ object: 'list', data: models });
});

app.post(['/chat/completions', '/v1/chat/completions'], async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;

    let orModel = MODEL_MAPPING[model] || 'openrouter/free';

    const orRequest = {
      model: orModel,
      messages: messages,
      temperature: temperature || 0.6,
      max_tokens: max_tokens || 9024,
      stream: stream || false
    };

    if (ENABLE_THINKING_MODE) {
      orRequest.extra_body = { chat_template_kwargs: { thinking: true } };
    }

    const response = await axios.post(`${OPENROUTER_API_BASE}/chat/completions`, orRequest, {
      headers: {
        'Authorization': `Bearer ${OPENROUTER_API_KEY}`,
        'Content-Type': 'application/json',
        'HTTP-Referer': 'https://openai-nim-proxy-1-421w.onrender.com',
        'X-Title': 'OpenAI to OpenRouter Proxy'
      },
      responseType: stream ? 'stream' : 'json'
    });

    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      let buffer = '';
      let reasoningStarted = false;

      response.data.on('data', (chunk) => {
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

      response.data.on('end', () => res.end());
      response.data.on('error', (err) => {
        console.error('Stream error:', err);
        res.end();
      });

    } else {
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
    console.error('Proxy error:', error.message);
    console.error('API key present:', !!OPENROUTER_API_KEY);
    console.error('API key length:', OPENROUTER_API_KEY ? OPENROUTER_API_KEY.length : 0);
    console.error('API key first 8 chars:', OPENROUTER_API_KEY ? OPENROUTER_API_KEY.slice(0, 8) : 'none');
    if (error.response?.data) {
      try {
        console.error('OpenRouter response body:', JSON.stringify(error.response.data));
      } catch (e) {
        console.error('OpenRouter response body (raw):', error.response.data);
      }
    }
    res.status(error.response?.status || 500).json({
      error: {
        message: error.message || 'Internal server error',
        type: 'invalid_request_error',
        code: error.response?.status || 500
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
  console.log(`Health check: http://localhost:${PORT}/health`);
});
