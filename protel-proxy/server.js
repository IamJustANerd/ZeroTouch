const express = require('express');

const app = express();
app.use(express.json());

const PROTEL_API_URL = "https://protel.danielkazukixd.workers.dev/";
const PROTEL_SECRET_KEY = "haloprotel";

app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { messages } = req.body;
    
    // Extract prompt from messages array
    let userPrompt = "";
    if (messages && Array.isArray(messages)) {
      // Just take the last user message, or concatenate all.
      // Concatenating provides context if the user has a multi-turn conversation.
      userPrompt = messages
        .filter(m => m.role === 'user' || m.role === 'assistant')
        .map(m => {
          let text = "";
          if (typeof m.content === 'string') text = m.content;
          else if (Array.isArray(m.content)) {
            text = m.content.map(c => {
              if (typeof c === 'string') return c;
              if (c.type === 'text' || c.type === 'input_text') return c.text;
              return JSON.stringify(c);
            }).join('\n');
          } else {
            text = JSON.stringify(m.content);
          }
          return `${m.role === 'user' ? 'User' : 'AI'}: ${text}`;
        })
        .join('\n');
      console.log("Extracted Prompt Preview:", userPrompt.substring(0, 500));
      
      // Append a prompt for the AI to respond
      userPrompt += '\nAI:';
    } else {
      userPrompt = req.body.prompt || "Hello";
    }

    console.log("Sending prompt to Protel:", userPrompt);

    const response = await fetch(PROTEL_API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Custom-Auth-Key": PROTEL_SECRET_KEY
      },
      body: JSON.stringify({
        prompt: userPrompt
      })
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error(`Protel API error: ${response.status} - ${errorText}`);
      return res.status(response.status).json({ error: errorText });
    }

    const data = await response.json();
    console.log("Protel API raw response:", JSON.stringify(data, null, 2));

    // Try multiple common response shapes from the Protel API
    const aiResponseText =
      (typeof data?.response === 'string' ? data.response : null) ||
      data?.response?.response ||
      data?.choices?.[0]?.message?.content ||
      data?.text ||
      data?.output ||
      data?.result ||
      "No response received (unknown API shape — check raw log above)";

    if (req.body.stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      
      const chunk = {
        id: "chatcmpl-protel-" + Date.now(),
        object: "chat.completion.chunk",
        created: Math.floor(Date.now() / 1000),
        model: req.body.model || "protel",
        choices: [
          {
            index: 0,
            delta: {
              role: "assistant",
              content: aiResponseText
            },
            finish_reason: null
          }
        ]
      };
      
      res.write(`data: ${JSON.stringify(chunk)}\n\n`);
      
      const finishChunk = {
        id: "chatcmpl-protel-" + Date.now(),
        object: "chat.completion.chunk",
        created: Math.floor(Date.now() / 1000),
        model: req.body.model || "protel",
        choices: [
          {
            index: 0,
            delta: {},
            finish_reason: "stop"
          }
        ]
      };
      
      res.write(`data: ${JSON.stringify(finishChunk)}\n\n`);
      res.write(`data: [DONE]\n\n`);
      return res.end();
    }

    // Return in OpenAI-compatible format
    const openAiResponse = {
      id: "chatcmpl-protel-" + Date.now(),
      object: "chat.completion",
      created: Math.floor(Date.now() / 1000),
      model: req.body.model || "protel",
      choices: [
        {
          index: 0,
          message: {
            role: "assistant",
            content: aiResponseText
          },
          finish_reason: "stop"
        }
      ],
      usage: {
        prompt_tokens: 0,
        completion_tokens: 0,
        total_tokens: 0
      }
    };

    res.json(openAiResponse);
  } catch (error) {
    console.error("Proxy error:", error);
    res.status(500).json({ error: "Internal Proxy Error", details: error.message });
  }
});

// Healthcheck endpoint for Docker
app.get('/health', (req, res) => res.send('OK'));

const PORT = 11435;
app.listen(PORT, () => {
  console.log(`Protel OpenAI-compatible proxy running on port ${PORT}`);
});
