# LLM Setup Guide

This guide explains how to configure different LLM providers for EchoAI Helper.

## Supported Providers

- **OpenAI** - GPT-4, GPT-3.5 (default)
- **Google Gemini** - Gemini 1.5 Flash/Pro (fast & cheap)
- **Ollama** - Local LLMs (100% free, private)
- **Claude** - Anthropic's Claude models
- **100+ other providers** via LiteLLM

---

## Quick Start

### 1. Install Dependencies

```bash
pip install litellm pyyaml
```

### 2. Choose Your Provider

Edit `conf.yaml` and set the `provider` field:

```yaml
LLM:
  provider: "openai"  # or "litellm"
```

---

## Configuration Examples

### Option 1: OpenAI (Default)

**Best for**: Production use, highest quality

```yaml
LLM:
  provider: "openai"
  openai:
    model: "gpt-4o-mini"  # or gpt-4o, gpt-4-turbo
```

**Setup**:
1. Get API key from https://platform.openai.com/api-keys
2. Set environment variable:
   ```bash
   # Windows
   setx OPENAI_API_KEY "sk-..."

   # Linux/Mac
   export OPENAI_API_KEY="sk-..."
   ```

**Cost**: ~$0.15-0.60 per 1M tokens

---

### Option 2: Google Gemini (Recommended)

**Best for**: Low cost, good quality, fast

```yaml
LLM:
  provider: "litellm"
  litellm:
    model: "gemini/gemini-1.5-flash"
    api_key: "YOUR_GEMINI_API_KEY"
```

**Setup**:
1. Get free API key from https://aistudio.google.com/app/apikey
2. Add to conf.yaml or set environment variable:
   ```bash
   setx GEMINI_API_KEY "your-key"
   ```

**Cost**: ~$0.075 per 1M tokens (10x cheaper than GPT-4)
**Free tier**: 15 requests/minute

**Models**:
- `gemini/gemini-1.5-flash` - Fast & cheap
- `gemini/gemini-1.5-pro` - High quality, 2M context

---

### Option 3: Ollama (100% Free & Private)

**Best for**: Privacy, no cost, offline use

```yaml
LLM:
  provider: "litellm"
  litellm:
    model: "ollama/gemma2:2b"
    api_base: "http://localhost:11434"
```

**Setup**:
1. Install Ollama: https://ollama.com/download
2. Pull a model:
   ```bash
   # Lightweight models (CPU-friendly)
   ollama pull gemma2:2b      # Google's 2B model
   ollama pull phi3:mini      # Microsoft's 3.8B model
   ollama pull qwen2:1.5b     # Alibaba's 1.5B model

   # Larger models (better quality, needs more RAM)
   ollama pull llama3:8b      # Meta's 8B model
   ollama pull gemma2:9b      # Google's 9B model
   ```
3. Start Ollama (usually auto-starts)

**Cost**: $0 (completely free)
**Privacy**: All processing happens locally

**Recommended models**:
- **gemma2:2b** - Best for CPU, fast
- **phi3:mini** - Good balance
- **llama3:8b** - Best quality (needs 8GB+ RAM)

---

### Option 4: Claude (Anthropic)

**Best for**: Long context, safety-focused

```yaml
LLM:
  provider: "litellm"
  litellm:
    model: "claude-3-haiku-20240307"
    api_key: "YOUR_ANTHROPIC_KEY"
```

**Setup**:
1. Get API key from https://console.anthropic.com/
2. Set environment variable:
   ```bash
   setx ANTHROPIC_API_KEY "sk-ant-..."
   ```

**Models**:
- `claude-3-haiku-20240307` - Fast & cheap
- `claude-3-sonnet-20240229` - Balanced
- `claude-3-opus-20240229` - Highest quality

---

## Switching Providers

You can easily switch between providers by changing the `provider` field in `conf.yaml`:

```yaml
# Use OpenAI
LLM:
  provider: "openai"

# Use Google Gemini
LLM:
  provider: "litellm"
  litellm:
    model: "gemini/gemini-1.5-flash"
    api_key: "your-key"

# Use local Ollama
LLM:
  provider: "litellm"
  litellm:
    model: "ollama/gemma2:2b"
    api_base: "http://localhost:11434"
```

No code changes needed - just restart the application!

---

## Performance Comparison

| Provider | Cost (1M tokens) | Speed | Quality | Privacy | CPU-friendly |
|----------|------------------|-------|---------|---------|--------------|
| GPT-4o-mini | $0.15-0.60 | Fast | ⭐⭐⭐⭐⭐ | ❌ | ✅ |
| Gemini Flash | $0.075 | Very Fast | ⭐⭐⭐⭐ | ❌ | ✅ |
| Ollama (gemma2:2b) | $0 | Fast | ⭐⭐⭐ | ✅ | ✅ |
| Ollama (llama3:8b) | $0 | Medium | ⭐⭐⭐⭐ | ✅ | ⚠️ |
| Claude Haiku | $0.25 | Fast | ⭐⭐⭐⭐ | ❌ | ✅ |

---

## Troubleshooting

### Error: "litellm not installed"

```bash
pip install litellm
```

### Error: "API key not found"

Make sure you've set the environment variable:
```bash
# Check if set
echo %OPENAI_API_KEY%    # Windows
echo $OPENAI_API_KEY     # Linux/Mac
```

### Ollama: "Connection refused"

1. Make sure Ollama is running
2. Check if accessible: http://localhost:11434
3. Try: `ollama serve`

### Slow response with Ollama

- Use smaller models (gemma2:2b, phi3:mini)
- Increase CPU threads in Ollama settings
- Consider using Gemini Flash instead

---

## Advanced Configuration

### Using environment variables for API keys

Instead of hardcoding in conf.yaml, use environment variables:

```yaml
LLM:
  provider: "litellm"
  litellm:
    model: "gemini/gemini-1.5-flash"
    # api_key will be read from GEMINI_API_KEY env var
```

### Custom parameters

You can pass additional parameters:

```yaml
LLM:
  litellm:
    model: "gemini/gemini-1.5-pro"
    api_key: "..."
    max_tokens: 1024
    top_p: 0.9
```

---

## Recommended Setup

**For development/testing:**
- Ollama with gemma2:2b (free, fast)

**For production (English):**
- Gemini 1.5 Flash (cheap, fast, good quality)

**For production (highest quality):**
- GPT-4o-mini or Claude Haiku

**For privacy-sensitive applications:**
- Ollama with llama3:8b (local, private)

---

## Support

For more LiteLLM providers and configurations, see:
https://docs.litellm.ai/docs/providers
