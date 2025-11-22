# Moondream Station Manual

Moondream Station is a local inference server that enables you to use Moondream locally with our Python, Node, cURL, and OpenAI clients.

## Getting Started

### First Launch
When you start moondream-station, it will:
1. Load available models from a manifest
2. Select a default model (if available)
3. Start the inference service automatically

### Basic Navigation
- Type commands and press Enter to execute them
- Press `Ctrl+C` or type `exit` to quit

## Essential Commands

### Model Management
- `models` - List all available models
- `models switch <name>` - Switch to a different model

### Service Control
- `start` - Start the REST API server (usually auto-starts)
- `stop` - Stop the server
- `restart` - Restart the server
- The green dot (🟢) means the service is running

#### Port Selection
The service will attempt to start on port 2020 by default. If that port is occupied, it will automatically try subsequent ports (2021, 2022, etc.) until it finds an available one. You can also specify a port manually with `start <port>`.

### Using the Service

Once running, the service provides a REST API at the displayed endpoint (e.g., `http://localhost:2020/v1`).

#### Direct Inference (from the CLI)
- `inference` - Enter interactive inference mode
- `infer <function> <image>` - Run a specific inference function

#### REST API (from any application)
Send POST requests to endpoints like:
- `/v1/caption` - Generate image captions
- `/v1/detect` - Detect objects in images
- `/v1/query` - Answer questions about images

Example:
```bash
curl -X POST http://localhost:2020/v1/caption \
  -H "Content-Type: application/json" \
  -d '{"image_url": "path/to/image.jpg"}'
```

## Other Useful Commands

- `settings` - View current configuration
- `settings set <key> <value>` - Change settings
- `update` - Check for updates
- `history` - View recent commands
- `manual` or `man` - Show this manual
- `reset` - Delete all app data and start fresh
- `help` - Show command list

## Configuration

Key settings you can adjust:
- `inference_workers` - Number of parallel workers (default: 2)
- `inference_timeout` - Request timeout in seconds (default: 60)
- `auto_start` - Auto-start service on model selection (default: true)

### Understanding Workers

Workers allow multiple requests to be processed simultaneously. Each worker is a **separate instance of the model** loaded in memory.

**Important:** This is NOT batched inference - each worker processes one request at a time independently.

#### Memory Usage
- Each worker loads a complete copy of the model
- Total memory usage = `number_of_workers × model_size`
- Example: A 5GB model with 3 workers will use approximately 15GB of memory

#### Configuring Workers
```
settings set inference_workers 4
```
After changing the worker count, the service will automatically restart to apply the new setting.

**Guidelines:**
- More workers = handle more concurrent requests
- More workers = higher memory usage
- Set workers based on your available RAM and expected load
- For single-user setups, 1-2 workers is usually sufficient

## Tips

- The service must be running (green dot) to handle requests
- Models are downloaded on first use and cached locally
- Check the API endpoint in the status box for integration with other apps
- Use `settings` to see current configuration and system status

## Need Help?

- Type `help` for a quick command reference
- Visit https://docs.moondream.ai for detailed documentation
- Report issues at https://github.com/moondream-ai/moondream-station/issues