import { TpaServer, TpaSession } from '@mentra/sdk';
import * as tf from '@tensorflow/tfjs-node'; // For ML model
import * as cocoSsd from '@tensorflow-models/coco-ssd'; // For object detection
import * as dotenv from 'dotenv';
import http from 'http';

dotenv.config(); // Loads .env variables like MENTRA_API_KEY (local only)

class SceneDescriberApp extends TpaServer {
  private model: cocoSsd.ObjectDetection | null = null;

  constructor(options: any) {
    super(options);
    // Load the model once on initialization
    this.loadModel();
  }

  private async loadModel() {
    try {
      this.model = await cocoSsd.load();
      console.log('COCO-SSD model loaded successfully.');
    } catch (error) {
      console.error('Error loading COCO-SSD model:', error);
    }
  }

  protected async onSession(
    session: TpaSession,
    sessionId: string,
    userId: string
  ): Promise<void> {
    console.log(`New session started: ${sessionId} for user ${userId}`);

    // Welcome message (private TTS in user's ear)
    await session.audio.speak('Scene describer ready. Say "describe scene" to scan.');

    // Listen for voice commands via transcription events
    session.events.onTranscription(async (data: { text: string; isFinal?: boolean }) => {
      const text = data.text.toLowerCase().trim();
      console.log(`User said: ${text}`);

      if (text.includes('describe scene')) {
        try {
          // Request photo from glasses camera
          const photo = await session.camera.requestPhoto();
          const imageBuffer = Buffer.from(photo.photoData);

          // Detect objects in the image
          const description = await this.describeImage(imageBuffer);

          // Announce results privately
          await session.audio.speak(description);
        } catch (error) {
          console.error('Scan error:', error);
          await session.audio.speak('Error describing scene. Try again.');
        }
      }
    });

    // Optional: Add more event handlers later, e.g., button presses
    // session.events.onButtonPress((data) => { ... });
  }

  // Describe the image using COCO-SSD object detection
  private async describeImage(imageBuffer: Buffer): Promise<string> {
    if (!this.model) {
      return 'Model not loaded. Unable to describe scene.';
    }

    try {
      // Decode image to tensor
      const imageTensor = tf.node.decodeImage(imageBuffer);

      // Run object detection
      const predictions = await this.model.detect(imageTensor);

      // Clean up tensor
      tf.dispose(imageTensor);

      if (predictions.length === 0) {
        return 'No objects detected in the scene.';
      }

      // Build a simple description
      const detectedObjects = predictions
        .filter((pred) => pred.score > 0.5) // Filter by confidence
        .map((pred) => `${pred.class} (confidence: ${Math.round(pred.score * 100)}%)`);

      return `I see: ${detectedObjects.join(', ')}.`;
    } catch (error) {
      console.error('Description error:', error);
      return 'Error processing image.';
    }
  }
}

// Create the Mentra app (no port — uses default 7010 for WebSocket)
const server = new SceneDescriberApp({
  packageName: 'com.yakov.scenedescriber',
  apiKey: process.env.MENTRA_API_KEY!,
});

// Health check for Railway - respond OK on any path (fixes 502 / failed response)
http.createServer((req, res) => {
  res.writeHead(200, { 'Content-Type': 'text/plain' });
  res.end('OK - Scene Describer is alive and running!');
}).listen(Number(process.env.PORT) || 3000, () => {
  console.log(`Health check server listening on port ${process.env.PORT || 3000}`);
});

// Start Mentra SDK server with error handling
try {
  server.start();
  console.log('Mentra AppServer started successfully (default port 7010)');
} catch (err: any) {
  console.error('Mentra startup failed:', err.message || err);
  process.exit(1);
}
