import { TpaServer, TpaSession } from '@mentra/sdk';
import * as tf from '@tensorflow/tfjs-node';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import * as dotenv from 'dotenv';
import http from 'http';

dotenv.config();

class SceneDescriberApp extends TpaServer {
  private model: cocoSsd.ObjectDetection | null = null;

  constructor(options: any) {
    super(options);
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

    // @ts-ignore
    await session.audio.speak('Scene describer ready. Say "describe scene" to scan.');

    session.events.onTranscription(async (data: { text: string; isFinal?: boolean }) => {
      const text = data.text.toLowerCase().trim();
      console.log(`User said: ${text}`);

      if (text.includes('describe scene')) {
        try {
          // @ts-ignore
          const photo = await session.camera.requestPhoto();
          const imageBuffer = Buffer.from(photo.photoData);

          const description = await this.describeImage(imageBuffer);

          // @ts-ignore
          await session.audio.speak(description);
        } catch (error) {
          console.error('Scan error:', error);
          // @ts-ignore
          await session.audio.speak('Error describing scene. Try again.');
        }
      }
    });
  }

  private async describeImage(imageBuffer: Buffer): Promise<string> {
    if (!this.model) {
      return 'Model not loaded. Unable to describe scene.';
    }

    try {
      const imageTensor = tf.node.decodeImage(imageBuffer);
      const predictions = await this.model.detect(imageTensor);
      tf.dispose(imageTensor);

      if (predictions.length === 0) {
        return 'No objects detected in the scene.';
      }

      const detectedObjects = predictions
        .filter((pred) => pred.score > 0.5)
        .map((pred) => `${pred.class} (confidence: ${Math.round(pred.score * 100)}%)`);

      return `I see: ${detectedObjects.join(', ')}.`;
    } catch (error) {
      console.error('Description error:', error);
      return 'Error processing image.';
    }
  }
}

const server = new SceneDescriberApp({
  packageName: 'com.yakov.scenedescriber',
  apiKey: process.env.MENTRA_API_KEY!,
});

http.createServer((req, res) => {
  res.writeHead(200, { 'Content-Type': 'text/plain' });
  res.end('OK - Scene Describer is alive and running!');
}).listen(Number(process.env.PORT) || 3000, () => {
  console.log(`Health check server listening on port ${process.env.PORT || 3000}`);
});

try {
  server.start();
  console.log('Mentra AppServer started successfully (default port 7010)');
} catch (err: any) {
  console.error('Mentra startup failed:', err.message || err);
  process.exit(1);
}
