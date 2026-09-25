import { solveMeshGeometry, type MeshParams } from './showcase-mesh';

self.onmessage = (event: MessageEvent<MeshParams>) => {
  try {
    const started = performance.now();
    const geometry = solveMeshGeometry(event.data, (progress) => {
      self.postMessage({ type: 'progress', progress });
    });
    self.postMessage({ type: 'done', geometry, elapsed: performance.now() - started });
  } catch (error) {
    self.postMessage({ type: 'error', error: error instanceof Error ? error.message : String(error) });
  }
};
