/**
 * The opening screen's showcase: the cloth lift (docs/opening-showcase-cloth-lift.md),
 * behind the intro. Each project's clip plays plain; then its name rises up
 * under the footage, holds draped in it, and slides away down the frame taking
 * the footage with it, leaving the next project's clip.
 *
 * Four projects in turn, each on its own canvas, drawn from its Blender bake.
 * The displays swap in a single paint at the start of each project's idle,
 * where every display draws the same frame: the next project's clip, whole.
 */

import { createMeshShowcase, cycleOf, MESH_DEFAULTS, type MeshParams, type MeshShowcase } from './showcase-mesh';

const PROJECTS = [
  { text: 'Yope3D', clip: 0 },
  { text: 'SpinStack', clip: 1 },
  { text: 'β-VAE', clip: 2 },
  { text: 'Glyph\nInterpreter', clip: 4 },
];

/** The cloth lift as /showcase-test/'s cloth-lift preset sets it. */
const CLOTH_LIFT: MeshParams = {
  ...MESH_DEFAULTS,
  footage: 'clothlift',
  commit: false,
  holdDrop: false,
  breath: 0,
  releaseShape: 'asymmetric',
  idle: 6,
  settle: 2.5,
  hold: 2.8,
  release: 4,
};

export function projectParams(index: number): MeshParams {
  const { text, clip } = PROJECTS[index];
  // Its own clip on the cloth, the next project's on the floor under it.
  const under = PROJECTS[(index + 1) % PROJECTS.length].clip;
  return { ...CLOTH_LIFT, text, clip, homeClip: clip, clothUnder: under };
}

/** Start the showcase in `root`, which holds one canvas per project and the
 *  clips in the engine's clip order. Leaves `root` hidden when motion is
 *  unwanted, the screen is narrow, or WebGL is missing. */
export function startShowcaseOpening(root: HTMLElement) {
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  if (!window.matchMedia('(min-width: 700px)').matches) return;
  const canvases = Array.from(root.querySelectorAll('canvas'));
  const videos = Array.from(root.querySelectorAll('video'));
  const engines: Array<MeshShowcase | null> = PROJECTS.map(() => null);
  const create = (index: number) => {
    const engine = createMeshShowcase(canvases[index], videos, undefined, true);
    engine?.setParams(projectParams(index));
    engine?.setVisible(index === 0);
    canvases[index].hidden = index !== 0;
    engines[index] = engine;
    return engine;
  };
  // Clips load only when asked for: the first project's now, and each other
  // one a project ahead of when it is seen.
  const asked = new Set<number>();
  const want = (clip: number) => {
    if (asked.has(clip)) return;
    asked.add(clip);
    const video = videos[clip];
    video.preload = 'auto';
    video.load();
    if (frameRequest) video.play().catch(() => {});
  };
  const clipOf = (index: number) => PROJECTS[(index + PROJECTS.length) % PROJECTS.length].clip;
  let frameRequest = 0;
  want(clipOf(0));
  const first = create(0);
  if (!first) return;
  root.hidden = false;

  const loop = cycleOf(CLOTH_LIFT).loop;
  const total = loop * PROJECTS.length;
  let clock = 0;
  let laps = 0;
  let current = 0;
  let last = 0;
  let onScreen = true;
  // The first project's floor, the second project's clip, once the page has
  // painted its first frames.
  let started = 0;

  function tick(now: number) {
    frameRequest = requestAnimationFrame(tick);
    const delta = last ? Math.min(0.1, (now - last) / 1000) : 0;
    last = now;
    clock += delta;
    if (clock >= total) {
      laps += Math.floor(clock / total);
      clock %= total;
    }
    // What the coming projects need, one project ahead: during project i,
    // soon after it starts, the clip on its floor (project i + 1's); and once
    // it is under way, project i + 1's display and bake, and the clip on
    // that one's floor, so all of it is in hand when project i + 1 begins.
    const local = clock - current * loop;
    if (!started) started = now;
    if (now - started > 3000 || current > 0) want(clipOf(current + 1));
    if (local >= 4 || current > 0) {
      const next = current + 1;
      if (next < PROJECTS.length && !engines[next]) create(next);
      want(clipOf(current + 2));
    }
    // Hidden displays advance their solves and prepare a first frame.
    engines.forEach((engine, index) => {
      if (index !== current) engine?.frame(0);
    });
    const wanted = Math.min(PROJECTS.length - 1, Math.floor(clock / loop));
    if (wanted !== current) {
      const next = engines[wanted]?.status();
      if (!next || next.solving || next.vertices === 0 || !next.ready) {
        // Hold the outgoing project's last frame until the next is ready.
        clock = wanted > current ? wanted * loop - 0.001 : total - 0.001;
        return;
      }
      engines[current]?.setVisible(false);
      canvases[current].hidden = true;
      current = wanted;
      engines[current]?.setVisible(true);
      canvases[current].hidden = false;
    }
    engines[current]?.frame(clock - current * loop, 9, clock + laps * total, laps * PROJECTS.length + current);
  }

  function run() {
    const wanted = onScreen && !document.hidden;
    if (wanted && !frameRequest) {
      last = 0;
      frameRequest = requestAnimationFrame(tick);
      for (const clip of asked) videos[clip].play().catch(() => {});
    } else if (!wanted && frameRequest) {
      cancelAnimationFrame(frameRequest);
      frameRequest = 0;
      for (const clip of asked) videos[clip].pause();
    }
  }
  new IntersectionObserver((entries) => {
    onScreen = entries.some((entry) => entry.isIntersecting);
    run();
  }).observe(root);
  document.addEventListener('visibilitychange', run);
  run();
}
