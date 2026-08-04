---
layout: archive
title: "Play"
permalink: /play/
author_profile: true
---

<p>Interactive p5.js sketches, toys built to visualize ideas from my
research.</p>

<style>
  .play-grid { display: flex; flex-direction: column; gap: 24px; margin-top: 20px; }
  .play-card { display: flex; gap: 18px; align-items: flex-start; }
  .play-card img { width: 200px; max-width: 40%; height: auto; border: 1px solid #ccc; border-radius: 6px; flex: 0 0 auto; }
  .play-card .play-body h3 { margin: 0 0 6px; }
  .play-card .play-body p { margin: 0; font-size: 14px; line-height: 1.5; }
  @media (max-width: 480px) { .play-card { flex-direction: column; } .play-card img { max-width: 100%; width: 100%; } }
</style>

<div class="play-grid">

  <div class="play-card">
    <a href="/p5_githubpages/cultural_diffusion_viz/">
      <img src="/p5_githubpages/cultural_diffusion_viz/interactive-diffusion.png" alt="Cultural diffusion sketch">
    </a>
    <div class="play-body">
      <h3><a href="/p5_githubpages/cultural_diffusion_viz/">Spatially-explicit cultural diffusion</a></h3>
      <p>Cultural diffusion in a moving population. Tweak parameters and
      click a node to seed an innovation.</p>
    </div>
  </div>

  <div class="play-card">
    <a href="/p5_githubpages/ewa_nbda_viz/">
      <img src="/p5_githubpages/ewa_nbda_viz/example_nbda_ewa.png" alt="EWA-NBDA sketch">
    </a>
    <div class="play-body">
      <h3><a href="/p5_githubpages/ewa_nbda_viz/">EWA-NBDA: acquisition & production</a></h3>
      <p>A novel behaviour competes with an established tradition in a structured population. Social
      transmission (NBDA) governs acquisition; experience-weighted attraction
      (EWA) governs how often agents use it once learned.</p>
    </div>
  </div>

</div>
