
//define colors from mariana sublime theme
mar_black = [0, 0, 0];
mar_blue = [210, 50, 60];
mar_blue2 = [209, 13, 35];
mar_blue3 = [210, 15, 24];
mar_blue4 = [210, 13, 45];
mar_blue5 = [180, 36, 54];
mar_blue6 = [221, 12, 69];
mar_green = [114, 31, 68];
mar_grey = [0, 0, 20];
mar_orange = [32, 93, 66];
mar_orange2 = [32, 85, 55];
mar_orange3 = [40, 94, 68];
mar_pink = [300, 30, 68];
mar_red = [357, 79, 65];
mar_red2 = [13, 93, 66];
mar_white = [0, 0, 100];
mar_white2 = [0, 0, 97];
mar_white3 = [219, 28, 88];

function softmax(logits) {
    const maxLogit = Math.max(...logits);
    const scores = logits.map(l => Math.exp(l - maxLogit));
    const denom = scores.reduce((a, b) => a + b);
    return scores.map(s => s / denom);
}

function *enumerate(array) {
   for (let i = 0; i < array.length; i += 1) {
      yield [i, array[i]];
   }
}

// generate cumulative distribution function from weights
function cdf(weights) {
    // calculate total
    var total = 0;
    for(var i=0; i<weights.length; i++) {
        total += weights[i];
    }
    // generate CDF, normalizing with total
    var cumul = [];
    cumul[0] = weights[0]/total;
    for(i=1; i<weights.length; i++) {
        cumul[i] = cumul[i-1] + (weights[i]/total);
    }
    return cumul;
}

// pick the index using the random value
function selectInd(cumul,rand) {
    for(var i=0; (i < cumul.length) && (rand > cumul[i]); ++i) {};
    return i;
}

class Agent{
  constructor(ID){
    this.ID=ID;
    this.position = new p5.Vector(random(260,width-70), random(45,height-90));
    this.velocity = p5.Vector.random2D().mult(0.5);
    this.r_const = random(15,25);
    this.r = this.r_const;
    this.color = color(...mar_white2);
    this.neighbor_vector = [];
    this.naive = true;
    this.repertoire = [{name: "a", Evec: 0, Ivec: 0, Svec: 0, Pvec: 0}]; //name Evec Ivec Svec Pvec
    this.rho = 0.5;
    this.alpha = alphaValue;      // set from the risk-appetite control
    this.tracked = true;          // false on the decorative example agent
    this.production_memory = [];
    this.social_memory = [];       // list of observed behavior names (feeds Svec)
    this.temp_social_memory = [];  // observations made during the current timestep only
    this.production_frame;
    this.learned_b=0;
    this.produced_b=0;
    this.fireFrame = -1;          // phaseFrame at which this agent fires during production
    this.firedThisStep = false;
  }

  Evec_update(produced_behavior){
    var reward = 1;
    let new_Evec;
    this.repertoire.forEach(element => {
      if (element.name === produced_behavior){
        new_Evec = (1 - this.rho) * element.Evec + this.rho * reward;
    }
    else {
        new_Evec = (1 - this.rho) * element.Evec;
    }
    element.Evec = new_Evec;
  });
  }

  Ivec_update(){
    var alpha=this.alpha;
    // alpha is a softmax temperature: low alpha -> sharp/near-deterministic (risk-averse),
    // high alpha -> flat/exploratory (risk-tolerant). Hence Evec / alpha, not Evec * alpha.
    var Evec_over_alpha = this.repertoire.map( function (element){ return element.Evec / alpha});
    var Ivec = softmax(Evec_over_alpha);
    this.repertoire.forEach(function callback(element,index){element.Ivec = Ivec[index]})
  }

  // record a neighbor's production, but only for behaviors this agent already knows
  observe(behavior){
    if (this.repertoire.some(element => element.name === behavior)){
      this.temp_social_memory.push(behavior);
    }
  }

  // social production probabilities: frequency of each known behavior in social memory,
  // raised to the conformity exponent CHI (=1, linear) and normalized. Empty memory -> 0.
  Svec_update(){
    var mem = this.social_memory;
    if (mem.length > 0){
      var denom = 0;
      this.repertoire.forEach(function(element){
        denom += Math.pow(mem.filter(x => x === element.name).length, CHI);
      });
      this.repertoire.forEach(function(element){
        element.Svec = denom > 0
          ? Math.pow(mem.filter(x => x === element.name).length, CHI) / denom
          : 0;
      });
    }
    else {
      this.repertoire.forEach(function(element){ element.Svec = 0; });
    }
  }

  // blend individual (Ivec) and social (Svec) probabilities by the social-info bias sigma.
  // Ivec is already a probability here (unlike the Python i_mat), so no exp() wrap.
  Pvec_update(){
    var socialPresent = this.repertoire.some(element => element.Svec > 0);
    if (socialPresent){
      this.repertoire.forEach(function(element){
        element.Pvec = (1 - sigmaValue) * element.Ivec + sigmaValue * element.Svec;
      });
    }
    else {
      this.repertoire.forEach(function(element){ element.Pvec = element.Ivec; });
    }
  }

  // fold this timestep's observations into the windowed social memory (temp -> long + prune)
  consolidate_social_memory(){
    for (var i = 0; i < this.temp_social_memory.length; i++){
      this.social_memory.push(this.temp_social_memory[i]);
    }
    this.temp_social_memory = [];
    while (this.social_memory.length > SOCIAL_MEMORY_WINDOW){
      this.social_memory.shift();
    }
  }

  produce_behavior(){
    var names = this.repertoire.map( function (element){ return element.name});
    var Pvec = this.repertoire.map( function (element){ return element.Pvec});
    var dist = cdf(Pvec)
    var dice = Math.random(); // 0 : 1
    var produced_behavior = names[selectInd(dist,dice)];
    if (produced_behavior=="b" && this.produced_b==0 && this.tracked){
      num_produced_b +=1;
      this.produced_b=num_produced_b;
    }
    return produced_behavior;
  }

  production_submodel(){
    var produced_behavior = this.produce_behavior();
    this.Evec_update(produced_behavior);
    this.Ivec_update()
    this.production_memory.push(produced_behavior);
    this.production_frame = frameCount;
    return produced_behavior
  }

  trim_production_memory(){
    if (this.production_memory.length < 25) {
    }
    else{
        this.production_memory.shift();
    }
    if (this.production_memory.length >= 25) { print("memory window not respected")}
  }

  seeded(){
    this.repertoire.push({name: "b", Evec: 0, Ivec: 0, Svec: 0, Pvec: 0})
    this.naive = false;
    num_know_b += 1;
    this.learned_b = num_know_b;
    // seed production probabilities so the first strum has a valid Pvec (avoids NaN)
    this.Ivec_update();
    this.Svec_update();
    this.Pvec_update();
  }

  viz_behavior(){
    if ( this.production_memory[this.production_memory.length-1] == "a" ){
      noFill();
      strokeWeight(3);

      if(frameCount - this.production_frame < ARC_FRAMES){
        stroke(...mar_red, (ARC_FRAMES - (frameCount-this.production_frame)) / ARC_FRAMES);
        arc(this.position.x, this.position.y, this.r_const + 10, this.r_const + 10, PI, TWO_PI);
      }

    }

    else if ( this.production_memory[this.production_memory.length-1] == "b" ){
      noFill();
      strokeWeight(3);

      if(frameCount - this.production_frame < ARC_FRAMES){
      stroke(...mar_green, (ARC_FRAMES - (frameCount-this.production_frame)) / ARC_FRAMES);
      arc(this.position.x, this.position.y, this.r_const + 10, this.r_const + 10, PI, TWO_PI);
    }

    }
  }

  acquire(behavior){
    //sum knowledgable neighbors
    let sum_knowledgable = 0;
    this.neighbor_vector.forEach(element =>{
        if (agents[element].production_memory.length === 0) return; // nothing produced yet -> avoid /0 NaN
        agents[element].repertoire.forEach(entry =>{
            if (entry.name === behavior){
                sum_knowledgable += 1 * agents[element].production_memory.filter(x => x == behavior).length / agents[element].production_memory.length;
            }
        })
    }
    )

    let taizt = 0;

    if (rule=="sum"){
      taizt = sum_knowledgable
    }

    else {
      console.log("unknown transmission rule")
    }

    //base_rate * (s * num_know + asocial learning)
    let acq_rate = ACQ_BASE_RATE * 1 * taizt;

    let acq_prob = 1 - exp( - acq_rate);

    let dice_roll = random(0,1);

    if (dice_roll < acq_prob){
      this.repertoire.push({name: behavior, Evec: 0, Ivec: 0, Svec: 0, Pvec: 0});
      this.naive = false;
      num_know_b += 1;
      this.learned_b = num_know_b;
      // seed production probabilities for the newly known behavior (avoids NaN)
      this.Ivec_update();
      this.Svec_update();
      this.Pvec_update();
    }

  }

  // Custom method for updating the variables
  moveParticle() {
    if(this.position.x < 0 || this.position.x > width){
      this.velocity.x*=-1;}
    if(this.position.y < 0 || this.position.y > height){
      this.velocity.y*=-1;
    }

    this.position.add(p5.Vector.random2D().mult(.4));
  }


  // this function creates the connections(lines)
// between particles which are less than a certain distance apart
  connectAgents(agents) {
    agents.forEach(element =>{
    if (this.ID !== element.ID){
      var dice = random();
      if (dice < .20){
        if (!this.neighbor_vector.includes(element.ID)){
          this.neighbor_vector.push(element.ID)
        }
        if (!element.neighbor_vector.includes(this.ID)){
          element.neighbor_vector.push(this.ID)
        }
      }
  }})
  }

drawEdges(agents) {
    agents.forEach(element =>{
    if (this.ID !== element.ID && this.neighbor_vector.includes(element.ID)){
      stroke(...mar_black);
        strokeWeight(1);
        line(this.position.x,this.position.y,element.position.x,element.position.y);
      }
  })}

draw() {
    strokeWeight(0.75)

    fill(...mar_blue2);
    strokeWeight(0.75);
    stroke(this.color);
    circle(this.position.x, this.position.y, this.r);

    if (this.repertoire.length === 1){
      fill(...mar_red);
      arc(this.position.x, this.position.y, this.r_const, this.r_const, 0, PI);
    }
    else if (this.repertoire.length === 2) {
      noStroke();
      fill(...mar_green);
      arc(this.position.x, this.position.y, this.r_const, this.r_const, PI, TWO_PI);
      fill(...mar_red);
      arc(this.position.x, this.position.y, this.r_const, this.r_const, 0, PI);
    }

    this.viz_behavior()
    this.trim_production_memory()

  }
}


let num_agents = 16;
let agents = [];
var cnv;

//sum, proportional, threshold, conformity
let rule="sum";
var num_know_b = 0;
var num_produced_b = 0;

const W = 640; // matches the body text column width
const H = 528; // sized so the left column has equal ~30px padding top and bottom

let example_agent;

// --- run lifecycle -----------------------------------------------------------
// Each timestep is choreographed as a transmission phase followed by a production
// phase. Within the production phase every agent fires its behaviour arc exactly
// once, staggered by x-position so the arcs sweep left->right as a "strum".
// Sim-speed presets (slow -> fast), selected with the "sim speed" slider.
// T/P are transmission/production phase lengths in frames. Slowest is 30/60,
// default is 15/30.
const SPEEDS = [ 
  { T: 20, P: 40 },
  { T: 15, P: 30 }, 
  { T: 12, P: 24 },
  { T: 10, P: 20 },
  { T: 8,  P: 16 },
  { T: 4,  P: 8 }, 
];
let speedIndex = 4;                        // default -> 1s timestep

// Phase-length state, derived from the selected speed by applySpeed().
let T_FRAMES, P_FRAMES, CYCLE, ARC_FRAMES, FIRE_SPAN;
function applySpeed() {
  T_FRAMES   = SPEEDS[speedIndex].T;       // transmission phase length
  P_FRAMES   = SPEEDS[speedIndex].P;       // production phase length
  CYCLE      = T_FRAMES + P_FRAMES;        // one full timestep
  ARC_FRAMES = constrain(round(P_FRAMES * 2 / 3), 8, 20); // arc fade 
  FIRE_SPAN  = P_FRAMES - ARC_FRAMES;      // strum window so every arc finishes within the cycle
}

const PULSE_FRAMES = 12;                  // how long an observation pulse takes to cross an edge
const ACQ_BASE_RATE = 0.5;                // per-timestep social transmission base rate
const MAX_TIMESTEPS = 60;                 // abandon a run that never completes (e.g. isolated node)
const CHI = 1;                            // conformity exponent for social info (1 = linear, fixed)
const SOCIAL_MEMORY_WINDOW = 25;          // observations retained for Svec

let phaseFrame = 0;         // 0..CYCLE-1, position within the current timestep
let timestep   = 0;         // completed timesteps in this run
let pulses     = [];        // observation pulses {fromID, toID, startFrame}, spawned on "b" production

// --- risk appetite -> EWA sensitivity (alpha) --------------------------------
const RISK = [
  { alpha: 0.5, label: "risk-averse" },
  { alpha: 1,   label: "risk-neutral" },
  { alpha: 2,   label: "risk-tolerant" },
];
let riskIndex = 1;
let alphaValue = RISK[riskIndex].alpha;

// --- social information bias (sigma) ------------------------------------------
// 0 -> production is purely individual (EWA); 1 -> production purely follows the
// observed frequency of behaviors in social memory. Read live inside Pvec_update.
let sigmaValue = 0;

// --- accumulated Oa (x) vs Op (y) heatmap ------------------------------------
let counts;        // num_agents x num_agents, counts[oa-1][op-1]
let cellOffsets;   // matching grid of {x,y} brownian jitter
let maxCount = 0;
let runsRecorded = 0;

// DOM references, filled in during setup
let riskSlider, riskLabel, speedSlider, sigmaSlider;

function makeGrid(factory) {
  return Array.from({ length: num_agents }, () =>
    Array.from({ length: num_agents }, factory));
}

// --- control panel -----------------------------------------------------------
function addControlRow(panel, labelText) {
  let row = createDiv().class("control-row").parent(panel);
  let label = createElement("label", labelText).parent(row);
  return { row, label };
}

function buildControls(panel) {
  // sim-speed slider (left = slow, right = fast; does not clear the heatmap)
  {
    let { row } = addControlRow(panel, "sim speed");
    speedSlider = createSlider(0, SPEEDS.length - 1, speedIndex, 1).parent(row);
  }
  // risk-appetite slider (3 notches -> alpha 0.5 / 1 / 2)
  {
    let { row } = addControlRow(panel, "risk appetite (α)");
    riskSlider = createSlider(0, 2, riskIndex, 1).parent(row);
  }
  // social-info slider (sigma, 0 = individual only -> 1 = follow the crowd)
  {
    let { row } = addControlRow(panel, "sensitivity to social info (σ)");
    sigmaSlider = createSlider(0, 1, sigmaValue, 0.05).parent(row);
  }
  // live status: regime + accumulated run count
  {
    let { label } = addControlRow(panel, "");
    riskLabel = label;
  }
  // reset button (clears the accumulated heatmap)
  {
    let { row } = addControlRow(panel, "");
    createButton("Reset").parent(row).mousePressed(fullReset);
  }
}

function readControls() {
  let sidx = speedSlider.value();
  if (sidx !== speedIndex) {
    speedIndex = sidx;
    applySpeed(); // cosmetic — retimes the phases, keeps the accumulated heatmap
  }
  let idx = riskSlider.value();
  if (idx !== riskIndex) {
    riskIndex = idx;
    alphaValue = RISK[idx].alpha;
    if (example_agent) example_agent.alpha = alphaValue;
    fullReset(); // one risk regime per accumulated heatmap
  }
  let sig = sigmaSlider.value();
  if (sig !== sigmaValue) {
    sigmaValue = sig;
    fullReset(); // one sigma regime per accumulated heatmap
  }
  riskLabel.html(
    "&alpha; = " + alphaValue + " (" + RISK[riskIndex].label + ") &middot; " +
    "&sigma; = " + sigmaValue.toFixed(2) + " &mdash; " +
    runsRecorded + (runsRecorded === 1 ? " run recorded" : " runs recorded"));
}

// rebuild the network for a fresh run; KEEP the accumulated counts
function resetSim() {
  agents = [];
  for (let i = 0; i < num_agents; i++) agents[i] = new Agent(i);
  for (let i = 0; i < num_agents; i++) agents[i].connectAgents(agents);
  num_know_b = 0;
  num_produced_b = 0;
  phaseFrame = 0;
  timestep = 0;
  pulses = [];
  // seed one random agent with the novel behavior to kick off the diffusion
  agents[floor(random(0, num_agents))].seeded();
}

// full wipe: fresh run AND empty heatmap (Reset button / risk change)
function fullReset() {
  resetSim();
  counts = makeGrid(() => 0);
  cellOffsets = makeGrid(() => ({ x: 0, y: 0 }));
  maxCount = 0;
  runsRecorded = 0;
}

// fold a completed run's 16 agents into the heatmap
function recordRun() {
  for (let a of agents) {
    let i = a.learned_b - 1;  // order of acquisition
    let j = a.produced_b - 1; // order of production
    if (i >= 0 && j >= 0 && i < num_agents && j < num_agents) {
      counts[i][j] += 1;
      if (counts[i][j] > maxCount) maxCount = counts[i][j];
    }
  }
  runsRecorded += 1;
}

function setup() {
  let wrapper = createDiv().id("viz-wrapper").parent("sketch-holder");

  cnv = createCanvas(W, H);
  cnv.parent(wrapper);
  colorMode(HSL);
  frameRate(30);
  applySpeed(); // initialise phase lengths from the default speed

  let panel = createDiv().id("control-panel").parent(wrapper);
  buildControls(panel);

  for (let i = 0; i < num_agents; i++) agents[i] = new Agent(i);
  for (let i = 0; i < num_agents; i++) agents[i].connectAgents(agents);
  agents[floor(random(0, num_agents))].seeded(); // kick off the diffusion

  // decorative "anatomy of an agent" figure on the left — not part of the sim
  example_agent = new Agent(25);
  example_agent.tracked = false;
  example_agent.position.x = 140; // centered on the heatmap box (x0 + boxW/2)
  example_agent.position.y = 152;
  example_agent.repertoire = [
    { name: "a", Evec: 0, Ivec: 0, Svec: 0, Pvec: 0 },
    { name: "b", Evec: 0, Ivec: 0, Svec: 0, Pvec: 0 },
  ];
  example_agent.r_const = 100;
  // seed the decorative agent's production probabilities (it has no neighbors, so its
  // social memory stays empty and Pvec = Ivec forever — this just avoids a NaN first draw)
  example_agent.Ivec_update();
  example_agent.Svec_update();
  example_agent.Pvec_update();

  counts = makeGrid(() => 0);
  cellOffsets = makeGrid(() => ({ x: 0, y: 0 }));
}

let legendJit = { fillx: 0, filly: 0, a1x: 0, a1y: 0, a2x: 0, a2y: 0 };

function drawExampleLegend() {
  example_agent.draw(); // static "anatomy" diagram — do not moveParticle (keeps it centered)
  if (frameCount % CYCLE == T_FRAMES) example_agent.production_submodel(); // once per timestep

  // refresh the decorative text jitter every other frame -> ~50% slower shake
  if (frameCount % 2 === 0) {
    legendJit = {
      fillx: random(-.5, .5), filly: random(-.5, .5),
      a1x:   random(-.5, .5), a1y:   random(-.5, .5),
      a2x:   random(-.5, .5), a2y:   random(-.5, .5),
    };
  }

  // all labels share the agent's center x so they stack on one vertical axis
  push();
  textAlign(CENTER, CENTER);
  let cx = example_agent.position.x, cy = example_agent.position.y;

  strokeWeight(.5);

  // title: two overlapping orange shades give it a subtle shimmer
  textSize(30);
  stroke(...mar_orange2);
  fill(...mar_orange2);
  text("agent", cx + legendJit.a1x, cy - 100 + legendJit.a1y);
  stroke(...mar_orange);
  fill(...mar_orange);
  text("agent", cx + legendJit.a2x, cy - 100 + legendJit.a2y);

  // arc label (above the circle)
  textSize(15);
  stroke(...mar_black);
  fill(...mar_black);
  text("arc:", cx, cy - 75);
  text("behavioural production", cx, cy - 60);

  // fill label (over the circle center)
  text("fill: repertoire", cx + legendJit.fillx, cy + 5 + legendJit.filly);

  // repertoire key (below the circle) — novel above established to match the split circle
  stroke(...mar_green);
  fill(...mar_green);
  text("novel behaviour", cx, cy + 75);
  stroke(...mar_red);
  fill(...mar_red);
  text("established behaviour", cx, cy + 90);
  pop();
}

// --- timestep phases ---------------------------------------------------------
// Transmission: naive agents attempt to acquire "b" once. The pulse visual is
// no longer spawned here — it fires from an agent that actually demonstrates
// "b" during the production phase (that is the observation opportunity).
function startTransmission() {
  for (let i = 0; i < num_agents; i++) {
    let a = agents[i];
    if (a.repertoire.length >= 2) continue; // already knows the novel behavior
    a.acquire("b");
  }
}

// Production: freeze an x-snapshot and assign each agent a fire frame so the
// arcs sweep left->right across the network.
function startProduction() {
  let xmin = Infinity, xmax = -Infinity;
  for (let i = 0; i < num_agents; i++) {
    xmin = min(xmin, agents[i].position.x);
    xmax = max(xmax, agents[i].position.x);
  }
  let span = xmax - xmin || 1; // guard against all-equal x
  for (let i = 0; i < num_agents; i++) {
    let t = (agents[i].position.x - xmin) / span;
    agents[i].fireFrame = T_FRAMES + round(t * FIRE_SPAN);
    agents[i].firedThisStep = false;
  }
}

// Green glow traveling from a demonstrator to each naive neighbor. Each pulse
// carries its own startFrame, so it animates independently of the phase clock
// and expires after PULSE_FRAMES.
function updateAndDrawPulses() {
  push();
  noStroke();
  for (let k = pulses.length - 1; k >= 0; k--) {
    let pulse = pulses[k];
    let p = (frameCount - pulse.startFrame) / PULSE_FRAMES;
    if (p >= 1) { pulses.splice(k, 1); continue; }
    let from = agents[pulse.fromID].position;
    let to = agents[pulse.toID].position;
    let x = lerp(from.x, to.x, p);
    let y = lerp(from.y, to.y, p);
    fill(...mar_green, 0.35);
    circle(x, y, 12);
    fill(...mar_green, 0.9);
    circle(x, y, 6);
  }
  pop();
}

function draw() {
  readControls();
  background(...mar_blue2);

  drawExampleLegend();

  // advance the timestep clock -----------------------------------------------
  if (phaseFrame >= CYCLE) {
    phaseFrame = 0;
    timestep++;
    // housekeeping: fold this timestep's observations into social memory and refresh the
    // production probabilities from it. Svec reads consolidated memory,
    // Pvec reads Svec. Runs before record/reset so next ts has a valid Pvec.
    for (let i = 0; i < num_agents; i++) {
      let a = agents[i];
      a.consolidate_social_memory();
      a.Ivec_update();
      a.Svec_update();
      a.Pvec_update();
    }
    // run ends once every agent has produced the novel behavior at least once,
    // checked here at the timestep boundary so the final strum plays out fully
    if (num_produced_b >= num_agents) { recordRun(); resetSim(); }
    else if (timestep >= MAX_TIMESTEPS) resetSim(); // stalled (e.g. isolated node) — don't record
  }
  if (phaseFrame === 0)        startTransmission();
  if (phaseFrame === T_FRAMES) startProduction();
  let inProduction = phaseFrame >= T_FRAMES;

  // network simulation --------------------------------------------------------
  for (let i = 0; i < num_agents; i++) agents[i].drawEdges(agents);

  updateAndDrawPulses();

  for (let i = 0; i < num_agents; i++) {
    let a = agents[i];
    if (inProduction && !a.firedThisStep && phaseFrame === a.fireFrame) {
      let produced = a.production_submodel();
      a.firedThisStep = true;
      // neighbors observe this production (both behaviors) into their social memory;
      // only behaviors they already know are retained (Agent.observe). Fed into
      // Svec/Pvec at the next timestep boundary, never this same strum.
      a.neighbor_vector.forEach(nID => agents[nID].observe(produced));
      // an actual demonstration of the novel behavior is an observation
      // opportunity: send a pulse to each naive neighbor
      if (produced === "b") {
        a.neighbor_vector.forEach(nID => {
          if (agents[nID].repertoire.length < 2) {
            pulses.push({ fromID: a.ID, toID: nID, startFrame: frameCount });
          }
        });
      }
    }
    a.draw();
    a.moveParticle();
  }

  phaseFrame++;

  drawHeatmap();
}

function drawHeatmap() {
  const x0 = 50, y0 = 282, boxW = 180, boxH = 180; // left quarter, centered under the example agent
  const cell = boxW / num_agents;

  push();
  rectMode(CORNER);

  // 80%-opacity white panel over the network region
  stroke(...mar_black);
  strokeWeight(1);
  fill(0, 0, 100, 0.8);
  rect(x0, y0, boxW, boxH);

  // populated cells: theme-red ramp + a little brownian jiggle
  noStroke();
  for (let i = 0; i < num_agents; i++) {
    for (let j = 0; j < num_agents; j++) {
      let c = counts[i][j];
      if (c <= 0) continue;

      let t = maxCount > 0 ? Math.sqrt(c / maxCount) : 0;

      let off = cellOffsets[i][j];
      off.x = constrain(off.x + random(-0.25, 0.25), -1.5, 1.5);
      off.y = constrain(off.y + random(-0.25, 0.25), -1.5, 1.5);

      let cx = x0 + i * cell + off.x;                     // x = order of acquisition
      let cy = y0 + (num_agents - 1 - j) * cell + off.y;  // y = order of production (inverted)

      fill(mar_red2[0], lerp(20, 93, t), lerp(92, 50, t));
      rect(cx, cy, cell - 1, cell - 1);
    }
  }

  // axes (black) ------------------------------------------------------------
  fill(...mar_black);
  noStroke();

  textAlign(CENTER, TOP);
  textSize(14);
  [1, 4, 8, 12, 16].forEach(v => {
    text(v, x0 + (v - 0.5) * cell, y0 + boxH + 3);
  });

  textAlign(RIGHT, CENTER);
  [1, 4, 8, 12, 16].forEach(v => {
    text(v, x0 - 4, y0 + (num_agents - v + 0.5) * cell);
  });

  textAlign(CENTER, TOP);
  textSize(15);
  text("order of acquisition", x0 + boxW / 2, y0 + boxH + 21);

  push();
  translate(x0 - 30, y0 + boxH / 2);
  rotate(-HALF_PI);
  textAlign(CENTER, BOTTOM);
  text("order of production", 0, 0);
  pop();

  textAlign(CENTER, BOTTOM);
  textSize(14);
  text("Oa vs Op (" + runsRecorded + (runsRecorded === 1 ? " run)" : " runs)"),
       x0 + boxW / 2, y0 - 6);

  pop();
}
