// ---------------------------------------------------------------------------
// Color themes
// Each theme defines named color *roles* (RGB, alpha optional) so the rest of
// the sketch never references a raw palette color. Switching the theme just
// changes `currentTheme` and everything recolors on the next frame.
// ---------------------------------------------------------------------------
const THEMES = {
  mariana: {
    label: "Mariana",
    background: [216, 222, 233],
    nodeNaive: [78, 90, 101],
    nodeKnow: [236, 95, 102],
    edge: [255, 255, 255, 140],
    graphBoxFill: [255, 255, 255, 190],
    graphBoxStroke: [0, 0, 0],
    graphLine: [153, 199, 148],
  },
  "gruvbox-dark": {
    label: "Gruvbox dark",
    background: [40, 40, 40],
    nodeNaive: [131, 165, 152],
    nodeKnow: [251, 73, 52],
    edge: [124, 111, 100, 170],
    graphBoxFill: [60, 56, 54, 210],
    graphBoxStroke: [235, 219, 178],
    graphLine: [184, 187, 38],
  },
  "gruvbox-light": {
    label: "Gruvbox light",
    background: [251, 241, 199],
    nodeNaive: [69, 133, 136],
    nodeKnow: [204, 36, 29],
    edge: [213, 196, 161, 200],
    graphBoxFill: [235, 219, 178, 210],
    graphBoxStroke: [60, 56, 54],
    graphLine: [152, 151, 26],
  },
  "solarized-dark": {
    label: "Solarized dark",
    background: [0, 43, 54],
    nodeNaive: [252, 215, 87],
    nodeKnow: [220, 50, 47],
    edge: [238, 232, 213, 150],
    graphBoxFill: [7, 54, 66, 210],
    graphBoxStroke: [147, 161, 161],
    graphLine: [133, 153, 0],
  },
  "solarized-light": {
    label: "Solarized light",
    background: [253, 246, 227],
    nodeNaive: [0, 43, 54],
    nodeKnow: [220, 50, 47],
    edge: [147, 161, 161, 160],
    graphBoxFill: [238, 232, 213, 210],
    graphBoxStroke: [88, 110, 117],
    graphLine: [133, 153, 0],
  },
};

let currentTheme = "solarized-light";
const T = () => THEMES[currentTheme];

// ---------------------------------------------------------------------------
// Learning parameters (mutable via the control panel)
// ---------------------------------------------------------------------------
const LAMBDA_STEPS = [0.0001, 0.001, 0.005, 0.01, 0.1];
let lambda_0 = 0.001; // index 1 of LAMBDA_STEPS
let s = 3;
let asoc = 0;

// higher = longer trails. drives the
// alpha of the per-frame background redraw.
let trails = 120;
let fadeAlpha = 255 - trails;

class Agent {
  constructor(ID) {
    this.ID = ID;
    this.position = new p5.Vector(random(0, width), random(0, height));
    this.velocity = p5.Vector.random2D().mult(0.5);
    this.r_const = random(3, 11);
    this.r = this.r_const;
    this.m = this.r * 0.1;
    this.neighbor_vector = [];
    this.naive = true;
  }

  // is the particle clicked
  clicked(m_x, m_y) {
    let d = dist(m_x, m_y, this.position.x, this.position.y);
    if (d < this.r) {
      this.naive = false;
      num_know += 1;
    }
  }

  seeded() {
    this.naive = false;
    num_know += 1;
  }

  acquire() {
    //sum knowledgable neighbors
    let sum_knowledgable = 0;
    this.neighbor_vector.forEach((element) => {
      if (agents[element].naive === false) {
        sum_knowledgable += 1;
      }
    });

    let taizt = 0;

    if (rule === "sum") {
      taizt = sum_knowledgable;
    } else if (rule === "threshold") {
      let threshold = 5;
      let sharpness = 3;

      let term1 = 1 / (1 - 1 / (1 + exp(threshold * sharpness)));
      let term2 = 1 / (1 + exp(-sharpness * (sum_knowledgable - threshold)));
      let term3 = 1 / (1 + exp(threshold * sharpness));

      taizt = term1 * (term2 - term3);
    } else if (rule === "proportional") {
      taizt = sum_knowledgable / this.neighbor_vector.length;
    } else if (rule === "conformity") {
      let f = 2;
      let sum_naive = this.neighbor_vector.length - sum_knowledgable;
      taizt = sum_knowledgable ** f * (1 / (sum_knowledgable ** f + sum_naive ** f));
    } else if (rule === "anti-conformity") {
      let f = .75;
      let sum_naive = this.neighbor_vector.length - sum_knowledgable;
      taizt = sum_knowledgable ** f * (1 / (sum_knowledgable ** f + sum_naive ** f));
    }

    //base_rate * (s * num_know + asocial learning)
    let acq_rate = lambda_0 * (asoc + s * taizt);

    let acq_prob = 1 - exp(-acq_rate);

    let dice_roll = random(0, 1);

    if (dice_roll < acq_prob) {
      this.naive = false;
      num_know += 1;
    }
  }

  // Custom method for updating the variables
  moveParticle() {
    if (this.position.x < 0 || this.position.x > width) this.velocity.x *= -1;
    if (this.position.y < 0 || this.position.y > height) this.velocity.y *= -1;
  }

  // this function creates the connections(lines)
  // between particles which are less than a certain distance apart
  connectAgents(agents) {
    agents.forEach((element) => {
      if (this.ID !== element.ID) {
        let dist_ij = dist(this.position.x, this.position.y, element.position.x, element.position.y);
        if (dist_ij < max_dist) {
          stroke(...T().edge);
          strokeWeight(sqrt(1 / dist_ij) * 5);
          line(this.position.x, this.position.y, element.position.x, element.position.y);
          if (!this.neighbor_vector.includes(element.ID)) {
            this.neighbor_vector.push(element.ID);
          }
        } else if (dist_ij >= max_dist & this.neighbor_vector.includes(element.ID)) {
          this.neighbor_vector = this.neighbor_vector.filter((v) => v != element.ID);
        }
      }
    });
  }

  draw() {
    this.position.add(this.velocity);
    noStroke();
    fill(...(this.naive ? T().nodeNaive : T().nodeKnow));
    circle(this.position.x, this.position.y, this.r);
  }
}

let num_agents = 200;
let agents = [];
var cnv;

// transmission dynamics: internal rule value -> user-facing label (menu order)
const RULES = [
  { value: "sum", label: "simple: additive" },
  { value: "proportional", label: "complex: proportional" },
  { value: "conformity", label: "complex: conformity" },
  { value: "anti-conformity", label: "complex: anti-conformity" },
  { value: "threshold", label: "complex: threshold" },
];
let rule = "sum";
let num_know = 0;
let simFrame = 0; // frames since last reset (frameCount can't be reset)
let hintAgentID = 0; // agent that carries the "click me!" prompt while still naive
let hintDismissed = false; // true once the user has clicked (seeded) any node

let data_y = [];
let data_x = [];

const W = 500;
const H = 500;

// DOM references, filled in during setup
let themeSel, ruleSel, trailsSlider, lambdaSlider, sSlider, asocSlider;
let trailsVal, lambdaVal, sVal, asocVal;

function mousePressed() {
  let before = num_know;
  for (let i = 0; i < num_agents; i++) {
    agents[i].clicked(mouseX, mouseY);
  }
  if (num_know > before) hintDismissed = true; // a click landed on a node
}

function resetSim() {
  agents = [];
  for (let i = 0; i < num_agents; i++) {
    agents[i] = new Agent(i);
  }
  num_know = 0;
  simFrame = 0;
  data_x = [];
  data_y = [];
  hintAgentID = floor(random(num_agents));
  hintDismissed = false;
}

// --- control panel -----------------------------------------------------------
function addControlRow(panel, labelText) {
  let row = createDiv().class("control-row").parent(panel);
  let label = createElement("label", labelText).parent(row);
  return { row, label };
}

function buildControls(panel) {
  // theme dropdown
  {
    let { row } = addControlRow(panel, "color theme");
    themeSel = createSelect().parent(row);
    for (const id in THEMES) themeSel.option(THEMES[id].label, id);
    themeSel.selected(currentTheme);
    themeSel.changed(() => (currentTheme = themeSel.value()));
  }

  // transmission dynamic dropdown
  {
    let { row } = addControlRow(panel, "transmission dynamic");
    ruleSel = createSelect().parent(row);
    RULES.forEach((r) => ruleSel.option(r.label, r.value));
    ruleSel.selected(rule);
    ruleSel.changed(() => (rule = ruleSel.value()));
  }

  // trails (higher = longer trails)
  {
    let { row, label } = addControlRow(panel, "");
    trailsVal = label;
    trailsSlider = createSlider(0, 250, trails, 1).parent(row);
  }

  // lambda_0 — discrete/notched (index into LAMBDA_STEPS)
  {
    let { row, label } = addControlRow(panel, "");
    lambdaVal = label;
    lambdaSlider = createSlider(0, LAMBDA_STEPS.length - 1, LAMBDA_STEPS.indexOf(lambda_0), 1).parent(row);
  }

  // s — social transmission strength
  {
    let { row, label } = addControlRow(panel, "");
    sVal = label;
    sSlider = createSlider(0, 5, s, 0.1).parent(row);
  }

  // asoc — asocial learning on/off
  {
    let { row, label } = addControlRow(panel, "");
    asocVal = label;
    asocSlider = createSlider(0, 1, asoc, 1).parent(row);
  }

  // reset button
  {
    let { row } = addControlRow(panel, "");
    createButton("Reset").parent(row).mousePressed(resetSim);
  }
}

function readControls() {
  trails = trailsSlider.value();
  fadeAlpha = 255 - trails; // higher trails => lower repaint alpha => longer trails
  lambda_0 = LAMBDA_STEPS[lambdaSlider.value()];
  s = sSlider.value();
  asoc = asocSlider.value();

  trailsVal.html("trails: " + trails);
  lambdaVal.html("intrinsic rate (&lambda;&#8320;): " + lambda_0);
  sVal.html("social learning strength (s): " + nf(s, 1, 1));
  asocVal.html("asocial learning (0/1): " + asoc);
}

// responsive square canvas: capped at 500 on desktop, shrinks to fit narrow phones
function sketchSize() {
  return Math.floor(Math.min(windowWidth * 0.94, 500));
}

function windowResized() {
  let sz = sketchSize();
  resizeCanvas(sz, sz);
}

function setup() {
  let wrapper = createDiv().id("viz-wrapper").parent("sketch-holder");

  colorMode(RGB, 255, 255, 255, 255);
  let sz = sketchSize();
  cnv = createCanvas(sz, sz);
  cnv.parent(wrapper);
  frameRate(30);

  let panel = createDiv().id("control-panel").parent(wrapper);
  buildControls(panel);

  agent_speed = 1;
  max_dist = 50;

  for (let i = 0; i < num_agents; i++) {
    agents[i] = new Agent(i);
  }
  hintAgentID = floor(random(num_agents));
}

function draw() {
  readControls();

  // scale factor vs. the original 500px design so everything stays proportional
  const sc = width / 500;
  max_dist = 50 * sc;

  background(...T().background, fadeAlpha);
  simFrame += 1;

  for (let i = 0; i < num_agents; i++) {
    agents[i].moveParticle();
    agents[i].connectAgents(agents);
  }

  for (let i = 0; i < num_agents; i++) {
    if (agents[i].naive === true) {
      agents[i].acquire();
    }
    agents[i].draw();
  }

  // "click me!" prompt that follows a still-naive agent until it's seeded
  let hint = agents[hintAgentID];
  if (hint && hint.naive && !hintDismissed) {
    noStroke();
    fill(...T().nodeKnow);
    textSize(15 * sc);
    textAlign(LEFT, CENTER);
    text("click us!", hint.position.x + hint.r + 4 * sc, hint.position.y);
  }

  // inset knowledge-over-time graph (coords scaled from the 500px design)
  strokeWeight(0.5);
  stroke(...T().graphBoxStroke);
  fill(...T().graphBoxFill);
  rect(25 * sc, 430 * sc, 135 * sc, 65 * sc); // rect(top left x, top left y, width, height)

  data_y.push(num_know);
  data_x.push(simFrame);

  if (simFrame > 1) {
    for (let i = 0; i < data_y.length; i++) {
      let y1 = (485 - (data_y[i] / num_agents) * 50) * sc;
      let y2 = (485 - (data_y[i + 1] / num_agents) * 50) * sc;
      let x1 = ((data_x[i] / simFrame) * 125 + 30) * sc;
      let x2 = ((data_x[i + 1] / simFrame) * 125 + 30) * sc;

      // vertical lines that fill the area under the curve — same green as the line
      stroke(...T().graphLine);
      strokeWeight(0.2);
      line(x1, 490 * sc, x1, y1 + 2 * sc);

      // polyline
      strokeWeight(2 * sc);
      stroke(...T().graphLine);
      line(x1, y1, x2, y2);
    }
  }
}
