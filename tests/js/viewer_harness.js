// Runs the viewer's JavaScript against a stub 3Dmol and DOM.
//
// The page's logic is not decoration: it decides which atoms are labelled,
// which can be clicked, and whether a drag pans or rotates. Asserting that the
// generated HTML *contains* a function name proves none of that. So the real
// code is executed here against a stub renderer and the decisions are checked.
//
// Driven by tests/test_mol3d_viewer.py, which generates the page first.
// Reads the extracted script from the path given as argv[2].
const calls = {labels: [], styles: [], zoomTo: [], translate: [], render: 0, bind: 0, highlights: [], opts: []};
const ATOMS = [0,1,2,3].map(s => ({serial:s, x:s, y:0, z:0, elem:'C'}));

global.document = {
  getElementById: (id) => ({ id, style:{}, innerHTML:'',
    addEventListener: (type, fn, cap) => { (listeners[type] ||= []).push(fn); } })
};
const listeners = {};
const model = { selectedAtoms: (sel) => {
    if (!sel || Object.keys(sel).length === 0) return ATOMS;
    const want = Array.isArray(sel.serial) ? sel.serial : [sel.serial];
    return ATOMS.filter(a => want.includes(a.serial)); } };
global.$3Dmol = { createViewer: () => VIEW };
const VIEW = {
  addModel(){}, removeAllModels(){}, getModel: () => model,
  setStyle(sel, sty){ calls.styles.push([sel, sty]); },
  addStyle(sel, sty){ calls.highlights.push([sel, sty]); },
  removeAllLabels(){ calls.labels.length = 0; calls.opts.length = 0; },
  addLabel(text, opt){ calls.labels.push(text); calls.opts.push(opt); },
  setClickable(sel, on, fn){ global.__click = fn; calls.bind++; },
  zoomTo(sel){ calls.zoomTo.push(sel); },
  translate(dx, dy){ calls.translate.push([dx, dy]); },
  getView: () => [1,2,3], setView(){}, render(){ calls.render++; },
};
global.QWebChannel = function(){}; global.qt = {webChannelTransport:{}};
global.bridgeCalls = [];

const code = require('fs').readFileSync(process.argv[2], 'utf8');
eval(code);
boot();
bridge = { atom_clicked: (s) => global.bridgeCalls.push(s) };

function check(name, cond, extra='') {
  console.log((cond ? 'PASS  ' : 'FAIL  ') + name + (extra ? '  -> ' + extra : ''));
  if (!cond) process.exitCode = 1;
}

// --- label modes
check('selected mode with nothing picked labels nothing',
      calls.labels.length === 0, JSON.stringify(calls.labels));

// The default is 'clicked': the atom you picked, and nothing else. It used
// to be 'selected' (atom plus neighbours), which meant one click produced
// three or four labels when all you wanted was to read one type.
select(0);
check('the default labels only the atom you clicked',
      calls.labels.length === 1 && calls.labels[0] === 'h0',
      JSON.stringify(calls.labels));

DATA.labelMode = 'selected'; render();
check('clicked + neighbours labels the atom and its VISIBLE neighbours',
      calls.labels.includes('h0') && calls.labels.includes('0')
      && !calls.labels.includes('t4'),
      JSON.stringify(calls.labels));
DATA.labelMode = 'clicked'; render();

DATA.labelMode = 'all'; render();
check('all mode labels every visible atom, still not the hidden one',
      calls.labels.includes('h0') && calls.labels.includes('0')
      && !calls.labels.includes('t4'), JSON.stringify(calls.labels));

DATA.labelMode = 'none'; render();
check('none mode labels nothing', calls.labels.length === 0);

// --- hidden atoms are not clickable
DATA.labelMode = 'selected';
global.bridgeCalls.length = 0;
__click({serial: 2});
check('a hidden atom cannot be clicked', global.bridgeCalls.length === 0);
__click({serial: 1});
check('a visible atom can be clicked', global.bridgeCalls[0] === 1);

// --- panning
DATA.panMode = true;
const down = listeners['mousedown'][0], move = listeners['mousemove'][0];
const ev = (x,y,extra={}) => Object.assign(
  {clientX:x, clientY:y, button:0, shiftKey:false,
   preventDefault(){}, stopPropagation(){}}, extra);
calls.translate.length = 0;
down(ev(100,100)); move(ev(140,90));
check('pan mode drag translates the view',
      calls.translate.length === 1 && calls.translate[0][0] === 40
      && calls.translate[0][1] === -10, JSON.stringify(calls.translate));

listeners['mouseup'][0](ev(140,90));
calls.translate.length = 0;
move(ev(200,200));
check('no translation after the drag ends', calls.translate.length === 0);

DATA.panMode = false;
down(ev(10,10)); move(ev(20,20));
check('without pan mode a plain drag does not translate (it rotates)',
      calls.translate.length === 0);
down(ev(10,10,{shiftKey:true})); move(ev(30,10,{shiftKey:true}));
check('shift-drag translates even with pan mode off',
      calls.translate.length === 1 && calls.translate[0][0] === 20);

// --- focusVisible frames only what is shown
calls.zoomTo.length = 0;
focusVisible();
check('fit frames only the visible atoms',
      JSON.stringify(calls.zoomTo[0]) === JSON.stringify({serial:[0,1,3]}),
      JSON.stringify(calls.zoomTo));

// --- selection is dropped if its atom gets hidden
select(1);
apply(JSON.stringify(Object.assign({}, DATA, {hidden:[1,2]})));
check('selecting then hiding that atom clears the selection', selected === null);

// --- REGRESSION: clicking died after the first state update.
//
// 3Dmol binds clickability to the atoms of the model present when
// setClickable runs. apply() swaps the model, so binding once in boot() left
// every atom unclickable from the first update onward. It went unnoticed
// because the old code reloaded the whole page for every change, re-running
// boot() each time.
calls.bind = 0;
apply(JSON.stringify(Object.assign({}, DATA, {hidden: []})));
check('clicks are re-bound when the model is replaced', calls.bind === 1,
      'setClickable called ' + calls.bind + ' times');

global.bridgeCalls.length = 0;
__click({serial: 3});
check('an atom is still clickable after a state update',
      global.bridgeCalls[0] === 3 && selected === 3,
      'bridge got ' + JSON.stringify(global.bridgeCalls));

// --- REGRESSION: pan mode swallowed plain clicks, so nothing could be
// selected while it was on.
DATA.panMode = true;
calls.translate.length = 0;
global.bridgeCalls.length = 0;
var swallowed = false;
var clickEv = (x, y) => ({clientX:x, clientY:y, button:0, shiftKey:false,
                          preventDefault(){ swallowed = true; },
                          stopPropagation(){ swallowed = true; }});
down(clickEv(50, 50));
check('a press in pan mode is not swallowed, so it can still be a click',
      swallowed === false);
listeners['mouseup'][0](clickEv(50, 50));
check('releasing without moving is not swallowed either', swallowed === false);

// A press that moves IS a pan, and its release must not also select an atom.
swallowed = false;
down(clickEv(50, 50));
move(clickEv(90, 50));
check('moving past the slop threshold pans', calls.translate.length === 1);
swallowed = false;
listeners['mouseup'][0](clickEv(90, 50));
check('the end of a pan is swallowed, so it does not select an atom',
      swallowed === true);

// A tiny tremor must not count as a drag.
calls.translate.length = 0;
down(clickEv(10, 10));
move(clickEv(11, 11));
check('a 2px tremor is a click, not a pan', calls.translate.length === 0);
listeners['mouseup'][0](clickEv(11, 11));

// --- the selected atom is drawn prominently enough to find at a glance
calls.highlights.length = 0;
select(0);
var onSelected = calls.highlights.filter(h => h[0].serial === 0);
var opaque = onSelected.filter(h => h[1].sphere && h[1].sphere.opacity >= 1.0);
var halo = onSelected.filter(h => h[1].sphere && h[1].sphere.opacity < 1.0);
check('the selected atom gets exactly one translucent halo',
      halo.length === 1, JSON.stringify(onSelected));
// A solid ball over the atom reads as an EXTRA ATOM bonded into the chain --
// which is exactly how it was reported. The atom must stay visible through it.
check('nothing opaque is drawn over the selected atom',
      opaque.length === 0, JSON.stringify(opaque));
check('the halo is drawn larger than any normal atom',
      halo[0][1].sphere.scale > 0.30,
      'scale ' + (halo.length ? halo[0][1].sphere.scale : 'n/a'));
// The halo must read as a marker on the atom, not as an atom. Two things
// make that true: it stays close to atom size, and you can see through it.
check('the halo stays close to atom size rather than swamping it',
      halo[0][1].sphere.scale <= 0.40,
      'scale ' + halo[0][1].sphere.scale);
check('the halo is see-through',
      halo[0][1].sphere.opacity < 0.6,
      'opacity ' + halo[0][1].sphere.opacity);

// A highlight on a hidden atom would be a marker floating in empty space.
calls.highlights.length = 0;
apply(JSON.stringify(Object.assign({}, DATA, {hidden: [0]})));
check('no highlight is drawn for a selected atom that is now hidden',
      calls.highlights.filter(h => h[0].serial === 0).length === 0);

// ---------------------------------------------------------------- labels
//
// Reported: clicking an atom labelled its neighbours too, the label showed a
// raw internal key like 20000010, every label sat in a white box, and the
// selection highlight was so large it was mistaken for an extra atom.
DATA.labels = {"0": "136", "1": "140", "2": "108"};
DATA.names  = {"0": "C0",  "1": "H1",  "2": "O2"};
DATA.neighbours = {"0": [1, 2], "1": [0]};
DATA.hidden = [];
DATA.labelColour = '#000000';

DATA.labelMode = 'clicked'; DATA.labelContent = 'type';
select(0);
check('clicked mode labels ONLY the atom you clicked',
      calls.labels.length === 1 && calls.labels[0] === '136',
      JSON.stringify(calls.labels));
check('the label has no white panel behind it',
      calls.opts[0].backgroundOpacity === 0.0,
      'backgroundOpacity ' + calls.opts[0].backgroundOpacity);
check('the label uses the chosen colour',
      calls.opts[0].fontColor === '#000000', calls.opts[0].fontColor);

DATA.labelMode = 'selected'; render();
check('clicked + neighbours adds the bonded atoms',
      calls.labels.length === 3, JSON.stringify(calls.labels));

DATA.labelMode = 'clicked'; DATA.labelContent = 'name'; render();
check('the name option shows the atom name, never an internal key',
      calls.labels[0] === 'C0', JSON.stringify(calls.labels));

DATA.labelContent = 'xyz'; render();
check('the xyz option shows coordinates',
      /^-?\d+\.\d\d, -?\d+\.\d\d, -?\d+\.\d\d$/.test(calls.labels[0]),
      JSON.stringify(calls.labels));

DATA.labelContent = 'type'; DATA.labelColour = '#ffffff'; render();
check('a different label colour reaches the renderer',
      calls.opts[0].fontColor === '#ffffff', calls.opts[0].fontColor);

DATA.labelMode = 'none'; render();
check('none still means none', calls.labels.length === 0);
