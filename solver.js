// solver.js
// Breadth-first search solver for Water-Sort bottle puzzles.

function cloneState(state) {
  return state.map((b) => b.slice());
}

export function contentsToStacksTopBottom(contentsTopToBottom, capacity) {
  // contentsTopToBottom: Array of length capacity with color ids or null (top -> bottom)
  // returns: stack array bottom->top
  const stack = [];
  for (let i = capacity - 1; i >= 0; i--) {
    const v = contentsTopToBottom[i];
    if (v === null || v === undefined) continue;
    stack.push(v);
  }
  return stack; // bottom->top
}

export function stacksToContentsTopBottom(stacks, capacity) {
  // stacks: Array<stack bottom->top>
  // returns: Array<Array<color|null>> top->bottom
  const out = [];
  for (const st of stacks) {
    const slots = new Array(capacity).fill(null);
    for (let i = 0; i < st.length; i++) {
      const slotIdx = capacity - 1 - i;
      slots[slotIdx] = st[i];
    }
    out.push(slots);
  }
  return out;
}

function isMonoOrEmpty(stack) {
  if (stack.length <= 1) return true;
  const c = stack[0];
  for (let i = 1; i < stack.length; i++) if (stack[i] !== c) return false;
  return true;
}

export function isGoalStateMono(stacks) {
  for (const st of stacks) {
    if (st.length === 0) continue;
    if (!isMonoOrEmpty(st)) return false;
  }
  return true;
}

function keyOfState(stacks) {
  // stable string key. stacks are arrays of small ints.
  // Example: "1,2,3| |4" etc
  return stacks.map((st) => st.join(',')).join('|');
}

function topColor(stack) {
  if (stack.length === 0) return null;
  return stack[stack.length - 1];
}

function countTopRun(stack) {
  if (stack.length === 0) return 0;
  const c = stack[stack.length - 1];
  let k = 1;
  for (let i = stack.length - 2; i >= 0; i--) {
    if (stack[i] !== c) break;
    k++;
  }
  return k;
}

export function getLegalMoves(stacks, capacity, rockSet) {
  const moves = [];
  const n = stacks.length;

  for (let i = 0; i < n; i++) {
    if (rockSet.has(i)) continue; // cannot pour out
    const src = stacks[i];
    if (src.length === 0) continue;

    const c = topColor(src);
    const run = countTopRun(src);

    for (let j = 0; j < n; j++) {
      if (j === i) continue;
      const dst = stacks[j];
      if (dst.length >= capacity) continue;

      if (dst.length !== 0) {
        const dTop = topColor(dst);
        if (dTop !== c) continue;
      }
      const space = capacity - dst.length;
      const amt = Math.min(space, run);
      if (amt <= 0) continue;

      moves.push({ src: i, dst: j, amt, color: c });
    }
  }
  return moves;
}

export function applyMove(stacks, move) {
  const { src, dst, amt } = move;
  const next = cloneState(stacks);
  const s = next[src];
  const d = next[dst];
  for (let k = 0; k < amt; k++) {
    d.push(s.pop());
  }
  return next;
}

export function solveBfs(initialStacks, capacity, rockSet, opts = {}) {
  const {
    maxStates = 200000,
    goalMode = 'mono', // mono or full (not implemented)
  } = opts;

  if (goalMode !== 'mono') {
    throw new Error(`Unsupported goalMode: ${goalMode}`);
  }

  const startKey = keyOfState(initialStacks);
  const queue = [initialStacks];
  const prev = new Map(); // key -> { prevKey, move }
  prev.set(startKey, null);

  let head = 0;
  let explored = 0;

  while (head < queue.length) {
    const cur = queue[head++];
    explored++;

    if (isGoalStateMono(cur)) {
      // reconstruct moves
      const moves = [];
      let k = keyOfState(cur);
      while (k !== startKey) {
        const rec = prev.get(k);
        if (!rec) break;
        moves.push(rec.move);
        k = rec.prevKey;
      }
      moves.reverse();
      return { moves, explored, solved: true };
    }

    if (explored > maxStates) {
      return { moves: [], explored, solved: false, reason: 'max_states' };
    }

    const curMoves = getLegalMoves(cur, capacity, rockSet);

    for (const mv of curMoves) {
      const nxt = applyMove(cur, mv);
      const nk = keyOfState(nxt);
      if (prev.has(nk)) continue;
      prev.set(nk, { prevKey: keyOfState(cur), move: mv });
      queue.push(nxt);
    }
  }

  return { moves: [], explored, solved: false, reason: 'exhausted' };
}
