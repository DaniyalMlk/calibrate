# calibrate

An adaptive testing engine built on item response theory.

A fixed-length quiz spends most of its questions telling you what it already
knows: easy items are wasted on strong candidates, hard ones on weak candidates.
An adaptive test picks each next question to maximise what it learns from the
answer, and stops when the estimate is precise enough. `calibrate` implements
that loop — the response models, the ability estimators, the item-selection
policies and the stopping rules — as a dependency-free TypeScript library.

## Status

Early. Phase 1 of [ROADMAP.md](ROADMAP.md) is in progress.

## Install and run

```bash
npm install
npm test        # vitest
npm run typecheck
npm run build
```

## License

MIT
