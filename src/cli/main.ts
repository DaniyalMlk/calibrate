#!/usr/bin/env node
import { main } from './run.js';

process.exitCode = main(process.argv.slice(2));
