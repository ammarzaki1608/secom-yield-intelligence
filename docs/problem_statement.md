# Problem Statement

> Short version. The cost assumption will be worked out in Phase 6 (modelling).

## The problem

In semiconductor manufacturing, each unit passes through hundreds of measured process
steps, but failures are only confirmed at final test. If unusual process behaviour can be
spotted earlier, fewer bad units travel further down the line.

SECOM has 1,567 units with 590 anonymous sensor readings each; 104 (6.6%) failed. The
sensors have no names, units or spec limits, so results can point to a sensor number
but not say what it physically measures.

## What this project does

1. Checks whether the process is stable over time (SPC control charts)
2. Estimates how much room the process has relative to proxy spec limits, since no
   real limits are available (process capability)
3. Looks for sensors whose behaviour differs between failing and passing units
4. Predicts failures using only information available before final test, and checks
   that the sensors the model relies on are consistent with steps 1-3

## Errors and their cost (open question)

A missed failure (a bad unit passed as good) is likely more costly than a false alarm
(a good unit sent for extra checks), but SECOM gives no cost data. Phase 6 will choose
the decision threshold under a range of stated cost ratios and show how much the result
depends on that choice, rather than relying on one assumed number.