# Learning Log

## Phase 0: understanding the raw data
- SECOM has 1,567 units x 590 sensors, plus a separate labels file
  (-1 = pass, 1 = fail, with a timestamp). The two files share no ID, so
  they are linked only by row order.
- Only 104 units fail (6.6%), so accuracy is misleading: always predicting
  "pass" scores 93.4% and catches no failures.
- 4.5% of sensor cells are missing (41,951 of 924,530), but unevenly: most
  sensors are nearly complete, a few are over 90% empty.
- `.mean()` of a True/False column is the fraction of True, which is how I
  get the failure rate and the missing rate per sensor.
- Bug I hit: `DataFrame.value_counts()` on the sensor table returned an empty
  result, because it counts whole identical rows and skips any row containing
  NaN. Check which variable you are looking at before trusting a result.

## Phase 1: data loading and cleaning
- Joined sensors and labels by row order; sort only after joining.
- Bug I hit: copied text timestamps before converting, so the sort was alphabetical
  while `is_monotonic_increasing` still said True. Look at the data, not the check.
- Used the median to fill missing values because it is not affected by outliers.
- Constant (zero-variance) sensors are dropped because many calculations divide by
  the standard deviation, including Cp/Cpk. The result would be infinite or NaN.
- Fitting medians on all rows leaks future information into training: the test rows
  influence the values used to fill the training data. Rules should be learned from
  the training period only.
- Bug I hit: mixed up `df` and `df_clean`. The shape was right (444 columns) but
  8,008 missing values were left. Check the content, not only the shape.
- Missingness can carry information: sensors 72, 73 and 345 are missing on about 35% of
  failing units but 52% of passing units. Dropping them throws that signal away.

## Phase 1b: train/test split
- Split by position after sorting, never randomly: a random split puts future
  units in training and near-twin neighbours across the split, so scores look
  better than they will in real use.
- 32 timestamps are shared by 65 different units (none are copies), so do not
  deduplicate by timestamp. The cut must not fall between two tied rows, so the
  function checks it.
- Test set is locked until the end: no exploration, no threshold or feature
  choices, no repeated peeking. With only 26 test failures, recall estimates
  are wide (about 32%-68% for a true 50%).
- Bug I hit: `load_secom(raw_dir)` accepted an argument but hardcoded the path,
  so it only worked from one folder. A parameter that is ignored is worse than
  none. Also a typo in a column name (`"timetamp"`) crashed the split: run the code.

## Phase 2: leakage-safe cleaning
- Cleaning is split into two functions: `fit_cleaning(train)` learns the rules
  from the training period only, and `apply_cleaning(df, rules)` applies those
  fixed rules to any table without recomputing anything.
- The rules store the sensors to keep (not to drop), so train and test always
  come out with the same columns. A new or unknown sensor is ignored on purpose.
- The rules also store the train median of each kept sensor. Test gaps are filled
  with the train median, never the test median.
- If a sensor the rules need is missing from the input, `apply_cleaning` raises an
  error instead of quietly filling the gap.
- Wrote the tests before the code (test-first): watch them fail, then write the
  code until they pass. Tiny tables with known answers, plus one boundary sensor at
  exactly 20% missing to catch `<` vs `<=` mistakes.
- Result on SECOM: train-only rules drop 52 sensors for missing (baseline 32) and
  122 as constant (baseline 116), keeping 416 (baseline 442). The rules depend on
  how the data is distributed over time, not only on the number of rows.
- 20 sensors (546-557 and 562-569) are about 24% missing in train but about 1% in
  test. The baseline kept them only because it averaged in the later, complete data.
- Learned to read list comprehensions: `[c for c in keep if c not in df.columns]`
  is a short `for` loop with an `if` that builds a list.
- Bug I hit: wrote `>= 0.2` and `== 1` when counting drops, instead of `> 0.2` and
  `<= 1`. The numbers happened to be right, because no sensor sits exactly on those
  boundaries in this data. Boundaries must match the function, even when the
  output looks correct.