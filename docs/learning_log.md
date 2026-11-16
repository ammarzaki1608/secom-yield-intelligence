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
