% ============================================================
% MiniZinc Model Template
% Generic structure for optimization problems
% ============================================================

include "globals.mzn";

% ============================================================
% PARAMETERS - Fixed Constants (with values)
% ============================================================

int: constant_int = 10;
float: constant_float = 480.0;

% ============================================================
% PARAMETERS - From Data (no values - provided via .dzn)
% ============================================================

int: param_int;                          % scalar parameter
array[int] of int: param_array;          % array parameter
array[int] of float: param_float_array;  % float array

% Derive sizes from data using length()
int: n_items = length(param_array);
set of int: ITEMS = 1..n_items;

% ============================================================
% DECISION VARIABLES
% ============================================================

array[ITEMS] of var 0..100: decisions;
array[ITEMS] of var bool: selected;

% ============================================================
% CONSTRAINTS
% ============================================================

% Example constraint
constraint forall(i in ITEMS)(
    decisions[i] >= 0
);

% ============================================================
% OBJECTIVE
% ============================================================

var int: objective = sum(i in ITEMS)(decisions[i]);
solve minimize objective;

% ============================================================
% IMPORTANT NOTES
% ============================================================
%
% 1. TYPE CONSISTENCY:
%    - If param is float (e.g., cycle_time), use float for comparisons
%    - WRONG:  sum(...) * float_param <= int_constant  (coercion error)
%    - RIGHT:  sum(...) * float_param <= float_constant
%
% 2. DERIVE SIZES FROM DATA:
%    - Use length() to get array sizes: int: n = length(data_array);
%    - Never hardcode counts
%
% 3. INDEX SETS:
%    - Define: set of int: ITEMS = 1..n;
%    - Use in arrays: array[ITEMS] of var int: x;
%
% 4. FUNCTION SYNTAX:
%    - Parameters need colon after type: function int: f(int: x) = x + 1;
%    - WRONG: function int: f(int x) = x + 1;   (missing colon)
%    - RIGHT: function int: f(int: x) = x + 1;
%
% 5. AVOID abs() IN CONSTRAINTS:
%    - abs() can cause parsing issues with var expressions
%    - Instead, use auxiliary variables with two inequalities:
%      var int: diff;
%      constraint diff >= a - b;
%      constraint diff >= b - a;
%
% 6. BOOLEAN SYNTAX:
%    - MiniZinc uses lowercase: true, false
%    - WRONG: True, False (Python style)
%    - RIGHT: true, false
%
% 7. LOGICAL OPERATORS:
%    - AND: /\  (single backslash, not double)
%    - OR:  \/
%    - NOT: not
%    - WRONG: /\\  (double backslash causes "invalid token" error)
%    - RIGHT: /\
%
